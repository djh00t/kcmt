from __future__ import annotations

import os
import re
from types import ModuleType
from typing import Any, Callable

import httpx

from kcmt.config import Config
from kcmt.exceptions import LLMError
from kcmt.providers.base import BaseDriver

# Optional dependency: import module, not symbols, for easier test stubbing
_openai: ModuleType | None
try:  # pragma: no cover - optional import
    import openai as _openai
except Exception:  # pragma: no cover
    _openai = None


class OpenAIDriver(BaseDriver):
    """Driver encapsulating OpenAI / OpenAI-compatible chat completions.

    Mirrors prior logic from LLMClient._call_openai plus adaptive retry and
    enrichment triggers. Diff pre-processing and enrichment helpers will
    still live in LLMClient for now (to avoid code duplication across
    drivers) but actual API invocation + model-specific param handling
    reside here.
    """

    def __init__(self, config: Config, debug: bool = False) -> None:
        super().__init__(config, debug)
        timeout_env = os.environ.get("KCMT_LLM_REQUEST_TIMEOUT")
        try:
            self._request_timeout = float(timeout_env) if timeout_env else 60.0
        except ValueError:
            self._request_timeout = 60.0
        # Prefer factory from kcmt.llm so tests can monkeypatch kcmt.llm.OpenAI
        client_factory: Callable[..., Any] | None = None
        try:  # pragma: no cover - relies on package layout at runtime
            from kcmt import llm as _llm_mod

            client_factory = getattr(_llm_mod, "OpenAI", None)
        except Exception:  # pragma: no cover
            client_factory = None

        self._client: Any
        if client_factory is not None:
            try:
                # Keep kwargs minimal to support narrow test doubles
                self._client = client_factory(
                    base_url=config.llm_endpoint,
                    api_key=config.resolve_api_key(),
                )
            except TypeError:
                # Fall back to passing timeout when supported
                self._client = client_factory(
                    base_url=config.llm_endpoint,
                    api_key=config.resolve_api_key(),
                    timeout=self._request_timeout,
                )
        elif _openai is not None:
            self._client = _openai.OpenAI(
                base_url=config.llm_endpoint,
                api_key=config.resolve_api_key(),
                timeout=self._request_timeout,
            )
        else:  # pragma: no cover - missing dependency entirely
            raise LLMError("OpenAI SDK not available")
        max_tokens_env = os.environ.get("KCMT_OPENAI_MAX_TOKENS")
        try:
            self._max_completion_tokens = int(max_tokens_env) if max_tokens_env else 512
        except ValueError:
            self._max_completion_tokens = 512
        self._minimal_prompt = False  # orchestrator can flip; kept for compat

    # The message-building stays orchestrated; we accept already-built messages
    def _invoke(self, messages: list[dict[str, Any]], minimal_ok: bool) -> str:
        max_tokens = getattr(self, "_max_completion_tokens", 512)
        model = self.config.model
        is_gpt5 = model.startswith("gpt-5")
        token_param = "max_completion_tokens" if is_gpt5 else "max_tokens"

        if self.debug:
            print("DEBUG(Driver:OpenAI): invoke")
            print(f"  model={model} token_param={token_param} " f"value={max_tokens}")
            print(f"  minimal_prompt={self._minimal_prompt}")

        # Prepare kwargs; for gpt-5 try once WITHOUT token limit first
        base_kwargs: dict[str, Any] = {
            "messages": messages,
            "model": model,
        }
        if is_gpt5:
            base_kwargs["temperature"] = 1
        else:
            base_kwargs[token_param] = max_tokens

        def _call_with_kwargs(k: dict[str, Any]) -> Any:
            create_fn = self._client.chat.completions.create
            return create_fn(**k)

        try:
            # For gpt-5, first attempt without token limit
            resp = _call_with_kwargs(base_kwargs)
        except Exception as e:  # noqa: BLE001 - broadened to support stubs
            msg = str(e)
            if (not is_gpt5) and "Unsupported parameter" in msg and "max_tokens" in msg:
                # Some servers only accept max_completion_tokens
                if self.debug:
                    print("DEBUG(Driver:OpenAI): fallback to " "max_completion_tokens")
                base_kwargs.pop("max_tokens", None)
                base_kwargs["max_completion_tokens"] = max_tokens
                resp = _call_with_kwargs(base_kwargs)
            else:
                raise LLMError(f"OpenAI client error: {e}") from e
        try:
            choice0 = resp.choices[0]
        except (AttributeError, IndexError):  # pragma: no cover - defensive
            raise LLMError("Missing choices in OpenAI response") from None

        # Extract content robustly: handle string or list-of-fragments
        raw_msg = getattr(choice0, "message", None)
        content: str = ""
        if raw_msg is not None:
            msg_content = getattr(raw_msg, "content", "")
            if isinstance(msg_content, str):
                content = msg_content
            elif isinstance(msg_content, list):  # new SDK list fragments
                fragments: list[str] = []
                for part in msg_content:
                    if isinstance(part, dict):
                        txt = part.get("text") or part.get("content") or ""
                        fragments.append(str(txt))
                    else:  # object with .text maybe
                        txt = getattr(part, "text", "") or getattr(part, "content", "")
                        if txt:
                            fragments.append(str(txt))
                content = "".join(fragments).strip()
        finish_reason = getattr(choice0, "finish_reason", None)

        if self.debug:
            print(
                "DEBUG(Driver:OpenAI): finish_reason={} len={}".format(
                    finish_reason, len(content)
                )
            )
            if not content:
                # Deep diagnostic dump of first choice
                diag = {}
                for attr in [
                    "index",
                    "finish_reason",
                    "logprobs",
                    "message",
                ]:
                    diag[attr] = getattr(choice0, attr, "<missing>")
                print("DEBUG(Driver:OpenAI): empty content diag keys=")
                # Avoid huge output; truncate string repr
                for k, v in diag.items():
                    vs = str(v)
                    if len(vs) > 400:
                        vs = vs[:400] + "…"
                    print(f"  {k}: {vs}")

        # If gpt-5 chat yields empty content, try token-limited retry, then
        # attempt Responses API fallback
        if not content and is_gpt5:
            if self.debug:
                print("DEBUG(Driver:OpenAI): gpt-5 retry with token limit")
            # Retry with explicit token limit first
            retry_kwargs = dict(base_kwargs)
            retry_kwargs["max_completion_tokens"] = max_tokens
            try:
                resp_retry = _call_with_kwargs(retry_kwargs)
                try:
                    choice_r = resp_retry.choices[0]
                    raw_r = getattr(choice_r, "message", None)
                    msg_r = getattr(raw_r, "content", "")
                    if isinstance(msg_r, str):
                        content = msg_r
                except (AttributeError, IndexError, TypeError):
                    pass
            except (ValueError, RuntimeError) as err:
                if self.debug:
                    print("DEBUG(Driver:OpenAI): token-limited retry error " + str(err))
            if not content:
                if self.debug:
                    print("DEBUG(Driver:OpenAI): attempting responses API " "fallback")
            # Combine messages into single prompt (preserve system parts)
            system_parts: list[str] = []
            user_parts: list[str] = []
            for m in messages:
                role = m.get("role")
                c = m.get("content", "")
                if isinstance(c, list):  # unlikely here but be safe
                    c = "\n".join(str(x) for x in c)
                if role == "system":
                    system_parts.append(str(c))
                else:
                    user_parts.append(str(c))
            combined_input = (
                ("\n\n".join(system_parts).strip() + "\n\n") if system_parts else ""
            ) + "\n\n".join(user_parts)
            try:
                resp_create = self._client.responses.create
                resp_alt = resp_create(
                    model=model,
                    input=combined_input,
                    max_output_tokens=max_tokens,
                    temperature=1,
                )
                # Attempt to extract text from various potential attributes
                alt_content = ""
                for attr in ["output_text", "content", "text"]:
                    val = getattr(resp_alt, attr, None)
                    if isinstance(val, str) and val.strip():
                        alt_content = val.strip()
                        break
                if not alt_content:
                    # Newer SDK: .output -> list of content blocks
                    blocks = getattr(resp_alt, "output", None)
                    if isinstance(blocks, list):
                        resp_fragments: list[str] = []
                        for b in blocks:
                            if isinstance(b, dict):
                                txt = (
                                    b.get("text")
                                    or b.get("content")
                                    or b.get("value")
                                    or ""
                                )
                                if txt:
                                    resp_fragments.append(str(txt))
                            else:
                                txt = getattr(b, "text", "") or getattr(
                                    b, "content", ""
                                )
                                if txt:
                                    resp_fragments.append(str(txt))
                        if resp_fragments:
                            alt_content = "".join(resp_fragments).strip()
                if self.debug:
                    prev_len = len(content)
                    print(
                        (
                            "DEBUG(Driver:OpenAI): responses fallback "
                            "prev_len={} new_len={}"
                        ).format(prev_len, len(alt_content))
                    )
                if alt_content:
                    content = alt_content
            except Exception as resp_err:  # noqa: BLE001 - support stubs
                if self.debug:
                    msg = str(resp_err)
                    if len(msg) > 200:
                        msg = msg[:200] + "…"
                    print("DEBUG(Driver:OpenAI): responses fallback error " + msg)

        # Adaptive strategies
        if not content and finish_reason == "length":
            # Path A: non-gpt5 -> signal minimal prompt retry
            if (not is_gpt5) and minimal_ok and not self._minimal_prompt:
                if self.debug:
                    print(
                        "DEBUG(Driver:OpenAI): enabling minimal prompt + "
                        "halving tokens"
                    )
                self._minimal_prompt = True
                self._max_completion_tokens = max(64, max_tokens // 2)
                raise LLMError("RETRY_MINIMAL_PROMPT")
            # Path B: gpt-5 cannot use minimal prompt; reduce tokens & retry
            if is_gpt5:
                if self.debug:
                    print(
                        "DEBUG(Driver:OpenAI): gpt-5 empty length -> "
                        "shrinking tokens and immediate internal retry"
                    )
                reduced = max(64, max_tokens // 2)
                if reduced < max_tokens:
                    self._max_completion_tokens = reduced
                    # Ensure kwargs carry the reduced token param
                    base_kwargs.pop("max_tokens", None)
                    base_kwargs["max_completion_tokens"] = reduced
                    try:
                        resp2 = _call_with_kwargs(base_kwargs)
                        try:
                            choice2 = resp2.choices[0]
                            raw_msg2 = getattr(choice2, "message", None)
                            msg_content2 = getattr(raw_msg2, "content", "")
                            if isinstance(msg_content2, str):
                                content = msg_content2
                            elif isinstance(msg_content2, list):
                                frags2: list[str] = []
                                for part in msg_content2:
                                    if isinstance(part, dict):
                                        frags2.append(
                                            str(
                                                part.get("text")
                                                or part.get("content")
                                                or ""
                                            )
                                        )
                                    else:
                                        frags2.append(
                                            str(
                                                getattr(part, "text", "")
                                                or getattr(part, "content", "")
                                            )
                                        )
                                content = "".join(frags2).strip()
                            if self.debug:
                                print(
                                    "DEBUG(Driver:OpenAI): second attempt "
                                    f"len={len(content)}"
                                )
                        except (AttributeError, TypeError):
                            pass
                    except (ValueError, RuntimeError) as retry_err:
                        if self.debug:
                            print("DEBUG(Driver:OpenAI): retry error " f"{retry_err}")
                        # fall through; content still empty -> final raise

            if not content:
                # Signal orchestrator to rebuild with simplified system prompt
                # for gpt-5 specifically, once.
                if is_gpt5:
                    raise LLMError("RETRY_SIMPLE_PROMPT")
                raise LLMError("Empty OpenAI response")
        return str(content)

    # Public wrapper to avoid accessing a protected member from orchestrator
    def invoke_messages(
        self, messages: list[dict[str, Any]], *, minimal_ok: bool
    ) -> str:
        return self._invoke(messages, minimal_ok)

    def generate(self, diff: str, context: str, style: str) -> str:  # noqa: D401,E501
        # The higher-level orchestration (messages building, sanitation,
        # enrichment) still resides in LLMClient for now. So this driver only
        # provides the raw model text given already-built messages and retry
        # semantics. We expose a thin wrapper so future refactor can migrate
        # more logic here.
        raise LLMError(
            "OpenAIDriver.generate should not be called directly; "
            "LLMClient still orchestrates messaging."
        )

    def list_models(self) -> list[dict[str, object]]:
        """Return filtered/normalized models from `/models`.

                - Excludes models with date-like tokens in their ID
                    (e.g. 2025-08-07 or -1106 suffix).
                - Excludes models whose IDs start with blocked prefixes in
                    DISALLOWED_PREFIXES (plus mapped extras).
        - Normalizes fields:
          - set owned_by="openai"
          - drop object/type
          - convert created -> created_at (ISO 8601 UTC, like Anthropic)
        """
        base = self.config.llm_endpoint.rstrip("/")
        url = f"{base}/models"
        key = self.config.resolve_api_key() or ""
        headers = {"Authorization": f"Bearer {key}"}
        items: list[Any] = []
        try:
            resp = httpx.get(url, headers=headers, timeout=self._request_timeout)
            resp.raise_for_status()
            data = resp.json()
            payload_items = data.get("data") if isinstance(data, dict) else None
            if isinstance(payload_items, list):
                items = payload_items
        except (httpx.HTTPError, ValueError, KeyError):
            # Defer to dataset-based fallback below
            items = []
        out: list[dict[str, object]] = []
        ids: list[str] = []
        for m in items:
            if not isinstance(m, dict):
                continue
            mid_val = m.get("id")
            if not mid_val:
                continue
            mid = str(mid_val)
            # Exclude date-like ids
            if self._DATE_YMD_RE.search(mid) or self._MD_SUFFIX_RE.search(mid):
                continue
            # Exclude disallowed strings anywhere in id
            if self._contains_disallowed_string(mid):
                continue
            entry: dict[str, object] = {"id": mid, "owned_by": "openai"}
            created = m.get("created")
            if isinstance(created, (int, float)):
                try:
                    import datetime as _dt

                    ts = int(created)
                    dt = _dt.datetime.utcfromtimestamp(ts)
                    entry["created_at"] = dt.strftime("%Y-%m-%dT%H:%M:%SZ")
                except (ValueError, OverflowError):
                    pass
            out.append(entry)
            ids.append(mid)
        # If remote listing failed or returned nothing, fall back to dataset
        if not out:
            try:
                try:
                    from kcmt.providers.pricing import build_enrichment_context
                except ImportError as import_err:
                    raise RuntimeError("pricing helper not available") from import_err
                alias_lut, _ctx, _mx = build_enrichment_context()
                seen: set[str] = set()
                for (prov, mid), canon in alias_lut.items():
                    if prov != "openai":
                        continue
                    for mm in (str(canon), str(mid)):
                        if not mm or mm in seen:
                            continue
                        # Apply same filters as remote path
                        if self._DATE_YMD_RE.search(mm) or self._MD_SUFFIX_RE.search(
                            mm
                        ):
                            continue
                        if self._contains_disallowed_string(mm):
                            continue
                        out.append({"id": mm, "owned_by": "openai"})
                        ids.append(mm)
                        seen.add(mm)
                # keep it reasonable
                if len(out) > 200:
                    out = out[:200]
                    ids = ids[:200]
            except (RuntimeError, ValueError, KeyError, TypeError):
                # Leave out empty; enrichment step below will no-op and we
                # will return [] to caller rather than raising.
                pass

        # Enrich with pricing/context/max_output (non-fatal on errors)
        try:
            from kcmt.providers.pricing import enrich_ids as _enrich
        except ImportError:
            return out
        try:
            emap = _enrich("openai", ids)
        except (
            ValueError,
            TypeError,
            KeyError,
            RuntimeError,
            AttributeError,
        ):
            return out
        enriched: list[dict[str, object]] = []
        for item in out:
            mid = str(item.get("id", ""))
            em = emap.get(mid) or {}
            if not em or not em.get("_has_pricing", False):
                if self.debug:
                    print(
                        "DEBUG(Driver:OpenAI): skipping %s due to missing "
                        "pricing" % mid
                    )
                continue
            payload = dict(em)
            payload.pop("_has_pricing", None)
            enriched.append({**item, **payload})
        return enriched

    # Alias/date filters
    _DATE_YMD_RE = re.compile(r"20\d{2}(?:-\d{2}){1,2}")
    _MD_SUFFIX_RE = re.compile(r"-(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])($|[^0-9])")

    # Families/strings to exclude anywhere in the id (wildcards semantics)
    DISALLOWED_STRINGS: list[str] = [
        "ada",
        "babbage",
        "chatgpt",
        "computer-use",
        "dall-e",
        "davinci",
        "gpt-3.5",
        "gpt-4-",
        "gpt-4o",
        "o1",
        "o3",
        "o4",
        "omni",
        "embedding",
        "tts",
        "whisper",
        "gpt-image",
        "gpt-audio",
        "gpt-realtime",
    ]

    @classmethod
    def _is_alias_id(cls, model_id: str) -> bool:
        """Heuristic: treat IDs without date-like tokens as stable aliases.

        Excludes IDs containing:
        - Year-month[-day] like 2025-08-07
        - MonthDay suffixes like -1106 or -0125
        """
        if cls._DATE_YMD_RE.search(model_id):
            return False
        if cls._MD_SUFFIX_RE.search(model_id):
            return False
        return True

    def list_alias_models(self) -> list[dict[str, object]]:
        """Return only alias-style models (no date-like tokens).

        Leaves the payload shape intact: list of dicts with at least 'id'.
        """
        models = self.list_models()
        out: list[dict[str, object]] = []
        for m in models:
            if not isinstance(m, dict):
                continue
            mid = str(m.get("id", ""))
            if not mid:
                continue
            if self._is_alias_id(mid):
                out.append(m)
        return out

    @classmethod
    def _contains_disallowed_string(cls, model_id: str) -> bool:
        """Check if model id contains any disallowed family string.

        Uses substring matching (as if surrounded by wildcards). Includes a
        couple of explicit rules for gpt-4 family aliases.
        """
        # Explicit rules around gpt-4 family: allow gpt-4.1*, but block
        # the bare alias and dash-variants per request.
        if model_id == "gpt-4":
            return True
        if model_id.startswith("gpt-4-"):
            return True
        if model_id.startswith("gpt-4o-"):
            return True
        tokens = list(cls.DISALLOWED_STRINGS) + [
            "text-embedding",  # ensure embedding family caught
        ]
        for token in tokens:
            if token and token in model_id:
                return True
        return False

    def list_filtered_alias_models(self) -> list[dict[str, object]]:
        """Alias models limited to known families/prefixes.

        Combines the date-token alias filter with DISALLOWED_STRINGS.
        """
        alias_models = self.list_alias_models()
        out: list[dict[str, object]] = []
        for m in alias_models:
            mid = str(m.get("id", ""))
            # Keep only those NOT containing disallowed strings
            if not self._contains_disallowed_string(mid):
                out.append(m)
        return out

    @classmethod
    def is_allowed_model_id(cls, model_id: str) -> bool:
        """Public helper to evaluate if a model id should be shown.

            Applies the same rules as list_models():
            - filters out date-like ids (year-month[-day], monthday suffix)
        - filters out ids containing disallowed strings/families
        """
        if not model_id:
            return False
        if cls._DATE_YMD_RE.search(model_id) or cls._MD_SUFFIX_RE.search(model_id):
            return False
        if cls._contains_disallowed_string(model_id):
            return False
        return True
