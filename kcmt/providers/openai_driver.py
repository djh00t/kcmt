from __future__ import annotations

import asyncio
import os
import re
from typing import Any, Callable

import httpx

from kcmt._optional import OpenAIModule, import_openai
from kcmt.config import Config
from kcmt.exceptions import LLMError
from kcmt.providers.base import BaseDriver, resolve_default_request_timeout

# Optional dependency: import module, not symbols, for easier test stubbing
_openai: OpenAIModule | None = import_openai()


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
        provider = getattr(config, "provider", None)
        default_timeout = resolve_default_request_timeout(provider)
        try:
            self._request_timeout = (
                float(timeout_env) if timeout_env else default_timeout
            )
        except ValueError:
            self._request_timeout = default_timeout

        api_key = config.resolve_api_key()

        llm_module = None
        client_factory: Callable[..., Any] | None = None
        try:  # pragma: no cover - relies on package layout at runtime
            from kcmt import llm as _llm_mod

            llm_module = _llm_mod
            client_factory = getattr(_llm_mod, "OpenAI", None)
        except Exception:  # pragma: no cover
            client_factory = None

        def _instantiate(factory: Callable[..., Any]) -> Any:
            base_kwargs = {
                "base_url": config.llm_endpoint,
                "api_key": api_key,
            }
            last_type_error: Exception | None = None
            for include_timeout in (False, True):
                kwargs = dict(base_kwargs)
                if include_timeout:
                    kwargs["timeout"] = self._request_timeout
                try:
                    return factory(**kwargs)
                except TypeError as exc:
                    last_type_error = exc
                    continue
            if last_type_error is not None:
                raise LLMError(
                    "OpenAI client factory rejected provided arguments"
                ) from last_type_error
            raise LLMError("Failed to instantiate OpenAI client")

        self._client: Any
        self._client_async: Any | None = None

        async_factory: Callable[..., Any] | None = None
        if client_factory is not None:
            self._client = _instantiate(client_factory)
            if llm_module is not None:
                async_factory = getattr(llm_module, "AsyncOpenAI", None)
        elif _openai is not None:
            self._client = _instantiate(_openai.OpenAI)
            async_factory = getattr(_openai, "AsyncOpenAI", None)
        else:  # pragma: no cover - missing dependency entirely
            raise LLMError("OpenAI SDK not available")

        if async_factory is not None:
            try:
                self._client_async = _instantiate(async_factory)
            except Exception as exc:  # pragma: no cover - defensive
                if self.debug:
                    msg = str(exc)
                    if len(msg) > 200:
                        msg = msg[:200] + "…"
                    print(
                        "DEBUG(Driver:OpenAI): failed to construct AsyncOpenAI client: "
                        + msg
                    )
                self._client_async = None
        elif self.debug:
            print(
                "DEBUG(Driver:OpenAI): Async client factory not available; using sync fallback"
            )
        max_tokens_env = os.environ.get("KCMT_OPENAI_MAX_TOKENS")
        try:
            self._max_completion_tokens = int(max_tokens_env) if max_tokens_env else 512
        except ValueError:
            self._max_completion_tokens = 512
        self._minimal_prompt = False  # orchestrator can flip; kept for compat

        # Pooled HTTP client for model listing and any direct REST calls
        limits = httpx.Limits(max_connections=60, max_keepalive_connections=30)
        base_url = self.config.llm_endpoint.rstrip("/")
        self._http = httpx.Client(
            base_url=base_url,
            timeout=self._request_timeout,
            http2=True,
            limits=limits,
            headers={
                "Authorization": f"Bearer {api_key}",
            },
        )

    # The message-building stays orchestrated; we accept already-built messages
    def _invoke(
        self,
        messages: list[dict[str, Any]],
        minimal_ok: bool,
        request_timeout: float | None = None,
    ) -> str:
        max_tokens = getattr(self, "_max_completion_tokens", 512)
        model = self.config.model
        is_gpt5 = model.startswith("gpt-5")
        token_param = "max_completion_tokens" if is_gpt5 else "max_tokens"
        timeout_value = request_timeout or self._request_timeout

        if self.debug:
            print("DEBUG(Driver:OpenAI): invoke")
            print(f"  model={model} token_param={token_param} value={max_tokens}")
            print(f"  minimal_prompt={self._minimal_prompt}")
            print(f"  timeout={timeout_value:.2f}s")

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
            call_kwargs = dict(k)
            call_kwargs.setdefault("timeout", timeout_value)
            create_fn = self._client.chat.completions.create
            return create_fn(**call_kwargs)

        try:
            resp = _call_with_kwargs(base_kwargs)
        except Exception as e:  # noqa: BLE001 - broadened to support stubs
            msg = str(e)
            if (not is_gpt5) and "Unsupported parameter" in msg and "max_tokens" in msg:
                if self.debug:
                    print("DEBUG(Driver:OpenAI): fallback to max_completion_tokens")
                base_kwargs.pop("max_tokens", None)
                base_kwargs["max_completion_tokens"] = max_tokens
                resp = _call_with_kwargs(base_kwargs)
            else:
                raise LLMError(f"OpenAI client error: {e}") from e

        choice0 = self._first_choice(resp)
        content, finish_reason = self._extract_choice_content(choice0)

        if self.debug:
            print(
                "DEBUG(Driver:OpenAI): finish_reason={} len={}".format(
                    finish_reason, len(content)
                )
            )
            if not content:
                self._debug_dump_choice(choice0)

        if not content and is_gpt5:
            if self.debug:
                print("DEBUG(Driver:OpenAI): gpt-5 retry with token limit")
            retry_kwargs = dict(base_kwargs)
            retry_kwargs["max_completion_tokens"] = max_tokens
            try:
                resp_retry = _call_with_kwargs(retry_kwargs)
                choice_r = self._first_choice(resp_retry)
                candidate, _finish_retry = self._extract_choice_content(choice_r)
                if candidate:
                    content = candidate
            except Exception as err:  # noqa: BLE001 - broad for stubs
                if self.debug:
                    print("DEBUG(Driver:OpenAI): token-limited retry error " + str(err))

            if not content:
                if self.debug:
                    print("DEBUG(Driver:OpenAI): attempting responses API fallback")
                combined_input = self._combine_messages(messages)
                try:
                    resp_create = getattr(self._client, "responses", None)
                    if resp_create is None:
                        raise AttributeError("responses API not available")
                    resp_alt = resp_create.create(
                        model=model,
                        input=combined_input,
                        max_output_tokens=max_tokens,
                        temperature=1,
                        timeout=timeout_value,
                    )
                    alt_content = self._extract_responses_content(resp_alt)
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

        if not content and finish_reason == "length":
            if (not is_gpt5) and minimal_ok and not self._minimal_prompt:
                if self.debug:
                    print(
                        "DEBUG(Driver:OpenAI): enabling minimal prompt + halving tokens"
                    )
                self._minimal_prompt = True
                self._max_completion_tokens = max(64, max_tokens // 2)
                raise LLMError("RETRY_MINIMAL_PROMPT")
            if is_gpt5:
                if self.debug:
                    print(
                        "DEBUG(Driver:OpenAI): gpt-5 empty length -> "
                        "shrinking tokens and immediate internal retry"
                    )
                reduced = max(64, max_tokens // 2)
                if reduced < max_tokens:
                    self._max_completion_tokens = reduced
                    base_kwargs.pop("max_tokens", None)
                    base_kwargs["max_completion_tokens"] = reduced
                    try:
                        resp2 = _call_with_kwargs(base_kwargs)
                        choice2 = self._first_choice(resp2)
                        candidate2, _finish2 = self._extract_choice_content(choice2)
                        if candidate2:
                            content = candidate2
                        if self.debug:
                            print(
                                "DEBUG(Driver:OpenAI): second attempt "
                                f"len={len(content)}"
                            )
                    except Exception as retry_err:  # noqa: BLE001
                        if self.debug:
                            print(f"DEBUG(Driver:OpenAI): retry error {retry_err}")
        if not content:
            if is_gpt5:
                raise LLMError("RETRY_SIMPLE_PROMPT")
            raise LLMError("Empty OpenAI response")
        return str(content)

    def _first_choice(self, resp: Any) -> Any:
        try:
            return resp.choices[0]
        except (AttributeError, IndexError):  # pragma: no cover - defensive
            raise LLMError("Missing choices in OpenAI response") from None

    def _combine_messages(self, messages: list[dict[str, Any]]) -> str:
        system_parts: list[str] = []
        user_parts: list[str] = []
        for message in messages:
            role = message.get("role")
            content = message.get("content", "")
            if isinstance(content, list):
                content = "\n".join(str(chunk) for chunk in content)
            if role == "system":
                system_parts.append(str(content))
            else:
                user_parts.append(str(content))
        system_block = (
            ("\n\n".join(system_parts).strip() + "\n\n") if system_parts else ""
        )
        return system_block + "\n\n".join(user_parts)

    def _extract_choice_content(self, choice: Any) -> tuple[str, Any]:
        raw_msg = getattr(choice, "message", None)
        if raw_msg is None and isinstance(choice, dict):
            raw_msg = choice.get("message")
        content_field: Any = ""
        if raw_msg is not None:
            content_field = getattr(raw_msg, "content", None)
            if content_field is None and isinstance(raw_msg, dict):
                content_field = raw_msg.get("content")
        content = self._coerce_content_field(content_field)
        finish_reason = getattr(choice, "finish_reason", None)
        if finish_reason is None and isinstance(choice, dict):
            finish_reason = choice.get("finish_reason")
        return content, finish_reason

    def _coerce_content_field(self, content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            fragments: list[str] = []
            for part in content:
                if isinstance(part, dict):
                    txt = (
                        part.get("text")
                        or part.get("content")
                        or part.get("value")
                        or ""
                    )
                else:
                    txt = getattr(part, "text", "") or getattr(part, "content", "")
                if txt:
                    fragments.append(str(txt))
            return "".join(fragments).strip()
        if content is None:
            return ""
        return str(content)

    def _extract_responses_content(self, response: Any) -> str:
        for attr in ("output_text", "content", "text"):
            val = getattr(response, attr, None)
            if isinstance(val, str) and val.strip():
                return val.strip()
        output = getattr(response, "output", None)
        if isinstance(output, list):
            fragments: list[str] = []
            for item in output:
                if isinstance(item, dict):
                    txt = (
                        item.get("text")
                        or item.get("content")
                        or item.get("value")
                        or ""
                    )
                else:
                    txt = getattr(item, "text", "") or getattr(item, "content", "")
                if txt:
                    fragments.append(str(txt))
            if fragments:
                return "".join(fragments).strip()
        return ""

    def _debug_dump_choice(self, choice: Any) -> None:
        if not self.debug:
            return
        diag: dict[str, Any] = {}
        for attr in ["index", "finish_reason", "logprobs", "message"]:
            diag[attr] = getattr(choice, attr, "<missing>")
        print("DEBUG(Driver:OpenAI): empty content diag keys=")
        for key, value in diag.items():
            text = str(value)
            if len(text) > 400:
                text = text[:400] + "…"
            print(f"  {key}: {text}")

    async def _invoke_async(
        self,
        messages: list[dict[str, Any]],
        minimal_ok: bool,
        request_timeout: float | None = None,
    ) -> str:
        if self._client_async is None:
            return await asyncio.to_thread(
                self._invoke,
                messages,
                minimal_ok,
                request_timeout=request_timeout,
            )

        max_tokens = getattr(self, "_max_completion_tokens", 512)
        model = self.config.model
        is_gpt5 = model.startswith("gpt-5")
        token_param = "max_completion_tokens" if is_gpt5 else "max_tokens"
        timeout_value = request_timeout or self._request_timeout

        if self.debug:
            print("DEBUG(Driver:OpenAI): invoke_async")
            print(f"  model={model} token_param={token_param} value={max_tokens}")
            print(f"  minimal_prompt={self._minimal_prompt}")
            print(f"  timeout={timeout_value:.2f}s")

        base_kwargs: dict[str, Any] = {
            "messages": messages,
            "model": model,
        }
        if is_gpt5:
            base_kwargs["temperature"] = 1
        else:
            base_kwargs[token_param] = max_tokens

        async def _call_with_kwargs_async(k: dict[str, Any]) -> Any:
            call_kwargs = dict(k)
            call_kwargs.setdefault("timeout", timeout_value)
            create_fn = self._client_async.chat.completions.create
            return await create_fn(**call_kwargs)

        try:
            resp = await _call_with_kwargs_async(base_kwargs)
        except Exception as e:  # noqa: BLE001 - stubs
            msg = str(e)
            if (not is_gpt5) and "Unsupported parameter" in msg and "max_tokens" in msg:
                if self.debug:
                    print(
                        "DEBUG(Driver:OpenAI): async fallback to max_completion_tokens"
                    )
                base_kwargs.pop("max_tokens", None)
                base_kwargs["max_completion_tokens"] = max_tokens
                resp = await _call_with_kwargs_async(base_kwargs)
            else:
                raise LLMError(f"OpenAI async client error: {e}") from e

        choice0 = self._first_choice(resp)
        content, finish_reason = self._extract_choice_content(choice0)

        if self.debug:
            print(
                "DEBUG(Driver:OpenAI): async finish_reason={} len={}".format(
                    finish_reason, len(content)
                )
            )
            if not content:
                self._debug_dump_choice(choice0)

        if not content and is_gpt5:
            if self.debug:
                print("DEBUG(Driver:OpenAI): async gpt-5 retry with token limit")
            retry_kwargs = dict(base_kwargs)
            retry_kwargs["max_completion_tokens"] = max_tokens
            try:
                resp_retry = await _call_with_kwargs_async(retry_kwargs)
                choice_r = self._first_choice(resp_retry)
                candidate, _finish_retry = self._extract_choice_content(choice_r)
                if candidate:
                    content = candidate
            except Exception as err:  # noqa: BLE001
                if self.debug:
                    print("DEBUG(Driver:OpenAI): async token-limited error " + str(err))

            if not content:
                if self.debug:
                    print("DEBUG(Driver:OpenAI): async attempting responses fallback")
                combined_input = self._combine_messages(messages)
                try:
                    responses_attr = getattr(self._client_async, "responses", None)
                    if responses_attr is None:
                        raise AttributeError(
                            "responses API not available on async client"
                        )
                    resp_alt = await responses_attr.create(
                        model=model,
                        input=combined_input,
                        max_output_tokens=max_tokens,
                        temperature=1,
                        timeout=timeout_value,
                    )
                    alt_content = self._extract_responses_content(resp_alt)
                    if self.debug:
                        prev_len = len(content)
                        print(
                            (
                                "DEBUG(Driver:OpenAI): async responses fallback "
                                "prev_len={} new_len={}"
                            ).format(prev_len, len(alt_content))
                        )
                    if alt_content:
                        content = alt_content
                except Exception as resp_err:  # noqa: BLE001
                    if self.debug:
                        msg = str(resp_err)
                        if len(msg) > 200:
                            msg = msg[:200] + "…"
                        print("DEBUG(Driver:OpenAI): async responses error " + msg)

        if not content and finish_reason == "length":
            if (not is_gpt5) and minimal_ok and not self._minimal_prompt:
                if self.debug:
                    print("DEBUG(Driver:OpenAI): async enabling minimal prompt")
                self._minimal_prompt = True
                self._max_completion_tokens = max(64, max_tokens // 2)
                raise LLMError("RETRY_MINIMAL_PROMPT")
            if is_gpt5:
                if self.debug:
                    print("DEBUG(Driver:OpenAI): async gpt-5 shrink tokens for retry")
                reduced = max(64, max_tokens // 2)
                if reduced < max_tokens:
                    self._max_completion_tokens = reduced
                    base_kwargs.pop("max_tokens", None)
                    base_kwargs["max_completion_tokens"] = reduced
                    try:
                        resp2 = await _call_with_kwargs_async(base_kwargs)
                        choice2 = self._first_choice(resp2)
                        candidate2, _finish2 = self._extract_choice_content(choice2)
                        if candidate2:
                            content = candidate2
                        if self.debug:
                            print(
                                "DEBUG(Driver:OpenAI): async second attempt len={}".format(
                                    len(content)
                                )
                            )
                    except Exception as retry_err:  # noqa: BLE001
                        if self.debug:
                            print(
                                "DEBUG(Driver:OpenAI): async retry error {}".format(
                                    retry_err
                                )
                            )
            if not content:
                if is_gpt5:
                    raise LLMError("RETRY_SIMPLE_PROMPT")
                raise LLMError("Empty OpenAI response")
        return str(content)

    # Public wrapper to avoid accessing a protected member from orchestrator
    def invoke_messages(
        self,
        messages: list[dict[str, Any]],
        *,
        minimal_ok: bool,
        request_timeout: float | None = None,
    ) -> str:
        return self._invoke(messages, minimal_ok, request_timeout=request_timeout)

    async def invoke_messages_async(
        self,
        messages: list[dict[str, Any]],
        *,
        minimal_ok: bool,
        request_timeout: float | None = None,
    ) -> str:
        return await self._invoke_async(
            messages,
            minimal_ok,
            request_timeout=request_timeout,
        )

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
        url = "/models"
        key = self.config.resolve_api_key() or ""
        headers = {"Authorization": f"Bearer {key}"}
        items: list[Any] = []
        try:
            resp = self._http.get(url, headers=headers, timeout=self._request_timeout)
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
                        "DEBUG(Driver:OpenAI): skipping %s due to missing pricing" % mid
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
