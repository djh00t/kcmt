from __future__ import annotations

import os
from typing import Any, Dict

import httpx

from ..config import Config
from ..exceptions import LLMError
from .base import BaseDriver, resolve_default_request_timeout


class AnthropicDriver(BaseDriver):
    """Driver handling Anthropic API calls (messages endpoint)."""

    # Wildcard-style strings to exclude anywhere in model id
    DISALLOWED_STRINGS: list[str] = [
        "claude-opus-",
        "claude-3-opus",
        "claude-3-5-sonnet",
        "claude-3-haiku",
    ]

    def __init__(self, config: Config, debug: bool = False) -> None:
        super().__init__(config, debug)
        timeout_env = os.environ.get("KCMT_LLM_REQUEST_TIMEOUT")
        default_timeout = resolve_default_request_timeout(config.provider)
        try:
            self._request_timeout = (
                float(timeout_env) if timeout_env else default_timeout
            )
        except ValueError:
            self._request_timeout = default_timeout
        self._api_key = config.resolve_api_key()

    def generate(self, diff: str, context: str, style: str) -> str:  # noqa: D401,E501
        # Driver is low-level; expects already rendered prompt text.
        raise LLMError(
            "AnthropicDriver.generate should not be called directly; "
            "LLMClient orchestrates prompts."
        )

    def invoke(self, prompt: str, request_timeout: float | None = None) -> str:
        url, headers, payload = self._build_messages_request(prompt)
        timeout = request_timeout or self._request_timeout
        try:
            response = httpx.post(
                url, headers=headers, json=payload, timeout=timeout
            )
        except httpx.HTTPError as e:  # pragma: no cover - network handling
            raise LLMError(
                f"Anthropic network error during messages request: {e}"
            ) from e
        return self._handle_messages_response(response)

    async def invoke_async(
        self, prompt: str, request_timeout: float | None = None
    ) -> str:
        url, headers, payload = self._build_messages_request(prompt)
        timeout = request_timeout or self._request_timeout
        async with httpx.AsyncClient(timeout=timeout) as client:
            try:
                response = await client.post(url, headers=headers, json=payload)
            except httpx.HTTPError as e:  # pragma: no cover - network handling
                raise LLMError(
                    f"Anthropic network error during messages request: {e}"
                ) from e
        return self._handle_messages_response(response)

    def _build_messages_request(self, prompt: str) -> tuple[str, Dict[str, str], Dict[str, Any]]:
        url = self.config.llm_endpoint.rstrip("/") + "/v1/messages"
        headers = {
            "x-api-key": self._api_key or "",
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        payload: Dict[str, Any] = {
            "model": self.config.model,
            "max_output_tokens": 512,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Generate a conventional commit "
                                "message diff.\n" + prompt
                            ),
                        }
                    ],
                }
            ],
            "system": (
                "You are a strict conventional commit message generator. "
                "Always respond with type(scope): description."
            ),
        }
        return url, headers, payload

    def _handle_messages_response(self, response: httpx.Response) -> str:
        status = getattr(response, "status_code", 200)
        if status and int(status) >= 400:
            raise LLMError(
                "Anthropic error {}: {}".format(
                    status, getattr(response, "text", "<no body>")
                )
            )
        data = response.json()
        content = data.get("content") or []
        texts: list[str] = []
        for chunk in content:
            if isinstance(chunk, dict) and chunk.get("type") == "text":
                texts.append(str(chunk.get("text", "")))
        return "\n".join(filter(None, texts))

    def list_models(self) -> list[dict[str, object]]:
        """Return models from Anthropic `/v1/models`.

        Requires valid API key; returns a simplified list of dicts with 'id'
        and any other attributes present in the provider response object.
        """
        url = self.config.llm_endpoint.rstrip("/") + "/v1/models"
        headers = {
            "x-api-key": self._api_key or "",
            "anthropic-version": "2023-06-01",
        }
        try:
            resp = httpx.get(url, headers=headers, timeout=self._request_timeout)
            resp.raise_for_status()
        except Exception as e:  # noqa: BLE001
            raise LLMError(f"Anthropic list_models failed: {e}") from e
        data = resp.json()
        items = data.get("data") or data.get("models") or []
        out: list[dict[str, object]] = []
        ids: list[str] = []
        for m in items:
            if not isinstance(m, dict):
                continue
            mid = m.get("id") or m.get("name")
            if not mid:
                continue
            smid = str(mid)
            if not self.is_allowed_model_id(smid):
                continue
            entry = {"id": mid, "owned_by": "anthropic"}
            # include a few common attrs if present (omit 'type' for
            # uniformity)
            for k in ("display_name", "context_window", "created_at"):
                if k in m:
                    entry[k] = m[k]
            out.append(entry)
            ids.append(smid)
        # Enrich (non-fatal if helper missing)
        try:
            from kcmt.providers.pricing import enrich_ids as _enrich

            emap = _enrich("anthropic", ids)
            enriched: list[dict[str, object]] = []
            for item in out:
                mid = str(item.get("id", ""))
                em = emap.get(mid) or {}
                if not em or not em.get("_has_pricing", False):
                    if self.debug:
                        print(
                            "DEBUG(Driver:Anthropic): skipping %s due to "
                            "missing pricing" % mid
                        )
                    continue
                payload = dict(em)
                payload.pop("_has_pricing", None)
                enriched.append({**item, **payload})
            return enriched
        except (
            ImportError,
            ModuleNotFoundError,
            ValueError,
            TypeError,
            KeyError,
        ):
            return out

    @classmethod
    def _contains_disallowed_string(cls, model_id: str) -> bool:
        if not model_id:
            return False
        for token in cls.DISALLOWED_STRINGS:
            if token and token in model_id:
                return True
        return False

    @classmethod
    def is_allowed_model_id(cls, model_id: str) -> bool:
        if not model_id:
            return False
        if not cls.DISALLOWED_STRINGS:
            return True
        return not cls._contains_disallowed_string(model_id)
