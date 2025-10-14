from __future__ import annotations

from typing import Any

import httpx

from kcmt.providers.openai_driver import OpenAIDriver

# Currently XAI path reused OpenAI semantics with slight diff cleaning.
# This driver serves as a placeholder for future divergence (rate limits,
# parameter names, system prompt shaping). For now it defers invocation to
# OpenAIDriver but retains a distinct class for clarity & future extension.


class XAIDriver(OpenAIDriver):
    """Alias driver for XAI/Grok style API (OpenAI-compatible)."""

    # Wildcard-style strings to exclude anywhere in model id
    DISALLOWED_STRINGS: list[str] = [
        "grok-2-",
    ]

    # Override to mark ownership as XAI
    def list_models(self) -> list[dict[str, object]]:
        # Query XAI endpoint directly to avoid OpenAI-specific enrichment
        url = "/models"
        key = self.config.resolve_api_key() or ""
        headers = {"Authorization": f"Bearer {key}"}
        out: list[dict[str, object]] = []
        ids: list[str] = []
        items: list[Any] = []
        try:
            # Reuse parent's pooled client if present, else a one-off
            http = getattr(self, "_http", None)
            if http is None:
                with httpx.Client(
                    base_url=self.config.llm_endpoint.rstrip("/"),
                    timeout=self._request_timeout,
                    http2=True,
                ) as http:
                    resp = http.get(url, headers=headers, timeout=self._request_timeout)
                    resp.raise_for_status()
                    data = resp.json()
            else:
                resp = http.get(url, headers=headers, timeout=self._request_timeout)
                resp.raise_for_status()
                data = resp.json()
            payload_items = data.get("data") if isinstance(data, dict) else None
            if isinstance(payload_items, list):
                items = payload_items
        except (httpx.HTTPError, ValueError, KeyError):
            items = []

        for m in items:
            if not isinstance(m, dict):
                continue
            mid_val = m.get("id")
            if not mid_val:
                continue
            mid = str(mid_val)
            if not self.is_allowed_model_id(mid):
                continue
            entry: dict[str, object] = {"id": mid, "owned_by": "xai"}
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
        # If nothing came back, try dataset fallback for xai
        if not out:
            try:
                from kcmt.providers.pricing import build_enrichment_context
            except ImportError:
                pass
            else:
                try:
                    alias_lut, _ctx, _mx = build_enrichment_context()
                    seen: set[str] = set()
                    for (prov, mid), canon in alias_lut.items():
                        if prov != "xai":
                            continue
                        for candidate in (str(canon), str(mid)):
                            if candidate and candidate not in seen:
                                out.append({"id": candidate, "owned_by": "xai"})
                                ids.append(candidate)
                                seen.add(candidate)
                    if len(out) > 200:
                        out = out[:200]
                        ids = ids[:200]
                except (RuntimeError, ValueError, KeyError, TypeError):
                    pass
        # Enrich as xai
        try:
            from kcmt.providers.pricing import enrich_ids as _enrich
        except ImportError:
            return out
        try:
            emap = _enrich("xai", ids)
        except (
            ValueError,
            TypeError,
            KeyError,
            RuntimeError,
            AttributeError,
        ):
            emap = {}
        enriched: list[dict[str, object]] = []
        for item in out:
            mid = str(item.get("id", ""))
            em = emap.get(mid) or {}
            if not em or not em.get("_has_pricing", False):
                if self.debug:
                    print("DEBUG(Driver:XAI): skipping %s due to missing pricing" % mid)
                continue
            payload = dict(em)
            payload.pop("_has_pricing", None)
            enriched.append({**item, **payload})
        return enriched

    @classmethod
    def is_allowed_model_id(cls, model_id: str) -> bool:
        if not model_id:
            return False
        if not cls.DISALLOWED_STRINGS:
            return True
        return not cls._contains_disallowed_string(model_id)
