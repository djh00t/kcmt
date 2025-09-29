from __future__ import annotations

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
        base = self.config.llm_endpoint.rstrip("/")
        url = f"{base}/models"
        key = self.config.resolve_api_key() or ""
        headers = {"Authorization": f"Bearer {key}"}
        out: list[dict[str, object]] = []
        ids: list[str] = []
        try:
            resp = httpx.get(
                url, headers=headers, timeout=self._request_timeout
            )
            resp.raise_for_status()
            data = resp.json()
            items = data.get("data") or []
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
                try:
                    from kcmt.providers.pricing import (
                        build_enrichment_context as _bctx,  # type: ignore
                    )
                except ImportError:
                    _bctx = None  # type: ignore[assignment]
                if _bctx is not None:
                    alias_lut, _ctx, _mx = _bctx()
                    seen: set[str] = set()
                    for (prov, mid), canon in alias_lut.items():
                        if prov != "xai":
                            continue
                        for candidate in (str(canon), str(mid)):
                            if candidate and candidate not in seen:
                                out.append(
                                    {"id": candidate, "owned_by": "xai"}
                                )
                                ids.append(candidate)
                                seen.add(candidate)
                    if len(out) > 200:
                        out = out[:200]
                        ids = ids[:200]
            except (RuntimeError, ValueError, KeyError, TypeError):
                pass
        # Enrich as xai
        try:
            try:
                from kcmt.providers.pricing import enrich_ids as _enrich  # type: ignore
            except ImportError:
                _enrich = None  # type: ignore[assignment]
            if _enrich is None:
                return out
            emap = _enrich("xai", ids)
        except (
            ImportError,
            ModuleNotFoundError,
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
                    print(
                        "DEBUG(Driver:XAI): skipping %s due to missing "
                        "pricing"
                        % mid
                    )
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
