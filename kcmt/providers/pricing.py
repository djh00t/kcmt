from __future__ import annotations

import logging
import os
import re
import threading
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation
from functools import lru_cache
from typing import Any, cast

import httpx
from genai_prices import UpdatePrices, data_snapshot
from genai_prices.types import ModelInfo, TieredPrices

TIMEOUT = 25
_SUPPORTED = {"openai", "anthropic", "xai"}
_DATE_SUFFIX_RE = re.compile(r"[-_](?:\d{8}|\d{6}|\d{4})(?:\d{2}\d{2})?$")

logger = logging.getLogger(__name__)


@dataclass
class _SnapshotState:
    cache: data_snapshot.DataSnapshot | None = None
    last_attempt: date | None = None
    last_success: date | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)


_STATE = _SnapshotState()


def _coerce_int(value: object) -> int | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        try:
            return int(value)
        except (ValueError, OverflowError):
            return None
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return None
    return None


def _normalize_provider(provider: str) -> str:
    p = (provider or "").strip().lower()
    if p in {"openai", "open-ai"}:
        return "openai"
    if p == "anthropic":
        return "anthropic"
    if p in {"xai", "x-ai", "x.ai", "x"}:
        return "xai"
    return p


def _snapshot_provider_id(canonical: str) -> str | None:
    if canonical == "openai":
        return "openai"
    if canonical == "anthropic":
        return "anthropic"
    if canonical == "xai":
        return "x-ai"
    return None


def _ensure_snapshot() -> data_snapshot.DataSnapshot:
    today = datetime.now(timezone.utc).date()
    with _STATE.lock:
        if _STATE.cache is not None and _STATE.last_success == today:
            return _STATE.cache

        if _STATE.last_attempt != today:
            _STATE.last_attempt = today
            try:
                snapshot = UpdatePrices().fetch()
            except (httpx.HTTPError, ValueError, OSError) as exc:
                logger.debug("genai-prices refresh failed: %s", exc)
            else:
                if snapshot is not None:
                    data_snapshot.set_custom_snapshot(snapshot)
                    _STATE.cache = snapshot
                    _STATE.last_success = today

        if _STATE.cache is None:
            _STATE.cache = data_snapshot.get_snapshot()
        return _STATE.cache


def _resolve_model(
    snapshot: data_snapshot.DataSnapshot,
    provider_id: str | None,
    model_id: str,
) -> tuple[str, ModelInfo | None]:
    if provider_id is None:
        return model_id, None

    attempts = [model_id]
    stripped = _DATE_SUFFIX_RE.sub("", model_id)
    if stripped and stripped != model_id:
        attempts.append(stripped)

    for candidate in attempts:
        try:
            _, model_info = snapshot.find_provider_model(
                candidate.lower(),
                None,
                provider_id,
                None,
            )
            return model_info.id, model_info
        except LookupError:
            continue
    return attempts[-1], None


def _mtok_value(val: Decimal | TieredPrices | None) -> float | None:
    if val is None:
        return None
    if isinstance(val, TieredPrices):
        val = val.base
    return _dec(val)


def _prices_from_model(
    model_info: ModelInfo | None,
) -> dict[str, float | None] | None:
    if model_info is None:
        return None
    now = datetime.now(timezone.utc)
    model_price = model_info.get_prices(now)
    prices = {
        "input_price_per_mtok": _mtok_value(model_price.input_mtok),
        "output_price_per_mtok": _mtok_value(model_price.output_mtok),
        "cache_read_per_mtok": _mtok_value(model_price.cache_read_mtok),
    }
    if all(v is None for v in prices.values()):
        return None
    return prices


def _dec(val: Decimal | str | float | int | None) -> float | None:
    if val is None:
        return None
    try:
        dv = Decimal(str(val)).quantize(Decimal("0.000001"))
    except (InvalidOperation, ValueError):
        return None
    s = f"{dv}".rstrip("0").rstrip(".")
    return float(s) if s else 0.0


def _get(url: str, headers: dict[str, str] | None = None) -> dict[str, Any]:
    response = httpx.get(url, headers=(headers or {}), timeout=TIMEOUT)
    response.raise_for_status()
    data = response.json()
    if isinstance(data, dict):
        return data
    raise ValueError("Expected JSON object from pricing endpoint")


def _scale_openrouter_price(value: float | str | None) -> Decimal | None:
    if value is None:
        return None
    try:
        dv = Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None
    return dv * Decimal(1_000_000)


@lru_cache(maxsize=1)
def openrouter_max_output_lookup() -> dict[str, int | None]:
    try:
        headers = {}
        api_key = os.getenv("OPENROUTER_API_KEY")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        payload = _get("https://openrouter.ai/api/v1/models", headers=headers)
        lut: dict[str, int | None] = {}
        for model in payload.get("data", []):
            mid = model.get("id")
            if not mid:
                continue
            provider_info = model.get("top_provider") or {}
            lut[mid] = provider_info.get("max_completion_tokens")
        return lut
    except (httpx.HTTPError, ValueError, KeyError, OSError):
        return {}


@lru_cache(maxsize=1)
def openrouter_metadata_lookup() -> dict[str, dict[str, object]]:
    try:
        headers = {}
        api_key = os.getenv("OPENROUTER_API_KEY")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        payload = _get("https://openrouter.ai/api/v1/models", headers=headers)
        lut: dict[str, dict[str, object]] = {}
        for model in payload.get("data", []):
            mid = model.get("id")
            if not mid:
                continue
            meta: dict[str, object] = {}
            provider_info = model.get("top_provider") or {}
            meta["max_output"] = provider_info.get("max_completion_tokens")
            ctx_len = model.get("context_length")
            if ctx_len is None and isinstance(provider_info, dict):
                ctx_len = provider_info.get("context_length")
            meta["context"] = ctx_len
            pricing = model.get("pricing") or {}
            if not pricing and isinstance(provider_info, dict):
                pricing = provider_info.get("pricing") or {}
            if isinstance(pricing, dict):
                meta["input_price_per_mtok"] = _dec(
                    _scale_openrouter_price(pricing.get("prompt"))
                )
                meta["output_price_per_mtok"] = _dec(
                    _scale_openrouter_price(pricing.get("completion"))
                )
                cached = pricing.get("cached") or pricing.get("cache_read")
                meta["cache_read_per_mtok"] = _dec(_scale_openrouter_price(cached))
            lut[mid] = meta
        return lut
    except (httpx.HTTPError, ValueError, KeyError, OSError):
        return {}


def candidate_openrouter_ids(
    prefix: str | list[str],
    mid: str,
    canon: str,
) -> list[str]:
    prefixes = prefix if isinstance(prefix, list) else [prefix]
    cands: list[str] = []
    for pref in prefixes:
        for model_key in {mid, canon}:
            if not model_key:
                continue
            cands.append(f"{pref}/{model_key}")
            cands.append(f"{pref}/{model_key}-latest")
            stripped = _DATE_SUFFIX_RE.sub("", model_key)
            if stripped != model_key:
                cands.append(f"{pref}/{stripped}")
    seen, out = set(), []
    for value in cands:
        if value not in seen:
            out.append(value)
            seen.add(value)
    return out


def _router_prefix_variants(canonical: str) -> list[str]:
    if canonical == "openai":
        return ["openai", "open-ai"]
    if canonical == "anthropic":
        return ["anthropic"]
    if canonical == "xai":
        return ["xai", "x-ai", "x.ai", "x"]
    return [canonical]


def list_known_model_ids(provider: str) -> list[str]:
    canonical = _normalize_provider(provider)
    if canonical not in _SUPPORTED:
        return []
    snapshot = _ensure_snapshot()
    provider_id = _snapshot_provider_id(canonical)
    if provider_id is None:
        return []
    for provider_info in snapshot.providers:
        if provider_info.id == provider_id:
            return [model.id for model in provider_info.models]
    return []


def enrich_ids(provider: str, ids: list[str]) -> dict[str, dict[str, object]]:
    snapshot = _ensure_snapshot()
    canonical = _normalize_provider(provider)
    provider_id = _snapshot_provider_id(canonical)
    router_prefixes = _router_prefix_variants(canonical)
    meta_lut = openrouter_metadata_lookup()
    max_out_lut = openrouter_max_output_lookup()

    enriched: dict[str, dict[str, object]] = {}
    for mid in ids:
        resolved_id, model_info = _resolve_model(snapshot, provider_id, mid)
        prices = _prices_from_model(model_info)
        total_ctx = model_info.context_window if model_info else None

        max_output_tokens = None
        candidates = candidate_openrouter_ids(
            router_prefixes,
            mid,
            resolved_id,
        )
        for candidate in candidates:
            if max_output_tokens is None and candidate in max_out_lut:
                max_output_tokens = max_out_lut[candidate]
            meta = meta_lut.get(candidate) or {}
            if total_ctx is None:
                ctx_value = _coerce_int(meta.get("context"))
                if ctx_value is not None:
                    total_ctx = ctx_value
            if prices is None and any(
                meta.get(key) is not None
                for key in (
                    "input_price_per_mtok",
                    "output_price_per_mtok",
                    "cache_read_per_mtok",
                )
            ):
                prices = {
                    "input_price_per_mtok": cast(
                        float | None, meta.get("input_price_per_mtok")
                    ),
                    "output_price_per_mtok": cast(
                        float | None, meta.get("output_price_per_mtok")
                    ),
                    "cache_read_per_mtok": cast(
                        float | None, meta.get("cache_read_per_mtok")
                    ),
                }

        record: dict[str, object] = {
            "id": mid,
            "total_context": total_ctx,
            "max_output": max_output_tokens,
            "_has_pricing": prices is not None,
        }
        if prices:
            record.update(prices)
        enriched[mid] = record
    return enriched


def build_enrichment_context() -> tuple[
    dict[tuple[str, str], str],
    dict[str, int | None],
    dict[str, int | None],
]:
    """Pre-compute alias mapping and metadata for offline enrichment."""

    snapshot = _ensure_snapshot()
    alias_map: dict[tuple[str, str], str] = {}
    context_map: dict[str, int | None] = {}
    max_output_map = openrouter_max_output_lookup()

    for provider_info in snapshot.providers:
        canonical = _normalize_provider(provider_info.id)
        if canonical not in _SUPPORTED:
            continue
        prefixes = _router_prefix_variants(canonical)
        for model in provider_info.models:
            model_id = model.id
            alias_map[(canonical, model_id)] = model_id
            context_map[f"{canonical}:{model_id}"] = getattr(
                model, "context_window", None
            )
            for alias in candidate_openrouter_ids(prefixes, model_id, model_id):
                alias_map[(canonical, alias)] = model_id

    return alias_map, context_map, max_output_map
