from __future__ import annotations

import importlib
import sys
from typing import Any, Callable, Protocol, cast


class OpenAIModule(Protocol):
    """Protocol describing the subset of the OpenAI SDK we rely on."""

    OpenAI: Callable[..., Any]


_cached_openai: OpenAIModule | None = None
_cached_source: Any | None = None


def _resolve_openai() -> OpenAIModule | None:
    """Import and cache the OpenAI SDK module when available."""

    global _cached_openai, _cached_source

    module_in_sys = sys.modules.get("openai")
    if module_in_sys is not None and module_in_sys is not _cached_source:
        openai_ctor = getattr(module_in_sys, "OpenAI", None)
        if callable(openai_ctor):
            _cached_openai = cast(OpenAIModule, module_in_sys)
            _cached_source = module_in_sys
            return _cached_openai
        # Fallback to fresh import to avoid caching unusable stubs

    try:
        module = importlib.import_module("openai")
    except Exception:  # pragma: no cover - optional dependency
        _cached_openai = None
        _cached_source = None
        return None
    openai_ctor = getattr(module, "OpenAI", None)
    if callable(openai_ctor):
        _cached_openai = cast(OpenAIModule, module)
    else:
        _cached_openai = None
    _cached_source = module
    return _cached_openai


def import_openai() -> OpenAIModule | None:
    """Return the cached OpenAI module if available."""

    return _resolve_openai()


def reset_openai_cache() -> None:
    """Clear the cached OpenAI module (primarily for tests)."""

    global _cached_openai, _cached_source
    _cached_openai = None
    _cached_source = None
