from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Protocol

from ..config import Config

DEFAULT_PROVIDER_TIMEOUTS: dict[str, float] = {
    "openai": 5.0,
    "github": 5.0,
    "xai": 12.0,
    "anthropic": 12.0,
}


def resolve_default_request_timeout(provider: str | None) -> float:
    if not provider:
        return 5.0
    return DEFAULT_PROVIDER_TIMEOUTS.get(provider.lower(), 5.0)


class DriverResult(Protocol):  # simple protocol if we later enrich
    ...


class BaseDriver(ABC):
    """Abstract base for provider-specific commit generation.

    Each driver encapsulates one provider's HTTP/client call patterns,
    parameter semantics, adaptive retry strategies, and any provider-
    specific prompt shaping. This isolates branching logic from the
    higher-level orchestration in LLMClient.
    """

    def __init__(self, config: Config, debug: bool = False) -> None:
        self.config = config
        self.debug = debug

    @abstractmethod
    def generate(self, diff: str, context: str, style: str) -> str:
        """Return a (possibly multi-line) conventional commit message.

        Must raise provider-specific exceptions as LLMError upstream if
        unrecoverable. Should not perform global diff classification
        (binary/minimal/large) – that remains in the orchestrator for
        cross-provider consistency.
        """
        raise NotImplementedError

    @abstractmethod
    def list_models(self) -> list[dict[str, Any]]:
        """Return a list of models with attributes available from provider.

        Each item should be a dict minimally containing an 'id' key, with any
        other attributes passed through from the provider where possible.
        """
        raise NotImplementedError
