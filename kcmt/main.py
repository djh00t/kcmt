"""Main entry point for kcmt."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Callable

__all__ = ["main"]

if TYPE_CHECKING:  # pragma: no cover - type checking aid only
    from .cli import main as _cli_main_type

_cached_cli_main: Callable[[], int] | None = None


def _load_cli_main() -> Callable[[], int]:
    """Import the CLI entry point lazily to minimise startup overhead."""

    global _cached_cli_main
    if _cached_cli_main is None:
        from .cli import main as cli_main

        _cached_cli_main = cli_main
    return _cached_cli_main


def main() -> int:
    """Main entry point that delegates to CLI and returns its exit code."""

    return _load_cli_main()()


if __name__ == "__main__":  # pragma: no cover - manual invocation
    sys.exit(main())
