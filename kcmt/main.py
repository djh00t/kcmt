"""Main entry point for kcmt."""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys
from typing import TYPE_CHECKING, Callable

__all__ = ["main"]

if TYPE_CHECKING:  # pragma: no cover - type checking aid only
    pass

_cached_cli_main: Callable[[], int] | None = None


def _load_cli_main() -> Callable[[], int]:
    """Import the CLI entry point lazily to minimise startup overhead."""

    global _cached_cli_main
    if _cached_cli_main is None:
        from .cli import main as cli_main

        _cached_cli_main = cli_main
    return _cached_cli_main


def _should_use_rust_runtime() -> bool:
    runtime = os.getenv("KCMT_RUNTIME", "python").strip().lower()
    canary = os.getenv("KCMT_RUST_CANARY", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if runtime == "rust":
        return True
    if runtime == "auto" and canary:
        return True
    return False


def _resolve_rust_binary() -> str:
    configured = os.getenv("KCMT_RUST_BIN", "").strip()
    if configured:
        return configured

    repo_root = pathlib.Path(__file__).resolve().parents[1]
    return str(repo_root / "rust" / "target" / "release" / "kcmt")


def _run_rust_runtime() -> int | None:
    if not _should_use_rust_runtime():
        return None

    rust_bin = _resolve_rust_binary()
    if not os.path.exists(rust_bin):
        return None

    completed = subprocess.run([rust_bin, *sys.argv[1:]], check=False)
    return completed.returncode


def main() -> int:
    """Main entry point that delegates to CLI and returns its exit code."""

    rust_code = _run_rust_runtime()
    if rust_code is not None:
        return rust_code
    return _load_cli_main()()


if __name__ == "__main__":  # pragma: no cover - manual invocation
    sys.exit(main())
