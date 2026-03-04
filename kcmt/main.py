"""Main entry point for kcmt."""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
from typing import TYPE_CHECKING, Any, Callable

__all__ = ["main"]

if TYPE_CHECKING:  # pragma: no cover - type checking aid only
    pass

_cached_cli_main: Callable[[], int] | None = None
_TRUTHY_VALUES = {"1", "true", "yes", "on"}


def _load_cli_main() -> Callable[[], int]:
    """Import the CLI entry point lazily to minimise startup overhead."""

    global _cached_cli_main
    if _cached_cli_main is None:
        from .cli import main as cli_main

        _cached_cli_main = cli_main
    return _cached_cli_main


def _runtime_mode() -> str:
    raw_mode = os.getenv("KCMT_RUNTIME", "python").strip().lower()
    return raw_mode or "python"


def _is_truthy(value: str) -> bool:
    return value.strip().lower() in _TRUTHY_VALUES


def _canary_enabled() -> bool:
    return _is_truthy(os.getenv("KCMT_RUST_CANARY", ""))


def _trace_enabled() -> bool:
    return _is_truthy(os.getenv("KCMT_RUNTIME_TRACE", ""))


def _should_use_rust_runtime() -> bool:
    runtime = _runtime_mode()
    canary = _canary_enabled()
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


def _build_runtime_decision(rust_bin: str) -> dict[str, Any]:
    runtime_mode = _runtime_mode()
    canary = _canary_enabled()
    rust_binary_exists = os.path.exists(rust_bin)
    selected_runtime = "python"
    decision_reason = "runtime_python"

    if runtime_mode not in {"python", "auto", "rust"}:
        decision_reason = "invalid_runtime_value"
    elif _should_use_rust_runtime():
        if rust_binary_exists:
            selected_runtime = "rust"
            if runtime_mode == "rust":
                decision_reason = "runtime_forced_rust"
            else:
                decision_reason = "auto_canary_enabled"
        else:
            decision_reason = "rust_binary_missing"
    elif runtime_mode == "auto":
        decision_reason = "auto_canary_disabled"

    return {
        "selected_runtime": selected_runtime,
        "decision_reason": decision_reason,
        "runtime_mode": runtime_mode,
        "canary_enabled": canary,
        "rust_binary": rust_bin,
        "rust_binary_exists": rust_binary_exists,
    }


def _emit_runtime_trace(decision: dict[str, Any]) -> None:
    if not _trace_enabled():
        return
    # Trace is stderr-only to avoid breaking stdout CLI contracts.
    sys.stderr.write(f"{json.dumps(decision, separators=(',', ':'))}\n")


def _run_rust_runtime() -> int | None:
    rust_bin = _resolve_rust_binary()
    decision = _build_runtime_decision(rust_bin)
    _emit_runtime_trace(decision)

    if decision["selected_runtime"] != "rust":
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
