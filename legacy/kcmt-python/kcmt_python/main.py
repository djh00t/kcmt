"""Main entry point for kcmt-python."""

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
_CONFIG_OVERRIDE_FLAGS = {
    "--provider",
    "--model",
    "--endpoint",
    "--api-key-env",
    "--batch",
    "--no-batch",
    "--batch-model",
    "--batch-timeout",
    "--auto-push",
    "--no-auto-push",
    "--max-commit-length",
}


def _load_cli_main() -> Callable[[], int]:
    """Import the CLI entry point lazily to minimise startup overhead."""

    global _cached_cli_main
    if _cached_cli_main is None:
        from .cli import main as cli_main

        _cached_cli_main = cli_main
    return _cached_cli_main


def _runtime_mode() -> str:
    raw_mode = os.getenv("KCMT_RUNTIME", "auto").strip().lower()
    return raw_mode or "auto"


def _is_truthy(value: str) -> bool:
    return value.strip().lower() in _TRUTHY_VALUES


def _canary_enabled() -> bool:
    return _is_truthy(os.getenv("KCMT_RUST_CANARY", ""))


def _trace_enabled() -> bool:
    return _is_truthy(os.getenv("KCMT_RUNTIME_TRACE", ""))


def _is_rust_covered_invocation(args: list[str] | None = None) -> bool:
    arg_list = list(sys.argv[1:] if args is None else args)
    if not arg_list:
        return not sys.stdout.isatty()
    if "--oneshot" in arg_list or "--file" in arg_list:
        return True
    is_configure = "--configure" in arg_list or "--configure-all" in arg_list
    if is_configure:
        return True
    if (
        "--list-models" in arg_list
        or "--verify-keys" in arg_list
        or "--benchmark" in arg_list
    ):
        return True
    if arg_list[0] == "status":
        return True
    if len(arg_list) >= 2 and arg_list[0] == "benchmark" and arg_list[1] == "runtime":
        return True
    if not is_configure and arg_list[0] not in {"status", "benchmark"}:
        return True
    return False


def _should_use_rust_runtime(args: list[str] | None = None) -> bool:
    runtime = _runtime_mode()
    canary = _canary_enabled()
    if runtime == "rust":
        return True
    if runtime == "python":
        return False
    if runtime == "auto" and _is_rust_covered_invocation(args):
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


def _build_runtime_decision(
    rust_bin: str, args: list[str] | None = None
) -> dict[str, Any]:
    runtime_mode = _runtime_mode()
    canary = _canary_enabled()
    covered_invocation = _is_rust_covered_invocation(args)
    rust_binary_exists = os.path.exists(rust_bin)
    selected_runtime = "python"
    decision_reason = "runtime_python"

    if runtime_mode not in {"python", "auto", "rust"}:
        decision_reason = "invalid_runtime_value"
    elif _should_use_rust_runtime(args):
        if rust_binary_exists:
            selected_runtime = "rust"
            if runtime_mode == "rust":
                decision_reason = "runtime_forced_rust"
            elif covered_invocation:
                decision_reason = "auto_covered_workflow"
            else:
                decision_reason = "auto_canary_enabled"
        else:
            decision_reason = "rust_binary_missing"
    elif runtime_mode == "auto":
        decision_reason = "auto_unsupported_invocation"

    return {
        "selected_runtime": selected_runtime,
        "decision_reason": decision_reason,
        "runtime_mode": runtime_mode,
        "canary_enabled": canary,
        "covered_invocation": covered_invocation,
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
    args = sys.argv[1:]
    decision = _build_runtime_decision(rust_bin, args)
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
