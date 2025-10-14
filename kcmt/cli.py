"""Hybrid CLI entrypoint that orchestrates the Ink UI and legacy parser."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
import shutil
from typing import Optional

from . import legacy_cli as _legacy_module
from .commit import CommitGenerator  # noqa: F401 - re-exported for tests
from .core import KlingonCMTWorkflow  # noqa: F401 - module level alias for tests
from .git import GitRepo  # noqa: F401 - re-exported for tests

from .legacy_cli import LegacyCLI

INK_APP_PATH = Path(__file__).resolve().parent / "ui" / "ink" / "index.mjs"


class CLI:
    """CLI facade that prefers the Ink UI when available."""

    def __init__(self) -> None:
        self._legacy = LegacyCLI()
        self.parser = self._legacy.parser

    def run(self, args: Optional[list[str]] = None) -> int:
        """Dispatch to the Ink UI when interactive, otherwise fallback."""

        _legacy_module.KlingonCMTWorkflow = KlingonCMTWorkflow
        _legacy_module.CommitGenerator = CommitGenerator
        _legacy_module.GitRepo = GitRepo
        effective_args = args if args is not None else sys.argv[1:]
        if self._should_use_ink(effective_args):
            code = self._run_with_ink(effective_args)
            if code is not None:
                return code
        return self._legacy.run(effective_args)

    # ------------------------------------------------------------------
    # Ink orchestration
    # ------------------------------------------------------------------
    def _should_use_ink(self, args: Optional[list[str]]) -> bool:
        env_flag = os.environ.get("KCMT_USE_INK", "")
        if env_flag and env_flag.lower() in {"0", "false", "no", "off"}:
            return False
        if os.environ.get("PYTEST_CURRENT_TEST"):
            return False
        if not sys.stdout.isatty():
            return False
        arg_list = args or []
        if any(
            token
            in {
                "status",
                "--raw",
                "--list-models",
                "--benchmark-json",
                "--benchmark-csv",
                "--oneshot",
                "--file",
                "--configure-all",
            }
            for token in arg_list
        ):
            return False
        if not INK_APP_PATH.exists():
            return False
        return self._ink_runtime_available()

    def _ink_runtime_available(self) -> bool:
        """Best-effort probe for a usable Node+Ink runtime.

        We only enable the Ink UI when:
        - `node` is available on PATH, and
        - required packages (react, ink) are resolvable from the Ink app dir.

        This avoids surfacing a noisy Node stack trace when dependencies
        aren't installed, and gracefully falls back to the legacy CLI.
        """
        # Check for node executable
        if shutil.which("node") is None:
            return False

        # Quick dependency resolution check using ESM import semantics.
        # Run from the Ink app directory so local node_modules (if present)
        # and its package.json resolution rules are used.
        probe = (
            "import('react').then(() => import('ink'))"
            ".then(() => process.exit(0))"
            ".catch(() => process.exit(2))"
        )
        try:
            completed = subprocess.run(
                [
                    "node",
                    "--input-type=module",
                    "-e",
                    probe,
                ],
                cwd=str(INK_APP_PATH.parent),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
                timeout=2.0,
            )
        except Exception:
            return False
        return completed.returncode == 0

    def _run_with_ink(self, args: Optional[list[str]]) -> Optional[int]:
        if not INK_APP_PATH.exists():
            return None
        env = os.environ.copy()
        env.setdefault("KCMT_PYTHON_EXECUTABLE", sys.executable)
        env.setdefault("KCMT_BACKEND_MODULE", "kcmt.ink_backend")
        command = ["node", str(INK_APP_PATH)]
        if args:
            command.append("--")
            command.extend(args)
        try:
            completed = subprocess.run(command, check=False, env=env)
        except FileNotFoundError:
            return None
        return completed.returncode


def main(argv: Optional[list[str]] = None) -> int:
    return CLI().run(argv)


if __name__ == "__main__":
    sys.exit(main())
