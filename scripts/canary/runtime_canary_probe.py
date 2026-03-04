#!/usr/bin/env python3
"""Runtime canary probe scenarios for kcmt wrapper routing."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

TRACE_REQUIRED_FIELDS = (
    "selected_runtime",
    "decision_reason",
    "runtime_mode",
    "canary_enabled",
    "rust_binary",
    "rust_binary_exists",
)


@dataclass(frozen=True)
class ProbeScenario:
    """One runtime routing probe with expectations."""

    id: str
    command: tuple[str, ...]
    env: Mapping[str, str]
    expected_runtime: str
    expected_reason: str
    expected_exit_codes: tuple[int, ...] = (0,)


def parse_trace_record(stderr_text: str) -> dict[str, Any]:
    """Parse the trace JSON object from stderr text."""

    for line in reversed(stderr_text.splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict) and all(k in parsed for k in TRACE_REQUIRED_FIELDS):
            return parsed
    raise ValueError("runtime trace record not found in stderr output")


def validate_trace_record(record: Mapping[str, Any]) -> list[str]:
    """Validate required runtime trace fields and value shapes."""

    errors: list[str] = []

    for field_name in TRACE_REQUIRED_FIELDS:
        if field_name not in record:
            errors.append(f"missing required field: {field_name}")

    selected_runtime = record.get("selected_runtime")
    if selected_runtime not in {"python", "rust"}:
        errors.append("selected_runtime must be 'python' or 'rust'")

    if not isinstance(record.get("decision_reason"), str) or not record.get(
        "decision_reason"
    ):
        errors.append("decision_reason must be a non-empty string")

    if not isinstance(record.get("runtime_mode"), str) or not record.get("runtime_mode"):
        errors.append("runtime_mode must be a non-empty string")

    if not isinstance(record.get("canary_enabled"), bool):
        errors.append("canary_enabled must be a boolean")

    if not isinstance(record.get("rust_binary"), str) or not record.get("rust_binary"):
        errors.append("rust_binary must be a non-empty string")

    if not isinstance(record.get("rust_binary_exists"), bool):
        errors.append("rust_binary_exists must be a boolean")

    return errors


def build_probe_scenarios(rust_bin: str) -> list[ProbeScenario]:
    """Build deterministic canary probe scenarios."""

    missing_bin = "/tmp/kcmt-missing-rust-bin"

    def _resolve_entrypoint(binary_name: str) -> str:
        repo_root = Path(__file__).resolve().parents[2]
        venv_candidate = repo_root / ".venv" / "bin" / binary_name
        if venv_candidate.exists():
            return str(venv_candidate)

        in_path = shutil.which(binary_name)
        if in_path:
            return in_path

        return str(Path(sys.executable).resolve().parent / binary_name)

    kcmt_cmd = _resolve_entrypoint("kcmt")
    commit_cmd = _resolve_entrypoint("commit")
    kc_cmd = _resolve_entrypoint("kc")

    return [
        ProbeScenario(
            id="default_python_help",
            command=(kcmt_cmd, "--help"),
            env={
                "KCMT_RUNTIME": "python",
                "KCMT_RUNTIME_TRACE": "1",
            },
            expected_runtime="python",
            expected_reason="runtime_python",
        ),
        ProbeScenario(
            id="auto_canary_missing_bin",
            command=(kcmt_cmd, "--help"),
            env={
                "KCMT_RUNTIME": "auto",
                "KCMT_RUST_CANARY": "1",
                "KCMT_RUST_BIN": missing_bin,
                "KCMT_RUNTIME_TRACE": "1",
            },
            expected_runtime="python",
            expected_reason="rust_binary_missing",
        ),
        ProbeScenario(
            id="auto_canary_rust_kcmt_help",
            command=(kcmt_cmd, "--help"),
            env={
                "KCMT_RUNTIME": "auto",
                "KCMT_RUST_CANARY": "1",
                "KCMT_RUST_BIN": rust_bin,
                "KCMT_RUNTIME_TRACE": "1",
            },
            expected_runtime="rust",
            expected_reason="auto_canary_enabled",
        ),
        ProbeScenario(
            id="auto_canary_rust_commit_help",
            command=(commit_cmd, "--help"),
            env={
                "KCMT_RUNTIME": "auto",
                "KCMT_RUST_CANARY": "1",
                "KCMT_RUST_BIN": rust_bin,
                "KCMT_RUNTIME_TRACE": "1",
            },
            expected_runtime="rust",
            expected_reason="auto_canary_enabled",
        ),
        ProbeScenario(
            id="auto_canary_rust_kc_help",
            command=(kc_cmd, "--help"),
            env={
                "KCMT_RUNTIME": "auto",
                "KCMT_RUST_CANARY": "1",
                "KCMT_RUST_BIN": rust_bin,
                "KCMT_RUNTIME_TRACE": "1",
            },
            expected_runtime="rust",
            expected_reason="auto_canary_enabled",
        ),
        ProbeScenario(
            id="forced_python_rollback",
            command=(kcmt_cmd, "--help"),
            env={
                "KCMT_RUNTIME": "python",
                "KCMT_RUST_CANARY": "1",
                "KCMT_RUST_BIN": rust_bin,
                "KCMT_RUNTIME_TRACE": "1",
            },
            expected_runtime="python",
            expected_reason="runtime_python",
        ),
    ]


def _run_scenario(scenario: ProbeScenario) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    for key in ("KCMT_RUNTIME", "KCMT_RUST_CANARY", "KCMT_RUST_BIN", "KCMT_RUNTIME_TRACE"):
        env.pop(key, None)
    env.update(scenario.env)

    return subprocess.run(
        list(scenario.command),
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )


def _evaluate_scenario(
    scenario: ProbeScenario,
    completed: subprocess.CompletedProcess[str],
) -> tuple[bool, list[str], dict[str, Any] | None]:
    errors: list[str] = []

    if completed.returncode not in scenario.expected_exit_codes:
        errors.append(
            f"unexpected exit code {completed.returncode}; expected {scenario.expected_exit_codes}"
        )

    trace_record: dict[str, Any] | None = None
    try:
        trace_record = parse_trace_record(completed.stderr)
    except ValueError as exc:
        errors.append(str(exc))

    if trace_record is not None:
        errors.extend(validate_trace_record(trace_record))
        if trace_record.get("selected_runtime") != scenario.expected_runtime:
            errors.append(
                "selected_runtime mismatch: "
                f"{trace_record.get('selected_runtime')} != {scenario.expected_runtime}"
            )
        if trace_record.get("decision_reason") != scenario.expected_reason:
            errors.append(
                "decision_reason mismatch: "
                f"{trace_record.get('decision_reason')} != {scenario.expected_reason}"
            )

    return (len(errors) == 0, errors, trace_record)


def run_canary_probe(scenarios: Sequence[ProbeScenario]) -> tuple[bool, list[dict[str, Any]]]:
    """Execute all canary scenarios and return success + structured report."""

    report_items: list[dict[str, Any]] = []
    overall_passed = True

    for scenario in scenarios:
        completed = _run_scenario(scenario)
        passed, errors, trace = _evaluate_scenario(scenario, completed)

        if not passed:
            overall_passed = False

        report_items.append(
            {
                "id": scenario.id,
                "command": list(scenario.command),
                "returncode": completed.returncode,
                "passed": passed,
                "errors": errors,
                "trace": trace,
            }
        )

        status = "PASS" if passed else "FAIL"
        print(f"[{status}] {scenario.id}")
        if errors:
            for error in errors:
                print(f"  - {error}")

    return overall_passed, report_items


def main() -> int:
    parser = argparse.ArgumentParser(description="Run kcmt Rust runtime canary probes")
    parser.add_argument(
        "--rust-bin",
        default=str(Path("rust/target/release/kcmt").resolve()),
        help="Path to Rust kcmt binary used for canary scenarios",
    )
    parser.add_argument(
        "--report",
        default="",
        help="Optional path for JSON report output",
    )
    args = parser.parse_args()

    scenarios = build_probe_scenarios(args.rust_bin)
    all_passed, report_items = run_canary_probe(scenarios)

    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_payload = {
            "all_passed": all_passed,
            "scenario_count": len(report_items),
            "results": report_items,
        }
        report_path.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")
        print(f"Report written to {report_path}")

    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
