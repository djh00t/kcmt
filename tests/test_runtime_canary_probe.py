from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path


def _load_probe_module():
    root = Path(__file__).resolve().parents[1]
    script_path = root / "scripts" / "canary" / "runtime_canary_probe.py"
    spec = importlib.util.spec_from_file_location("runtime_canary_probe", script_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_parse_trace_record_picks_json_line_from_stderr():
    probe = _load_probe_module()
    stderr_text = 'warning line\n{"selected_runtime":"python","decision_reason":"runtime_python","runtime_mode":"python","canary_enabled":false,"rust_binary":"/tmp/kcmt","rust_binary_exists":false}\n'

    record = probe.parse_trace_record(stderr_text)

    assert record["selected_runtime"] == "python"
    assert record["decision_reason"] == "runtime_python"


def test_parse_trace_record_raises_when_trace_missing():
    probe = _load_probe_module()

    try:
        probe.parse_trace_record("no json here")
        raised = False
    except ValueError:
        raised = True

    assert raised is True


def test_validate_trace_record_reports_missing_fields():
    probe = _load_probe_module()

    errors = probe.validate_trace_record({"selected_runtime": "python"})

    assert any("missing required field" in err for err in errors)


def test_build_probe_scenarios_contains_required_ids():
    probe = _load_probe_module()

    scenarios = probe.build_probe_scenarios("/tmp/rust-kcmt")
    ids = {scenario.id for scenario in scenarios}

    assert "default_python_help" in ids
    assert "auto_canary_missing_bin" in ids
    assert "auto_canary_rust_kcmt_help" in ids
    assert "forced_python_rollback" in ids


def test_run_canary_probe_reports_failures(monkeypatch):
    probe = _load_probe_module()

    scenario = probe.ProbeScenario(
        id="unit-scenario",
        command=("kcmt", "--help"),
        env={"KCMT_RUNTIME": "python", "KCMT_RUNTIME_TRACE": "1"},
        expected_runtime="python",
        expected_reason="runtime_python",
    )

    def _fake_run(_scenario):
        return subprocess.CompletedProcess(
            args=["kcmt", "--help"],
            returncode=9,
            stdout="",
            stderr='{"selected_runtime":"python","decision_reason":"runtime_python","runtime_mode":"python","canary_enabled":false,"rust_binary":"/tmp/kcmt","rust_binary_exists":false}\n',
        )

    monkeypatch.setattr(probe, "_run_scenario", _fake_run)

    passed, report = probe.run_canary_probe([scenario])

    assert passed is False
    assert report[0]["passed"] is False
    assert report[0]["returncode"] == 9
