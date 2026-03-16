from __future__ import annotations

import json
import pathlib

from kcmt import main as main_mod


def test_should_use_rust_runtime_explicit_rust(monkeypatch):
    monkeypatch.setenv("KCMT_RUNTIME", "rust")
    monkeypatch.delenv("KCMT_RUST_CANARY", raising=False)
    assert main_mod._should_use_rust_runtime() is True


def test_should_use_rust_runtime_auto_requires_canary(monkeypatch):
    monkeypatch.setenv("KCMT_RUNTIME", "auto")
    monkeypatch.setenv("KCMT_RUST_CANARY", "yes")
    assert main_mod._should_use_rust_runtime() is True

    monkeypatch.setenv("KCMT_RUST_CANARY", "0")
    assert main_mod._should_use_rust_runtime() is False


def test_should_use_rust_runtime_defaults_to_python(monkeypatch):
    monkeypatch.delenv("KCMT_RUNTIME", raising=False)
    monkeypatch.delenv("KCMT_RUST_CANARY", raising=False)
    assert main_mod._should_use_rust_runtime() is False


def test_resolve_rust_binary_prefers_env(monkeypatch):
    monkeypatch.setenv("KCMT_RUST_BIN", "/tmp/custom-kcmt")
    assert main_mod._resolve_rust_binary() == "/tmp/custom-kcmt"


def test_resolve_rust_binary_uses_repo_default(monkeypatch):
    monkeypatch.delenv("KCMT_RUST_BIN", raising=False)
    resolved = pathlib.Path(main_mod._resolve_rust_binary())
    assert resolved.name == "kcmt"
    assert resolved.parts[-4:] == ("rust", "target", "release", "kcmt")


def test_build_runtime_decision_invalid_runtime_value(monkeypatch):
    monkeypatch.setenv("KCMT_RUNTIME", "mystery")
    monkeypatch.delenv("KCMT_RUST_CANARY", raising=False)
    monkeypatch.setattr(main_mod.os.path, "exists", lambda _: True)

    decision = main_mod._build_runtime_decision("/tmp/kcmt-rust")

    assert decision["selected_runtime"] == "python"
    assert decision["decision_reason"] == "invalid_runtime_value"
    assert decision["runtime_mode"] == "mystery"


def test_build_runtime_decision_auto_canary_disabled(monkeypatch):
    monkeypatch.setenv("KCMT_RUNTIME", "auto")
    monkeypatch.setenv("KCMT_RUST_CANARY", "0")
    monkeypatch.setattr(main_mod.os.path, "exists", lambda _: True)

    decision = main_mod._build_runtime_decision("/tmp/kcmt-rust")

    assert decision["selected_runtime"] == "python"
    assert decision["decision_reason"] == "auto_canary_disabled"
    assert decision["canary_enabled"] is False


def test_run_rust_runtime_returns_none_when_disabled(monkeypatch):
    monkeypatch.setattr(main_mod, "_should_use_rust_runtime", lambda: False)
    assert main_mod._run_rust_runtime() is None


def test_run_rust_runtime_returns_none_when_binary_missing(monkeypatch):
    monkeypatch.setattr(main_mod, "_should_use_rust_runtime", lambda: True)
    monkeypatch.setattr(main_mod, "_resolve_rust_binary", lambda: "/tmp/missing-kcmt")
    monkeypatch.setattr(main_mod.os.path, "exists", lambda _: False)
    assert main_mod._run_rust_runtime() is None


def test_run_rust_runtime_executes_binary_and_returns_code(monkeypatch):
    called: list[list[str]] = []

    def _fake_run(args: list[str], check: bool):
        called.append(args)
        assert check is False
        return type("CompletedProcessStub", (), {"returncode": 7})()

    monkeypatch.setattr(main_mod, "_should_use_rust_runtime", lambda: True)
    monkeypatch.setattr(main_mod, "_resolve_rust_binary", lambda: "/tmp/kcmt-rust")
    monkeypatch.setattr(main_mod.os.path, "exists", lambda _: True)
    monkeypatch.setattr(main_mod.sys, "argv", ["kcmt", "status", "--repo-path", "."])
    monkeypatch.setattr(main_mod.subprocess, "run", _fake_run)

    assert main_mod._run_rust_runtime() == 7
    assert called == [["/tmp/kcmt-rust", "status", "--repo-path", "."]]


def test_run_rust_runtime_preserves_status_raw_args(monkeypatch):
    called: list[list[str]] = []

    def _fake_run(args: list[str], check: bool):
        called.append(args)
        assert check is False
        return type("CompletedProcessStub", (), {"returncode": 0})()

    monkeypatch.setattr(main_mod, "_should_use_rust_runtime", lambda: True)
    monkeypatch.setattr(main_mod, "_resolve_rust_binary", lambda: "/tmp/kcmt-rust")
    monkeypatch.setattr(main_mod.os.path, "exists", lambda _: True)
    monkeypatch.setattr(
        main_mod.sys,
        "argv",
        ["kcmt", "status", "--repo-path", "/tmp/repo", "--raw"],
    )
    monkeypatch.setattr(main_mod.subprocess, "run", _fake_run)

    assert main_mod._run_rust_runtime() == 0
    assert called == [["/tmp/kcmt-rust", "status", "--repo-path", "/tmp/repo", "--raw"]]


def test_emit_runtime_trace_enabled_writes_json(monkeypatch, capsys):
    monkeypatch.setenv("KCMT_RUNTIME_TRACE", "1")
    decision = {
        "selected_runtime": "python",
        "decision_reason": "runtime_python",
        "runtime_mode": "python",
        "canary_enabled": False,
        "rust_binary": "/tmp/kcmt-rust",
        "rust_binary_exists": False,
    }

    main_mod._emit_runtime_trace(decision)
    out = capsys.readouterr()
    parsed = json.loads(out.err.strip())

    assert out.out == ""
    assert parsed["selected_runtime"] == "python"
    assert parsed["decision_reason"] == "runtime_python"


def test_emit_runtime_trace_disabled_has_no_output(monkeypatch, capsys):
    monkeypatch.delenv("KCMT_RUNTIME_TRACE", raising=False)
    decision = {
        "selected_runtime": "python",
        "decision_reason": "runtime_python",
        "runtime_mode": "python",
        "canary_enabled": False,
        "rust_binary": "/tmp/kcmt-rust",
        "rust_binary_exists": False,
    }

    main_mod._emit_runtime_trace(decision)
    out = capsys.readouterr()

    assert out.out == ""
    assert out.err == ""


def test_run_rust_runtime_emits_trace_on_fallback(monkeypatch, capsys):
    monkeypatch.setenv("KCMT_RUNTIME", "auto")
    monkeypatch.setenv("KCMT_RUST_CANARY", "1")
    monkeypatch.setenv("KCMT_RUNTIME_TRACE", "1")
    monkeypatch.setattr(main_mod, "_resolve_rust_binary", lambda: "/tmp/missing-kcmt")
    monkeypatch.setattr(main_mod.os.path, "exists", lambda _: False)

    assert main_mod._run_rust_runtime() is None
    out = capsys.readouterr()
    parsed = json.loads(out.err.strip())

    assert parsed["selected_runtime"] == "python"
    assert parsed["decision_reason"] == "rust_binary_missing"


def test_main_returns_rust_runtime_code(monkeypatch):
    monkeypatch.setattr(main_mod, "_run_rust_runtime", lambda: 5)
    assert main_mod.main() == 5


def test_main_falls_back_to_python_cli(monkeypatch):
    invoked = {"count": 0}

    def _fake_cli_main() -> int:
        invoked["count"] += 1
        return 0

    monkeypatch.setattr(main_mod, "_run_rust_runtime", lambda: None)
    monkeypatch.setattr(main_mod, "_load_cli_main", lambda: _fake_cli_main)

    assert main_mod.main() == 0
    assert invoked["count"] == 1
