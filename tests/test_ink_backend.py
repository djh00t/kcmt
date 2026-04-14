from __future__ import annotations

from types import SimpleNamespace

import kcmt.ink_backend as ink_backend


def test_action_benchmark_rejects_runtime_mode_payload(monkeypatch):
    events: list[tuple[str, dict[str, str]]] = []

    monkeypatch.setattr(
        ink_backend,
        "_emit",
        lambda event, payload: events.append((event, payload)),
    )

    code = ink_backend._action_benchmark(".", {"benchmarkMode": "runtime"})

    assert code == 1
    assert events == [
        (
            "error",
            {
                "message": (
                    "Ink benchmark UI only supports provider benchmarking. "
                    "Use `kcmt benchmark runtime ...` in the legacy CLI for "
                    "runtime timing."
                )
            },
        )
    ]


def test_action_benchmark_rejects_runtime_specific_options(monkeypatch):
    events: list[tuple[str, dict[str, str]]] = []

    monkeypatch.setattr(
        ink_backend,
        "_emit",
        lambda event, payload: events.append((event, payload)),
    )

    code = ink_backend._action_benchmark(
        ".",
        {"providers": ["openai"], "runtime": "both", "iterations": 1},
    )

    assert code == 1
    assert events == [
        (
            "error",
            {
                "message": (
                    "Runtime benchmark options are not supported in the Ink "
                    "benchmark UI. Use `kcmt benchmark runtime ...` instead."
                )
            },
        )
    ]


def _build_ink_workflow(monkeypatch):
    def fake_init(self, **_kwargs):
        self._config = SimpleNamespace(provider="xai", model="grok-code-fast-1")
        self._progress_history = []

    monkeypatch.setattr(ink_backend.KlingonCMTWorkflow, "__init__", fake_init)

    events: list[tuple[str, dict[str, str]]] = []
    workflow = ink_backend.InkWorkflow(
        lambda event, payload: events.append((event, payload)),
        repo_path=".",
    )
    return workflow, events


def test_ink_workflow_tracks_stage_metadata_and_snapshot_stability(monkeypatch):
    workflow, _events = _build_ink_workflow(monkeypatch)

    times = iter([10.0, 13.5])
    monkeypatch.setattr(ink_backend.time, "time", lambda: next(times))

    workflow._progress_event("diff-ready", file="alpha.py")
    first = workflow.file_states_snapshot()["alpha.py"]

    assert first["current_stage"] == "diff"
    assert first["active_label"] == "collecting diff"
    assert first["stage_started_at"] == 10.0
    assert first["last_update_at"] == 10.0
    assert first["diff"] == "yes"

    first["current_stage"] = "mutated"
    snapshot = workflow.file_states_snapshot()
    assert snapshot["alpha.py"]["current_stage"] == "diff"

    workflow._progress_event("request-sent", file="alpha.py")
    second = workflow.file_states_snapshot()["alpha.py"]

    assert second["current_stage"] == "llm_wait"
    assert second["active_label"] == "waiting for LLM response"
    assert second["stage_started_at"] == 13.5
    assert second["last_update_at"] == 13.5
    assert second["req"] == "sent"
    assert second["batch"] == "validating"


def test_ink_workflow_tracks_terminal_stage_transitions(monkeypatch):
    workflow, _events = _build_ink_workflow(monkeypatch)

    times = iter([20.0, 25.0, 30.0, 45.0])
    monkeypatch.setattr(ink_backend.time, "time", lambda: next(times))

    workflow._progress_event("response", file="alpha.py")
    response_snapshot = workflow.file_states_snapshot()["alpha.py"]
    assert response_snapshot["current_stage"] == "prepared"
    assert response_snapshot["active_label"] == "message prepared"
    assert response_snapshot["res"] == "ok"

    workflow._progress_event("commit-start", file="alpha.py")
    commit_snapshot = workflow.file_states_snapshot()["alpha.py"]
    assert commit_snapshot["current_stage"] == "commit"
    assert commit_snapshot["active_label"] == "writing commit"
    assert commit_snapshot["commit"] == "running"
    assert commit_snapshot["stage_started_at"] == 25.0

    workflow._progress_event("commit-done", file="alpha.py")
    done_snapshot = workflow.file_states_snapshot()["alpha.py"]
    assert done_snapshot["current_stage"] == "done"
    assert done_snapshot["active_label"] == "commit complete"
    assert done_snapshot["commit"] == "ok"
    assert done_snapshot["stage_started_at"] == 30.0

    workflow._progress_event("commit-error", file="bravo.py", detail="boom")
    error_snapshot = workflow.file_states_snapshot()["bravo.py"]
    assert error_snapshot["current_stage"] == "failed"
    assert error_snapshot["active_label"] == "commit failed"
    assert error_snapshot["commit"] == "err"
    assert error_snapshot["stage_started_at"] == 45.0
