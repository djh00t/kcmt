from __future__ import annotations

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
