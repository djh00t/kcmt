from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import kcmt.benchmark as bench


@dataclass
class _StubClient:
    provider: str
    model: str

    def generate_commit_message(
        self,
        diff: str,
        context: str = "",
        style: str = "conventional",
        request_timeout: float | None = None,
    ) -> str:
        # Deterministic, valid conventional header.
        return f"test({self.model}): update"


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )


def _create_runtime_corpus(tmp_path: Path) -> Path:
    repo = tmp_path / "runtime-corpus"
    repo.mkdir()
    subprocess.run(
        ["git", "init", "--initial-branch=main", str(repo)],
        check=False,
        capture_output=True,
        text=True,
    )
    if not (repo / ".git").exists():
        _git(repo, "init")
        _git(repo, "branch", "-M", "main")
    _git(repo, "config", "user.name", "Tester")
    _git(repo, "config", "user.email", "tester@example.com")

    source_file = repo / "src" / "app.py"
    source_file.parent.mkdir(parents=True, exist_ok=True)
    source_file.write_text(
        "def greet() -> str:\n    return 'hello'\n", encoding="utf-8"
    )
    metadata = {
        "id": "pytest-runtime-corpus",
        "kind": "synthetic",
        "file_count": 1,
        "git_history_state": "seeded-history",
        "change_shape": ["modified", "nested-paths"],
        "default_file_target": "src/app.py",
    }
    (repo / bench.RUNTIME_BENCHMARK_METADATA_FILENAME).write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "chore(repo): seed")
    source_file.write_text(
        "def greet() -> str:\n    return 'hello runtime'\n", encoding="utf-8"
    )
    return repo


def test_run_benchmark_produces_results(monkeypatch):
    def _fake_build_config(provider: str, model: str):
        # Not used by the stub client factory; only present for signature.
        return object()

    def _fake_llm_client(cfg, debug: bool = False):  # noqa: ARG001
        # cfg isn't used; our benchmark code only needs .generate_commit_message.
        # Use a predictable model id via closure.
        raise AssertionError("LLMClient constructor should be monkeypatched per model")

    # Monkeypatch factory points used by run_benchmark.
    monkeypatch.setattr(bench, "_build_config", _fake_build_config)

    # Create a per-model stub client by monkeypatching LLMClient to a callable
    # that closes over a module-level mutable.
    current = {"provider": None, "model": None}

    class _LLMClientFactory:
        def __init__(self, cfg, debug: bool = False):  # noqa: ARG002
            provider = current["provider"]
            model = current["model"]
            assert provider is not None
            assert model is not None
            self._client = _StubClient(provider=provider, model=model)

        def generate_commit_message(self, *args, **kwargs):
            return self._client.generate_commit_message(*args, **kwargs)

    monkeypatch.setattr(bench, "LLMClient", _LLMClientFactory)

    models_map = {
        "openai": [
            {
                "id": "m1",
                "input_price_per_mtok": 1.0,
                "output_price_per_mtok": 2.0,
            },
            {
                "id": "m2",
                "input_price_per_mtok": 1.0,
                "output_price_per_mtok": 2.0,
            },
        ]
    }

    # Set the current model before each client instantiation.
    orig_build_config = bench._build_config

    def _wrapped_build_config(provider: str, model: str):
        current["provider"] = provider
        current["model"] = model
        return orig_build_config(provider, model)

    monkeypatch.setattr(bench, "_build_config", _wrapped_build_config)

    results, exclusions = bench.run_benchmark(models_map, per_provider_limit=1)

    assert not exclusions
    assert results
    assert results[0].provider == "openai"
    assert results[0].model == "m1"
    assert results[0].runs == len(bench.sample_diffs())


def test_run_benchmark_detailed_respects_allowlist(monkeypatch):
    # Minimal stub that always returns a valid conventional header.
    current = {"provider": None, "model": None}

    def _fake_build_config(provider: str, model: str):
        current["provider"] = provider
        current["model"] = model
        return object()

    class _LLMClientFactory:
        def __init__(self, cfg, debug: bool = False):  # noqa: ARG002
            provider = current["provider"]
            model = current["model"]
            assert provider is not None
            assert model is not None
            self._client = _StubClient(provider=provider, model=model)

        def generate_commit_message(self, *args, **kwargs):
            return self._client.generate_commit_message(*args, **kwargs)

    monkeypatch.setattr(bench, "_build_config", _fake_build_config)
    monkeypatch.setattr(bench, "LLMClient", _LLMClientFactory)

    models_map = {
        "openai": [
            {"id": "m1", "input_price_per_mtok": 1.0, "output_price_per_mtok": 2.0},
            {"id": "m2", "input_price_per_mtok": 1.0, "output_price_per_mtok": 2.0},
        ]
    }

    allowlist = {"openai": {"m2"}}
    results, exclusions, details = bench.run_benchmark_detailed(
        models_map,
        provider_model_allowlist=allowlist,
    )

    assert not exclusions
    assert results
    assert {r.model for r in results} == {"m2"}
    assert details
    assert all(d.model == "m2" for d in details)


def test_run_runtime_benchmark_produces_python_results(tmp_path):
    repo = _create_runtime_corpus(tmp_path)

    report = bench.run_runtime_benchmark(
        repo,
        runtime="python",
        iterations=1,
    )
    payload = bench.runtime_benchmark_report_to_dict(report)

    schema_path = (
        Path(__file__).resolve().parents[1]
        / "specs"
        / "003-bring-rust-cli"
        / "contracts"
        / "runtime-benchmark.schema.json"
    )
    schema = json.loads(schema_path.read_text(encoding="utf-8"))

    assert payload["schema_version"] == schema["properties"]["schema_version"]["const"]
    assert set(schema["required"]).issubset(payload)
    assert payload["corpora"] == ["pytest-runtime-corpus"]
    assert len(payload["results"]) == 3
    assert {item["runtime"] for item in payload["results"]} == {"python"}
    assert all(item["status"] == "passed" for item in payload["results"])
    required_result_fields = set(schema["properties"]["results"]["items"]["required"])
    assert all(required_result_fields.issubset(item) for item in payload["results"])
    assert payload["summary"]["python"]["passed"] == 3
    assert payload["summary"]["rust"]["scenario_count"] == 0


def test_run_runtime_benchmark_records_missing_rust_binary_as_excluded(tmp_path):
    repo = _create_runtime_corpus(tmp_path)

    report = bench.run_runtime_benchmark(
        repo,
        runtime="rust",
        iterations=1,
        rust_bin=tmp_path / "missing-kcmt",
    )
    payload = bench.runtime_benchmark_report_to_dict(report)

    assert len(payload["results"]) == 3
    assert all(item["status"] == "excluded" for item in payload["results"])
    assert all(
        "Rust binary not available" in (item["failure_reason"] or "")
        for item in payload["results"]
    )
    assert payload["summary"]["rust"]["excluded"] == 3


def test_render_benchmark_markdown_report_remains_provider_focused() -> None:
    results = [
        bench.BenchResult(
            provider="openai",
            model="gpt-5-mini",
            avg_latency_ms=125.0,
            avg_cost_usd=0.0042,
            quality=91.5,
            success_rate=1.0,
            runs=5,
        )
    ]
    exclusions = [
        bench.BenchExclusion(
            provider="anthropic",
            model="claude-3-5-haiku-latest",
            reason="missing_api_key",
            detail="ANTHROPIC_API_KEY is not set",
        )
    ]

    report = bench.render_benchmark_markdown_report(
        results,
        exclusions,
        timestamp="2026-03-16T10:00:00Z",
        repo_path="/tmp/provider-benchmark-repo",
        params={"providers": ["openai"], "limit": 1},
    )

    assert "# kcmt Benchmark Report" in report
    assert "## Run Summary" in report
    assert "Providers: openai" in report
    assert "gpt-5-mini" in report
    assert "Excluded Models" in report
    assert "local-workflows-v1" not in report
    assert "command_set" not in report
