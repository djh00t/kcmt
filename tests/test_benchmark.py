from __future__ import annotations

from dataclasses import dataclass

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
