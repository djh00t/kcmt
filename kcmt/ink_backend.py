# coverage: ignore-file
"""JSON-stream backend powering the Ink UI."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import re
import sys
import threading
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TextIO, Tuple

from .benchmark import BenchResult, run_benchmark_detailed
from .config import (
    BATCH_TIMEOUT_MIN_SECONDS,
    DEFAULT_MODELS,
    PROVIDER_DISPLAY_NAMES,
    Config,
    detect_available_providers,
    load_config,
    load_persisted_config,
    load_preferences,
    save_config,
    save_preferences,
    state_dir,
)
from .core import KlingonCMTWorkflow
from .exceptions import GitError, KlingonCMTError, LLMError
from .git import find_git_repo_root

try:  # Optional imports only needed for model enrichment
    from .providers.anthropic_driver import AnthropicDriver
    from .providers.openai_driver import OpenAIDriver
    from .providers.pricing import build_enrichment_context, enrich_ids
    from .providers.xai_driver import XAIDriver
except Exception:  # pragma: no cover - guard optional deps for packaging
    AnthropicDriver = OpenAIDriver = XAIDriver = None  # type: ignore[assignment,misc]
    build_enrichment_context = enrich_ids = None  # type: ignore[assignment]


def _serialise(value: Any) -> Any:
    if is_dataclass(value):
        return {key: _serialise(val) for key, val in asdict(value).items()}  # type: ignore[arg-type]
    if isinstance(value, dict):
        return {str(key): _serialise(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialise(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _emit(event: str, payload: Dict[str, Any]) -> None:
    message = {"event": event, "payload": _serialise(payload)}
    # NOTE: use the original interpreter streams so our JSON event protocol is
    # not swallowed by redirect_stdout/redirect_stderr contexts used elsewhere
    # to suppress incidental prints from third-party code.
    target: TextIO
    if event == "tick":
        target = sys.__stderr__ or sys.stderr
    else:
        target = sys.__stdout__ or sys.stdout
    target.write(json.dumps(message) + "\n")
    target.flush()


def _resolve_repo_root(repo_path: Optional[str]) -> Path:
    base = Path(repo_path or ".").expanduser().resolve(strict=False)
    detected = find_git_repo_root(base)
    return (detected or base).resolve(strict=False)


def _load_provider_config(provider: str, repo_root: Path) -> Config:
    overrides = {"provider": provider}
    try:
        cfg = load_config(repo_root=repo_root, overrides=overrides)
    except Exception:  # pragma: no cover - defensive fallback
        meta = DEFAULT_MODELS.get(provider, {})
        cfg = Config(
            provider=provider,
            model=str(meta.get("model", "")),
            llm_endpoint=str(meta.get("endpoint", "")),
            api_key_env=str(meta.get("api_key_env", "")),
        )
    cfg.git_repo_path = str(repo_root)
    return cfg


def _driver_for_provider(provider: str, cfg: Config) -> Any:
    if provider in {"openai", "github"} and OpenAIDriver is not None:
        return OpenAIDriver(cfg, debug=False)
    if provider == "xai" and XAIDriver is not None:
        return XAIDriver(cfg, debug=False)
    if AnthropicDriver is not None:
        return AnthropicDriver(cfg, debug=False)
    raise RuntimeError("Required provider drivers are unavailable")


def _list_enriched_models(provider: str, repo_root: Path) -> list[dict[str, Any]]:
    # Cache provider model catalogs on disk to avoid repeated network calls.
    try:
        ttl_seconds = int(os.environ.get("KCMT_MODEL_CACHE_TTL_SECONDS", "86400"))
    except Exception:
        ttl_seconds = 86400

    cache_root = state_dir(repo_root) / "model_catalog"
    cache_path = cache_root / f"{provider}.json"
    now = time.time()
    if ttl_seconds > 0:
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if isinstance(cached, dict):
                ts = float(cached.get("ts", 0.0))
                cached_models_raw = cached.get("models")
                if (
                    ts
                    and (now - ts) < ttl_seconds
                    and isinstance(cached_models_raw, list)
                    and all(isinstance(item, dict) for item in cached_models_raw)
                ):
                    return [dict(item) for item in cached_models_raw]
        except Exception:
            pass

    cfg = _load_provider_config(provider, repo_root)
    models: list[dict[str, Any]] = []
    try:
        driver = _driver_for_provider(provider, cfg)
        fetched = driver.list_models()
        if isinstance(fetched, list):
            models = [dict(item) for item in fetched if isinstance(item, dict)]
    except Exception:  # pragma: no cover - providers may reject list calls
        models = []

    if not models and build_enrichment_context and enrich_ids:  # type: ignore[truthy-function]
        try:
            alias_lut, _ctx, _meta = build_enrichment_context()
            ids: list[str] = []
            seen: set[str] = set()
            for (prov, model_id), canonical in alias_lut.items():
                if prov != provider:
                    continue
                for candidate in (str(canonical), str(model_id)):
                    if candidate and candidate not in seen:
                        ids.append(candidate)
                        seen.add(candidate)
            enriched = enrich_ids(provider, ids)
            for model_id in ids:
                details = dict(enriched.get(model_id) or {})
                if not details:
                    continue
                details.pop("_has_pricing", None)
                entry = {"id": model_id, **details}
                models.append(entry)
        except Exception:  # pragma: no cover - enrichment is best effort
            models = []

    filtered: list[dict[str, Any]] = []
    for item in models:
        if not isinstance(item, dict):
            continue
        identifier = str(item.get("id") or item.get("model") or "").strip()
        if not identifier:
            continue
        if (
            item.get("input_price_per_mtok") is None
            and item.get("output_price_per_mtok") is None
        ):
            continue
        filtered.append({"id": identifier, **{k: item[k] for k in item if k != "id"}})

    if ttl_seconds > 0:
        try:
            cache_root.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(
                json.dumps({"ts": now, "models": filtered}, indent=2, sort_keys=True),
                encoding="utf-8",
            )
        except Exception:
            pass
    return filtered


def _build_model_catalog(repo_root: Path) -> dict[str, list[dict[str, Any]]]:
    catalog: dict[str, list[dict[str, Any]]] = {}
    for provider in DEFAULT_MODELS:
        catalog[provider] = _list_enriched_models(provider, repo_root)
    return catalog


def _benchmark_progress(stage: str, info: dict[str, Any]) -> None:
    label = stage
    if stage == "provider":
        label = f"Scanning {info.get('provider', '')}"
    elif stage == "model_start":
        label = f"Testing {info.get('provider', '')}/{info.get('model', '')}"
    elif stage == "tick":
        label = (
            f"{info.get('done', 0)} of {info.get('total', 0)} "
            f"({info.get('provider', '')}/{info.get('model', '')})"
        )
    elif stage == "done":
        label = "Benchmark complete"
    _emit("progress", {"stage": stage, "label": label, "info": info})


def _build_leaderboards(results: list[BenchResult]) -> dict[str, Any]:
    if not results:
        return {"overall": [], "fastest": [], "cheapest": [], "best_quality": []}

    def _fmt(result: BenchResult) -> dict[str, Any]:
        return {
            "provider": result.provider,
            "model": result.model,
            "avg_latency_ms": result.avg_latency_ms,
            "avg_cost_usd": result.avg_cost_usd,
            "quality": result.quality,
            "success_rate": result.success_rate,
            "runs": result.runs,
        }

    fastest = sorted(results, key=lambda r: (r.avg_latency_ms, r.avg_cost_usd))[:10]
    cheapest = sorted(results, key=lambda r: (r.avg_cost_usd, r.avg_latency_ms))[:10]
    best_quality = sorted(results, key=lambda r: (-r.quality, r.avg_latency_ms))[:10]

    min_lat = min(r.avg_latency_ms for r in results)
    max_lat = max(r.avg_latency_ms for r in results)
    min_cost = min(r.avg_cost_usd for r in results)
    max_cost = max(r.avg_cost_usd for r in results)

    def _norm(value: float, low: float, high: float) -> float:
        if high <= low:
            return 1.0
        return (value - low) / (high - low)

    overall_pairs: list[tuple[float, BenchResult]] = []
    for item in results:
        quality_score = item.quality / 100.0
        cost_score = 1.0 - _norm(item.avg_cost_usd, min_cost, max_cost)
        latency_score = 1.0 - _norm(item.avg_latency_ms, min_lat, max_lat)
        overall_score = 0.4 * quality_score + 0.3 * cost_score + 0.3 * latency_score
        overall_pairs.append((overall_score, item))
    overall = [
        entry for _score, entry in sorted(overall_pairs, key=lambda kv: -kv[0])[:10]
    ]

    return {
        "overall": [_fmt(r) for r in overall],
        "fastest": [_fmt(r) for r in fastest],
        "cheapest": [_fmt(r) for r in cheapest],
        "best_quality": [_fmt(r) for r in best_quality],
    }


class InkWorkflow(KlingonCMTWorkflow):
    """Workflow subclass that emits structured progress instead of printing."""

    def __init__(
        self, emitter: Callable[[str, Dict[str, Any]], None], **kwargs: Any
    ) -> None:
        super().__init__(show_progress=True, **kwargs)
        self._emitter = emitter
        # Maintain a compact per-file state map for the Ink UI without printing
        # to stdout. Keys mirror the legacy renderer but avoid terminal output.
        self._file_states: dict[str, dict[str, str]] = {}

    def _clear_progress_line(self) -> None:  # pragma: no cover - disabled
        return

    def _refresh_progress_line(self) -> None:  # pragma: no cover - disabled
        return

    def _print_progress(self, stage: str) -> None:
        snapshot = self._stats.snapshot()
        payload = {"stage": stage, "stats": snapshot}
        self._emitter("progress", payload)

    def _finalize_progress(self) -> None:
        snapshot = self._stats.snapshot()
        payload = {"stage": "done", "stats": snapshot}
        self._emitter("progress", payload)

    def _print_commit_generated(self, file_path: str, commit_message: str) -> None:
        lines = (commit_message or "").splitlines()
        subject = lines[0] if lines else commit_message
        body = "\n".join(lines[1:]) if len(lines) > 1 else ""
        self._emitter(
            "commit-generated",
            {
                "file": file_path,
                "subject": subject,
                "body": body,
            },
        )

    def _print_prepare_error(self, file_path: str, error: str) -> None:
        self._emitter(
            "prepare-error",
            {
                "file": file_path,
                "error": error,
            },
        )

    async def _prepare_commit_messages_async(
        self, file_changes: List[Any]
    ) -> List[Tuple[int, Any]]:
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            result = await super()._prepare_commit_messages_async(file_changes)
        leftover = buffer.getvalue().strip()
        if leftover:
            self._emitter("log", {"message": leftover})
        return result

    def _progress_event(self, kind: str, **info: object) -> None:
        # Update our in-memory file state map without writing to stdout.
        file_path = str(info.get("file") or "")
        if file_path:
            entry = self._file_states.setdefault(
                file_path,
                {"diff": "-", "req": "-", "res": "-", "batch": "-", "commit": "-"},
            )
            if kind == "diff-ready":
                entry["diff"] = "yes"
            elif kind == "request-sent":
                entry["req"] = "sent"
                entry["batch"] = "validating"
            elif kind == "response":
                entry["res"] = "ok"
                # For non-batch flows, response implies ready
                if entry.get("batch", "-") == "-":
                    entry["batch"] = "completed"
            elif kind == "llm" and info.get("detail"):
                detail = str(info.get("detail") or "").strip().lower()
                label = detail
                if detail.startswith("batch status"):
                    parts = detail.replace("batch status:", "").strip().split()
                    label = (parts[0] if parts else "").strip().lower()
                if label in {"validating", "queued"}:
                    entry["batch"] = "validating"
                elif label in {"running", "in_progress", "in-progress"}:
                    entry["batch"] = "in_progress"
                elif label == "finalizing":
                    entry["batch"] = "finalizing"
                elif label == "completed":
                    entry["batch"] = "completed"
                elif label:
                    entry["batch"] = label[:12]
            elif kind == "commit-start":
                entry["commit"] = "running"
            elif kind == "commit-done":
                entry["commit"] = "ok"
            elif kind == "commit-error":
                entry["commit"] = "err"

        message = self._format_progress_message(kind, info)
        if not message:
            return
        payload = {"message": message, "stage": kind, **info}
        self._emitter("status", payload)

    def file_states_snapshot(self) -> dict[str, dict[str, str]]:
        # Return a shallow copy safe for JSON serialisation
        return {k: dict(v) for k, v in self._file_states.items()}


def _action_bootstrap(repo_path: str, payload: dict[str, Any]) -> int:
    repo_root = _resolve_repo_root(repo_path)
    overrides = payload.get("overrides") or {}
    argv_payload = payload.get("argv")
    argv = argv_payload if isinstance(argv_payload, dict) else {}
    persisted = load_persisted_config(repo_root)
    if persisted:
        config = persisted
    else:
        config = load_config(repo_root=repo_root, overrides=overrides)
    detection = detect_available_providers()
    try:
        preferences = load_preferences(repo_root)
    except Exception:  # pragma: no cover - prefs optional
        preferences = {}
    # Skip expensive model catalog build for fast startup; Configure view will lazy-load
    response = {
        "repoRoot": str(repo_root),
        "config": _serialise(config),
        "persisted": persisted is not None,
        "defaultModels": DEFAULT_MODELS,
        "providerDetection": detection,
        "preferences": preferences,
        "modelCatalog": {},
        "argv": argv,
    }
    _emit("complete", response)
    return 0


def _action_save_config(repo_path: str, payload: dict[str, Any]) -> int:
    config_payload = payload.get("config")
    if not isinstance(config_payload, dict):
        _emit("error", {"message": "config payload missing"})
        return 2

    repo_root = _resolve_repo_root(repo_path)
    base = load_config(repo_root=repo_root)

    def looks_like_url(value: str) -> bool:
        return bool(value) and (
            "://" in value
            or value.startswith("http://")
            or value.startswith("https://")
        )

    def looks_like_env(value: str) -> bool:
        return (
            bool(value) and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value) is not None
        )

    # Start from existing providers map, ensuring default structure.
    providers_map: dict[str, dict[str, Any]] = {
        prov: dict(entry or {}) for prov, entry in (base.providers or {}).items()
    }
    for prov, meta in DEFAULT_MODELS.items():
        entry = providers_map.get(prov, {}) or {}
        entry.setdefault("name", PROVIDER_DISPLAY_NAMES.get(prov, prov))
        entry.setdefault("endpoint", meta["endpoint"])
        entry.setdefault("api_key_env", meta["api_key_env"])
        entry.setdefault("preferred_model", None)
        providers_map[prov] = entry

    incoming_providers = config_payload.get("providers")
    if isinstance(incoming_providers, dict):
        for prov, raw_entry in incoming_providers.items():
            if prov not in DEFAULT_MODELS or not isinstance(raw_entry, dict):
                continue
            default_endpoint = DEFAULT_MODELS[prov]["endpoint"]
            default_env = DEFAULT_MODELS[prov]["api_key_env"]
            endpoint_val = str(
                raw_entry.get("endpoint", providers_map[prov]["endpoint"])
            )
            api_env_val = str(
                raw_entry.get("api_key_env", providers_map[prov]["api_key_env"])
            )

            if (
                looks_like_url(api_env_val)
                and looks_like_env(endpoint_val)
                and not looks_like_url(endpoint_val)
            ):
                api_env_val, endpoint_val = endpoint_val, api_env_val
            if not looks_like_env(api_env_val):
                api_env_val = default_env
            if not looks_like_url(endpoint_val):
                endpoint_val = default_endpoint

            providers_map[prov]["endpoint"] = endpoint_val
            providers_map[prov]["api_key_env"] = api_env_val
            if raw_entry.get("preferred_model"):
                providers_map[prov]["preferred_model"] = str(
                    raw_entry["preferred_model"]
                )

    # Top-level provider/model overrides (may be omitted when using priority list)
    provider = str(config_payload.get("provider", base.provider)) or base.provider
    model = str(config_payload.get("model", base.model)) or base.model
    endpoint_raw = str(
        config_payload.get(
            "llm_endpoint",
            providers_map.get(provider, {}).get("endpoint", base.llm_endpoint),
        )
    )
    api_env_raw = str(
        config_payload.get(
            "api_key_env",
            providers_map.get(provider, {}).get("api_key_env", base.api_key_env),
        )
    )

    if (
        looks_like_url(api_env_raw)
        and looks_like_env(endpoint_raw)
        and not looks_like_url(endpoint_raw)
    ):
        api_env_raw, endpoint_raw = endpoint_raw, api_env_raw
    if provider in DEFAULT_MODELS and not looks_like_env(api_env_raw):
        api_env_raw = DEFAULT_MODELS[provider]["api_key_env"]
    if provider in DEFAULT_MODELS and not looks_like_url(endpoint_raw):
        endpoint_raw = DEFAULT_MODELS[provider]["endpoint"]

    priority_payload = config_payload.get("model_priority")
    priority_list: list[dict[str, str]] = []
    if isinstance(priority_payload, list):
        for item in priority_payload:
            if not isinstance(item, dict):
                continue
            prov = str(item.get("provider", "")).strip()
            model_name = str(item.get("model", "")).strip()
            if prov in DEFAULT_MODELS and model_name:
                priority_list.append({"provider": prov, "model": model_name})

    if not priority_list and provider in DEFAULT_MODELS and model:
        priority_list.append({"provider": provider, "model": model})

    seen_pairs: set[tuple[str, str]] = set()
    sanitised_priority: list[dict[str, str]] = []
    for pref in priority_list:
        pair = (pref["provider"], pref["model"])
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        sanitised_priority.append(pref)
        if len(sanitised_priority) >= 5:
            break
    if not sanitised_priority and provider in DEFAULT_MODELS:
        sanitised_priority.append(
            {"provider": provider, "model": DEFAULT_MODELS[provider]["model"]}
        )

    active_provider = sanitised_priority[0]["provider"]
    active_model = sanitised_priority[0]["model"]

    # Ensure providers map aligns with active selection
    for prov, entry in providers_map.items():
        entry.setdefault("endpoint", DEFAULT_MODELS[prov]["endpoint"])
        entry.setdefault("api_key_env", DEFAULT_MODELS[prov]["api_key_env"])
        entry.setdefault("preferred_model", DEFAULT_MODELS[prov]["model"])
    providers_map[active_provider]["endpoint"] = endpoint_raw
    providers_map[active_provider]["api_key_env"] = api_env_raw
    providers_map[active_provider]["preferred_model"] = active_model

    first_by_provider: dict[str, str] = {}
    for pref in sanitised_priority:
        first_by_provider.setdefault(pref["provider"], pref["model"])
    for prov, entry in providers_map.items():
        if prov in first_by_provider:
            entry["preferred_model"] = first_by_provider[prov]

    base.provider = active_provider
    base.model = active_model
    base.llm_endpoint = endpoint_raw
    base.api_key_env = api_env_raw
    base.git_repo_path = str(repo_root)
    base.providers = providers_map
    base.model_priority = sanitised_priority
    use_batch_val = (
        bool(config_payload.get("use_batch")) if active_provider == "openai" else False
    )
    base.use_batch = use_batch_val
    if active_provider == "openai":
        batch_model_val = (
            config_payload.get("batch_model")
            or providers_map["openai"].get("preferred_model")
            or base.model
        )
        base.batch_model = str(batch_model_val)
        timeout_raw = (
            config_payload.get("batch_timeout_seconds") or base.batch_timeout_seconds
        )
        try:
            base.batch_timeout_seconds = int(float(timeout_raw))
        except (TypeError, ValueError):
            pass
        base.batch_timeout_seconds = max(
            BATCH_TIMEOUT_MIN_SECONDS, base.batch_timeout_seconds
        )
    else:
        base.batch_model = None

    save_config(base, repo_root)
    _emit("complete", {"config": _serialise(base)})
    return 0


def _action_save_preferences(repo_path: str, payload: dict[str, Any]) -> int:
    prefs = payload.get("preferences")
    if not isinstance(prefs, dict):
        _emit("error", {"message": "preferences payload missing"})
        return 2
    repo_root = _resolve_repo_root(repo_path)
    save_preferences(prefs, repo_root)
    _emit("complete", {"preferences": prefs})
    return 0


def _action_benchmark(repo_path: str, payload: dict[str, Any]) -> int:
    repo_root = _resolve_repo_root(repo_path)
    providers_payload = payload.get("providers")
    providers_explicit = isinstance(providers_payload, list) and bool(providers_payload)
    providers = providers_payload
    if not providers:
        providers = list(DEFAULT_MODELS.keys())
    limit = payload.get("limit")
    if isinstance(limit, str) and limit.isdigit():
        limit = int(limit)
    timeout = payload.get("timeout")
    try:
        timeout_value = float(timeout) if timeout is not None else None
    except (TypeError, ValueError):
        timeout_value = None

    only_models = payload.get("onlyModels")
    only_providers = payload.get("onlyProviders")

    allowlist: dict[str, set[str]] | None = None
    disabled_providers: set[str] = set()
    if not only_models:
        try:
            prefs = load_preferences(repo_root)
        except Exception:  # pragma: no cover - prefs optional
            prefs = {}

        bench_cfg = prefs.get("benchmark") if isinstance(prefs, dict) else None
        providers_cfg = None
        if isinstance(bench_cfg, dict):
            providers_cfg = bench_cfg.get("providers")

        if isinstance(providers_cfg, dict):
            allowlist = {}
            for prov, entry in providers_cfg.items():
                if not prov:
                    continue
                models: list[str] = []
                if isinstance(entry, list):
                    models = [str(m).strip() for m in entry if str(m).strip()]
                elif isinstance(entry, dict):
                    if entry.get("enabled") is False:
                        disabled_providers.add(str(prov))
                    raw_models = entry.get("models")
                    if isinstance(raw_models, list):
                        models = [str(m).strip() for m in raw_models if str(m).strip()]
                if models:
                    allowlist[str(prov)] = set(models)
            if not allowlist:
                allowlist = None

    # Apply disabled providers only when the benchmark did not explicitly
    # request providers via flags.
    if disabled_providers and not providers_explicit and not only_providers:
        providers = [prov for prov in providers if str(prov) not in disabled_providers]

    catalog = {
        provider: _list_enriched_models(provider, repo_root) for provider in providers
    }

    results, exclusions, details = run_benchmark_detailed(
        catalog,
        per_provider_limit=int(limit) if limit else None,
        request_timeout=timeout_value,
        debug=bool(payload.get("debug")),
        progress=_benchmark_progress,
        only_models=only_models,
        only_providers=only_providers,
        provider_model_allowlist=allowlist,
    )
    leaderboards = _build_leaderboards(results)
    response = {
        **leaderboards,
        "exclusions": [_serialise(ex) for ex in exclusions],
        "details": [_serialise(item) for item in details],
    }
    if payload.get("includeRaw"):
        response["raw"] = [_serialise(item) for item in results]
    _emit("complete", response)
    return 0


def _action_workflow(repo_path: str, payload: dict[str, Any]) -> int:
    repo_root = _resolve_repo_root(payload.get("repoPath") or repo_path)

    # Ink UI prioritizes responsiveness; avoid the optional "second call" used
    # to add a commit body (header is sufficient for committing).
    os.environ.setdefault("KCMT_DISABLE_BODY_ENRICHMENT", "1")

    overrides = payload.get("overrides") or {}
    config = load_config(repo_root=repo_root, overrides=overrides)
    config.git_repo_path = str(repo_root)
    auto_push = payload.get("autoPush")
    if auto_push is not None:
        config.auto_push = bool(auto_push)
    max_commit_length = payload.get("maxCommitLength")
    if max_commit_length:
        try:
            config.max_commit_length = int(max_commit_length)
        except (TypeError, ValueError):  # pragma: no cover - validation
            pass

    max_retries = payload.get("maxRetries")
    try:
        max_retries_value = int(max_retries) if max_retries is not None else 3
    except (TypeError, ValueError):
        max_retries_value = 3

    limit = payload.get("limit")
    try:
        limit_value = int(limit) if limit is not None else None
    except (TypeError, ValueError):
        limit_value = None

    workers = payload.get("workers")
    try:
        workers_value = int(workers) if workers is not None else None
    except (TypeError, ValueError):
        workers_value = None

    stage_tracker = {"value": "prepare"}

    def _emitter(event: str, info: Dict[str, Any]) -> None:
        if event == "progress" and isinstance(info, dict):
            stage_tracker["value"] = str(info.get("stage", stage_tracker["value"]))
        _emit(event, info)

    workflow = InkWorkflow(
        _emitter,
        repo_path=str(repo_root),
        max_retries=max_retries_value,
        config=config,
        file_limit=limit_value,
        debug=bool(payload.get("debug")),
        profile=bool(payload.get("profile")),
        verbose=bool(payload.get("verbose")),
        workers=workers_value,
    )

    result_holder: dict[str, Any] = {}
    error_holder: list[str] = []

    def _runner() -> None:
        try:
            result_holder["result"] = workflow.execute_workflow()
        except (GitError, KlingonCMTError, LLMError) as exc:
            error_holder.append(str(exc))
        except Exception as exc:  # pragma: no cover - unexpected failure
            error_holder.append(str(exc))

    thread = threading.Thread(target=_runner, daemon=True)
    thread.start()

    last_sent: Optional[dict[str, Any]] = None
    while thread.is_alive():
        snapshot = workflow.stats_snapshot()
        if snapshot != last_sent:
            files = {}
            try:
                files = workflow.file_states_snapshot()
            except Exception:  # pragma: no cover - best-effort UI telemetry
                files = {}
            _emit(
                "tick",
                {
                    "stage": stage_tracker["value"],
                    "stats": snapshot,
                    "files": files,
                },
            )
            last_sent = snapshot
        time.sleep(0.2)
    thread.join()

    if error_holder:
        _emit("error", {"message": error_holder[0]})
        return 1

    result = result_holder.get("result", {})
    metrics_summary: str | None = None
    metrics_obj = getattr(workflow, "_metrics", None)
    if metrics_obj is not None:
        try:
            metrics_summary = str(metrics_obj.summary())
        except Exception:  # pragma: no cover - defensive
            metrics_summary = None
    response = {
        "result": _serialise(result),
        "stats": workflow.stats_snapshot(),
        "commit_subjects": workflow.commit_subjects(),
        "metrics_summary": metrics_summary,
    }
    _emit("complete", response)
    return 0


def _action_list_models(repo_path: str, payload: dict[str, Any]) -> int:
    """Lazy-load model catalog for specific providers on demand."""
    repo_root = _resolve_repo_root(repo_path)
    providers = payload.get("providers")
    if not providers:
        providers = list(DEFAULT_MODELS.keys())
    elif isinstance(providers, str):
        providers = [providers]

    catalog: dict[str, list[dict[str, Any]]] = {}
    for provider in providers:
        if provider in DEFAULT_MODELS:
            catalog[provider] = _list_enriched_models(provider, repo_root)

    response = {"modelCatalog": catalog}
    _emit("complete", response)
    return 0


_ACTIONS = {
    "bootstrap": _action_bootstrap,
    "save-config": _action_save_config,
    "save-preferences": _action_save_preferences,
    "benchmark": _action_benchmark,
    "workflow": _action_workflow,
    "list-models": _action_list_models,
}


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="kcmt Ink backend")
    parser.add_argument("action", choices=sorted(_ACTIONS.keys()))
    parser.add_argument("--repo-path", dest="repo_path", default=".")
    parser.add_argument("--payload", default="{}")
    args = parser.parse_args(argv)

    try:
        payload = json.loads(args.payload or "{}")
    except json.JSONDecodeError as exc:
        _emit("error", {"message": f"Invalid payload: {exc}"})
        return 2

    handler = _ACTIONS[args.action]
    return handler(args.repo_path, payload)


if __name__ == "__main__":  # pragma: no cover - manual execution
    sys.exit(main())
