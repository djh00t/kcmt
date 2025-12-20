from __future__ import annotations

import re
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

from .config import (
    Config,
    DEFAULT_MODELS,
    PROVIDER_DISPLAY_NAMES,
    load_config,
    state_dir,
)
from .exceptions import LLMError
from .llm import LLMClient

# -------------------------------
# Sample diffs for benchmarking
# -------------------------------


def sample_diffs() -> list[tuple[str, str]]:
    """Return a fixed set of (name, diff) pairs for benchmarking.

    The diffs are intentionally small but representative: feature add,
    bug fix, docs update, refactor, and tests.
    """
    return [
        (
            "feature-add",
            """diff --git a/app/math.py b/app/math.py
index 111..222 100644
--- a/app/math.py
+++ b/app/math.py
@@
 def add(a, b):
     return a + b

@@
 def multiply(a, b):
     return a * b

@@
 +def divide(a, b):
 +    if b == 0:
 +        raise ValueError("division by zero")
 +    return a / b
""",
        ),
        (
            "bugfix-conditional",
            """diff --git a/app/auth.py b/app/auth.py
index 333..444 100644
--- a/app/auth.py
+++ b/app/auth.py
@@
-if user.is_admin:
-    allow = True
+if user and user.is_admin:
+    allow = True
 else:
     allow = False
""",
        ),
        (
            "docs-update",
            """diff --git a/README.md b/README.md
index 555..666 100644
--- a/README.md
+++ b/README.md
@@
 # Project Title
 
-A tool.
+A tool for atomic git commits with AI assistance.
 
 ## Usage
""",
        ),
        (
            "refactor-utils",
            '''diff --git a/app/utils.py b/app/utils.py
index 777..888 100644
--- a/app/utils.py
+++ b/app/utils.py
@@
-def slugify(s):
-    return s.lower().replace(" ", "-")
+def slugify(text: str) -> str:
+    """Return URL-safe slug."""
+    s = text.strip().lower()
+    return re.sub(r"[^a-z0-9-]", "-", s.replace(" ", "-"))
''',
        ),
        (
            "tests-add",
            """diff --git a/tests/test_math.py b/tests/test_math.py
index 999..aaa 100644
--- a/tests/test_math.py
+++ b/tests/test_math.py
@@
 def test_divide():
     assert divide(6, 3) == 2
""",
        ),
    ]


# -------------------------------
# Scoring & utilities
# -------------------------------


_CONVENTIONAL_RE = re.compile(
    r"^(feat|fix|docs|style|refactor|test|chore|perf|ci|build|revert)(\([^)]+\))?:\s+.+"
)

_GENERIC_WORDS = {
    "update",
    "updates",
    "change",
    "changes",
    "fixes",
    "stuff",
    "minor",
    "tweak",
    "misc",
}

_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "from",
    "this",
    "that",
    "into",
    "return",
    "raise",
    "value",
    "error",
    "true",
    "false",
    "none",
    "import",
    "def",
    "class",
}


def approx_tokens(text: str) -> int:
    # Simple heuristic: ~4 chars per token
    return max(1, int(len(text) / 4))


def extract_keywords_from_diff(diff: str, limit: int = 12) -> list[str]:
    words: dict[str, int] = {}
    for line in diff.splitlines():
        if not line:
            continue
        if line.startswith("+++") or line.startswith("---"):
            continue
        if not (line.startswith("+") or line.startswith("-")):
            continue
        for raw in re.split(r"[^a-zA-Z0-9_]+", line):
            token = raw.strip().lower()
            if len(token) < 4 or token in _STOPWORDS:
                continue
            words[token] = words.get(token, 0) + 1
    sorted_items = sorted(words.items(), key=lambda kv: (-kv[1], kv[0]))
    return [w for (w, _cnt) in sorted_items[:limit]]


def score_quality(diff: str, message: str, max_subject: int = 72) -> dict[str, Any]:
    """Score conventional commit quality with a heuristic, 0..100.

    Breakdown keys: format, scope, subject_len, specificity, body, penalties.
    """
    score = 0.0
    breakdown: dict[str, float] = {
        "format": 0.0,
        "scope": 0.0,
        "subject_len": 0.0,
        "specificity": 0.0,
        "body": 0.0,
        "penalties": 0.0,
    }

    lines = [ln for ln in (message or "").strip().splitlines() if ln is not None]
    subject = lines[0].strip() if lines else ""

    m = _CONVENTIONAL_RE.match(subject)
    if m:
        breakdown["format"] = 30.0
        score += 30.0
        if "(" in subject.split(":", 1)[0]:
            breakdown["scope"] = 10.0
            score += 10.0

    # Subject length adherence
    if subject:
        if len(subject) <= max_subject:
            breakdown["subject_len"] = 10.0
            score += 10.0
        else:
            over = len(subject) - max_subject
            penalty = min(10.0, over * 0.2)
            breakdown["subject_len"] = 10.0 - penalty
            score += 10.0 - penalty

    # Specificity: overlap between subject words and diff keywords
    kws = set(extract_keywords_from_diff(diff))
    subj_words = {
        w.lower() for w in re.split(r"[^a-zA-Z0-9_]+", subject) if len(w) >= 4
    }
    matches = len(kws.intersection(subj_words))
    if matches >= 3:
        pts = 30.0
    elif matches == 2:
        pts = 20.0
    elif matches == 1:
        pts = 12.0
    else:
        pts = 0.0
    breakdown["specificity"] = pts
    score += pts

    # Body presence for larger diffs
    changed_lines = 0
    for line in diff.splitlines():
        if line and (line.startswith("+") or line.startswith("-")):
            if not (line.startswith("+++") or line.startswith("---")):
                changed_lines += 1
    if changed_lines >= 10:
        has_body = any(ln.strip() == "" for ln in lines[1:]) and (len(lines) > 1)
        if has_body:
            breakdown["body"] = 10.0
            score += 10.0

    # Penalties: generic words, trailing period
    penalty = 0.0
    subj_lower = subject.lower()
    for word in _GENERIC_WORDS:
        if word in subj_lower:
            penalty += 5.0
    if subject.endswith("."):
        penalty += 2.0
    if penalty:
        breakdown["penalties"] = -penalty
        score -= penalty

    score = max(0.0, min(100.0, score))
    return {"score": score, "breakdown": breakdown}


# -------------------------------
# Benchmark runner
# -------------------------------


@dataclass
class BenchResult:
    provider: str
    model: str
    avg_latency_ms: float
    avg_cost_usd: float
    quality: float
    success_rate: float
    runs: int


@dataclass
class BenchExclusion:
    provider: str
    model: str
    reason: str
    detail: str | None = None


@dataclass
class BenchSampleDetail:
    provider: str
    model: str
    sample: str
    diff: str
    success: bool
    latency_ms: float
    cost_usd: float
    quality: float
    quality_breakdown: dict[str, float]
    message: str
    error: str | None = None


@dataclass
class _PreparedModel:
    provider: str
    model: str
    client: LLMClient
    input_price_per_mtok: float
    output_price_per_mtok: float


def _build_config(provider: str, model: str) -> Config:
    """Build a provider-specific Config using loader heuristics.

    Uses load_config(overrides={"provider": provider}) to resolve the best
    api_key_env for the current environment and repo, then swaps in the
    specific model under test.
    """
    try:
        cfg = load_config(overrides={"provider": provider})
    except Exception:
        meta = DEFAULT_MODELS.get(provider, {})
        cfg = Config(
            provider=provider,
            model=model,
            llm_endpoint=str(meta.get("endpoint", "")),
            api_key_env=str(meta.get("api_key_env", "")),
        )
    cfg.model = model
    return cfg


def _run_benchmark_impl(
    models_map: dict[str, list[dict[str, Any]]],
    *,
    per_provider_limit: int | None = None,
    request_timeout: float | None = None,
    debug: bool = False,
    progress: Any | None = None,
    only_providers: Iterable[str] | None = None,
    only_models: Iterable[str] | None = None,
    include_details: bool = False,
    provider_model_allowlist: dict[str, set[str]] | None = None,
) -> tuple[list[BenchResult], list[BenchExclusion], list[BenchSampleDetail] | None]:
    diffs = sample_diffs()
    results: list[BenchResult] = []
    exclusions: list[BenchExclusion] = []
    details: list[BenchSampleDetail] | None = [] if include_details else None
    provider_model_counts: dict[str, int] = {}
    total_runs = 0

    provider_filter = {str(p) for p in only_providers} if only_providers else None
    model_filter = {str(m) for m in only_models} if only_models else None

    if provider_filter is None:
        ordered_providers = list(models_map.keys())
    else:
        ordered_providers = [
            prov for prov in models_map.keys() if prov in provider_filter
        ]

    prepared: dict[str, list[_PreparedModel]] = {}

    for provider in ordered_providers:
        items = list(models_map.get(provider, []) or [])
        if not items:
            exclusions.append(
                BenchExclusion(
                    provider=provider,
                    model="*",
                    reason="no_models_available",
                    detail="provider returned no benchmarkable models",
                )
            )
            prepared[provider] = []
            continue

        subset = items

        allowlist = (
            provider_model_allowlist.get(provider) if provider_model_allowlist else None
        )
        if allowlist is not None:
            subset = [
                item for item in subset if str(item.get("id", "")).strip() in allowlist
            ]

        if model_filter is not None:
            subset = [
                item
                for item in subset
                if str(item.get("id", "")).strip() in model_filter
            ]
        if per_provider_limit:
            subset = subset[:per_provider_limit]

        if model_filter is not None and not subset:
            if provider_filter is not None:
                if provider in provider_filter:
                    for target in sorted(model_filter):
                        exclusions.append(
                            BenchExclusion(
                                provider=provider,
                                model=target,
                                reason="model_not_listed",
                                detail="model not present in provider catalog",
                            )
                        )
            prepared[provider] = []
            continue

        prepared_models: list[_PreparedModel] = []
        for entry in subset:
            model = str(entry.get("id", "")).strip()
            if not model:
                continue

            try:
                in_price = float(entry.get("input_price_per_mtok") or 0.0)
                out_price = float(entry.get("output_price_per_mtok") or 0.0)
            except (TypeError, ValueError):
                in_price = 0.0
                out_price = 0.0

            try:
                cfg = _build_config(provider, model)
                client = LLMClient(cfg, debug=debug)
            except LLMError as exc:
                exclusions.append(
                    BenchExclusion(
                        provider=provider,
                        model=model,
                        reason="client_init_failed",
                        detail=str(exc),
                    )
                )
                continue
            except Exception as exc:  # pragma: no cover - defensive
                exclusions.append(
                    BenchExclusion(
                        provider=provider,
                        model=model,
                        reason="client_init_failed",
                        detail=str(exc),
                    )
                )
                continue

            prepared_models.append(
                _PreparedModel(
                    provider=provider,
                    model=model,
                    client=client,
                    input_price_per_mtok=in_price,
                    output_price_per_mtok=out_price,
                )
            )

        if not prepared_models:
            prepared[provider] = []
            continue

        prepared[provider] = prepared_models
        provider_model_counts[provider] = len(prepared_models)
        total_runs += len(prepared_models) * len(diffs)

    if callable(progress):
        try:
            progress(
                "init",
                {
                    "total_runs": total_runs,
                    "providers": ordered_providers,
                    "provider_model_counts": provider_model_counts,
                    "samples": len(diffs),
                },
            )
        except Exception:  # pragma: no cover - progress is best-effort
            pass

    done = 0
    provider_index = 0
    total_providers = len(ordered_providers)
    for provider in ordered_providers:
        prepared_models = prepared.get(provider, [])
        if not prepared_models:
            continue
        provider_index += 1
        if callable(progress):
            try:
                progress(
                    "provider",
                    {
                        "provider": provider,
                        "index": provider_index,
                        "total_providers": total_providers,
                        "models": provider_model_counts.get(provider, 0),
                    },
                )
            except Exception:  # pragma: no cover
                pass
        model_index = 0
        for prepared_model in prepared_models:
            model_index += 1
            if callable(progress):
                try:
                    progress(
                        "model_start",
                        {
                            "provider": prepared_model.provider,
                            "model": prepared_model.model,
                            "index": model_index,
                            "total_models": provider_model_counts.get(provider, 0),
                            "samples": len(diffs),
                        },
                    )
                except Exception:
                    pass
            latencies: list[float] = []
            costs: list[float] = []
            scores: list[float] = []
            successes = 0
            for name, diff_text in diffs:
                start = time.perf_counter()
                try:
                    msg = prepared_model.client.generate_commit_message(
                        diff_text,
                        context=f"Benchmark sample: {name}",
                        style="conventional",
                        request_timeout=request_timeout,
                    )
                    successes += 1
                    err_detail = None
                except LLMError:
                    msg = ""
                    err_detail = "LLM request failed"
                elapsed_ms = (time.perf_counter() - start) * 1000.0
                latencies.append(elapsed_ms)
                in_tokens = approx_tokens(diff_text)
                out_tokens = approx_tokens(msg or "")
                cost = (
                    in_tokens * (prepared_model.input_price_per_mtok / 1_000_000.0)
                ) + (out_tokens * (prepared_model.output_price_per_mtok / 1_000_000.0))
                costs.append(cost)
                score_payload = score_quality(diff_text, msg)
                q = float(score_payload.get("score", 0.0))
                scores.append(q)
                if details is not None:
                    diff_for_details = diff_text
                    if len(diff_for_details) > 2000:
                        diff_for_details = (
                            diff_for_details[:1500] + "\n…\n" + diff_for_details[-400:]
                        )
                    breakdown_raw = score_payload.get("breakdown")
                    breakdown: dict[str, float] = {}
                    if isinstance(breakdown_raw, dict):
                        for key, val in breakdown_raw.items():
                            try:
                                breakdown[str(key)] = float(val)
                            except (TypeError, ValueError):
                                continue
                    details.append(
                        BenchSampleDetail(
                            provider=prepared_model.provider,
                            model=prepared_model.model,
                            sample=name,
                            diff=str(diff_for_details),
                            success=bool(msg),
                            latency_ms=float(elapsed_ms),
                            cost_usd=float(cost),
                            quality=float(q),
                            quality_breakdown=breakdown,
                            message=str(msg or "").strip(),
                            error=err_detail,
                        )
                    )
                done += 1
                if callable(progress):
                    try:
                        progress(
                            "tick",
                            {
                                "done": done,
                                "total": total_runs,
                                "provider": prepared_model.provider,
                                "model": prepared_model.model,
                                "sample": name,
                            },
                        )
                    except Exception:
                        pass
            if not latencies:
                continue
            avg_lat = sum(latencies) / len(latencies)
            avg_cost = sum(costs) / len(costs)
            avg_quality = sum(scores) / len(scores) if scores else 0.0
            success_rate = successes / len(diffs) if diffs else 0.0
            results.append(
                BenchResult(
                    provider=prepared_model.provider,
                    model=prepared_model.model,
                    avg_latency_ms=avg_lat,
                    avg_cost_usd=avg_cost,
                    quality=avg_quality,
                    success_rate=success_rate,
                    runs=len(diffs),
                )
            )
    if callable(progress):
        try:
            progress("done", {"done": done, "total": total_runs})
        except Exception:
            pass
    return results, exclusions, details


def run_benchmark(
    models_map: dict[str, list[dict[str, Any]]],
    *,
    per_provider_limit: int | None = None,
    request_timeout: float | None = None,
    debug: bool = False,
    progress: Any | None = None,
    only_providers: Iterable[str] | None = None,
    only_models: Iterable[str] | None = None,
) -> tuple[list[BenchResult], list[BenchExclusion]]:
    results, exclusions, _details = _run_benchmark_impl(
        models_map,
        per_provider_limit=per_provider_limit,
        request_timeout=request_timeout,
        debug=debug,
        progress=progress,
        only_providers=only_providers,
        only_models=only_models,
        include_details=False,
        provider_model_allowlist=None,
    )
    return results, exclusions


def run_benchmark_detailed(
    models_map: dict[str, list[dict[str, Any]]],
    *,
    per_provider_limit: int | None = None,
    request_timeout: float | None = None,
    debug: bool = False,
    progress: Any | None = None,
    only_providers: Iterable[str] | None = None,
    only_models: Iterable[str] | None = None,
    provider_model_allowlist: dict[str, set[str]] | None = None,
) -> tuple[list[BenchResult], list[BenchExclusion], list[BenchSampleDetail]]:
    results, exclusions, details = _run_benchmark_impl(
        models_map,
        per_provider_limit=per_provider_limit,
        request_timeout=request_timeout,
        debug=debug,
        progress=progress,
        only_providers=only_providers,
        only_models=only_models,
        include_details=True,
        provider_model_allowlist=provider_model_allowlist,
    )
    return results, exclusions, list(details or [])


def _fmt_money(value: float) -> str:
    if value < 1.0:
        return f"${value:.4f}"
    return f"${value:.2f}"


def _escape_md(value: object) -> str:
    text = str(value).replace("\r", " ").replace("\n", " ").strip()
    if not text:
        return "-"
    return text.replace("|", "\\|")


def _build_benchmark_leaderboards(
    results: list[BenchResult],
) -> dict[str, list[BenchResult]]:
    if not results:
        return {
            "overall": [],
            "fastest": [],
            "cheapest": [],
            "best_quality": [],
            "most_stable": [],
        }

    fastest = sorted(results, key=lambda r: (r.avg_latency_ms, r.avg_cost_usd))[:10]
    cheapest = sorted(results, key=lambda r: (r.avg_cost_usd, r.avg_latency_ms))[:10]
    best_quality = sorted(results, key=lambda r: (-r.quality, r.avg_latency_ms))[:10]
    most_stable = sorted(results, key=lambda r: (-r.success_rate, r.avg_latency_ms))[
        :10
    ]

    min_lat = min(r.avg_latency_ms for r in results)
    max_lat = max(r.avg_latency_ms for r in results)
    min_cost = min(r.avg_cost_usd for r in results)
    max_cost = max(r.avg_cost_usd for r in results)

    def _norm(val: float, low: float, high: float) -> float:
        if high <= low:
            return 1.0
        return (val - low) / (high - low)

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
        "overall": overall,
        "fastest": fastest,
        "cheapest": cheapest,
        "best_quality": best_quality,
        "most_stable": most_stable,
    }


def render_benchmark_markdown_report(
    results: list[BenchResult],
    exclusions: list[BenchExclusion],
    *,
    timestamp: str,
    repo_path: str,
    params: dict[str, Any] | None = None,
) -> str:
    params = params or {}
    clean_ts = timestamp.strip() if timestamp else ""
    if not clean_ts:
        clean_ts = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    repo_display = repo_path or "."

    provider_list: list[str] = []
    providers_raw = params.get("providers")
    if isinstance(providers_raw, (list, tuple, set)):
        provider_list = [str(item) for item in providers_raw if str(item).strip()]
    elif isinstance(providers_raw, str) and providers_raw.strip():
        provider_list = [providers_raw.strip()]
    if not provider_list:
        provider_list = sorted({r.provider for r in results})

    model_list: list[str] = []
    models_raw = params.get("models")
    if isinstance(models_raw, (list, tuple, set)):
        model_list = [str(item) for item in models_raw if str(item).strip()]
    elif isinstance(models_raw, str) and models_raw.strip():
        model_list = [models_raw.strip()]

    total_models = len(results)
    total_runs = sum(r.runs for r in results)
    sample_count = params.get("samples")
    if sample_count is None and results:
        sample_count = max(r.runs for r in results)

    lines: list[str] = [
        "# kcmt Benchmark Report",
        "",
        "## Run Summary",
        f"- Timestamp: {clean_ts}",
        f"- Repository: {repo_display}",
    ]
    if provider_list:
        lines.append(f"- Providers: {', '.join(sorted(provider_list))}")
    if model_list:
        lines.append(f"- Model filter: {', '.join(sorted(model_list))}")

    limit = params.get("limit")
    if isinstance(limit, int) and limit > 0:
        lines.append(f"- Per-provider limit: {limit}")

    timeout = params.get("timeout")
    if timeout is not None:
        try:
            timeout_val = float(timeout)
            lines.append(f"- Request timeout (s): {timeout_val:.2f}")
        except (TypeError, ValueError):
            lines.append(f"- Request timeout (s): {timeout}")

    if isinstance(sample_count, int) and sample_count > 0:
        lines.append(f"- Samples per model: {sample_count}")

    lines.append(f"- Total models: {total_models}")
    lines.append(f"- Total runs: {total_runs}")
    if exclusions:
        lines.append(f"- Exclusions: {len(exclusions)}")

    if not results:
        lines.append("")
        lines.append("_No benchmarkable models were run._")
    else:
        leaderboards = _build_benchmark_leaderboards(results)
        lines.append("")
        lines.append("## Leaderboards")

        def _leaderboard_rows(items: list[BenchResult]) -> list[list[str]]:
            rows: list[list[str]] = []
            for idx, item in enumerate(items, start=1):
                rows.append(
                    [
                        str(idx),
                        _escape_md(item.provider),
                        _escape_md(item.model),
                        f"{item.avg_latency_ms:.1f}",
                        _fmt_money(item.avg_cost_usd),
                        f"{item.quality:.1f}",
                        f"{item.success_rate:.0%}",
                        str(item.runs),
                    ]
                )
            return rows

        def _render_table(headers: list[str], rows: list[list[str]]) -> None:
            if not rows:
                lines.append("_No results._")
                return
            lines.append("| " + " | ".join(headers) + " |")
            lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
            for row in rows:
                lines.append("| " + " | ".join(row) + " |")

        sections = [
            ("Overall", leaderboards.get("overall", [])),
            ("Fastest", leaderboards.get("fastest", [])),
            ("Cheapest", leaderboards.get("cheapest", [])),
            ("Best Quality", leaderboards.get("best_quality", [])),
            ("Most Stable", leaderboards.get("most_stable", [])),
        ]
        for title, rows in sections:
            lines.append("")
            lines.append(f"### {title}")
            _render_table(
                [
                    "Rank",
                    "Provider",
                    "Model",
                    "Latency (ms)",
                    "Cost",
                    "Quality",
                    "Success",
                    "Runs",
                ],
                _leaderboard_rows(rows),
            )

        lines.append("")
        lines.append("## Results by Provider")
        grouped: dict[str, list[BenchResult]] = {}
        for item in results:
            grouped.setdefault(item.provider, []).append(item)
        for provider in sorted(grouped.keys()):
            rows = sorted(
                grouped[provider], key=lambda r: (r.avg_latency_ms, r.avg_cost_usd)
            )
            label = PROVIDER_DISPLAY_NAMES.get(provider, provider)
            lines.append("")
            lines.append(f"### {label}")
            lines.append(
                "| Model | Latency (ms) | Cost | Quality | Success | Runs |"
            )
            lines.append("| --- | --- | --- | --- | --- | --- |")
            for item in rows:
                lines.append(
                    "| {model} | {lat:.1f} | {cost} | {quality:.1f} | {success:.0%} | {runs} |".format(
                        model=_escape_md(item.model),
                        lat=item.avg_latency_ms,
                        cost=_fmt_money(item.avg_cost_usd),
                        quality=item.quality,
                        success=item.success_rate,
                        runs=item.runs,
                    )
                )

    if exclusions:
        lines.append("")
        lines.append("## Excluded Models")
        lines.append("| Provider | Model | Reason | Detail |")
        lines.append("| --- | --- | --- | --- |")
        for item in exclusions:
            lines.append(
                "| {provider} | {model} | {reason} | {detail} |".format(
                    provider=_escape_md(item.provider),
                    model=_escape_md(item.model),
                    reason=_escape_md(item.reason),
                    detail=_escape_md(item.detail) if item.detail else "-",
                )
            )

    return "\n".join(lines).rstrip() + "\n"


def write_benchmark_markdown_report(
    *,
    results: list[BenchResult],
    exclusions: list[BenchExclusion],
    repo_root: Path | str | None,
    timestamp: str,
    params: dict[str, Any] | None = None,
) -> Path | None:
    try:
        repo_path = Path(repo_root) if repo_root is not None else Path(".")
    except TypeError:
        repo_path = Path(".")

    report = render_benchmark_markdown_report(
        results,
        exclusions,
        timestamp=timestamp,
        repo_path=str(repo_path),
        params=params,
    )
    safe_ts = timestamp.strip() if timestamp else ""
    if not safe_ts:
        safe_ts = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    safe_ts = safe_ts.replace(":", "")
    try:
        report_dir = state_dir(repo_path) / "benchmarks"
        report_dir.mkdir(parents=True, exist_ok=True)
        path = report_dir / f"benchmark-{safe_ts}.md"
        path.write_text(report, encoding="utf-8")
        return path
    except OSError:
        return None
