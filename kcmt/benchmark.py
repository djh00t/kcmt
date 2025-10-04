from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .config import Config, DEFAULT_MODELS, load_config
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
            '''diff --git a/app/math.py b/app/math.py
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
''',
        ),
        (
            "bugfix-conditional",
            '''diff --git a/app/auth.py b/app/auth.py
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
''',
        ),
        (
            "docs-update",
            '''diff --git a/README.md b/README.md
index 555..666 100644
--- a/README.md
+++ b/README.md
@@
 # Project Title
 
-A tool.
+A tool for atomic git commits with AI assistance.
 
 ## Usage
''',
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
            '''diff --git a/tests/test_math.py b/tests/test_math.py
index 999..aaa 100644
--- a/tests/test_math.py
+++ b/tests/test_math.py
@@
 def test_divide():
     assert divide(6, 3) == 2
''',
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
    subj_words = {w.lower() for w in re.split(r"[^a-zA-Z0-9_]+", subject) if len(w) >= 4}
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
    diffs = sample_diffs()
    results: list[BenchResult] = []
    exclusions: list[BenchExclusion] = []

    provider_filter = {p for p in only_providers} if only_providers else None
    model_filter = {str(m) for m in only_models} if only_models else None

    ordered_providers = [
        prov
        for prov in models_map.keys()
        if provider_filter is None or prov in provider_filter
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

        if model_filter:
            subset = [
                item for item in items if str(item.get("id", "")) in model_filter
            ]
        elif per_provider_limit:
            subset = items[:per_provider_limit]
        else:
            subset = items

        if model_filter and not subset:
            if provider_filter is not None and provider in provider_filter:
                for target in model_filter:
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

        prepared_list: list[_PreparedModel] = []
        for item in subset:
            model_id = str(item.get("id", ""))
            in_price = float(item.get("input_price_per_mtok") or 0.0)
            out_price = float(item.get("output_price_per_mtok") or 0.0)
            cfg = _build_config(provider, model_id)
            api_key = cfg.resolve_api_key()
            if not api_key:
                detail = (
                    f"set {cfg.api_key_env} in your environment"
                    if cfg.api_key_env
                    else "configure api_key_env for this provider"
                )
                exclusions.append(
                    BenchExclusion(
                        provider=provider,
                        model=model_id,
                        reason="missing_api_key",
                        detail=detail,
                    )
                )
                continue
            try:
                client = LLMClient(cfg, debug=debug)
            except LLMError as exc:
                exclusions.append(
                    BenchExclusion(
                        provider=provider,
                        model=model_id,
                        reason="client_init_error",
                        detail=str(exc),
                    )
                )
                continue
            prepared_list.append(
                _PreparedModel(
                    provider=provider,
                    model=model_id,
                    client=client,
                    input_price_per_mtok=in_price,
                    output_price_per_mtok=out_price,
                )
            )
        prepared[provider] = prepared_list

    provider_model_counts = {
        prov: len(prepared.get(prov, [])) for prov in ordered_providers
    }
    total_runs = sum(provider_model_counts.values()) * len(diffs)

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
            for (name, diff_text) in diffs:
                start = time.perf_counter()
                try:
                    msg = prepared_model.client.generate_commit_message(
                        diff_text,
                        context=f"Benchmark sample: {name}",
                        style="conventional",
                        request_timeout=request_timeout,
                    )
                    successes += 1
                except LLMError:
                    msg = ""
                elapsed_ms = (time.perf_counter() - start) * 1000.0
                latencies.append(elapsed_ms)
                in_tokens = approx_tokens(diff_text)
                out_tokens = approx_tokens(msg or "")
                cost = (
                    in_tokens * (prepared_model.input_price_per_mtok / 1_000_000.0)
                ) + (
                    out_tokens * (prepared_model.output_price_per_mtok / 1_000_000.0)
                )
                costs.append(cost)
                q = score_quality(diff_text, msg).get("score", 0.0)
                scores.append(float(q))
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
    return results, exclusions
