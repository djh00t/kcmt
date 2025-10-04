# kcmt Benchmark Guide

The `kcmt --benchmark` command exercises every configured provider/model pair against a
fixed suite of repository diffs to measure latency, estimated cost, message
quality, and stability. Use this guide to understand how the suite is
constructed, how scores are produced, and how to focus the benchmark on
specific providers or models when tuning.

## Test corpus

Each benchmark run replays the same five synthetic diffs so results remain
comparable across providers:

| Sample | Scenario | Intent |
| ------ | -------- | ------ |
| `feature-add` | math helper gain | Adds a guarded `divide` helper including error handling | 
| `bugfix-conditional` | auth guard fix | Tightens a conditional to avoid null dereferences |
| `docs-update` | README polish | Rewords usage copy in Markdown |
| `refactor-utils` | slugify cleanup | Refactors a helper to normalize whitespace and characters |
| `tests-add` | regression tests | Adds a focused regression test |

Every provider/model combination that can be initialized runs all five samples.

## Metrics collected

For each provider/model the benchmark records:

- **Average latency (ms)** – wall-clock time per diff, averaged across the
  corpus.
- **Average cost (USD)** – estimated spend derived from the provider's listed
  per-million-token pricing and the prompt/response lengths.
- **Quality score (0–100)** – conventional-commit adherence scored by the
  `score_quality` heuristic described below.
- **Success rate** – fraction of diffs that produced a commit message without
  raising an `LLMError`.

### Quality scoring rubric

The `score_quality` helper awards points for conventional-commit discipline and
subject specificity while deducting for generic phrasing:

- **Format (30 pts)** – subject matches `<type>(<scope>): <subject>` using
  standard conventional types.
- **Scope (10 pts)** – optional scope present in the prefix, e.g. `feat(ui):`.
- **Subject length (≤10 pts)** – subjects at or under the configured limit earn
  the full 10 points; going long subtracts up to 10 points.
- **Specificity (0/12/20/30 pts)** – keyword overlap with diff content rewards
  precise wording.
- **Body (10 pts)** – large diffs (≥10 changed lines) receive credit when a
  commit body accompanies the subject.
- **Penalties (−)** – subtracts 5 points per generic word (`update`, `fixes`,
  etc.) and 2 points for trailing punctuation.

The score is clamped to 0–100 and averaged across the diff corpus.

### Recommended pass criteria

Benchmarks are most useful when we enforce strict gates. Treat a provider/model
as "passing" only if **all** of the following hold:

- Quality score ≥ 80
- Success rate = 100%
- Average latency ≤ 4,000 ms
- Average cost ≤ $0.010 per run

Models falling short on any metric should be tuned (prompt adjustments, pricing
review, retries) before promoting them as defaults.

## Exclusions and missing credentials

When a provider/model cannot run, the benchmark lists it in an "Excluded Models"
section with a human-readable reason. Missing API keys, SDK initialization
errors, empty catalogs, and filtered models are all reported so it is obvious
why a provider did not contribute quality scores.

## Targeted benchmarking

Use the standard CLI overrides to narrow runs when debugging or comparing a
single model:

- `kcmt --benchmark --provider anthropic` – limit to a single provider.
- `kcmt --benchmark --model gpt-5-mini` – exercise any provider that offers the
  named model.
- `kcmt --benchmark --provider openai --model gpt-5-mini` – pin both provider
  and model.
- `kcmt --benchmark --benchmark-limit 2` – cap each provider to the first two
  priced models (ignored when `--model` is supplied).

All benchmark outputs (leaderboards, JSON, CSV) include the exclusions list so
missing credentials can be remedied quickly.

## Machine-readable exports

Add `--benchmark-json` or `--benchmark-csv` to emit structured reports that
include both the run results and the exclusions table. These outputs are ideal
for regression tracking in CI.
