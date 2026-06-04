# Rust Fastest Automated Commit Tool Implementation Plan

**Branch**: `005-rust-fastest-commit-tool` | **Date**: 2026-06-04 | **Spec**: `specs/005-rust-fastest-commit-tool/spec.md`

## Summary

Optimize only the Rust automated commit path. The Python implementation is reference material for behavior, prompts, validation, batch semantics, tests, and side-by-side benchmark comparison, but the hot path target is Rust from git discovery through LLM queueing, response validation, commit, push, telemetry, and reporting.

## Technical Context

**Language/Version**: Rust stable target 1.78+, Python 3.12 retained as reference and test harness during migration  
**Primary Dependencies**: Existing Rust workspace crates, `clap`, `tokio`, `reqwest`, `serde`/`serde_json`, `tracing`, `anyhow`/`thiserror`; add only small dependencies with clear performance value  
**Storage**: Local filesystem snapshots and reports only  
**Testing**: Rust unit/integration/contract tests, executable BDD feature tests, existing `pytest` gate while Python remains in repo  
**Target Platform**: macOS and Linux terminals with git CLI available  
**Project Type**: CLI application with reusable Rust core, provider, benchmark, and optional TUI crates  
**Performance Goals**: Fill the LLM queue as fast as possible, preserve quality, and reduce post-LLM overhead  
**Baseline Corpus**: Existing synthetic 1,000-file corpus from `scripts/benchmark/generate_uncommitted_repo.py`, plus new committed benchmark fixtures for modified/new/deleted/renamed paths  
**Constraints**: Do not remove the LLM; preserve commit quality; preserve git atomicity; keep secrets out of telemetry; do not depend on Python in the optimized Rust path  
**Scale/Scope**: Large developer repos with hundreds to thousands of changed files and OpenAI batch-mode operation

## Current Baseline

Captured before implementation on 2026-06-04 with release Rust binary and the synthetic 1,000-file corpus:

- `make check`: pass
- `cargo build --release`: pass
- `status-repo-path`: 35.921792 ms median
- `file-repo-path`: 321.260958 ms median
- `oneshot-repo-path`: 371.255916 ms median
- Static complexity scan of Rust tree: zero findings

Limitations:

- Current Rust production workflow uses `heuristic_commit_message`, not the real LLM.
- Current benchmark measures local git/commit plumbing, not real batch LLM queue fill.
- There are no `.feature` files yet, so BDD coverage must be added for behavior changes.
- Python-vs-Rust score board output is required for every iteration so each Rust stage can be compared against the equivalent Python stage where one exists.

## Constitution Check

- Principle I (CLI Contract Compatibility): PASS. Aliases and automation-facing output remain in scope.
- Principle II (Git Safety and Atomic Operations): PASS. File-scoped commits remain mandatory.
- Principle III (Quality Gates and Test Discipline): PASS. Plan includes BDD, Rust tests, quality scoring, `make check`, and later `make quality-gates`.
- Principle IV (Performance and Benchmark Accountability): PASS. Baselines, deterministic fixtures, and five iteration reports are required.
- Principle V (Security and Configuration Integrity): PASS. Telemetry excludes secrets and provider credentials.

Result: PASS. No known gate violations.

## Architecture

Rust workflow should be reorganized around a pipeline:

1. `RepoScanner`: find repo root, capture one porcelain status snapshot, classify changed files.
2. `ChangePreparer`: build diff/content summaries for changed files using bounded concurrency.
3. `LlmScheduler`: enqueue each prepared change immediately, with provider-aware rate limits and retry/backoff telemetry.
4. `MessageQualityGate`: sanitize, validate, score, and reject invalid Conventional Commit messages.
5. `CommitExecutor`: commit accepted messages atomically by file and push once after successful completion when configured.
6. `TelemetryRecorder`: record stage timings and output JSON/markdown reports.
7. `WorkflowComparisonScoreboard`: render Python and Rust stage timings side by side with deltas, percent change, quality scores, and comparability notes.
8. `BenchmarkHarness`: run deterministic corpora, fake-provider runs, real-provider runs when credentials exist, and iteration comparisons.

## Project Structure

```text
specs/005-rust-fastest-commit-tool/
├── spec.md
├── plan.md
└── tasks.md

docs/performance/
└── rust-fastest-commit-tool-iterations.md

features/
└── rust_fast_commit_tool.feature
```

## Five Iteration Loop

### Iteration 1: Telemetry and Real Baseline

- Add `WorkflowTelemetry`, stage timing, and JSON report output.
- Add deterministic fake-provider mode.
- Replace pure heuristic benchmark with real pipeline measurement points.
- Report Python and Rust baselines side by side for status, prep, queue, fake-provider response, validation, commit, snapshot, and push.

### Iteration 2: Single Scan and Concurrent Preparation

- Reuse one status snapshot for all files.
- Prepare diffs/content summaries concurrently.
- Measure time-to-first-enqueue and time-to-all-enqueued.

### Iteration 3: LLM Queue Scheduler and Provider Transport

- Implement Rust LLM request construction and batch/fake-provider queueing.
- Reuse async HTTP client state.
- Separate provider wait, retry, and backoff telemetry.

### Iteration 4: Quality Gate and Post-LLM Fast Path

- Port prompt preparation, sanitization, Conventional Commit validation, and quality scoring.
- Optimize validation-to-commit path with safe git command batching where behavior remains atomic.

### Iteration 5: Final Polish and Report

- Tighten report writing, snapshots, and CLI output.
- Run full deterministic benchmark suite.
- Produce final five-iteration performance report and remaining opportunities.

## Top Ten Improvements

1. Real Rust LLM-backed generation replaces `heuristic_commit_message`.
2. One status scan feeds all downstream stages.
3. Bounded concurrent diff/content preparation.
4. Immediate LLM enqueue per prepared file.
5. Shared async provider transport.
6. Provider wait separated from local work telemetry.
7. Fake-provider deterministic benchmark mode.
8. Safe reduction of git subprocess count.
9. Compact telemetry/snapshot serialization.
10. Strict prompt/postprocessing quality gate with measured quality score.

## Score Board Format

Every iteration report should include a table with these columns:

```text
Stage | Python median ms | Rust median ms | Delta ms | Rust change % | Quality impact | Comparable | Notes
```

Comparable stages include repo discovery, status scan, changed-file classification, diff/content preparation, time-to-first-enqueue, time-to-all-enqueued, provider wait, response validation, commit, snapshot, and push. Python wrapper or TUI-only stages are reported as `no` in the Comparable column with a reason.

## Validation Commands

- Narrow Rust tests: `rtk cargo test -p kcmt-core -p kcmt-provider -p kcmt-bench`
- Rust integration tests: `cd rust && rtk cargo test`
- Python compatibility gate while Python remains: `rtk make check`
- Full pre-push gate: `rtk make check && rtk make quality-gates`
- Runtime benchmark: `rtk ./rust/target/release/kcmt benchmark runtime --repo-path <corpus> --runtime rust --iterations 5 --json`

## Complexity Tracking

No current blocking violations. Main risk is scope size: Rust provider parity, quality scoring, queueing, telemetry, and git optimization can become too broad. The iteration loop limits each pass to one measurable bottleneck and one report.

