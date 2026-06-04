# Tasks: Rust Fastest Automated Commit Tool

**Input**: `specs/005-rust-fastest-commit-tool/spec.md` and `specs/005-rust-fastest-commit-tool/plan.md`

## Phase 1: Test and Baseline Scaffold

- [ ] T001 Add executable BDD feature `features/rust_fast_commit_tool.feature` covering telemetry, batch queueing, quality validation, and post-LLM commit behavior.
- [ ] T002 Add BDD step tests that run against deterministic fake-provider fixtures.
- [ ] T003 Add `docs/performance/rust-fastest-commit-tool-iterations.md` with baseline and iteration report sections.
- [ ] T004 Add Rust telemetry model tests for required stage names and secret-safe serialization.
- [ ] T005 Add benchmark model tests for iteration delta calculations.
- [ ] T006 Add Python-vs-Rust score board model tests for comparable, non-comparable, faster, slower, and missing-stage rows.

## Phase 2: Iteration 1 - Telemetry and Baseline

- [ ] T007 Add `rust/crates/kcmt-core/src/workflow/telemetry.rs`.
- [ ] T008 Wire telemetry through `run_file_workflow` and `run_oneshot_workflow`.
- [ ] T009 Add machine-readable telemetry output to benchmark results.
- [ ] T010 Add fake-provider benchmark mode for deterministic response timing.
- [ ] T011 Run Python and Rust baselines on the same corpus and update iteration 1 report with side-by-side score board, stage timings, and bottlenecks.

## Phase 3: Iteration 2 - Single Scan and Concurrent Preparation

- [ ] T012 Refactor Rust change discovery to reuse one status snapshot.
- [ ] T013 Add changed-file classification for modified, new, deleted, renamed, and ignored files.
- [ ] T014 Add bounded concurrent diff/content preparation.
- [ ] T015 Record time-to-first-enqueue and time-to-all-enqueued.
- [ ] T016 Run Python and Rust benchmarks and update iteration 2 score board.

## Phase 4: Iteration 3 - Rust LLM Queue Scheduler

- [ ] T017 Add Rust prompt request structs for provider queue items.
- [ ] T018 Add OpenAI-compatible batch/fake queue scheduler with bounded concurrency.
- [ ] T019 Reuse `reqwest::Client` through shared transport state.
- [ ] T020 Record provider wait, retry, and backoff telemetry.
- [ ] T021 Run Python and Rust benchmarks and update iteration 3 score board.

## Phase 5: Iteration 4 - Quality Gate and Post-LLM Path

- [ ] T022 Port Conventional Commit validation from Python to Rust.
- [ ] T023 Port prompt output sanitization and recovery rules from Python to Rust.
- [ ] T024 Add quality scoring parity tests against deterministic corpus examples.
- [ ] T025 Optimize validation-to-commit execution while preserving file-scoped atomic commits.
- [ ] T026 Run Python and Rust benchmarks and update iteration 4 score board.

## Phase 6: Iteration 5 - Final Report and Gate

- [ ] T027 Compact snapshot/report serialization where benchmark evidence shows overhead.
- [ ] T028 Run final Python and Rust benchmark suite against deterministic corpora.
- [ ] T029 Run `rtk make check`.
- [ ] T030 Run `rtk make quality-gates`.
- [ ] T031 Update final iteration report with all five score boards, remaining bottlenecks, and recommended next work.

## Acceptance Evidence

- `make check` must pass before handoff.
- `make quality-gates` must be run before push or PR creation.
- Each iteration must record baseline, changed files, measured improvement, quality result, and next bottleneck.
- Each iteration must include a Python-vs-Rust score board for comparable workflow stages.
- Any skipped or failed benchmark must be reported with the reason.

