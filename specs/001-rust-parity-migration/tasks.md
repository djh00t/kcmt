# Tasks: High-Performance Core Migration with Feature Parity

**Input**: Design documents from `/Users/djh/work/src/github.com_local/djh00t/kcmt/specs/001-rust-parity-migration/`
**Prerequisites**: plan.md (required), spec.md (required), research.md, data-model.md, contracts/, quickstart.md

**Tests**: Included because the specification defines independent test criteria and measurable regression outcomes.

**Organization**: Tasks are grouped by user story to support independent implementation, validation, and incremental delivery.

## Format: `[ID] [P?] [Story?] Description`

- **[P]**: Parallelizable (different files, no incomplete dependency overlap)
- **[Story]**: User story label (`[US1]`, `[US2]`, `[US3]`) used only in story phases
- Every task includes an exact file path

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Initialize the Rust workspace and baseline project scaffolding.

- [X] T001 Create Rust workspace manifest in rust/Cargo.toml
- [X] T002 Create core crate manifest and package metadata in rust/crates/kcmt-core/Cargo.toml
- [X] T003 [P] Create CLI crate manifest and binary targets in rust/crates/kcmt-cli/Cargo.toml
- [X] T004 [P] Create provider crate manifest with async HTTP dependencies in rust/crates/kcmt-provider/Cargo.toml
- [X] T005 [P] Create benchmark and TUI crate manifests in rust/crates/kcmt-bench/Cargo.toml and rust/crates/kcmt-tui/Cargo.toml
- [X] T006 [P] Create Rust test harness scaffold documentation and TUI skeleton in rust/tests/README.md and rust/crates/kcmt-tui/src/lib.rs

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Build core abstractions required by all user stories.

**⚠️ CRITICAL**: Complete this phase before story implementation.

- [X] T007 Create shared error and result types in rust/crates/kcmt-core/src/error.rs
- [X] T008 [P] Define shared domain models from data model entities in rust/crates/kcmt-core/src/model.rs
- [X] T009 [P] Implement git subprocess command runner abstraction in rust/crates/kcmt-core/src/git/runner.rs
- [X] T010 Implement git repository adapter interface on top of runner in rust/crates/kcmt-core/src/git/repo.rs
- [X] T011 [P] Implement configuration source precedence loader in rust/crates/kcmt-core/src/config/loader.rs
- [X] T012 [P] Implement shared async transport client with retry and rate-limit policy in rust/crates/kcmt-provider/src/transport.rs
- [X] T013 Implement provider adapter trait and registry wiring in rust/crates/kcmt-provider/src/registry.rs
- [X] T014 [P] Create CLI contract test harness using trycmd in rust/tests/contract/cli_contract.rs
- [X] T015 Create Python-vs-Rust parity runner scaffold in rust/tests/parity/parity_runner.rs
- [X] T053 [P] Define regression corpus and fixture classes in specs/001-rust-parity-migration/validation/regression-corpus.md
- [X] T054 [P] Define high-usage workflow catalog in specs/001-rust-parity-migration/validation/high-usage-workflows.md
- [X] T055 [P] Define exit/error baseline matrix in specs/001-rust-parity-migration/validation/exit-error-baseline.md

**Checkpoint**: Foundation complete; user stories can proceed.

---

## Phase 3: User Story 1 - Preserve Daily Commit Workflow (Priority: P1) 🎯 MVP

**Goal**: Preserve day-to-day commit commands and alias behavior with equivalent output and commit side effects.

**Independent Test**: Run core workflows (`kcmt`, `commit`, `kc`, `--file`, `--oneshot`, `status`) against fixture repos and verify equivalent parse behavior, commit format validity, and file-scoped commit semantics.

### Tests for User Story 1

- [X] T016 [P] [US1] Add CLI alias and flag parse contract snapshots in rust/tests/contract/us1_cli_aliases.trycmd
- [X] T017 [P] [US1] Add integration test for staged commit workflow in rust/tests/integration/us1_staged_commit.rs
- [X] T018 [P] [US1] Add integration test for file-targeted commit isolation in rust/tests/integration/us1_file_scoped_commit.rs
- [X] T056 [P] [US1] Add integration test for no-staged/no-relevant-changes behavior in rust/tests/integration/us1_no_changes_behavior.rs
- [X] T057 [P] [US1] Add compatibility test for legacy alias/flag combinations in rust/tests/contract/us1_legacy_flags.trycmd

### Implementation for User Story 1

- [X] T019 [US1] Implement clap command tree and global options in rust/crates/kcmt-cli/src/args.rs
- [X] T020 [US1] Implement canonical CLI entrypoint dispatch in rust/crates/kcmt-cli/src/bin/kcmt.rs
- [X] T021 [P] [US1] Implement `commit` alias binary forwarding to shared entrypoint in rust/crates/kcmt-cli/src/bin/commit.rs
- [X] T022 [P] [US1] Implement `kc` alias binary forwarding to shared entrypoint in rust/crates/kcmt-cli/src/bin/kc.rs
- [X] T023 [US1] Implement staged/working change-set collection in rust/crates/kcmt-core/src/workflow/changeset.rs
- [X] T024 [US1] Implement commit recommendation orchestration flow in rust/crates/kcmt-core/src/workflow/commit_flow.rs
- [X] T025 [US1] Implement pathspec-constrained commit execution in rust/crates/kcmt-core/src/git/commit_file.rs
- [X] T026 [US1] Implement `status` subcommand rendering and output mapping in rust/crates/kcmt-cli/src/commands/status.rs
- [X] T058 [US1] Implement partial-failure handling for atomic single-file commits in rust/crates/kcmt-core/src/workflow/atomic_failure_recovery.rs

**Checkpoint**: US1 is independently functional and testable.

---

## Phase 4: User Story 2 - Keep Provider and Config Compatibility (Priority: P2)

**Goal**: Preserve provider behavior, configuration precedence, and actionable error semantics for existing automation and team setups.

**Independent Test**: Replay provider/config test matrix with env/config/CLI override permutations and verify equivalent selected provider/model, deterministic errors, and no secret leakage.

### Tests for User Story 2

- [X] T027 [P] [US2] Add config precedence contract tests in rust/tests/contract/us2_config_precedence.rs
- [X] T028 [P] [US2] Add integration tests for provider fallback and selection in rust/tests/integration/us2_provider_fallback.rs
- [X] T029 [P] [US2] Add integration tests for invalid credential error behavior in rust/tests/integration/us2_invalid_credentials.rs
- [X] T059 [P] [US2] Add provider failure-mode tests (timeout/rate-limit/malformed response) in rust/tests/integration/us2_provider_failure_modes.rs
- [X] T060 [P] [US2] Add config compatibility tests for deprecated/partial values in rust/tests/integration/us2_legacy_config_values.rs

### Implementation for User Story 2

- [X] T030 [US2] Implement workflow configuration parsing and validation in rust/crates/kcmt-core/src/config/workflow_config.rs
- [X] T031 [US2] Implement backward-compatible persisted config loader in rust/crates/kcmt-core/src/config/persisted.rs
- [X] T032 [US2] Implement provider profile model and resolution logic in rust/crates/kcmt-provider/src/profile.rs
- [X] T033 [US2] Implement provider client modules (OpenAI, Anthropic, xAI, GitHub) in rust/crates/kcmt-provider/src/clients/mod.rs
- [X] T034 [US2] Implement provider error normalization and secret-safe messaging in rust/crates/kcmt-provider/src/error_map.rs
- [X] T035 [US2] Implement configure and override command wiring in rust/crates/kcmt-cli/src/commands/configure.rs
- [X] T036 [US2] Integrate provider registry into commit workflow dispatch in rust/crates/kcmt-core/src/workflow/provider_dispatch.rs

**Checkpoint**: US2 is independently functional and testable.

---

## Phase 5: User Story 3 - Improve Throughput for Large Repositories (Priority: P3)

**Goal**: Improve local processing throughput and benchmark execution performance while preserving score/report compatibility.

**Independent Test**: Run benchmark suite and representative large-diff scenarios; verify performance targets and schema-compatible benchmark outputs versus baseline.

### Tests for User Story 3

- [X] T037 [P] [US3] Add benchmark snapshot schema contract tests in rust/tests/contract/us3_benchmark_schema.rs
- [X] T038 [P] [US3] Add integration tests for benchmark table/json/csv outputs in rust/tests/integration/us3_benchmark_outputs.rs
- [X] T039 [P] [US3] Add performance regression baseline test harness in rust/tests/parity/us3_performance_baseline.rs
- [X] T061 [P] [US3] Add oversized diff/token-limit handling tests in rust/tests/integration/us3_oversized_diff_limits.rs

### Implementation for User Story 3

- [X] T040 [US3] Implement benchmark run and result models in rust/crates/kcmt-bench/src/model.rs
- [X] T041 [US3] Implement benchmark execution pipeline with exclusions in rust/crates/kcmt-bench/src/runner.rs
- [X] T042 [US3] Implement benchmark JSON and CSV exporters per contract in rust/crates/kcmt-bench/src/export.rs
- [X] T043 [US3] Integrate benchmark command into CLI surface in rust/crates/kcmt-cli/src/commands/benchmark.rs
- [X] T044 [US3] Implement local preprocessing timing instrumentation in rust/crates/kcmt-core/src/metrics/timing.rs
- [X] T045 [US3] Implement parallel diff preparation and worker scheduling in rust/crates/kcmt-core/src/workflow/scheduler.rs
- [X] T046 [US3] Implement benchmark quality scoring parity logic in rust/crates/kcmt-bench/src/quality.rs

**Checkpoint**: US3 is independently functional and testable.

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Complete rollout controls and cross-story improvements.

- [X] T047 [P] Document staged migration and rollback controls for FR-001/FR-009 and SC-003 in docs/rust-migration-rollout.md
- [X] T048 Implement Python wrapper runtime switch for canary cutover supporting FR-001/FR-009 in kcmt/main.py
- [X] T049 [P] Add experimental gitoxide read-only adapter behind feature flag to support FR-002/FR-003 risk reduction in rust/crates/kcmt-core/src/git/gitoxide_readonly.rs
- [X] T050 [P] Add optional Ratatui interactive mode behind TTY gate while preserving FR-001 compatibility in rust/crates/kcmt-tui/src/lib.rs
- [X] T051 Execute quickstart validation updates for FR-008 and SC-001..SC-006 in /Users/djh/work/src/github.com_local/djh00t/kcmt/specs/001-rust-parity-migration/quickstart.md
- [X] T052 Capture full regression and performance gate results for FR-007/FR-008 and SC-001..SC-006 in /Users/djh/work/src/github.com_local/djh00t/kcmt/specs/001-rust-parity-migration/tasks-validation.md
- [ ] T062 [P] Execute macOS/Linux/Windows parity matrix (`high-usage-workflows` + `exit-error-baseline`) and record evidence for NFR-005 in /Users/djh/work/src/github.com_local/djh00t/kcmt/specs/001-rust-parity-migration/tasks-validation.md

---

## Dependencies & Execution Order

### Phase Dependencies

- **Phase 1 (Setup)**: no dependencies
- **Phase 2 (Foundational)**: depends on Phase 1; blocks all user stories
- **Phase 3 (US1)**: depends on Phase 2 completion
- **Phase 4 (US2)**: depends on Phase 2 completion; can run in parallel with US1 after shared files are coordinated
- **Phase 5 (US3)**: depends on Phase 2 completion; can run in parallel with US1/US2 after shared files are coordinated
- **Phase 6 (Polish)**: depends on completion of targeted user stories

### User Story Dependency Graph

- **US1 (P1)**: MVP baseline, independent after Phase 2
- **US2 (P2)**: independently releasable after Phase 2; integrates with US1 only through stable core interfaces
- **US3 (P3)**: independently releasable after Phase 2; integrates with US1 via benchmark/CLI extension points without hard runtime dependency

### Cross-Cutting Requirement Mapping

- T047/T048/T050 map to FR-001, FR-009, and SC-003.
- T049 maps to FR-002 and FR-003 risk-managed backend evolution.
- T051 maps to FR-008 and quickstart validation discipline.
- T052 maps to FR-007, FR-008, and SC-001 through SC-006 evidence capture.
- T062 maps to NFR-005 with explicit `macos-latest`/`ubuntu-latest`/`windows-latest` parity evidence.

Recommended completion order: `US1 -> US2 -> US3`

### Within-Story Order

- Contract/integration/perf tests first
- Core models and adapters
- Service/workflow logic
- CLI integration
- Story checkpoint validation

---

## Parallel Execution Examples

### User Story 1

```bash
# Parallel test authoring
T016, T017, T018

# Parallel alias entrypoint work
T021, T022
```

### User Story 2

```bash
# Parallel compatibility test authoring
T027, T028, T029

# Parallel provider/core modeling
T032, T033, T034
```

### User Story 3

```bash
# Parallel performance and output tests
T037, T038, T039

# Parallel benchmark internals
T042, T044, T045
```

---

## Implementation Strategy

### MVP First (US1 Only)

1. Complete Phase 1 and Phase 2
2. Complete US1 tasks (Phase 3)
3. Validate US1 independent tests and parity checkpoints
4. Demo/deploy MVP-compatible Rust path behind feature flag

### Incremental Delivery

1. Deliver US1 (workflow parity)
2. Deliver US2 (provider/config parity)
3. Deliver US3 (performance and benchmark parity)
4. Finish Phase 6 rollout controls and optional capabilities

### Parallel Team Strategy

1. Team completes Phase 1-2 together
2. After foundation:
   - Engineer A: US1 core workflow
   - Engineer B: US2 provider/config compatibility
   - Engineer C: US3 benchmark/performance pipeline
3. Merge by contract and parity checkpoints

---

## Notes

- All checklist items use the required format: checkbox + task ID + optional `[P]` + optional `[USx]` + action + file path.
- `[P]` markers indicate parallel-safe tasks assuming no same-file concurrent edits.
- Use story checkpoints to validate independent completeness before advancing rollout scope.
