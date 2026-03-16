# Tasks: Rust CLI Feature Parity and Runtime Benchmark Mode

**Input**: Design documents from `/specs/003-bring-rust-cli/`  
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/

**Tests**: Tests are required for this feature because the specification and
constitution require parity, contract, and benchmark validation.

**Organization**: Tasks are grouped by user story to preserve independent delivery and
testing.

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Establish shared corpora and test scaffolding used by all stories.

- [X] T001 Create realistic runtime corpus fixture files in `tests/fixtures/runtime_corpus/mini_realistic_repo/`
- [X] T002 Create Python/Rust CLI parity harness scaffolding in `tests/test_cli.py` and `tests/test_main_entrypoint.py`
- [X] T003 [P] Create Rust contract/parity test scaffolding in `rust/crates/kcmt-cli/tests/status_contracts.rs`, `rust/crates/kcmt-cli/tests/workflow_modes.rs`, and `rust/crates/kcmt-cli/tests/benchmark_contracts.rs`

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Shared command-routing and benchmark scaffolding that blocks all stories.

**⚠️ CRITICAL**: No user story work should begin until this phase is complete.

- [X] T004 Add shared Rust CLI argument models for repo selection and benchmark mode routing in `rust/crates/kcmt-cli/src/args.rs`
- [X] T005 [P] Add alias-aware Rust dispatch and repo-path propagation in `rust/crates/kcmt-cli/src/lib.rs`
- [X] T006 [P] Add Rust workflow command module scaffolding in `rust/crates/kcmt-cli/src/commands/workflow.rs` and `rust/crates/kcmt-cli/src/commands/mod.rs`
- [X] T007 [P] Add runtime benchmark report scaffolding in `rust/crates/kcmt-bench/src/model.rs`, `rust/crates/kcmt-bench/src/export.rs`, and `rust/crates/kcmt-bench/src/runner.rs`
- [X] T008 [P] Add Python runtime benchmark entry scaffolding distinct from provider benchmarking in `kcmt/benchmark.py` and `kcmt/legacy_cli.py`

**Checkpoint**: Rust CLI routing and runtime benchmark foundations exist; user stories can now proceed.

---

## Phase 3: User Story 1 - Run the Same CLI Workflow on Rust (Priority: P1) 🎯 MVP

**Goal**: Make Rust a real runtime candidate by matching Python for the required-now
workflow catalog.

**Independent Test**: Build Rust binaries and run the parity catalog for help,
`status --repo-path`, `--file`, `--oneshot`, and invalid-flag scenarios, verifying
matching exit codes and expected side effects.

### Tests for User Story 1

- [X] T009 [P] [US1] Add Python contract assertions for status/raw and runtime dispatch behavior in `tests/test_cli.py` and `tests/test_main_entrypoint.py`
- [X] T010 [P] [US1] Add Rust contract tests for alias branding, parser errors, and repo-path flows in `rust/crates/kcmt-cli/tests/status_contracts.rs`

### Implementation for User Story 1

- [X] T011 [US1] Implement global `--repo-path` parsing and `status --raw` support in `rust/crates/kcmt-cli/src/args.rs`
- [X] T012 [US1] Implement alias-aware entrypoint branding and repo-path dispatch in `rust/crates/kcmt-cli/src/lib.rs` and `rust/crates/kcmt-cli/src/args.rs`
- [X] T013 [US1] Implement snapshot-backed `status` parity in `rust/crates/kcmt-cli/src/commands/status.rs`
- [X] T014 [US1] Implement real single-file commit workflow in `rust/crates/kcmt-cli/src/commands/workflow.rs` and `rust/crates/kcmt-core/src/git/commit_file.rs`
- [X] T015 [US1] Implement `--oneshot` file selection and preserve path-scoped commit safety in `rust/crates/kcmt-cli/src/commands/workflow.rs`
- [X] T016 [US1] Wire config loading and repo-root resolution for Rust workflows in `rust/crates/kcmt-core/src/config/loader.rs`, `rust/crates/kcmt-core/src/config/persisted.rs`, and `rust/crates/kcmt-core/src/git/repo.rs`
- [X] T017 [US1] Align Rust error and exit handling for repo and parser failures in `rust/crates/kcmt-cli/src/lib.rs` and `rust/crates/kcmt-cli/tests/status_contracts.rs`

**Checkpoint**: Rust can execute the required-now workflow catalog as a real runtime candidate.

---

## Phase 4: User Story 2 - Benchmark Python and Rust on the Same Repo Corpus (Priority: P2)

**Goal**: Add a separate runtime benchmark mode that compares Python and Rust on the
same deterministic repo corpora without requiring live provider traffic.

**Independent Test**: Run the runtime benchmark on the synthetic 1,000-file corpus and
the realistic fixture, then validate JSON output against
`contracts/runtime-benchmark.schema.json`.

### Tests for User Story 2

- [X] T018 [P] [US2] Add Python runtime benchmark tests and schema assertions in `tests/test_benchmark.py` and `tests/test_cli.py`
- [X] T019 [P] [US2] Add Rust runtime benchmark contract/parity tests in `rust/crates/kcmt-cli/tests/benchmark_contracts.rs`

### Implementation for User Story 2

- [X] T020 [US2] Implement Python runtime benchmark orchestration separate from provider benchmarking in `kcmt/benchmark.py` and `kcmt/legacy_cli.py`
- [X] T021 [US2] Implement Rust runtime benchmark command handling and direct-binary execution in `rust/crates/kcmt-cli/src/commands/benchmark.rs` and `rust/crates/kcmt-cli/src/args.rs`
- [X] T022 [US2] Implement runtime benchmark report models and exports in `rust/crates/kcmt-bench/src/model.rs`, `rust/crates/kcmt-bench/src/export.rs`, and `rust/crates/kcmt-bench/src/runner.rs`
- [X] T023 [US2] Integrate synthetic and realistic corpora into the runtime benchmark flow in `scripts/benchmark/generate_uncommitted_repo.py` and `tests/fixtures/runtime_corpus/mini_realistic_repo/`
- [X] T024 [US2] Record failed and excluded runtime scenarios explicitly in `kcmt/benchmark.py` and `rust/crates/kcmt-bench/src/runner.rs`

**Checkpoint**: Maintainers can produce reproducible Python-vs-Rust runtime benchmark artifacts.

---

## Phase 5: User Story 3 - Keep Runtime and LLM Benchmarks Coherent (Priority: P3)

**Goal**: Preserve the current provider-quality benchmark while introducing the new
runtime benchmark mode with clear UX and documentation separation.

**Independent Test**: Run the legacy provider benchmark path and the new runtime
benchmark path, confirming that outputs, help text, and persisted artifacts remain
distinct and understandable.

### Tests for User Story 3

- [X] T025 [P] [US3] Add backward-compatibility tests for provider benchmark flags and output contracts in `tests/test_benchmark.py`, `tests/test_cli.py`, and `tests/test_ink_backend.py`

### Implementation for User Story 3

- [X] T026 [US3] Preserve legacy `--benchmark*` provider behavior while adding explicit benchmark modes in `kcmt/legacy_cli.py` and `rust/crates/kcmt-cli/src/args.rs`
- [X] T027 [US3] Keep the Ink benchmark backend provider-centric while excluding runtime benchmarking from the existing UI path in `kcmt/cli.py`, `kcmt/ink_backend.py`, and `kcmt/ui/ink/src/components/benchmark-view.mjs`
- [X] T028 [US3] Document provider-vs-runtime benchmark usage and examples in `docs/benchmark.md`, `docs/rust-migration-rollout.md`, and `README.md`

**Checkpoint**: Benchmark UX and artifacts stay coherent, backward compatible, and separately interpretable.

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Close validation, documentation, and deferred implementation-risk items.

- [X] T029 [P] Update validation evidence and contract notes in `specs/003-bring-rust-cli/validation/workflow-parity-catalog.md` and `specs/003-bring-rust-cli/validation/exit-error-baseline.md`
- [X] T030 Run full quality gates and capture proof in `specs/003-bring-rust-cli/tasks-validation.md`
- [X] T031 [P] Document `gitoxide` readiness and default-backend deferral in `specs/003-bring-rust-cli/research.md` and `rust/crates/kcmt-core/src/git/gitoxide_readonly.rs`

---

## Dependencies & Execution Order

### Phase Dependencies

- **Phase 1: Setup**: Starts immediately.
- **Phase 2: Foundational**: Depends on Setup completion and blocks all user stories.
- **Phase 3: US1**: Depends on Foundational completion; this is the MVP slice.
- **Phase 4: US2**: Depends on US1 completion because runtime benchmarking is not
  valid until Rust workflow parity exists.
- **Phase 5: US3**: Depends on US2 because benchmark-mode coherence requires both
  provider and runtime paths to exist.
- **Phase 6: Polish**: Depends on all desired user stories being complete.

### User Story Dependencies

- **US1**: No story dependency beyond Foundational.
- **US2**: Requires US1 because direct Python-vs-Rust runtime comparison is misleading
  before core workflow parity exists.
- **US3**: Requires US2 because documentation and compatibility rules depend on the
  runtime benchmark contract being real.

### Parallel Opportunities

- T003 can run in parallel with T001-T002 after directories are created.
- T005-T008 can run in parallel after T004 lands.
- T009 and T010 can run in parallel inside US1.
- T018 and T019 can run in parallel inside US2.
- T025 can run while US3 implementation tasks are being prepared.
- T029 and T031 can run in parallel during Polish.

---

## Parallel Example: User Story 1

```bash
# Contract tests in parallel
Task: "Add Python contract assertions for status/raw and runtime dispatch behavior in tests/test_cli.py and tests/test_main_entrypoint.py"
Task: "Add Rust contract tests for alias branding, parser errors, and repo-path flows in rust/tests/contract/us1_cli_contracts.rs"
```

---

## Parallel Example: User Story 2

```bash
# Runtime benchmark tests in parallel
Task: "Add Python runtime benchmark tests and schema assertions in tests/test_benchmark.py"
Task: "Add Rust runtime benchmark contract/parity tests in rust/tests/contract/us2_runtime_benchmark_schema.rs and rust/tests/parity/us2_runtime_benchmark.rs"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational
3. Complete Phase 3: User Story 1
4. Stop and validate Rust parity for the required-now workflow catalog

### Incremental Delivery

1. Deliver US1 to make Rust a real runtime candidate
2. Add US2 to produce reproducible runtime benchmark artifacts
3. Add US3 to preserve benchmark UX clarity and backward compatibility
4. Finish Polish with validation evidence and deferred backend documentation

### Suggested First Merge Slice

1. T001-T017 only
2. Merge once parity tests and `make check` pass
3. Start runtime benchmark work after that baseline is stable
