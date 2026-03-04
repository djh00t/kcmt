# Tasks: Rust Runtime Canary Rollout and Observability

**Input**: Design documents from `/Users/djh/work/src/github.com_local/djh00t/kcmt/specs/002-rust-canary-rollout/`
**Prerequisites**: plan.md (required), spec.md (required), research.md, data-model.md, contracts/, quickstart.md

**Tests**: Included because this feature explicitly requires runtime routing and canary probe validation.

**Organization**: Tasks are grouped by user story to support independent implementation and validation.

## Format: `[ID] [P?] [Story?] Description`

- **[P]**: Parallelizable (different files, no incomplete dependency overlap)
- **[Story]**: Story label used only in user-story phases (`[US1]`, `[US2]`, `[US3]`)

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Create the canary probe and workflow scaffolding.

- [X] T001 Create canary probe module scaffold in scripts/canary/runtime_canary_probe.py
- [X] T002 Create canary smoke workflow scaffold in .github/workflows/rust-canary-smoke.yml
- [X] T003 [P] Add canary validation log scaffold in specs/002-rust-canary-rollout/tasks-validation.md

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Implement shared runtime trace primitives and basic test harness before story work.

**⚠️ CRITICAL**: Complete this phase before user story implementation.

- [X] T004 Implement runtime decision record builder and trace emitter helpers in kcmt/main.py
- [X] T005 [P] Add foundational unit tests for trace opt-in and fallback behavior in tests/test_main_entrypoint.py
- [X] T006 [P] Implement JSON trace parsing and assertion helpers in scripts/canary/runtime_canary_probe.py
- [X] T007 Wire canary workflow setup/build/probe invocation in .github/workflows/rust-canary-smoke.yml

**Checkpoint**: Foundation complete; user stories can proceed.

---

## Phase 3: User Story 1 - Automated Canary Safety Gate (Priority: P1) 🎯 MVP

**Goal**: Ensure CI can deterministically validate wrapper-routed Rust canary behavior.

**Independent Test**: Run canary workflow on PR and verify all scenarios pass with expected runtime selection and exits.

### Tests for User Story 1

- [X] T008 [P] [US1] Define canary probe scenarios for baseline/canary/fallback/rollback in scripts/canary/runtime_canary_probe.py
- [X] T009 [P] [US1] Add automated probe smoke test coverage for scenario contract in tests/test_runtime_canary_probe.py

### Implementation for User Story 1

- [X] T010 [US1] Implement scenario execution and failure reporting in scripts/canary/runtime_canary_probe.py
- [X] T011 [US1] Add artifact upload and workflow path filters in .github/workflows/rust-canary-smoke.yml
- [X] T012 [US1] Record canary workflow evidence links and outcomes in specs/002-rust-canary-rollout/tasks-validation.md

**Checkpoint**: US1 canary gate is independently functional.

---

## Phase 4: User Story 2 - Runtime Decision Observability (Priority: P2)

**Goal**: Provide safe machine-readable runtime traces for diagnostics.

**Independent Test**: With `KCMT_RUNTIME_TRACE=1`, verify trace records for Python default, Rust canary, and missing-binary fallback paths.

### Tests for User Story 2

- [X] T013 [P] [US2] Add tests for trace record fields and stderr-only behavior in tests/test_main_entrypoint.py

### Implementation for User Story 2

- [X] T014 [US2] Implement runtime decision reasons and normalized runtime-mode handling in kcmt/main.py
- [X] T015 [US2] Ensure trace payload excludes secrets and only emits allowlisted fields in kcmt/main.py
- [X] T016 [US2] Update runtime trace contract documentation in specs/002-rust-canary-rollout/contracts/canary-probe-contract.md

**Checkpoint**: US2 trace observability is independently functional.

---

## Phase 5: User Story 3 - Rollout Procedure and Rollback Controls (Priority: P3)

**Goal**: Deliver operator-ready staged rollout/rollback guidance using current env controls.

**Independent Test**: Follow quickstart and rollout docs on local repo and verify baseline/canary/rollback command paths.

### Tests for User Story 3

- [X] T017 [P] [US3] Validate quickstart canary commands and capture outcomes in specs/002-rust-canary-rollout/tasks-validation.md

### Implementation for User Story 3

- [X] T018 [US3] Update staged rollout and rollback procedures in docs/rust-migration-rollout.md
- [X] T019 [US3] Refine quickstart operational commands for baseline/canary/rollback in specs/002-rust-canary-rollout/quickstart.md

**Checkpoint**: US3 operational guidance is independently usable.

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Final verification and quality gate evidence.

- [X] T020 Run strict quality gate and capture results in specs/002-rust-canary-rollout/tasks-validation.md
- [X] T021 Verify workflow YAML validity and document final feature status in specs/002-rust-canary-rollout/tasks-validation.md

---

## Dependencies & Execution Order

### Phase Dependencies

- **Phase 1 (Setup)**: no dependencies
- **Phase 2 (Foundational)**: depends on Phase 1; blocks all user stories
- **Phase 3 (US1)**: depends on Phase 2 completion
- **Phase 4 (US2)**: depends on Phase 2 completion; can run with US1 when file overlap is coordinated
- **Phase 5 (US3)**: depends on Phases 3-4 outputs for evidence commands
- **Phase 6 (Polish)**: depends on completion of all user story phases

### User Story Dependency Graph

- **US1 (P1)**: MVP canary gate after foundation
- **US2 (P2)**: independent diagnostics layer after foundation
- **US3 (P3)**: operational rollout guidance dependent on implemented canary controls

### Parallel Opportunities

- T003, T005, T006 can run in parallel
- T008 and T009 can run in parallel
- T013 can run in parallel with T010/T011 once foundational helpers are complete

---

## Implementation Strategy

### MVP First

1. Complete Phase 1 and Phase 2
2. Complete US1 tasks (T008-T012)
3. Validate canary gate behavior before proceeding

### Incremental Delivery

1. Deliver US1 (CI canary gate)
2. Deliver US2 (trace diagnostics)
3. Deliver US3 (rollout/rollback operations)
4. Finalize polish with quality-gate evidence
