# Tasks: Ink Workflow Progress Telemetry and Slow-Step Visibility

**Input**: Design documents from `/specs/004-ink-progress-telemetry/`  
**Prerequisites**: plan.md, spec.md

**Tests**: Tests are required for this feature because the specification requires
backend telemetry coverage, frontend rendering coverage, and validation of large-run
progress visibility.

**Organization**: Tasks are grouped by user story so stage counters, slow-step
visibility, and large-run orientation can be implemented and validated
independently.

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Establish the shared Ink test harness and file layout used by all stories.

- [ ] T001 [P] Add Ink frontend test entry wiring in `kcmt/ui/ink/package.json`
- [ ] T002 [P] Create Node test scaffold for the new progress helper in `kcmt/ui/ink/src/components/workflow-progress-model.test.mjs`
- [ ] T003 [P] Extend Ink backend test scaffolding for workflow telemetry in `tests/test_ink_backend.py`

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Add the telemetry and helper scaffolding that all stories depend on.

**⚠️ CRITICAL**: No user story work should begin until this phase is complete.

- [ ] T004 Create the pure workflow progress helper scaffold in `kcmt/ui/ink/src/components/workflow-progress-model.mjs`
- [ ] T005 Add per-file stage and timestamp telemetry scaffolding in `kcmt/ink_backend.py`
- [ ] T006 [P] Import the progress helper into `kcmt/ui/ink/src/components/workflow-view.mjs` without changing displayed behavior yet

**Checkpoint**: Backend snapshots carry explicit per-file stage data, and the Ink
screen can consume the new helper module.

---

## Phase 3: User Story 1 - Understand Current Workflow Stage at a Glance (Priority: P1) 🎯 MVP

**Goal**: Show clear, full-text workflow counters and phase labels so the Ink UI
always explains what kcmt is doing.

**Independent Test**: Run the Ink workflow against a repo with multiple changed files
and verify the screen shows labeled counters for each stage, a current phase label,
and total elapsed runtime without relying on abbreviations.

### Tests for User Story 1

- [ ] T007 [P] [US1] Add backend assertions for `current_stage`, `active_label`, and snapshot stability in `tests/test_ink_backend.py`
- [ ] T008 [P] [US1] Add Node tests for full-text stage counters and phase labels in `kcmt/ui/ink/src/components/workflow-progress-model.test.mjs`

### Implementation for User Story 1

- [ ] T009 [US1] Implement labeled stage-counter derivation in `kcmt/ui/ink/src/components/workflow-progress-model.mjs`
- [ ] T010 [US1] Render the full-text counter strip, current phase label, and total elapsed runtime in `kcmt/ui/ink/src/components/workflow-view.mjs`
- [ ] T011 [US1] Preserve final-state rendering for completed runs in `kcmt/ui/ink/src/components/workflow-view.mjs` and `kcmt/ink_backend.py`

**Checkpoint**: The Ink UI clearly shows the current phase and each stage counter in
plain language.

---

## Phase 4: User Story 2 - Identify Slow or Stalled Work Quickly (Priority: P2)

**Goal**: Make the currently active file and slow-running step obvious during a live run.

**Independent Test**: Simulate a slow LLM or commit step and verify the screen shows
an `Active now` line plus a slow-step warning with stage label and elapsed time.

### Tests for User Story 2

- [ ] T012 [P] [US2] Add backend transition-timing tests in `tests/test_ink_backend.py`
- [ ] T013 [P] [US2] Add Node tests for active-item selection and slow-step warnings in `kcmt/ui/ink/src/components/workflow-progress-model.test.mjs`

### Implementation for User Story 2

- [ ] T014 [US2] Implement active-item selection and human-readable stage labels in `kcmt/ui/ink/src/components/workflow-progress-model.mjs`
- [ ] T015 [US2] Implement slow-step threshold logic and alert text in `kcmt/ui/ink/src/components/workflow-progress-model.mjs`
- [ ] T016 [US2] Render the `Active now`, slow-step, and footer status lines in `kcmt/ui/ink/src/components/workflow-view.mjs`

**Checkpoint**: A user can tell which file is active and whether a slow wait is due
to diffing, the LLM, commit, or push.

---

## Phase 5: User Story 3 - Stay Oriented During Large Repository Runs (Priority: P3)

**Goal**: Make off-screen work visible and prioritise active or stalled files during
large runs.

**Independent Test**: Run against a large repository fixture and verify the screen
shows `Visible X of Y files`, a not-yet-started count, and active-first file ordering.

### Tests for User Story 3

- [ ] T017 [P] [US3] Add Node tests for viewport summaries, not-started counts, and active-first ordering in `kcmt/ui/ink/src/components/workflow-progress-model.test.mjs`

### Implementation for User Story 3

- [ ] T018 [US3] Implement viewport summary and not-started count derivation in `kcmt/ui/ink/src/components/workflow-progress-model.mjs`
- [ ] T019 [US3] Implement active-first and slow-first file ordering in `kcmt/ui/ink/src/components/workflow-progress-model.mjs`
- [ ] T020 [US3] Render large-run orientation text and ordered file rows in `kcmt/ui/ink/src/components/workflow-view.mjs`

**Checkpoint**: Large runs remain understandable even when most files are outside the
visible viewport.

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Wire the new test gate into repo checks and validate the end-to-end experience.

- [ ] T021 [P] Add an Ink-specific frontend test target to `Makefile`
- [ ] T022 Run targeted validation in `tests/test_ink_backend.py` and `kcmt/ui/ink/src/components/workflow-progress-model.test.mjs`
- [ ] T023 Validate the live Ink workflow against `../kids_movie` using `uv run kcmt --repo-path ../kids_movie --limit 1` and a larger run without `--limit`

---

## Dependencies & Execution Order

### Phase Dependencies

- **Phase 1: Setup**: Starts immediately.
- **Phase 2: Foundational**: Depends on Phase 1 and blocks all user stories.
- **Phase 3: US1**: Depends on Foundational and delivers the MVP.
- **Phase 4: US2**: Depends on US1 because slow-step visibility needs the clear
  stage model and labeled counters to exist first.
- **Phase 5: US3**: Depends on US1 and US2 because large-run orientation builds on
  the same enriched progress model and active-item ordering.
- **Phase 6: Polish**: Depends on all desired stories being complete.

### User Story Dependencies

- **US1**: No dependency beyond Foundational.
- **US2**: Requires US1’s labeled counters and phase model.
- **US3**: Requires the shared progress model from US1 and benefits from the active
  file semantics from US2.

### Parallel Opportunities

- T001-T003 can run in parallel during Setup.
- T005 and T006 can run in parallel after T004 is established.
- T007 and T008 can run in parallel inside US1.
- T012 and T013 can run in parallel inside US2.
- T017 can run while US2 implementation work is being prepared.
- T021 and T022 can run in parallel during Polish once implementation is complete.

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational
3. Complete Phase 3: User Story 1
4. Validate that the Ink UI now clearly explains the workflow stages

### Incremental Delivery

1. Deliver US1 so the workflow stage counters and phase labels become trustworthy
2. Add US2 so slow or stalled work is visible
3. Add US3 so large-repo runs remain understandable
4. Finish with repo-level test wiring and live validation

### Suggested Subagent Sequence

1. Task group T001-T006 to establish the shared progress model and telemetry hooks
2. Task group T007-T011 to land the MVP counter and phase labeling
3. Task group T012-T016 to add active-item and slow-step visibility
4. Task group T017-T020 to finish large-run orientation
5. Task group T021-T023 to wire validation and perform live checks
