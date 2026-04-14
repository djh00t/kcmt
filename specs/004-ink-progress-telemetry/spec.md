# Feature Specification: Ink Workflow Progress Telemetry and Slow-Step Visibility

**Feature Branch**: `004-ink-progress-telemetry`  
**Created**: 2026-04-14  
**Status**: Draft  
**Input**: User description: "We need more frontend progress indication especially when there are large numbers of files or a step is running slowly. Option 2 plus we need counters for each stage of the process. Labels need to be clear of what each counter is as well."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Understand Current Workflow Stage at a Glance (Priority: P1)

As a developer running the Ink UI, I can see clearly labeled counters for each
workflow stage so I know whether kcmt is collecting diffs, waiting on the LLM,
preparing messages, committing, or pushing.

**Why this priority**: The current UI can look idle even when work is happening,
which makes the tool feel broken during normal runs.

**Independent Test**: Run `kcmt` on a repository with multiple changed files and
verify the Ink UI shows full-text counters for each stage, with values changing
as the workflow progresses.

**Acceptance Scenarios**:

1. **Given** a repository with many changed files, **When** the Ink workflow is
   running, **Then** the UI displays clear counters for `Files discovered`,
   `Diffs collected`, `LLM requests sent`, `LLM responses received`,
   `Messages prepared`, `Commits in progress`, `Commits completed`, `Failures`,
   and `Push`.
2. **Given** a user is watching a live run, **When** a stage is active,
   **Then** the current phase label matches the actual backend stage instead of
   collapsing everything into a generic in-progress state.
3. **Given** the workflow completes, **When** the final state is rendered,
   **Then** the counters remain understandable and reflect the completed run.

---

### User Story 2 - Identify Slow or Stalled Work Quickly (Priority: P2)

As a developer waiting on kcmt, I can see which file or step is currently active
and whether it is taking unusually long so I can distinguish a slow provider
response from a hung workflow.

**Why this priority**: The biggest trust issue is not just runtime length; it is
the inability to tell what kcmt is waiting on.

**Independent Test**: Simulate a long-running diff, LLM, commit, or push step
and verify the Ink UI shows a pinned active item plus a slow-step warning once a
threshold is crossed.

**Acceptance Scenarios**:

1. **Given** a file is waiting on an LLM response, **When** the response takes
   longer than the configured threshold, **Then** the UI shows a highlighted
   slow-step warning with the file path, stage label, and elapsed time.
2. **Given** a run is actively processing a file, **When** the user views the
   screen, **Then** the UI shows an `Active now` line naming the current file and
   the current stage in plain language.
3. **Given** no step has crossed a slow threshold, **When** the workflow is
   progressing normally, **Then** the UI still shows active work without a false
   slow warning.

---

### User Story 3 - Stay Oriented During Large Repository Runs (Priority: P3)

As a developer running kcmt on a repository with hundreds of files, I can tell
how many files are visible, how many remain untouched, and which files are most
important to watch right now.

**Why this priority**: Large runs overflow the visible file list, so work can be
occurring off-screen even when the visible rows look unchanged.

**Independent Test**: Run the Ink UI against a large fixture or real repository
with hundreds of files and verify the UI reports visible vs total files, active
vs pending counts, and sorts active or stalled files to the top during the run.

**Acceptance Scenarios**:

1. **Given** a run has more files than fit in the visible viewport, **When** the
   Ink UI is rendering, **Then** it shows `Visible X of Y files` and makes it
   clear that additional files are off-screen.
2. **Given** some files are active or stalled, **When** the file list is shown,
   **Then** those files are prioritised above untouched files during the live run.
3. **Given** most files have not started yet, **When** the run is early in the
   workflow, **Then** the UI explicitly reports how many files are not yet started.

### Edge Cases

- The run contains only one file, so the counter labels still need to read cleanly.
- A file fails during prepare or commit, and the failure must appear in both the
  per-file view and the aggregate stage counters.
- The backend emits state changes faster than the UI refresh cadence.
- A push step starts only after all commits are done, so the push state must not
  be confused with commit progress.
- A file leaves the visible list due to sorting changes while still being the
  active or slow item.
- The workflow is running on a narrow terminal where long labels and paths must
  still degrade gracefully without hiding meaning.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The Ink workflow view MUST display full-text counters for each
  major workflow stage instead of abbreviated labels such as `req` or `res`.
- **FR-002**: The UI MUST show the current workflow phase using the labels
  `DIFF`, `PREPARE`, `COMMIT`, and `PUSH`, with a completed state after the run.
- **FR-003**: The UI MUST show `Files discovered`, `Diffs collected`,
  `LLM requests sent`, `LLM responses received`, `Messages prepared`,
  `Commits in progress`, `Commits completed`, `Failures`, and `Push`.
- **FR-004**: The UI MUST show total elapsed runtime for the current run.
- **FR-005**: The UI MUST show a pinned `Active now` line that identifies the
  current active file and its current stage label in plain language.
- **FR-006**: The UI MUST show a highlighted slow-step warning when an active
  stage exceeds the defined threshold for that stage.
- **FR-007**: The file list MUST sort stalled active files first, then active
  files, then recently completed files, then untouched files while a run is live.
- **FR-008**: The UI MUST show large-run orientation cues, including
  `Visible X of Y files` and a count of files not yet started.
- **FR-009**: Backend telemetry emitted to the Ink frontend MUST include enough
  per-file state to derive clear stage counters, active-item labels, and elapsed
  timers without requiring the UI to guess from incomplete flags.
- **FR-010**: The existing legacy CLI path and command-line behavior MUST remain
  unchanged by this feature.

### Non-Functional Requirements *(mandatory)*

- **NFR-001**: The new progress telemetry MUST preserve existing CLI compatibility
  and MUST NOT change documented command flags, exit codes, or non-Ink output.
- **NFR-002**: Aggregate stage counters MUST remain internally consistent with
  the backend workflow state at every emitted update.
- **NFR-003**: The Ink UI MUST remain readable on large runs with at least 500
  files in scope.
- **NFR-004**: Slow-step warnings MUST avoid false positives during ordinary fast
  transitions while still surfacing sustained waits quickly enough to be useful.
- **NFR-005**: Additional progress telemetry MUST NOT introduce significant UI
  lag or visibly degrade normal workflow throughput.

### Key Entities *(include if feature involves data)*

- **Workflow Stage Summary**: Aggregate counters representing the current totals
  for each named workflow stage in a live run.
- **File Progress Record**: Per-file UI telemetry including current stage,
  stage start time, last update time, active label, and terminal outcome.
- **Slow Step Alert**: A derived warning for the currently active file when its
  stage duration exceeds the threshold defined for that stage.
- **Viewport Summary**: Large-repo orientation metadata describing total files,
  visible files, and not-yet-started files.

### Dependencies and Assumptions

- The backend progress emitter remains the source of truth for stage transitions.
- Stage thresholds can be fixed defaults for the first version and do not require
  user configuration in this feature.
- Per-file timestamps can be captured in the backend without changing the core
  git or LLM workflow contracts.
- The file list may reorder during live runs if that improves clarity for active
  or stalled work.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: In manual validation on a large repository run, the Ink UI always
  shows clearly labeled counters for every agreed workflow stage and no counter
  label relies on abbreviations alone.
- **SC-002**: In a forced slow-step validation scenario, the UI shows the active
  file and a slow-step warning within the threshold window for the affected stage.
- **SC-003**: In a repository run with at least 500 files, the UI reports total
  files, visible files, and not-yet-started files without hiding off-screen work.
- **SC-004**: Automated tests cover the backend telemetry transitions and the
  frontend rendering of stage counters, active-item status, and slow-step alerts.
