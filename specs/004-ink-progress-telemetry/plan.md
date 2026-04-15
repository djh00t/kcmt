# Ink Workflow Progress Telemetry Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add clear Ink progress telemetry for large repository runs, including full-text stage counters, active-work visibility, slow-step warnings, and viewport orientation, without changing legacy CLI behavior.

**Architecture:** Extend `InkWorkflow` in `kcmt/ink_backend.py` so every file snapshot includes explicit stage and timing metadata. Move the frontend progress math into a new pure helper module consumed by `workflow-view.mjs`, then render the labeled counters, active item, slow warning, and large-run summary from that model. Keep the legacy CLI path untouched and cover the new behavior with Python and Node tests.

**Tech Stack:** Python 3.12, Node ESM, Ink/React, `pytest`, built-in `node:test`

---

**Branch**: `004-ink-progress-telemetry` | **Date**: 2026-04-14 | **Spec**: [`specs/004-ink-progress-telemetry/spec.md`](./spec.md)  
**Input**: Feature specification from `/specs/004-ink-progress-telemetry/spec.md`

## Summary

The current Ink workflow screen exposes low-signal counters (`req`, `res`) and
does not make long waits or off-screen work obvious. This plan enriches backend
telemetry per file, derives a strongly typed workflow view model with
human-readable labels, and updates the Ink screen to show full-text stage
counters, current active work, slow-step warnings, and large-run orientation.

## Technical Context

**Language/Version**: Python 3.12, Node ESM, React 18, Ink 4  
**Primary Dependencies**: Python `pytest`; Ink UI dependencies from
`kcmt/ui/ink/package.json` (`react`, `ink`, `chalk`, `ink-spinner`)  
**Storage**: In-memory workflow telemetry only; no persisted state changes  
**Testing**: `pytest`, built-in `node:test`, existing `make check` with a new Ink
test target added to the repo gate  
**Target Platform**: Local macOS/Linux terminal sessions using the Ink UI  
**Project Type**: Hybrid Python CLI with Node-based Ink frontend  
**Performance Goals**: Progress telemetry updates must remain responsive during
large runs of at least 500 files without adding visible workflow lag  
**Baseline Corpus**: `../kids_movie` for large-run manual validation plus the
existing Python test suite and new Node unit tests for the view model  
**Constraints**: Preserve legacy CLI output and flags, avoid adding a browser or
DOM test harness, keep the Ink component testable through pure helper functions,
and do not depend on a live provider response to validate stage labels  
**Scale/Scope**: `kcmt/ink_backend.py`, `kcmt/ui/ink/src/components/workflow-view.mjs`,
new Ink helper tests, and root quality gate wiring

## Constitution Check

*GATE: Must pass before implementation. Re-check after the plan is executed.*

- [x] Principle I: CLI contract compatibility is preserved or explicitly versioned  
  Decision: only the Ink view and its telemetry payload change; command flags,
  exit codes, and legacy CLI behavior remain unchanged.
- [x] Principle II: Git safety/atomic behavior is preserved with parity strategy  
  Decision: no git-operation behavior changes are introduced; this feature is UI
  telemetry only.
- [x] Principle III: Required tests and strict quality gates are defined  
  Decision: add `pytest` coverage for backend telemetry and `node:test` coverage
  for the frontend progress model, then wire the Ink test target into `make check`.
- [x] Principle IV: Performance claims include baseline corpus and metrics  
  Decision: validate interactively against `../kids_movie` and keep telemetry
  derivation in pure helper code to avoid adding runtime overhead.
- [x] Principle V: Secrets/config precedence/error handling constraints are covered  
  Decision: no secrets or config precedence changes; provider names can be shown
  in the active-work line without logging secret values.

No constitution violations identified.

## Project Structure

### Documentation (this feature)

```text
specs/004-ink-progress-telemetry/
├── spec.md
└── plan.md
```

### Source Code (repository root)

```text
kcmt/
├── ink_backend.py
└── ui/ink/
    ├── package.json
    └── src/
        ├── backend-client.mjs
        └── components/
            ├── workflow-view.mjs
            ├── workflow-progress-model.mjs          # new
            └── workflow-progress-model.test.mjs     # new

tests/
└── test_ink_backend.py

Makefile
```

**Structure Decision**: Keep `workflow-view.mjs` as the Ink renderer but extract
all counter, timing, sorting, and alert derivation into a new
`workflow-progress-model.mjs` helper so the UI logic is unit-testable without
adding a separate frontend testing framework.

## Implementation Tasks

### Task 1: Enrich Backend File Telemetry

**Files:**
- Modify: `kcmt/ink_backend.py`
- Test: `tests/test_ink_backend.py`

- [ ] **Step 1: Write the failing backend telemetry test**

```python
from __future__ import annotations

import kcmt.ink_backend as ink_backend


def test_ink_workflow_progress_event_tracks_stage_metadata(monkeypatch):
    monkeypatch.setattr(
        ink_backend.KlingonCMTWorkflow,
        "__init__",
        lambda self, **_: None,
    )

    workflow = ink_backend.InkWorkflow(lambda *_: None, repo_path=".")
    workflow._progress_event("diff-ready", file="alpha.py")
    first = workflow.file_states_snapshot()["alpha.py"]

    assert first["current_stage"] == "diff"
    assert first["active_label"] == "collecting diff"
    assert first["stage_started_at"] > 0
    assert first["last_update_at"] >= first["stage_started_at"]

    workflow._progress_event("request-sent", file="alpha.py")
    second = workflow.file_states_snapshot()["alpha.py"]

    assert second["current_stage"] == "llm_wait"
    assert second["active_label"] == "waiting for LLM response"
    assert second["stage_started_at"] >= first["last_update_at"]
```

- [ ] **Step 2: Run the backend test and verify it fails**

Run: `uv run pytest -q tests/test_ink_backend.py -k telemetry`  
Expected: `FAIL` because the file snapshot only contains the old compact flags
and no `current_stage`, `active_label`, or timestamp fields.

- [ ] **Step 3: Implement the telemetry metadata in the Ink backend**

```python
def _mark_file_state(self, file_path: str, *, stage: str, label: str) -> None:
    now = time.time()
    entry = self._file_states.setdefault(
        file_path,
        {
            "diff": "-",
            "req": "-",
            "res": "-",
            "batch": "-",
            "commit": "-",
            "current_stage": "pending",
            "active_label": "pending",
            "stage_started_at": now,
            "last_update_at": now,
        },
    )
    if entry.get("current_stage") != stage:
        entry["stage_started_at"] = now
    entry["current_stage"] = stage
    entry["active_label"] = label
    entry["last_update_at"] = now
```

Update `InkWorkflow._progress_event()` so:
- `diff-ready` maps to `current_stage="diff"` and `active_label="collecting diff"`
- `request-sent` maps to `current_stage="llm_wait"` and `active_label="waiting for LLM response"`
- `response` maps to `current_stage="prepared"` and `active_label="message prepared"`
- `commit-start` maps to `current_stage="commit"` and `active_label="writing commit"`
- `commit-done` maps to `current_stage="done"` and `active_label="commit complete"`
- `commit-error` maps to `current_stage="failed"` and `active_label="commit failed"`
- `push-start`/`push-done`/`push-error` remain aggregate workflow events, not per-file state

- [ ] **Step 4: Re-run the backend test and verify it passes**

Run: `uv run pytest -q tests/test_ink_backend.py -k telemetry`  
Expected: `PASS`

- [ ] **Step 5: Commit the backend telemetry change**

```bash
git add tests/test_ink_backend.py kcmt/ink_backend.py
git commit -m "feat(ink): enrich workflow telemetry"
```

### Task 2: Build a Pure Progress View Model with Labeled Stage Counters

**Files:**
- Create: `kcmt/ui/ink/src/components/workflow-progress-model.mjs`
- Create: `kcmt/ui/ink/src/components/workflow-progress-model.test.mjs`
- Modify: `kcmt/ui/ink/package.json`

- [ ] **Step 1: Write the failing Node test for labeled counters**

```javascript
import test from 'node:test';
import assert from 'node:assert/strict';
import {buildWorkflowViewModel} from './workflow-progress-model.mjs';

test('buildWorkflowViewModel exposes full-text stage counters', () => {
  const model = buildWorkflowViewModel({
    stats: {
      total_files: 694,
      diffs_built: 212,
      requests: 18,
      responses: 11,
      prepared: 11,
      processed: 9,
      successes: 9,
      failures: 0,
    },
    fileStates: {
      'README.md': {
        current_stage: 'llm_wait',
        active_label: 'waiting for LLM response',
        stage_started_at: 1,
        last_update_at: 12,
      },
    },
    now: 12,
    viewportCount: 20,
    pushState: 'idle',
  });

  assert.deepEqual(
    model.stageCounters.map(item => item.label),
    [
      'Files discovered',
      'Diffs collected',
      'LLM requests sent',
      'LLM responses received',
      'Messages prepared',
      'Commits in progress',
      'Commits completed',
      'Failures',
      'Push',
    ],
  );
});
```

- [ ] **Step 2: Run the Node test and verify it fails**

Run: `node --test kcmt/ui/ink/src/components/workflow-progress-model.test.mjs`  
Expected: `FAIL` because `workflow-progress-model.mjs` does not exist yet.

- [ ] **Step 3: Implement the pure view-model helper**

```javascript
export function buildWorkflowViewModel({stats, fileStates = {}, fileMeta = {}, now, viewportCount, pushState}) {
  const totalFiles = Number(stats?.total_files ?? 0);
  const diffsCollected = Number(stats?.diffs_built ?? 0);
  const llmRequestsSent = Number(stats?.requests ?? 0);
  const llmResponsesReceived = Number(stats?.responses ?? 0);
  const messagesPrepared = Number(stats?.prepared ?? 0);
  const commitsCompleted = Number(stats?.successes ?? 0);
  const commitsInProgress = Object.values(fileStates).filter(
    entry => entry.current_stage === 'commit',
  ).length;
  const failures = Number(stats?.failures ?? 0);

  const stageCounters = [
    {label: 'Files discovered', value: totalFiles},
    {label: 'Diffs collected', value: diffsCollected},
    {label: 'LLM requests sent', value: llmRequestsSent},
    {label: 'LLM responses received', value: llmResponsesReceived},
    {label: 'Messages prepared', value: messagesPrepared},
    {label: 'Commits in progress', value: commitsInProgress},
    {label: 'Commits completed', value: commitsCompleted},
    {label: 'Failures', value: failures},
    {label: 'Push', value: pushState},
  ];

  return {
    stageCounters,
    activeNow: null,
    slowAlert: null,
    viewportSummary: null,
    orderedFiles: [],
  };
}
```

Also update `kcmt/ui/ink/package.json`:

```json
{
  "scripts": {
    "test": "node --test src/components/workflow-progress-model.test.mjs"
  }
}
```

- [ ] **Step 4: Re-run the Node test and verify it passes**

Run: `npm --prefix kcmt/ui/ink test`  
Expected: `PASS`

- [ ] **Step 5: Commit the progress model**

```bash
git add kcmt/ui/ink/package.json kcmt/ui/ink/src/components/workflow-progress-model.mjs kcmt/ui/ink/src/components/workflow-progress-model.test.mjs
git commit -m "feat(ink): add labeled progress model"
```

### Task 3: Add Active-Work, Slow-Step, and Large-Run Orientation Logic

**Files:**
- Modify: `kcmt/ui/ink/src/components/workflow-progress-model.mjs`
- Modify: `kcmt/ui/ink/src/components/workflow-progress-model.test.mjs`

- [ ] **Step 1: Write the failing Node tests for slow alerts and viewport summaries**

```javascript
test('buildWorkflowViewModel surfaces active work and slow-step warnings', () => {
  const model = buildWorkflowViewModel({
    stats: {total_files: 694, diffs_built: 694, requests: 1, responses: 0, prepared: 0, processed: 0, successes: 0, failures: 0},
    fileStates: {
      '.gitignore': {
        current_stage: 'llm_wait',
        active_label: 'waiting for LLM response',
        stage_started_at: 0,
        last_update_at: 12,
      },
    },
    now: 12,
    viewportCount: 20,
    pushState: 'idle',
  });

  assert.match(model.activeNow, /waiting for LLM response on \.gitignore/);
  assert.match(model.slowAlert, /Slow step: waiting for LLM response/);
});

test('buildWorkflowViewModel reports viewport and pending counts for large runs', () => {
  const fileStates = Object.fromEntries([
    [
      '.gitignore',
      {
        current_stage: 'llm_wait',
        active_label: 'waiting for LLM response',
        stage_started_at: 0,
        last_update_at: 12,
      },
    ],
    ...Array.from({length: 693}, (_, index) => [
      `file-${String(index + 1).padStart(3, '0')}.txt`,
      {
        current_stage: index < 211 ? 'diff' : 'pending',
        active_label: index < 211 ? 'collecting diff' : 'pending',
        stage_started_at: 0,
        last_update_at: 0,
      },
    ]),
  ]);

  const model = buildWorkflowViewModel({
    stats: {total_files: 694, diffs_built: 212, requests: 18, responses: 11, prepared: 11, processed: 9, successes: 9, failures: 0},
    fileStates,
    now: 12,
    viewportCount: 20,
    pushState: 'idle',
  });

  assert.equal(model.viewportSummary, 'Visible 20 of 694 files');
  assert.equal(model.notStartedCount, 482);
  assert.equal(model.orderedFiles[0].path, '.gitignore');
});
```

- [ ] **Step 2: Run the Node test and verify it fails**

Run: `npm --prefix kcmt/ui/ink test`  
Expected: `FAIL` because `buildWorkflowViewModel()` does not yet produce
`activeNow`, `slowAlert`, `viewportSummary`, `notStartedCount`, or active-first
ordering.

- [ ] **Step 3: Implement thresholds, ordering, and orientation output**

```javascript
const SLOW_THRESHOLDS_SECONDS = {
  diff: 5,
  llm_wait: 10,
  prepared: 10,
  commit: 5,
  push: 10,
};

function rankFile(record) {
  if (record.isSlow) return 0;
  if (record.isActive) return 1;
  if (record.isDone) return 2;
  return 3;
}
```

Extend the helper so it:
- computes `activeNow` from the slowest active file
- emits `slowAlert` when `now - stage_started_at` crosses the threshold for the
  current stage
- returns `viewportSummary` as `Visible X of Y files`
- returns `notStartedCount`
- sorts file rows as slow active, active, recently completed, untouched, then
  alphabetical for ties

- [ ] **Step 4: Re-run the Node test and verify it passes**

Run: `npm --prefix kcmt/ui/ink test`  
Expected: `PASS`

- [ ] **Step 5: Commit the alert and ordering logic**

```bash
git add kcmt/ui/ink/src/components/workflow-progress-model.mjs kcmt/ui/ink/src/components/workflow-progress-model.test.mjs
git commit -m "feat(ink): add slow-step and viewport summaries"
```

### Task 4: Render the New Progress Model in the Ink Workflow Screen and Wire the Repo Gate

**Files:**
- Modify: `kcmt/ui/ink/src/components/workflow-view.mjs`
- Modify: `Makefile`
- Test: `tests/test_ink_backend.py`
- Test: `kcmt/ui/ink/src/components/workflow-progress-model.test.mjs`

- [ ] **Step 1: Write a failing regression in the Node view-model test for the footer sentence**

```javascript
test('buildWorkflowViewModel emits a clear footer sentence for in-flight work', () => {
  const fileStates = Object.fromEntries([
    [
      '.gitignore',
      {
        current_stage: 'llm_wait',
        active_label: 'waiting for LLM response',
        stage_started_at: 0,
        last_update_at: 12,
      },
    ],
    ...Array.from({length: 693}, (_, index) => [
      `file-${String(index + 1).padStart(3, '0')}.txt`,
      {
        current_stage: index < 211 ? 'diff' : 'pending',
        active_label: index < 211 ? 'collecting diff' : 'pending',
        stage_started_at: 0,
        last_update_at: 0,
      },
    ]),
  ]);

  const model = buildWorkflowViewModel({
    stats: {total_files: 694, diffs_built: 694, requests: 18, responses: 11, prepared: 11, processed: 9, successes: 9, failures: 0},
    fileStates,
    now: 12,
    viewportCount: 20,
    pushState: 'idle',
  });

  assert.match(model.footerStatus, /waiting on 18 LLM requests/i);
});
```

- [ ] **Step 2: Run the regression tests and verify the new assertion fails**

Run: `npm --prefix kcmt/ui/ink test`  
Expected: `FAIL` because `footerStatus` is not yet computed.

- [ ] **Step 3: Integrate the helper into `workflow-view.mjs` and add the Ink test gate**

```javascript
import {buildWorkflowViewModel} from './workflow-progress-model.mjs';

const model = buildWorkflowViewModel({
  stats,
  fileStates,
  fileMeta,
  now: Date.now() / 1000,
  viewportCount: getFileViewportCount(),
  pushState,
});
```

Update the component so it:
- renders the stage counter strip using `model.stageCounters`
- renders `model.activeNow` and `model.slowAlert` above the file list
- renders `model.viewportSummary` and `model.notStartedCount`
- uses `model.orderedFiles` instead of alphabetical-only ordering
- renders `model.footerStatus` in the footer

Update `Makefile`:

```make
test-ink:
	npm --prefix kcmt/ui/ink test

check: ruff-fix format lint typecheck test-strict test-ink
```

- [ ] **Step 4: Re-run all relevant checks and verify they pass**

Run: `uv run pytest -q tests/test_ink_backend.py`  
Expected: `PASS`

Run: `npm --prefix kcmt/ui/ink test`  
Expected: `PASS`

Run: `make check`  
Expected: Python checks remain green and the new Ink test target passes.

- [ ] **Step 5: Commit the screen integration**

```bash
git add kcmt/ui/ink/src/components/workflow-view.mjs Makefile kcmt/ui/ink/src/components/workflow-progress-model.mjs kcmt/ui/ink/src/components/workflow-progress-model.test.mjs tests/test_ink_backend.py
git commit -m "feat(ink): surface detailed workflow progress"
```

## Spec Coverage Check

- FR-001 through FR-004 are covered by Tasks 2 and 4 via full-text counter labels,
  explicit phase rendering, and elapsed runtime output in the helper-backed view.
- FR-005 through FR-008 are covered by Tasks 1, 3, and 4 via active item labels,
  slow-step warnings, active-first sorting, and viewport summaries.
- FR-009 is covered by Task 1 through backend telemetry enrichment.
- FR-010 is covered by scope: only Ink files, tests, and quality gate wiring are
  changed; no legacy CLI contracts are modified.
- SC-001 through SC-004 are covered by the Node and Python tests plus `../kids_movie`
  manual validation in Task 4.

## Complexity Tracking

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| None | N/A | N/A |
