import assert from 'node:assert/strict';
import test from 'node:test';

import {buildWorkflowViewModel} from './workflow-progress-model.mjs';

test('buildWorkflowViewModel exposes full-text stage counters and current phase labels', () => {
  const model = buildWorkflowViewModel({
    stage: 'prepare',
    status: 'running',
    stats: {
      total_files: 694,
      diffs_built: 212,
      requests: 18,
      responses: 11,
      prepared: 11,
      processed: 9,
      successes: 9,
      failures: 0,
      elapsed: 12.4,
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
  assert.equal(model.currentPhaseLabel, 'PREPARE');
  assert.equal(model.totalElapsedLabel, '12.4s');
});

test('buildWorkflowViewModel surfaces active work and slow-step warnings', () => {
  const model = buildWorkflowViewModel({
    stage: 'prepare',
    status: 'running',
    stats: {
      total_files: 694,
      diffs_built: 694,
      requests: 1,
      responses: 0,
      prepared: 0,
      processed: 0,
      successes: 0,
      failures: 0,
      elapsed: 12,
    },
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

  assert.match(model.activeNow, /waiting for LLM response on \.gitignore/i);
  assert.match(model.slowAlert, /Slow step: waiting 12\.0s for LLM response on \.gitignore/i);
});

test('buildWorkflowViewModel prefers active file telemetry over a stale diff stage', () => {
  const model = buildWorkflowViewModel({
    stage: 'diff',
    status: 'running',
    stats: {
      total_files: 1,
      diffs_built: 1,
      requests: 1,
      responses: 0,
      prepared: 0,
      processed: 0,
      successes: 0,
      failures: 0,
      elapsed: 3,
    },
    fileStates: {
      '.gitignore': {
        current_stage: 'llm_wait',
        active_label: 'waiting for LLM response',
        stage_started_at: 0,
        last_update_at: 3,
      },
    },
    now: 3,
    viewportCount: 20,
    pushState: 'idle',
  });

  assert.equal(model.currentPhaseLabel, 'PREPARE');
});

test('buildWorkflowViewModel reports viewport, pending counts, and active-first ordering', () => {
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
    stage: 'prepare',
    status: 'running',
    stats: {
      total_files: 694,
      diffs_built: 212,
      requests: 18,
      responses: 11,
      prepared: 11,
      processed: 9,
      successes: 9,
      failures: 0,
      elapsed: 12,
    },
    fileStates,
    now: 12,
    viewportCount: 20,
    pushState: 'idle',
  });

  assert.equal(model.viewportSummary, 'Visible 20 of 694 files');
  assert.equal(model.notStartedCount, 482);
  assert.equal(model.orderedFiles[0].path, '.gitignore');
});

test('buildWorkflowViewModel emits a clear footer sentence for in-flight work', () => {
  const fileStates = {
    '.gitignore': {
      current_stage: 'llm_wait',
      active_label: 'waiting for LLM response',
      stage_started_at: 0,
      last_update_at: 12,
    },
  };

  const model = buildWorkflowViewModel({
    stage: 'prepare',
    status: 'running',
    stats: {
      total_files: 694,
      diffs_built: 694,
      requests: 18,
      responses: 11,
      prepared: 11,
      processed: 9,
      successes: 9,
      failures: 0,
      elapsed: 12,
    },
    fileStates,
    now: 12,
    viewportCount: 20,
    pushState: 'idle',
  });

  assert.match(model.footerStatus, /waiting on 7 LLM requests/i);
});
