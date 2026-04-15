const ACTIVE_STAGES = new Set(['diff', 'llm_wait', 'prepared', 'commit']);
const DONE_STAGES = new Set(['done', 'failed']);

const PHASE_LABELS = {
  diff: 'DIFF',
  prepare: 'PREPARE',
  commit: 'COMMIT',
  push: 'PUSH',
  done: 'COMPLETE',
};

const STAGE_PROGRESS = {
  pending: 0,
  diff: 20,
  llm_wait: 45,
  prepared: 80,
  commit: 90,
  done: 100,
  failed: 100,
};

const STAGE_LABELS = {
  pending: 'pending',
  diff: 'collecting diff',
  llm_wait: 'waiting for LLM response',
  prepared: 'message prepared',
  commit: 'writing commit',
  done: 'commit complete',
  failed: 'commit failed',
};

const SLOW_THRESHOLDS_SECONDS = {
  diff: 5,
  llm_wait: 10,
  prepared: 10,
  commit: 5,
};

function asNumber(value, fallback = 0) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

export function formatDuration(seconds) {
  const total = Math.max(0, asNumber(seconds));
  if (total < 60) {
    return `${total.toFixed(1)}s`;
  }
  const minutes = Math.floor(total / 60);
  const remaining = total - minutes * 60;
  if (minutes < 60) {
    return `${minutes}m ${remaining.toFixed(0).padStart(2, '0')}s`;
  }
  const hours = Math.floor(minutes / 60);
  const leftoverMinutes = minutes % 60;
  return `${hours}h ${String(leftoverMinutes).padStart(2, '0')}m`;
}

export function normaliseWorkflowStats(stats = {}) {
  return {
    total_files: asNumber(stats?.total_files ?? stats?.total),
    diffs_built: asNumber(stats?.diffs_built),
    requests: asNumber(stats?.requests),
    responses: asNumber(stats?.responses),
    prepared: asNumber(stats?.prepared ?? stats?.ready),
    processed: asNumber(stats?.processed ?? stats?.done),
    successes: asNumber(stats?.successes),
    failures: asNumber(stats?.failures),
    elapsed: asNumber(stats?.elapsed),
    rate: asNumber(stats?.rate),
  };
}

function humanizePushState(pushState) {
  const value = String(pushState || 'idle').toLowerCase();
  if (value === 'pushing') {
    return 'Pushing';
  }
  if (value === 'done') {
    return 'Done';
  }
  if (value === 'error') {
    return 'Failed';
  }
  return 'Idle';
}

function deriveStage(entry, meta) {
  if (meta?.error) {
    return 'failed';
  }
  const currentStage = String(entry?.current_stage || '').trim().toLowerCase();
  if (currentStage) {
    return currentStage;
  }
  if (entry?.commit === 'err') {
    return 'failed';
  }
  if (entry?.commit === 'ok') {
    return 'done';
  }
  if (entry?.commit === 'running') {
    return 'commit';
  }
  if (entry?.res === 'ok' || entry?.batch === 'completed') {
    return 'prepared';
  }
  if (entry?.req === 'sent' || entry?.batch === 'validating' || entry?.batch === 'in_progress' || entry?.batch === 'finalizing') {
    return 'llm_wait';
  }
  if (entry?.diff === 'yes') {
    return 'diff';
  }
  return 'pending';
}

function deriveActiveLabel(entry, meta, stage) {
  if (meta?.error) {
    return String(meta.error);
  }
  const label = String(entry?.active_label || '').trim();
  if (label) {
    return label;
  }
  return STAGE_LABELS[stage] || 'pending';
}

function buildFileRecord(path, entry, meta, nowSeconds) {
  const stage = deriveStage(entry, meta);
  const stageStartedAt = asNumber(entry?.stage_started_at, asNumber(entry?.last_update_at, nowSeconds));
  const lastUpdateAt = asNumber(entry?.last_update_at, stageStartedAt);
  const stageDuration = Math.max(0, nowSeconds - stageStartedAt);
  const activeLabel = deriveActiveLabel(entry, meta, stage);
  const isActive = ACTIVE_STAGES.has(stage);
  const isDone = DONE_STAGES.has(stage);
  const threshold = SLOW_THRESHOLDS_SECONDS[stage];
  const isSlow = isActive && Number.isFinite(threshold) && stageDuration >= threshold;

  return {
    path,
    entry,
    stage,
    activeLabel,
    stageStartedAt,
    lastUpdateAt,
    stageDuration,
    isActive,
    isSlow,
    isDone,
    isFailed: stage === 'failed',
    progressPercent: STAGE_PROGRESS[stage] ?? 0,
    statusText: STAGE_LABELS[stage] || activeLabel || 'pending',
    subject: meta?.subject ? String(meta.subject) : '',
    error: meta?.error ? String(meta.error) : '',
  };
}

function compareFileRecords(left, right) {
  const rank = record => {
    if (record.isSlow) {
      return 0;
    }
    if (record.isActive) {
      return 1;
    }
    if (record.isFailed) {
      return 2;
    }
    if (record.isDone) {
      return 3;
    }
    return 4;
  };

  const rankDelta = rank(left) - rank(right);
  if (rankDelta !== 0) {
    return rankDelta;
  }
  if (left.isActive || left.isSlow || right.isActive || right.isSlow) {
    const durationDelta = right.stageDuration - left.stageDuration;
    if (durationDelta !== 0) {
      return durationDelta;
    }
  }
  if (left.isDone || right.isDone || left.isFailed || right.isFailed) {
    const updateDelta = right.lastUpdateAt - left.lastUpdateAt;
    if (updateDelta !== 0) {
      return updateDelta;
    }
  }
  return left.path.localeCompare(right.path);
}

function buildCurrentPhaseLabel({stage, pushState, status, stats, orderedFiles}) {
  if (String(pushState || '').toLowerCase() === 'pushing') {
    return 'PUSH';
  }
  if (status === 'completed' && String(pushState || '').toLowerCase() !== 'pushing') {
    return 'COMPLETE';
  }

  const totalFiles = Math.max(stats.total_files, orderedFiles.length);
  const hasPrepareActivity =
    stats.requests > 0 ||
    stats.responses > 0 ||
    stats.prepared > 0 ||
    orderedFiles.some(record => record.stage === 'llm_wait' || record.stage === 'prepared');
  const hasCommitActivity =
    stats.processed > 0 ||
    orderedFiles.some(record => record.stage === 'commit' || record.stage === 'done' || record.stage === 'failed');
  const activeStages = orderedFiles.filter(record => record.isActive).map(record => record.stage);

  if (!hasPrepareActivity && !hasCommitActivity && totalFiles > 0 && stats.diffs_built < totalFiles) {
    return 'DIFF';
  }

  if (activeStages.includes('commit')) {
    return 'COMMIT';
  }
  if (activeStages.includes('llm_wait') || activeStages.includes('prepared')) {
    return 'PREPARE';
  }
  if (activeStages.includes('diff')) {
    return 'DIFF';
  }
  if (hasCommitActivity) {
    return 'COMMIT';
  }
  if (hasPrepareActivity) {
    return 'PREPARE';
  }

  const stageKey = String(stage || '').trim().toLowerCase();
  if (PHASE_LABELS[stageKey]) {
    return PHASE_LABELS[stageKey];
  }
  return 'DIFF';
}

function buildSlowAlert(record) {
  if (!record || !record.isSlow) {
    return null;
  }
  let label = record.activeLabel;
  if (label.toLowerCase().startsWith('waiting ')) {
    label = label.slice('waiting '.length);
  }
  return `Slow step: waiting ${formatDuration(record.stageDuration)} ${label} on ${record.path}`;
}

function buildFooterStatus({
  currentPhaseLabel,
  pushState,
  status,
  stats,
  commitsInProgress,
  failureCount,
}) {
  if (String(pushState || '').toLowerCase() === 'pushing') {
    return 'Pushing commits to remote';
  }
  if (status === 'completed') {
    if (String(pushState || '').toLowerCase() === 'error') {
      return 'Push failed';
    }
    if (failureCount > 0) {
      return `Run completed with ${failureCount} failures`;
    }
    return 'Run complete';
  }
  const pendingRequests = Math.max(0, stats.requests - stats.responses);
  if (currentPhaseLabel === 'PREPARE' && pendingRequests > 0) {
    return `Waiting on ${pendingRequests} LLM requests`;
  }
  if (currentPhaseLabel === 'COMMIT' && commitsInProgress > 0) {
    return `Writing ${commitsInProgress} commits`;
  }
  if (currentPhaseLabel === 'DIFF' && stats.total_files > stats.diffs_built) {
    return `Collecting diffs for ${stats.total_files - stats.diffs_built} files`;
  }
  return 'Waiting for workflow activity';
}

function buildOverallProgressPct(stats, pushState, status, commitsInProgress) {
  const total = Math.max(0, stats.total_files);
  if (!total) {
    return 0;
  }

  let progressPct = 0;
  progressPct += (Math.min(stats.diffs_built, total) / total) * 20;
  progressPct += (Math.min(stats.prepared, total) / total) * 40;
  progressPct += (Math.min(stats.successes + commitsInProgress, total) / total) * 35;
  if (String(pushState || '').toLowerCase() === 'pushing') {
    progressPct += 2.5;
  } else if (
    String(pushState || '').toLowerCase() === 'done' ||
    (status === 'completed' && stats.successes + stats.failures >= total)
  ) {
    progressPct += 5;
  }
  if (status === 'completed' && String(pushState || '').toLowerCase() !== 'pushing') {
    return 100;
  }
  return Math.max(0, Math.min(100, progressPct));
}

export function buildWorkflowViewModel({
  stage = 'diff',
  status = 'running',
  stats,
  fileStates = {},
  fileMeta = {},
  now,
  viewportCount = 0,
  pushState = 'idle',
}) {
  const snapshot = normaliseWorkflowStats(stats);
  const nowSeconds = asNumber(now, Date.now() / 1000);
  const fileKeys = new Set([
    ...Object.keys(fileStates || {}),
    ...Object.keys(fileMeta || {}),
  ]);

  const orderedFiles = [...fileKeys]
    .map(path => buildFileRecord(path, fileStates?.[path] || {}, fileMeta?.[path] || {}, nowSeconds))
    .sort(compareFileRecords);

  const totalFiles = Math.max(snapshot.total_files, orderedFiles.length);
  const commitsInProgress = orderedFiles.filter(record => record.stage === 'commit').length;
  const failureCount = Math.max(
    snapshot.failures,
    orderedFiles.filter(record => record.isFailed).length,
  );
  const currentPhaseLabel = buildCurrentPhaseLabel({
    stage,
    pushState,
    status,
    stats: snapshot,
    orderedFiles,
  });
  const activeRecord = orderedFiles.find(record => record.isSlow) || orderedFiles.find(record => record.isActive) || null;
  const visibleCount = Math.min(Math.max(0, viewportCount), totalFiles);
  const startedCount = orderedFiles.filter(record => record.stage !== 'pending').length;
  const notStartedCount = Math.max(0, totalFiles - startedCount);

  return {
    currentPhaseLabel,
    totalElapsedLabel: formatDuration(snapshot.elapsed),
    stageCounters: [
      {label: 'Files discovered', value: totalFiles},
      {label: 'Diffs collected', value: snapshot.diffs_built},
      {label: 'LLM requests sent', value: snapshot.requests},
      {label: 'LLM responses received', value: snapshot.responses},
      {label: 'Messages prepared', value: snapshot.prepared},
      {label: 'Commits in progress', value: commitsInProgress},
      {label: 'Commits completed', value: snapshot.successes},
      {label: 'Failures', value: failureCount},
      {label: 'Push', value: humanizePushState(pushState)},
    ],
    activeNow: activeRecord
      ? `Active now: ${activeRecord.activeLabel} on ${activeRecord.path}`
      : null,
    slowAlert: buildSlowAlert(activeRecord),
    viewportSummary: totalFiles ? `Visible ${visibleCount} of ${totalFiles} files` : null,
    notStartedCount,
    orderedFiles,
    footerStatus: buildFooterStatus({
      currentPhaseLabel,
      pushState,
      status,
      stats: snapshot,
      commitsInProgress,
      failureCount,
    }),
    overallProgressPct: buildOverallProgressPct(snapshot, pushState, status, commitsInProgress),
  };
}
