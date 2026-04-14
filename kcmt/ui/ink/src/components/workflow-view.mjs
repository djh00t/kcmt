import React, {useCallback, useContext, useEffect, useMemo, useRef, useState} from 'react';
import {Box, Text, useInput, useStdout} from 'ink';
import chalk from 'chalk';
import {AppContext} from '../app-context.mjs';
import {buildWorkflowViewModel} from './workflow-progress-model.mjs';
const h = React.createElement;

function ellipsize(text, maxLength) {
  const value = text == null ? '' : String(text);
  if (!maxLength || value.length <= maxLength) {
    return value;
  }
  const limit = Math.max(1, maxLength - 1);
  return `${value.slice(0, limit)}…`;
}

function normaliseStats(stats = {}) {
  if (!stats) {
    return {
      total_files: 0,
      diffs_built: 0,
      requests: 0,
      responses: 0,
      prepared: 0,
      processed: 0,
      successes: 0,
      failures: 0,
      rate: 0,
    };
  }
  return {
    total_files: stats.total_files ?? stats.total ?? 0,
    prepared: stats.prepared ?? stats.ready ?? 0,
    processed: stats.processed ?? stats.done ?? 0,
    successes: stats.successes ?? 0,
    failures: stats.failures ?? 0,
    rate: stats.rate ?? 0,
    diffs_built: stats.diffs_built ?? 0,
    requests: stats.requests ?? 0,
    responses: stats.responses ?? 0,
    elapsed: stats.elapsed ?? 0,
  };
}

function useMessageLog() {
  const idRef = useRef(0);
  const [messages, setMessages] = useState([]);

  const append = useCallback(lines => {
    const list = Array.isArray(lines) ? lines : [lines];
    if (!list.length) {
      return;
    }
    setMessages(prev => {
      const next = [...prev];
      list.forEach(line => {
        if (line === undefined || line === null) {
          return;
        }
        const text = String(line);
        next.push({id: idRef.current++, text});
      });
      // Cap the log to the last N entries to avoid unbounded growth
      const CAP = 200;
      if (next.length > CAP) {
        return next.slice(next.length - CAP);
      }
      return next;
    });
  }, []);

  return [messages, append];
}

function chunkItems(items, size) {
  const chunks = [];
  for (let index = 0; index < items.length; index += size) {
    chunks.push(items.slice(index, index + size));
  }
  return chunks;
}

export default function WorkflowView({onBack} = {}) {
  const {backend, bootstrap, argv} = useContext(AppContext);
  const {stdout} = useStdout();
  const stdoutRows = stdout && stdout.rows ? Number(stdout.rows) : undefined;
  const stdoutCols = stdout && stdout.columns ? Number(stdout.columns) : undefined;
  const lineWidth = stdoutCols ? Math.max(40, stdoutCols - 2) : undefined;

  const HEADER_ROWS = 13;
  const FOOTER_ROWS = 3;
  const FILE_ROWS = 2;
  const COMPLETION_EXIT_DELAY_MS = 1000;

  function getFileViewportCount() {
    const rows = stdoutRows || 30;
    const bodyRows = Math.max(0, rows - HEADER_ROWS - FOOTER_ROWS);
    return Math.max(1, Math.floor(bodyRows / FILE_ROWS));
  }

  const [stage, setStage] = useState('prepare');
  const [stats, setStats] = useState(normaliseStats());
  const [status, setStatus] = useState('running');
  const [messages, appendMessages] = useMessageLog();
  const emitterRef = useRef(null);
  const stageRef = useRef('prepare');
  const statsRef = useRef(normaliseStats());
  const [fileStates, setFileStates] = useState({});
  const [fileMeta, setFileMeta] = useState({}); // { [path]: {subject?, error?} }
  const [viewMode, setViewMode] = useState('files'); // 'files' | 'messages'
  const [scroll, setScroll] = useState(0);
  const [pushState, setPushState] = useState('idle'); // 'idle' | 'pushing' | 'done' | 'error'

  const overrides = useMemo(() => {
    const out = {};
    if (argv.provider) out.provider = argv.provider;
    if (argv.model) out.model = argv.model;
    if (argv.endpoint) out.endpoint = argv.endpoint;
    if (argv['api-key-env']) out.api_key_env = argv['api-key-env'];
    return out;
  }, [argv]);

  useEffect(() => {
    const payload = {
      overrides,
      repoPath: argv.repoPath || bootstrap?.repoRoot,
      limit: argv.limit,
      maxRetries: argv['max-retries'],
      workers: argv.workers,
      debug: argv.debug,
      verbose: argv.verbose,
      oneshot: Boolean(argv.oneshot),
      singleFile: argv.file,
      autoPush: argv['no-auto-push'] ? false : true,
      compact: Boolean(argv.compact || argv.summary),
    };
    const emitter = backend.runWorkflow(payload);
    emitterRef.current = emitter;

    emitter.on('event', message => {
      const {event, payload: data} = message;
      if (event === 'progress' || event === 'tick') {
        const nextStats = normaliseStats(data?.stats || statsRef.current);
        const nextStage = data?.stage || stageRef.current;
        statsRef.current = nextStats;
        stageRef.current = nextStage;
        setStats(nextStats);
        setStage(nextStage);
        if (data?.files && typeof data.files === 'object') {
          setFileStates(data.files);
        }
      }
      if (event === 'commit-generated') {
        const file = data?.file || 'unknown file';
        const raw = String(data?.subject || data?.body || '').trim();
        const subject = raw.split(/\r?\n/)[0] || '(no subject)';
        appendMessages([
          chalk.cyan(`Generated for ${file}:`),
          chalk.green(subject),
          '',
        ]);
        setFileMeta(prev => ({
          ...prev,
          [file]: {...(prev[file] || {}), subject},
        }));
      }
      if (event === 'status') {
        const msg = String(data?.message || '').trim();
        const stageKey = String(data?.stage || '').toLowerCase();
        const file = data?.file || '';
        if (msg) {
          appendMessages([chalk.dim(msg)]);
        }
        if (file) {
          if (stageKey === 'commit-error') {
            const detail = String(data?.detail || data?.message || 'commit failed');
            setFileMeta(prev => ({
              ...prev,
              [file]: {...(prev[file] || {}), error: detail},
            }));
          }
        }
        // Track push state
        if (stageKey === 'push-start') {
          setPushState('pushing');
        } else if (stageKey === 'push-done') {
          setPushState('done');
        } else if (stageKey === 'push-error') {
          setPushState('error');
        }
      }
      if (event === 'log') {
        return;
      }
      if (event === 'prepare-error') {
        const file = data?.file || 'unknown file';
        const detail = data?.error ? chalk.dim(data.error) : null;
        appendMessages([
          chalk.yellow(`Skipped ${file}`),
          detail,
          '',
        ]);
        if (data?.file && data?.error) {
          setFileMeta(prev => ({
            ...prev,
            [data.file]: {...(prev[data.file] || {}), error: String(data.error)},
          }));
        }
      }
      if (event === 'complete') {
        setStatus('completed');
        if (data?.files && typeof data.files === 'object') {
          setFileStates(data.files);
        }
        const doneStats = normaliseStats(data?.stats || statsRef.current);
        stageRef.current = 'done';
        statsRef.current = doneStats;
        setStage('done');
      }
      if (event === 'error') {
        const messageText = data?.message || 'Workflow failed';
        setStatus('error');
        appendMessages(chalk.red(`✖ ${messageText}`));
      }
    });

      emitter.on('error', err => {
        setStatus('error');
      });

    emitter.on('done', () => {
      setStatus(prev => (prev === 'running' ? 'completed' : prev));
    });

    return () => {
      emitter.cancel?.();
    };
  }, [appendMessages, backend, bootstrap, overrides, argv]);

  useInput((input, key) => {
    const char = (input || '').toLowerCase();
    if (key.escape || (key.ctrl && char === 'c') || char === 'q') {
      if (emitterRef.current && typeof emitterRef.current.cancel === 'function') {
        emitterRef.current.cancel();
      }
      emitterRef.current = null;
      if (status !== 'running') {
        const exitCode = status === 'error' ? 1 : 0;
        process.exit(exitCode);
        return;
      }
      onBack();
    }
    // Toggle file/messages view
    if (char === 'm') {
      setViewMode(prev => (prev === 'files' ? 'messages' : 'files'));
    }
    // Scrolling controls for file list view
    if (viewMode === 'files') {
      const fileCount = Object.keys(fileStates || {}).length;
      const viewport = getFileViewportCount();
      if (key.downArrow || char === 'j') {
        setScroll(prev => Math.min(Math.max(0, fileCount - viewport), prev + 1));
      } else if (key.upArrow || char === 'k') {
        setScroll(prev => Math.max(0, prev - 1));
      } else if (key.pageDown) {
        setScroll(prev => Math.min(Math.max(0, fileCount - viewport), prev + viewport));
      } else if (key.pageUp) {
        setScroll(prev => Math.max(0, prev - viewport));
      } else if (char === 'g' && !key.shift) {
        setScroll(0);
      } else if (char === 'g' && key.shift) {
        setScroll(Math.max(0, fileCount - viewport));
      }
    }
  });

  useEffect(() => {
    if (status === 'running') {
      return undefined;
    }
    if (pushState === 'pushing') {
      return undefined;
    }
    const exitCode = status === 'error' ? 1 : 0;
    const timer = setTimeout(() => {
      if (emitterRef.current && typeof emitterRef.current.cancel === 'function') {
        emitterRef.current.cancel();
      }
      emitterRef.current = null;
      process.exit(exitCode);
    }, COMPLETION_EXIT_DELAY_MS);

    return () => clearTimeout(timer);
  }, [status, pushState]);

  const provider = bootstrap?.config?.provider || 'openai';
  const repo = ellipsize(bootstrap?.repoRoot || '', lineWidth ? lineWidth - 15 : undefined);
  const model = bootstrap?.config?.model || '';
  const endpoint = ellipsize(bootstrap?.config?.llm_endpoint || '', lineWidth ? lineWidth - 12 : undefined);
  const maxRetries = argv['max-retries'] || bootstrap?.config?.max_retries || 3;

  const viewportFiles = getFileViewportCount();
  const viewModel = useMemo(
    () =>
      buildWorkflowViewModel({
        stage,
        status,
        stats,
        fileStates,
        fileMeta,
        now: Date.now() / 1000,
        viewportCount: viewportFiles,
        pushState,
      }),
    [stage, status, stats, fileStates, fileMeta, viewportFiles, pushState],
  );

  function formatStageCounter(counter) {
    const label = chalk.bold(counter.label);
    if (counter.label === 'Commits completed') {
      return `${label}: ${chalk.green(counter.value)}`;
    }
    if (counter.label === 'Commits in progress') {
      return `${label}: ${chalk.yellow(counter.value)}`;
    }
    if (counter.label === 'Failures') {
      return `${label}: ${chalk.red(counter.value)}`;
    }
    if (counter.label === 'Push') {
      const pushValue = String(counter.value || '');
      let painter = chalk.dim;
      if (pushValue === 'Pushing') {
        painter = chalk.yellow;
      } else if (pushValue === 'Done') {
        painter = chalk.green;
      } else if (pushValue === 'Failed') {
        painter = chalk.red;
      }
      return `${label}: ${painter(pushValue)}`;
    }
    if (counter.label.startsWith('LLM')) {
      return `${label}: ${chalk.cyan(counter.value)}`;
    }
    return `${label}: ${counter.value}`;
  }

  const headerElements = [
    h(Text, {key: 'hdr-banner'}, chalk.bold.cyan(`kcmt :: provider ${provider} :: repo ${repo}`)),
    h(Text, {key: 'hdr-provider'}, `Provider: ${provider}`),
    h(Text, {key: 'hdr-model'}, `Model: ${model}`),
    h(Text, {key: 'hdr-endpoint'}, `Endpoint: ${endpoint}`),
    h(Text, {key: 'hdr-retries'}, `Max retries: ${maxRetries}`),
    h(
      Text,
      {key: 'hdr-phase'},
      `${chalk.bold('Phase')}: ${chalk.yellow(viewModel.currentPhaseLabel)} │ ${chalk.bold('Elapsed')}: ${viewModel.totalElapsedLabel}`,
    ),
    ...chunkItems(viewModel.stageCounters, 3).map((group, index) =>
      h(
        Text,
        {key: `hdr-counters-${index}`, wrap: 'truncate'},
        group.map(formatStageCounter).join(' │ '),
      ),
    ),
  ];

  if (viewModel.activeNow) {
    headerElements.push(
      h(Text, {key: 'hdr-active', wrap: 'truncate'}, chalk.cyan(viewModel.activeNow)),
    );
  }
  if (viewModel.slowAlert) {
    headerElements.push(
      h(Text, {key: 'hdr-slow', wrap: 'truncate'}, chalk.yellow(viewModel.slowAlert)),
    );
  }
  const orientationParts = [];
  if (viewModel.viewportSummary) {
    orientationParts.push(viewModel.viewportSummary);
  }
  if (viewModel.notStartedCount > 0) {
    orientationParts.push(`${viewModel.notStartedCount} not started`);
  }
  if (orientationParts.length) {
    headerElements.push(
      h(Text, {key: 'hdr-orientation', wrap: 'truncate'}, chalk.dim(orientationParts.join(' │ '))),
    );
  }
  headerElements.push(
    h(
      Text,
      {key: 'hdr-hint', dimColor: true},
      viewMode === 'files'
        ? 'j/k, PgUp/PgDn to scroll • m to toggle messages • ESC to exit'
        : 'm to toggle files • ESC to exit',
    ),
  );
  headerElements.push(h(Text, {key: 'hdr-gap'}, ''));

  function renderBar(pct, width) {
    const w = Math.max(10, Math.min(30, width || 20));
    const filled = Math.round((pct / 100) * w);
    const empty = Math.max(0, w - filled);
    return `${chalk.green('█'.repeat(filled))}${chalk.dim('░'.repeat(empty))}`;
  }

  const filesArray = viewModel.orderedFiles;
  const start = Math.max(0, Math.min(scroll, Math.max(0, filesArray.length - viewportFiles)));
  const end = Math.min(filesArray.length, start + viewportFiles);
  const visibleFiles = filesArray.slice(start, end);

  const fileElements = visibleFiles.length
    ? visibleFiles.flatMap((item, idx) => {
        const pct = item.progressPercent;
        const bar = renderBar(pct, 20);
        const pathMax = Math.max(10, (lineWidth || 80) - 40);
        const shownPath = ellipsize(item.path, pathMax);
        const lines = [];
        lines.push(
          h(
            Text,
            {key: `file-${start + idx}-row1`, wrap: 'truncate'},
            `${shownPath.padEnd(pathMax)}  ${bar} ${String(pct).padStart(3)}%`,
          ),
        );
        if (item.subject) {
          const subMax = Math.max(10, (lineWidth || 80) - 4);
          lines.push(
            h(
              Text,
              {key: `file-${start + idx}-row2`, wrap: 'truncate'},
              chalk.greenBright(ellipsize(item.subject, subMax)),
            ),
          );
        } else if (item.error) {
          const errMax = Math.max(10, (lineWidth || 80) - 4);
          lines.push(
            h(
              Text,
              {key: `file-${start + idx}-row2-err`, wrap: 'truncate'},
              chalk.red(ellipsize(item.error, errMax)),
            ),
          );
        } else {
          let statusColor = chalk.dim;
          if (item.stage === 'done') {
            statusColor = chalk.green;
          } else if (item.stage === 'failed') {
            statusColor = chalk.red;
          } else if (item.stage === 'commit') {
            statusColor = chalk.yellow;
          } else if (item.stage === 'llm_wait' || item.stage === 'prepared') {
            statusColor = chalk.cyan;
          } else if (item.stage === 'diff') {
            statusColor = chalk.blue;
          }
          lines.push(
            h(
              Text,
              {key: `file-${start + idx}-status`, wrap: 'truncate'},
              statusColor(item.statusText),
            ),
          );
        }
        return lines;
      })
    : [h(Text, {key: 'files-empty', dimColor: true}, 'Waiting for workflow activity…')];

  // Messages view (capped)
  const messageElements = messages.map(entry =>
    h(Text, {key: `msg-${entry.id}`}, entry.text)
  );
  if (!messageElements.length) {
    messageElements.push(h(Text, {key: 'msg-placeholder', dimColor: true}, 'Waiting for workflow activity…'));
  }

  function buildOverallProgressParts() {
    const total = Number(viewModel.stageCounters[0]?.value || 0);
    if (total === 0) {
      return '';
    }
    const progressPct = Math.min(100, Math.max(0, viewModel.overallProgressPct));
    const pctStr = String(Math.round(progressPct)).padStart(3);

    const statusLabel =
      status === 'completed' && pushState !== 'pushing'
        ? 'Complete'
        : viewModel.currentPhaseLabel;
    const rightPlain = `${pctStr}% ${statusLabel}`;
    const right = `${pctStr}% ${chalk.dim(statusLabel)}`;
    const barWidth = Math.max(10, (lineWidth || 80) - rightPlain.length - 1);
    const filled = Math.round((progressPct / 100) * barWidth);
    const empty = Math.max(0, barWidth - filled);
    const bar = `${chalk.green('█'.repeat(filled))}${chalk.dim('░'.repeat(empty))}`;

    return {bar, right};
  }

  const footerElements = [];
  const overall = buildOverallProgressParts();
  if (overall) {
    footerElements.push(
      h(
        Box,
        {key: 'overall-progress', width: '100%'},
        h(Box, {flexGrow: 1, flexShrink: 1}, h(Text, {wrap: 'truncate'}, overall.bar)),
        h(Box, {flexShrink: 0}, h(Text, {wrap: 'truncate'}, ` ${overall.right}`)),
      ),
    );
  }
  if (viewModel.footerStatus) {
    footerElements.push(
      h(Text, {key: 'footer-status', wrap: 'truncate'}, chalk.dim(viewModel.footerStatus)),
    );
  }

  const rootProps = {flexDirection: 'column', paddingX: 0, paddingY: 0};
  if (stdoutRows) {
    rootProps.height = stdoutRows;
  } else {
    rootProps.flexGrow = 1;
    rootProps.alignItems = 'stretch';
  }

  return h(
    Box,
    rootProps,
    // Header
    h(
      Box,
      {flexDirection: 'column', flexGrow: 0, gap: 0},
      ...headerElements,
    ),
    // Body (files or messages)
    h(
      Box,
      {flexDirection: 'column', flexGrow: 1, gap: 0},
      ...(viewMode === 'files' ? fileElements : messageElements),
    ),
    // Footer
    h(
      Box,
      {flexDirection: 'column', flexGrow: 0, gap: 0, width: '100%'},
      ...footerElements,
    ),
  );
}
