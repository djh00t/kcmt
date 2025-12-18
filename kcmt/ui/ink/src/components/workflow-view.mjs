import React, {useCallback, useContext, useEffect, useMemo, useRef, useState} from 'react';
import {Box, Text, useInput, useStdout} from 'ink';
import Spinner from 'ink-spinner';
import chalk from 'chalk';
import {AppContext} from '../app.mjs';
const h = React.createElement;

const STAGE_ORDER = ['prepare', 'commit', 'done'];

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
  };
}

function buildProgressLine(stage, stats, maxWidth) {
  const snapshot = normaliseStats(stats);
  const total = Math.max(0, snapshot.total_files);
  const diffs = Math.max(0, Math.min(snapshot.diffs_built, total));
  const requests = Math.max(0, snapshot.requests);
  const responses = Math.max(0, snapshot.responses);
  const processed = Math.max(0, Math.min(snapshot.processed, total));
  const prepared = Math.max(0, Math.min(snapshot.prepared, total));
  const success = Math.max(0, snapshot.successes);
  const failures = Math.max(0, snapshot.failures);
  const rate = Number.isFinite(snapshot.rate) ? snapshot.rate : 0;

  const stageStyles = {
    prepare: {icon: '🧠', color: chalk.cyan},
    commit: {icon: '🚀', color: chalk.green},
    done: {icon: '🏁', color: chalk.yellow},
  };
  const {icon, color} = stageStyles[stage] || stageStyles.prepare;
  const stageLabel = (stage || 'progress').toUpperCase().padEnd(7);

  const diffStr = String(diffs).padStart(3);
  const processedStr = String(processed).padStart(3);
  const totalStr = String(total).padStart(3);
  const preparedStr = String(prepared).padStart(3);
  const reqStr = String(requests).padStart(3);
  const resStr = String(responses).padStart(3);
  const successStr = String(success).padStart(3);
  const failureStr = String(failures).padStart(3);
  const rateStr = rate.toFixed(2).padStart(5);

  const line = (
    `${chalk.bold(`${icon} kcmt`)} ` +
    `${color(stageLabel)} │ ` +
    `${chalk.dim(`Δ ${diffStr}`)}/${totalStr} │ ` +
    `${chalk.cyan(`req ${reqStr}`)}/${chalk.cyan(`${resStr} res`)} │ ` +
    `${chalk.green(`${preparedStr}/${totalStr} ready`)} │ ` +
    `${chalk.green(`✓ ${successStr}`)} │ ` +
    `${chalk.red(`✗ ${failureStr}`)} │ ` +
    `${chalk.dim(`${rateStr} commits/s`)}`
  );

  if (!maxWidth) {
    return line;
  }
  return ellipsize(line, maxWidth);
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

export default function WorkflowView({onBack}) {
  const {backend, bootstrap, argv} = useContext(AppContext);
  const {stdout} = useStdout();
  const stdoutRows = stdout && stdout.rows ? Number(stdout.rows) : undefined;
  const stdoutCols = stdout && stdout.columns ? Number(stdout.columns) : undefined;
  const lineWidth = stdoutCols ? Math.max(40, stdoutCols - 2) : undefined;

  const [stage, setStage] = useState('prepare');
  const [stats, setStats] = useState(normaliseStats());
  const [status, setStatus] = useState('running');
  const [summary, setSummary] = useState(null);
  const [errors, setErrors] = useState([]);
  const [progressSnapshots, setProgressSnapshots] = useState({});
  const [currentProgressLine, setCurrentProgressLine] = useState('');
  const [commitSubjects, setCommitSubjects] = useState([]);
  const [metricsSummary, setMetricsSummary] = useState('');
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
        const line = buildProgressLine(nextStage, nextStats, lineWidth);
        setCurrentProgressLine(line);
        setProgressSnapshots(prev => ({...prev, [nextStage]: line}));
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
        setErrors(prev => [...prev, data?.error || `Skipped ${file}`]);
        if (data?.file && data?.error) {
          setFileMeta(prev => ({
            ...prev,
            [data.file]: {...(prev[data.file] || {}), error: String(data.error)},
          }));
        }
      }
      if (event === 'complete') {
        setSummary(data);
        setStatus('completed');
        if (Array.isArray(data?.commit_subjects)) {
          setCommitSubjects(data.commit_subjects.map(item => String(item))); // already serialised
        }
        if (data?.metrics_summary) {
          setMetricsSummary(String(data.metrics_summary));
        }
        const doneStats = normaliseStats(data?.stats || statsRef.current);
        const doneLine = buildProgressLine('done', doneStats, lineWidth);
        setProgressSnapshots(prev => ({...prev, done: doneLine}));
        setCurrentProgressLine('');
        stageRef.current = 'done';
        statsRef.current = doneStats;
        setStage('done');
      }
      if (event === 'error') {
        const messageText = data?.message || 'Workflow failed';
        setStatus('error');
        setErrors(prev => [...prev, messageText]);
        appendMessages(chalk.red(`✖ ${messageText}`));
      }
    });

      emitter.on('error', err => {
        setStatus('error');
        setErrors(prev => [...prev, err.message]);
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
      const viewport = Math.max(5, (stdoutRows || 30) - 12);
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
    const exitCode = status === 'error' ? 1 : 0;
    const timer = setTimeout(() => {
      if (emitterRef.current && typeof emitterRef.current.cancel === 'function') {
        emitterRef.current.cancel();
      }
      emitterRef.current = null;
      process.exit(exitCode);
    }, 750);

    return () => clearTimeout(timer);
  }, [status]);

  const provider = bootstrap?.config?.provider || 'openai';
  const repo = ellipsize(bootstrap?.repoRoot || '', lineWidth ? lineWidth - 15 : undefined);
  const model = bootstrap?.config?.model || '';
  const endpoint = ellipsize(bootstrap?.config?.llm_endpoint || '', lineWidth ? lineWidth - 12 : undefined);
  const maxRetries = argv['max-retries'] || bootstrap?.config?.max_retries || 3;

  const headerElements = [
    h(Text, {key: 'hdr-banner'}, chalk.bold.cyan(`kcmt :: provider ${provider} :: repo ${repo}`)),
    h(Text, {key: 'hdr-provider'}, `Provider: ${provider}`),
    h(Text, {key: 'hdr-model'}, `Model: ${model}`),
    h(Text, {key: 'hdr-endpoint'}, `Endpoint: ${endpoint}`),
    h(Text, {key: 'hdr-retries'}, `Max retries: ${maxRetries}`),
    h(Text, {key: 'hdr-hint', dimColor: true}, viewMode === 'files' ? 'j/k, PgUp/PgDn to scroll • m to toggle messages • ESC to exit' : 'm to toggle files • ESC to exit'),
    h(Text, {key: 'hdr-gap'}, ''),
  ];
  // Build file list view
  const filesArray = useMemo(() => {
    const map = fileStates || {};
    const paths = Object.keys(map);
    paths.sort((a, b) => a.localeCompare(b));
    return paths.map(p => ({
      path: p,
      state: map[p] || {},
    }));
  }, [fileStates]);

  function computeProgress(entry) {
    const s = entry?.state || {};
    let pct = 0;
    if (s.diff === 'yes') pct = Math.max(pct, 20);
    if (s.req === 'sent') pct = Math.max(pct, 40);
    if (s.res === 'ok') pct = Math.max(pct, 60);
    if (s.batch === 'completed') pct = Math.max(pct, 80);
    if (s.commit === 'running') pct = Math.max(pct, 85);
    if (s.commit === 'ok') pct = 100;
    if (s.commit === 'err') pct = 100;
    return pct;
  }

  function computeStatus(entry) {
    const s = entry?.state || {};
    const meta = fileMeta[entry.path] || {};
    
    // Detailed workflow states - show each step clearly
    if (meta.error) return chalk.red('error');
    if (s.commit === 'err') return chalk.red('commit failed');
    if (s.commit === 'ok') return chalk.green('committed');
    if (s.commit === 'running') return chalk.yellow('committing');
    if (meta.subject) return chalk.cyan('ready to commit');
    if (s.res === 'ok' && s.req === 'sent') return chalk.cyan('generating message');
    if (s.res === 'ok') return chalk.cyan('response received');
    if (s.req === 'sent') return chalk.magenta('awaiting response');
    if (s.diff === 'yes') return chalk.blue('diff collected');
    return chalk.dim('pending');
  }

  function renderBar(pct, width) {
    const w = Math.max(10, Math.min(30, width || 20));
    const filled = Math.round((pct / 100) * w);
    const empty = Math.max(0, w - filled);
    return `${chalk.green('█'.repeat(filled))}${chalk.dim('░'.repeat(empty))}`;
  }

  const viewportRows = Math.max(5, (stdoutRows || 30) - 12);
  const start = Math.max(0, Math.min(scroll, Math.max(0, filesArray.length - viewportRows)));
  const end = Math.min(filesArray.length, start + viewportRows);
  const visibleFiles = filesArray.slice(start, end);

  const fileElements = visibleFiles.length
    ? visibleFiles.flatMap((item, idx) => {
        const pct = computeProgress(item);
        const bar = renderBar(pct, 20);
        const statusLabel = computeStatus(item);
        const pathMax = Math.max(10, (lineWidth || 80) - 40);
        const shownPath = ellipsize(item.path, pathMax);
        const meta = fileMeta[item.path] || {};
        const lines = [];
        lines.push(
          h(
            Text,
            {key: `file-${start + idx}-row1`},
            `${shownPath.padEnd(pathMax)}  ${bar} ${String(pct).padStart(3)}%`,
          ),
        );
        if (meta.subject) {
          const subMax = Math.max(10, (lineWidth || 80) - 4);
          lines.push(
            h(
              Text,
              {key: `file-${start + idx}-row2`},
              chalk.green(ellipsize(meta.subject, subMax)),
            ),
          );
        } else if (meta.error) {
          const errMax = Math.max(10, (lineWidth || 80) - 4);
          lines.push(
            h(
              Text,
              {key: `file-${start + idx}-row2-err`},
              chalk.red(ellipsize(meta.error, errMax)),
            ),
          );
        } else {
          lines.push(
            h(
              Text,
              {key: `file-${start + idx}-status`},
              chalk.dim(statusLabel),
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

  function buildAggregateLine() {
    const snapshot = normaliseStats(statsRef.current || stats);
    const totalFromFiles = Object.keys(fileStates || {}).length;
    const total = Math.max(snapshot.total_files || 0, totalFromFiles);

    let diffed = 0;
    let req = 0;
    let res = 0;
    let batchValidating = 0;
    let batchInProgress = 0;
    let batchFinalizing = 0;
    let batchCompleted = 0;
    let committing = 0;
    let committed = 0;
    let commitErr = 0;
    let metaErr = 0;
    const meta = fileMeta || {};
    const fmap = fileStates || {};
    for (const p of Object.keys(fmap)) {
      const s = fmap[p] || {};
      if (s.diff === 'yes') diffed += 1;
      if (s.req === 'sent') req += 1;
      if (s.res === 'ok') res += 1;
      if (s.batch === 'validating' || s.batch === 'queued') batchValidating += 1;
      if (s.batch === 'in_progress' || s.batch === 'running' || s.batch === 'in-progress') batchInProgress += 1;
      if (s.batch === 'finalizing') batchFinalizing += 1;
      if (s.batch === 'completed') batchCompleted += 1;
      if (s.commit === 'running') committing += 1;
      if (s.commit === 'ok') committed += 1;
      if (s.commit === 'err') commitErr += 1;
      if (meta[p]?.error) metaErr += 1;
    }
    const errors = commitErr + metaErr;
    const hasBatchActivity = batchValidating + batchInProgress + batchFinalizing + batchCompleted > 0;

    // Simplified format: files, states, results
    const parts = [
      `${chalk.bold('files')} ${String(total).padStart(3)}`,
      `${chalk.dim('Δ')} ${String(diffed).padStart(3)}`,
      `${chalk.cyan('req')} ${String(req).padStart(3)}`,
      `${chalk.cyan('res')} ${String(res).padStart(3)}`,
    ];

    // Only show batch stats if there's actual batch activity
    if (hasBatchActivity) {
      parts.push(
        `${chalk.magenta('batch')} ${String(batchValidating + batchInProgress + batchFinalizing).padStart(2)}/${String(batchCompleted).padStart(2)}`
      );
    }

    parts.push(
      `${chalk.yellow('committing')} ${String(committing).padStart(3)}`,
      `${chalk.green('✓')} ${String(committed).padStart(3)}`,
      `${chalk.red('✗')} ${String(errors).padStart(3)}`
    );

    const line = parts.join(' │ ');
    return lineWidth ? ellipsize(line, lineWidth) : line;
  }

  function buildOverallProgressBar() {
    const snapshot = normaliseStats(statsRef.current || stats);
    const total = Math.max(snapshot.total_files || 0, Object.keys(fileStates || {}).length);
    
    if (total === 0) {
      return '';
    }

    // Calculate overall progress: diff → prepare → commit → push
    // Weight each stage: diff 20%, prepare 40%, commit 35%, push 5%
    const committed = snapshot.successes || 0;
    const prepared = snapshot.prepared || 0;
    const diffed = snapshot.diffs_built || 0;
    
    let progressPct = 0;
    progressPct += (diffed / total) * 20;      // Diff stage: 20%
    progressPct += (prepared / total) * 40;    // Prepare stage: 40%
    progressPct += (committed / total) * 35;   // Commit stage: 35%
    
    // Push adds final 5%
    if (pushState === 'pushing') {
      progressPct += 2.5; // Half of push
    } else if (pushState === 'done') {
      progressPct += 5;   // Full push
    } else if (status === 'completed' && committed === total) {
      progressPct += 5;   // Assume push done if all committed and completed
    }
    
    progressPct = Math.min(100, Math.max(0, progressPct));
    
    const barWidth = Math.max(30, Math.min(50, (lineWidth || 80) - 25));
    const filled = Math.round((progressPct / 100) * barWidth);
    const empty = Math.max(0, barWidth - filled);
    const bar = `${chalk.green('█'.repeat(filled))}${chalk.dim('░'.repeat(empty))}`;
    
    let statusLabel = 'In progress';
    if (pushState === 'pushing') {
      statusLabel = 'Pushing';
    } else if (pushState === 'done' || (status === 'completed' && progressPct >= 100)) {
      statusLabel = 'Complete';
    } else if (committed > 0 && committed === total) {
      statusLabel = 'Committing complete';
    }
    
    return `${bar} ${String(Math.round(progressPct)).padStart(3)}% ${chalk.dim(statusLabel)}`;
  }

  const footerElements = [];
  if (status === 'running') {
    const overallProgress = buildOverallProgressBar();
    if (overallProgress) {
      footerElements.push(h(Text, {key: 'overall-progress'}, overallProgress));
    }
    footerElements.push(h(Text, {key: 'aggregate-live'}, buildAggregateLine()));
  }

  if (status !== 'running') {
    const overallProgress = buildOverallProgressBar();
    if (overallProgress) {
      footerElements.push(h(Text, {key: 'overall-progress-done'}, overallProgress));
    }
    footerElements.push(h(Text, {key: 'aggregate-done'}, buildAggregateLine()));
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
      {flexDirection: 'column', flexGrow: 0, gap: 0},
      ...footerElements,
    ),
  );
}
