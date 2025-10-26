import React, {useCallback, useContext, useEffect, useMemo, useRef, useState} from 'react';
import {Box, Text, useInput, useStdout} from 'ink';
import Spinner from 'ink-spinner';
import chalk from 'chalk';
import {AppContext} from '../app.mjs';
const h = React.createElement;

const STAGE_ORDER = ['prepare', 'commit', 'done'];

function normaliseStats(stats = {}) {
  if (!stats) {
    return {
      total_files: 0,
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
  };
}

function buildProgressLine(stage, stats) {
  const snapshot = normaliseStats(stats);
  const total = Math.max(0, snapshot.total_files);
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

  const processedStr = String(processed).padStart(3);
  const totalStr = String(total).padStart(3);
  const preparedStr = String(prepared).padStart(3);
  const successStr = String(success).padStart(3);
  const failureStr = String(failures).padStart(3);
  const rateStr = rate.toFixed(2).padStart(5);

  return (
    `${chalk.bold(`${icon} kcmt`)} ` +
    `${color(stageLabel)} │ ` +
    `${chalk.green(processedStr)}/${totalStr} files │ ` +
    `${chalk.cyan(preparedStr)}/${totalStr} ready │ ` +
    `${chalk.green(`✓ ${successStr}`)} │ ` +
    `${chalk.red(`✗ ${failureStr}`)} │ ` +
    `${chalk.dim(`${rateStr} commits/s`)}`
  );
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
      return next;
    });
  }, []);

  return [messages, append];
}

export default function WorkflowView({onBack}) {
  const {backend, bootstrap, argv} = useContext(AppContext);
  const {stdout} = useStdout();
  const stdoutRows = stdout && stdout.rows ? Number(stdout.rows) : undefined;

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
      autoPush: argv['auto-push'] ? true : argv['no-auto-push'] ? false : undefined,
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
        const line = buildProgressLine(nextStage, nextStats);
        setCurrentProgressLine(line);
        setProgressSnapshots(prev => ({...prev, [nextStage]: line}));
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
        const doneLine = buildProgressLine('done', doneStats);
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
    if (key.escape || (key.ctrl && input === 'c')) {
      emitterRef.current?.cancel?.();
      onBack();
    }
    if (input === 'q' && status !== 'running') {
      onBack();
    }
  });

  const provider = bootstrap?.config?.provider || 'openai';
  const repo = bootstrap?.repoRoot || '';
  const model = bootstrap?.config?.model || '';
  const endpoint = bootstrap?.config?.llm_endpoint || '';
  const maxRetries = argv['max-retries'] || bootstrap?.config?.max_retries || 3;

  const headerElements = [
    h(Text, {key: 'hdr-banner'}, chalk.bold.cyan(`kcmt :: provider ${provider} :: repo ${repo}`)),
    h(Text, {key: 'hdr-provider'}, `Provider: ${provider}`),
    h(Text, {key: 'hdr-model'}, `Model: ${model}`),
    h(Text, {key: 'hdr-endpoint'}, `Endpoint: ${endpoint}`),
    h(Text, {key: 'hdr-retries'}, `Max retries: ${maxRetries}`),
    h(Text, {key: 'hdr-gap'}, ''),
  ];

  const messageElements = messages.map(entry =>
    h(Text, {key: `msg-${entry.id}`}, entry.text)
  );

  if (!messageElements.length) {
    messageElements.push(h(Text, {key: 'msg-placeholder', dimColor: true}, 'Waiting for workflow activity…'));
  }

  const footerElements = [];
  if (status === 'running') {
    if (currentProgressLine) {
      footerElements.push(h(Text, {key: 'progress-live'}, currentProgressLine));
    }
    footerElements.push(h(Text, {key: 'footer-running', dimColor: true}, h(Spinner, {type: 'runner'}), ' Press ESC to abort.'));
  }

  if (summary) {
    const summaryLines = [];
    const resultSummary = summary?.result?.summary;
    if (resultSummary) {
      summaryLines.push(h(Text, {key: 'summary-head', color: 'greenBright'}, resultSummary));
    }
    if (commitSubjects.length) {
      commitSubjects.forEach((subject, idx) => {
        summaryLines.push(h(Text, {key: `summary-subject-${idx}`}, chalk.green(subject)));
      });
    }
    const metricsLine = metricsSummary ? chalk.dim(`metrics: ${metricsSummary}`) : null;
    if (metricsLine) {
      summaryLines.push(h(Text, {key: 'summary-metrics'}, metricsLine));
    }
    footerElements.push(
      h(Text, {key: 'summary-title'}, ''),
      h(Text, {key: 'summary-label'}, chalk.bold('Workflow Summary')),
      ...summaryLines,
      h(Text, {key: 'summary-gap'}, ''),
    );
  }

  if (errors.length) {
    footerElements.push(
      h(Text, {key: 'errors-title', color: 'redBright'}, 'Issues'),
      ...errors.slice(-5).map((err, idx) => h(Text, {key: `error-${idx}`, dimColor: true}, `• ${err}`)),
    );
  }

  if (status !== 'running') {
    const stageBlock = [];
    STAGE_ORDER.forEach(stageKey => {
      const line = progressSnapshots[stageKey];
      if (line) {
        stageBlock.push(h(Text, {key: `progress-${stageKey}`}, line));
      }
    });
    if (stageBlock.length) {
      footerElements.unshift(h(Text, {key: 'progress-gap-top'}, ''));
      footerElements.unshift(...stageBlock);
    }
    footerElements.push(h(Text, {key: 'footer-complete', dimColor: true}, 'Press q to return to the main menu.'));
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
    h(
      Box,
      {flexDirection: 'column', flexGrow: 0, gap: 0},
      ...headerElements,
      ...messageElements,
    ),
    h(Box, {flexGrow: 1}),
    h(
      Box,
      {flexDirection: 'column', flexGrow: 0, gap: 0},
      ...footerElements,
    ),
  );
}
