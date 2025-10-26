import React, {useContext, useEffect, useMemo, useRef, useState} from 'react';
import {Box, Text, useInput} from 'ink';
import Spinner from 'ink-spinner';
import chalk from 'chalk';
import gradient from 'gradient-string';
import {AppContext} from '../app.mjs';
const h = React.createElement;

const headerGradient = gradient(['#00c6ff', '#0072ff']);

function ProgressBar({label, stats}) {
  const total = Math.max(1, stats?.total_files || stats?.total || 1);
  const processed = Math.min(total, stats?.processed || stats?.done || 0);
  const percent = Math.min(1, Math.max(0, processed / total));
  const width = 32;
  const filled = Math.round(width * percent);
  const bar = `${'█'.repeat(filled)}${'░'.repeat(Math.max(0, width - filled))}`;
  return h(
    Box,
    {flexDirection: 'column'},
    h(Text, null, label),
    h(Text, null, `${chalk.cyan(bar)} ${Math.round(percent * 100)}% (${processed}/${total})`),
  );
}

function Timeline({events}) {
  if (!events.length) {
    return h(Text, {dimColor: true}, 'Waiting for the first AI-prepared commit…');
  }
  return h(
    Box,
    {flexDirection: 'column', gap: 0},
    ...events.slice(-6).map((event, idx) =>
      h(
        Box,
        {key: `${event.type}-${idx}`, flexDirection: 'column'},
        h(
          Text,
          {
            color:
              event.type === 'commit' ? 'greenBright' : event.type === 'warning' ? 'yellow' : 'cyan',
          },
          `${event.icon} ${event.title}`,
        ),
        event.detail ? h(Text, {dimColor: true}, event.detail) : null,
      ),
    ),
  );
}

export default function WorkflowView({onBack}) {
  const {backend, bootstrap, argv} = useContext(AppContext);
  const [stage, setStage] = useState('prepare');
  const [stats, setStats] = useState({total_files: 0, processed: 0});
  const [events, setEvents] = useState([]);
  const [errors, setErrors] = useState([]);
  const [status, setStatus] = useState('running');
  const [summary, setSummary] = useState(null);
  const emitterRef = useRef(null);

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
        if (data?.stats) {
          setStats(data.stats);
        } else {
          setStats(prev => ({...prev, ...data}));
        }
        if (data?.stage) {
          setStage(data.stage);
        }
      }
      if (event === 'commit-generated') {
        setEvents(prev => [
          ...prev,
          {
            type: 'commit',
            icon: '🚀',
            title: `Commit ready for ${data?.file}`,
            detail: data?.subject,
          },
        ]);
      }
      if (event === 'log') {
        setEvents(prev => [
          ...prev,
          {
            type: 'log',
            icon: '📝',
            title: data?.message || 'Workflow update',
          },
        ]);
      }
      if (event === 'prepare-error') {
        setEvents(prev => [
          ...prev,
          {
            type: 'warning',
            icon: '⚠️',
            title: `Skipped ${data?.file}`,
            detail: data?.error,
          },
        ]);
        setErrors(prev => [...prev, data?.error || 'Unknown error']);
      }
      if (event === 'complete') {
        setSummary(data);
        setStatus('completed');
        if (data?.stats) {
          setStats(data.stats);
        }
      }
      if (event === 'error') {
        setStatus('error');
        setErrors(prev => [...prev, data?.message || 'Workflow failed']);
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
  }, [backend, bootstrap, overrides, argv]);

  useInput((input, key) => {
    if (key.escape || (key.ctrl && input === 'c')) {
      emitterRef.current?.cancel?.();
      onBack();
    }
    if (input === 'q' && status !== 'running') {
      onBack();
    }
  });

  return h(
    Box,
    {flexDirection: 'column', padding: 1, gap: 1, borderStyle: 'round', borderColor: 'blueBright'},
    h(Text, null, headerGradient('🧠 kcmt workflow')),
    h(ProgressBar, {label: `Stage: ${stage.toUpperCase()}`, stats}),
    h(
      Box,
      {flexDirection: 'column', gap: 1},
      h(Text, {color: 'whiteBright'}, 'Live queue'),
      h(Timeline, {events}),
    ),
    status === 'running'
      ? h(Text, {dimColor: true}, h(Spinner, {type: 'runner'}), ' Press ESC to abort.')
      : null,
    summary
      ? h(
          Box,
          {flexDirection: 'column'},
          h(Text, {color: 'greenBright'}, summary?.result?.summary || 'Workflow complete'),
          Array.isArray(summary?.result?.file_commits)
            ? summary.result.file_commits.flatMap((item, idx) => {
                const file = item.file_path || item.filePath || '';
                const raw = String(item.message || item.subject || item.commit_message || '');
                const subject = raw.split(/\r?\n/)[0];
                return [
                  h(Text, {key: `commit-${idx}`, dimColor: true}, `${file}: ${subject}`),
                  h(Text, {key: `sp-${idx}`}, ''),
                ];
              })
            : null,
        )
      : null,
    errors.length
      ? h(
          Box,
          {flexDirection: 'column'},
          h(Text, {color: 'redBright'}, 'Issues'),
          ...errors.slice(-5).map((err, idx) => h(Text, {key: idx, dimColor: true}, `• ${err}`)),
        )
      : null,
    status !== 'running' ? h(Text, {dimColor: true}, 'Press q to return to the main menu.') : null,
  );
}
