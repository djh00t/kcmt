import React, {useContext, useEffect, useMemo, useRef, useState} from 'react';
import {Box, Text, useInput} from 'ink';
import Spinner from 'ink-spinner';
import chalk from 'chalk';
import gradient from 'gradient-string';
import {AppContext} from '../app.mjs';
const h = React.createElement;

const titleGradient = gradient(['#f6d365', '#fda085']);

function Leaderboard({title, rows}) {
  if (!rows?.length) {
    return null;
  }
  const headers = ['Provider', 'Model', 'Latency', 'Cost', 'Quality', 'Success'];
  return h(
    Box,
    {flexDirection: 'column', marginTop: 1},
    h(Text, {color: 'whiteBright'}, title),
    h(Text, {dimColor: true}, headers.join('  │  ')),
    ...rows.slice(0, 8).map((row, idx) => {
      const line = `${chalk.cyan(row.provider.padEnd(8))} │ ${chalk.green(
        row.model.padEnd(24),
      )} │ ${row.avg_latency_ms.toFixed(1)} ms │ $${row.avg_cost_usd.toFixed(4)} │ ${row.quality.toFixed(1)} │ ${(
        row.success_rate * 100
      ).toFixed(0)}%`;
      return h(Text, {key: `${row.provider}-${row.model}-${idx}`}, line);
    }),
  );
}

export default function BenchmarkView({onBack}) {
  const {backend, argv} = useContext(AppContext);
  const [status, setStatus] = useState('running');
  const [progress, setProgress] = useState({label: 'Preparing providers…'});
  const [payload, setPayload] = useState(null);
  const emitterRef = useRef(null);

  const options = useMemo(() => {
    const base = {
      providers: argv.provider ? [argv.provider] : undefined,
      limit: argv['benchmark-limit'],
      timeout: argv['benchmark-timeout'],
      debug: argv.debug,
      includeRaw: Boolean(argv['benchmark-json']),
    };
    if (argv.model) {
      base.onlyModels = [argv.model];
    }
    return base;
  }, [argv]);

  useEffect(() => {
    const emitter = backend.runBenchmark(options);
    emitterRef.current = emitter;

    emitter.on('event', message => {
      const {event, payload: data} = message;
      if (event === 'progress') {
        setProgress(data);
      }
      if (event === 'complete') {
        setPayload(data);
        setStatus('done');
      }
      if (event === 'error') {
        setStatus('error');
      }
    });

    emitter.on('error', err => {
      setStatus('error');
      setProgress({label: err.message});
    });

    return () => {
      emitter.cancel?.();
    };
  }, [backend, options]);

  useInput((input, key) => {
    if (key.escape || input === 'q') {
      emitterRef.current?.cancel?.();
      onBack();
    }
  });

  const statusLine =
    status === 'running'
      ? h(React.Fragment, null, h(Spinner, {type: 'dots'}), ' ', progress.label || 'Crunching diffs across providers…')
      : status === 'error'
        ? `⚠️ ${progress.label || 'Benchmark failed'}`
        : 'Benchmark complete';

  return h(
    Box,
    {flexDirection: 'column', padding: 1, gap: 1, borderStyle: 'round', borderColor: 'yellow'},
    h(Text, null, titleGradient('🧪 kcmt benchmark lab')),
    h(Text, {dimColor: true}, statusLine),
    payload
      ? h(
          React.Fragment,
          null,
          h(Leaderboard, {title: '🏆 Overall', rows: payload.overall}),
          h(Leaderboard, {title: '⚡ Fastest', rows: payload.fastest}),
          h(Leaderboard, {title: '💰 Cheapest', rows: payload.cheapest}),
          h(Leaderboard, {title: '🎯 Best Quality', rows: payload.best_quality}),
        )
      : null,
    h(Text, {dimColor: true}, 'Press q to return.'),
  );
}
