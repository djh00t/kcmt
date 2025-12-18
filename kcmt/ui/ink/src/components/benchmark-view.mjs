import React, {useContext, useEffect, useMemo, useRef, useState} from 'react';
import {Box, Text, useInput} from 'ink';
import Spinner from 'ink-spinner';
import chalk from 'chalk';
import gradient from 'gradient-string';
import {AppContext} from '../app.mjs';
const h = React.createElement;

const titleGradient = gradient(['#f6d365', '#fda085']);
const PROVIDER_LABELS = {
  openai: 'OpenAI',
  anthropic: 'Anthropic',
  xai: 'X.AI',
  github: 'GitHub Models',
};

const SCORE_COLUMNS = [
  {key: 'avg_latency_ms', label: 'Latency (ms)', width: 14, align: 'end', format: value => (value != null ? value.toFixed(1) : '—')},
  {key: 'avg_cost_usd', label: 'Cost ($)', width: 11, align: 'end', format: value => (value != null ? `$${value.toFixed(4)}` : '—')},
  {key: 'quality', label: 'Quality', width: 10, align: 'end', format: value => (value != null ? value.toFixed(1) : '—')},
  {key: 'success_rate', label: 'Success %', width: 11, align: 'end', format: value => (value != null ? `${(value * 100).toFixed(0)}%` : '—')},
  {key: 'runs', label: 'Runs', width: 6, align: 'end', format: value => (value != null ? String(value) : '—')},
];

const CATEGORY_COLUMNS = [
  {key: 'overall', label: 'Overall', width: 9},
  {key: 'fastest', label: 'Fastest', width: 9},
  {key: 'cheapest', label: 'Cheapest', width: 10},
  {key: 'best_quality', label: 'Best', width: 7},
];

const MODEL_COL_WIDTH = 28;

function pad(value, width, align = 'start') {
  const text = String(value ?? '');
  if (text.length >= width) {
    if (width <= 1) {
      return text.slice(0, 1);
    }
    return text.slice(0, width - 1) + '…';
  }
  const padding = ' '.repeat(width - text.length);
  return align === 'end' ? padding + text : text + padding;
}

function buildRegistry(payload) {
  const registry = new Map();

  function ensure(provider, model) {
    const key = `${provider}::${model}`;
    if (!registry.has(key)) {
      registry.set(key, {
        provider,
        model,
        avg_latency_ms: undefined,
        avg_cost_usd: undefined,
        quality: undefined,
        success_rate: undefined,
        runs: undefined,
        flags: {overall: null, fastest: null, cheapest: null, best_quality: null},
      });
    }
    return registry.get(key);
  }

  const categories = ['overall', 'fastest', 'cheapest', 'best_quality'];
  for (const category of categories) {
    const rows = Array.isArray(payload?.[category]) ? payload[category] : [];
    rows.forEach((entry, idx) => {
      const item = ensure(entry.provider, entry.model);
      item.avg_latency_ms = entry.avg_latency_ms ?? item.avg_latency_ms;
      item.avg_cost_usd = entry.avg_cost_usd ?? item.avg_cost_usd;
      item.quality = entry.quality ?? item.quality;
      item.success_rate = entry.success_rate ?? item.success_rate;
      item.runs = entry.runs ?? item.runs;
      item.flags[category] = idx + 1;
    });
  }

  const rawRows = Array.isArray(payload?.raw) ? payload.raw : [];
  rawRows.forEach(entry => {
    const item = ensure(entry.provider, entry.model);
    item.avg_latency_ms = entry.avg_latency_ms ?? item.avg_latency_ms;
    item.avg_cost_usd = entry.avg_cost_usd ?? item.avg_cost_usd;
    item.quality = entry.quality ?? item.quality;
    item.success_rate = entry.success_rate ?? item.success_rate;
    item.runs = entry.runs ?? item.runs;
  });

  return registry;
}

function renderProviderSections(payload) {
  const registry = buildRegistry(payload);
  const exclusions = Array.isArray(payload?.exclusions) ? payload.exclusions : [];
  if (!registry.size) {
    const messages = [];
    messages.push(h(Text, {key: 'bench-empty', dimColor: true}, 'No benchmark results yet.'));
    if (exclusions.length) {
      messages.push(h(Text, {key: 'bench-exc-title', color: 'yellow'}, 'Provider exclusions:'));
      exclusions.slice(0, 10).forEach((item, idx) => {
        const provider = item?.provider || 'unknown';
        const model = item?.model || '*';
        const reason = item?.reason || item?.message || 'unavailable';
        messages.push(
          h(
            Text,
            {key: `bench-exc-${idx}`, dimColor: true},
            `${provider} / ${model}: ${reason}`,
          ),
        );
      });
    }
    return messages;
  }

  const providers = new Map();
  for (const entry of registry.values()) {
    if (!providers.has(entry.provider)) {
      providers.set(entry.provider, []);
    }
    providers.get(entry.provider).push(entry);
  }

  const sections = [];
  const header = [
    pad('Model', MODEL_COL_WIDTH),
    ...SCORE_COLUMNS.map(col => pad(col.label, col.width)),
    ...CATEGORY_COLUMNS.map(col => pad(col.label, col.width)),
  ].join(' ');

  const divider = header.replace(/\S/g, '-');

  const sortedProviders = Array.from(providers.keys()).sort((a, b) => a.localeCompare(b));
  sortedProviders.forEach((provider, idx) => {
    const rows = providers.get(provider) || [];
    rows.sort((a, b) => {
      const aRank = a.flags.overall || 999;
      const bRank = b.flags.overall || 999;
      if (aRank !== bRank) return aRank - bRank;
      return (a.avg_latency_ms ?? Infinity) - (b.avg_latency_ms ?? Infinity);
    });

    const providerLabel = PROVIDER_LABELS[provider] || provider;
    sections.push(
      h(Text, {key: `prov-${provider}`, color: 'whiteBright'}, `${providerLabel}`),
      h(Text, {key: `head-${provider}`, dimColor: true}, header),
      h(Text, {key: `div-${provider}`, dimColor: true}, divider),
    );

    rows.forEach((row, rowIdx) => {
      const scoreCells = SCORE_COLUMNS.map(col => {
        const raw = row[col.key];
        const formatted = col.format(raw);
        const padded = pad(formatted, col.width, col.align || 'start');
        return raw != null ? chalk.cyan(padded) : padded;
      });

      const categoryCells = CATEGORY_COLUMNS.map(col => {
        const rank = row.flags[col.key];
        return pad(rank ? `#${rank}` : '—', col.width, 'end');
      });

      const line = [pad(row.model, MODEL_COL_WIDTH), ...scoreCells, ...categoryCells].join(' ');
      sections.push(h(Text, {key: `row-${provider}-${rowIdx}`}, line));
    });

    if (idx < sortedProviders.length - 1) {
      sections.push(h(Text, {key: `sp-${provider}`}, ''));
    }
  });

  return sections;
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
    payload ? renderProviderSections(payload) : null,
    h(Text, {dimColor: true}, 'Press q to return.'),
  );
}
