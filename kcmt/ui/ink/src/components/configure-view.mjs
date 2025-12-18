import React, {useCallback, useContext, useEffect, useRef, useState} from 'react';
import {Box, Text, useInput} from 'ink';
import SelectInput from 'ink-select-input';
import TextInput from 'ink-text-input';
import Spinner from 'ink-spinner';
import chalk from 'chalk';
import {AppContext} from '../app-context.mjs';

const h = React.createElement;

const PROVIDER_ORDER = ['openai', 'anthropic', 'xai', 'github'];
const PROVIDER_LABELS = {
  openai: 'OpenAI',
  anthropic: 'Anthropic',
  xai: 'X.AI',
  github: 'GitHub Models',
};
const MAX_PRIORITY = 5;

function looksLikeEnv(value) {
  return Boolean(value) && /^[A-Za-z_][A-Za-z0-9_]*$/.test(String(value));
}

function defaultEnv(bootstrap, provider) {
  return bootstrap?.defaultModels?.[provider]?.api_key_env || '';
}

function defaultModel(bootstrap, provider) {
  return bootstrap?.defaultModels?.[provider]?.model || '';
}

function providerSupportsBatch(provider) {
  return provider === 'openai';
}

function buildInitialProviders(bootstrap) {
  const defaults = bootstrap?.defaultModels || {};
  const existing = bootstrap?.config?.providers || {};
  const result = {};
  for (const prov of PROVIDER_ORDER) {
    const entry = existing[prov] || {};
    const meta = defaults[prov] || {};
    result[prov] = {
      name: entry.name || PROVIDER_LABELS[prov] || prov,
      endpoint: entry.endpoint || meta.endpoint || '',
      api_key_env: entry.api_key_env || meta.api_key_env || '',
      preferred_model: entry.preferred_model || meta.model || '',
    };
  }
  return result;
}

function buildInitialPriority(bootstrap) {
  const fromConfig = Array.isArray(bootstrap?.config?.model_priority)
    ? bootstrap.config.model_priority
    : [];
  const legacyPrimary = bootstrap?.config
    ? [{provider: bootstrap.config.provider, model: bootstrap.config.model}]
    : [];
  const merged = [...fromConfig, ...legacyPrimary];
  const deduped = [];
  const seen = new Set();
  for (const item of merged) {
    if (!item || typeof item !== 'object') continue;
    const prov = String(item.provider || '').trim();
    const model = String(item.model || '').trim();
    if (!prov || !model) continue;
    const key = `${prov}::${model}`;
    if (seen.has(key)) continue;
    seen.add(key);
    deduped.push({provider: prov, model});
    if (deduped.length >= MAX_PRIORITY) break;
  }
  const slots = new Array(MAX_PRIORITY).fill(null);
  deduped.forEach((entry, idx) => {
    slots[idx] = entry;
  });
  return slots;
}

function buildInitialBenchmark(bootstrap) {
  const prefs = bootstrap?.preferences && typeof bootstrap.preferences === 'object'
    ? bootstrap.preferences
    : {};
  const bench = prefs?.benchmark && typeof prefs.benchmark === 'object' ? prefs.benchmark : {};
  const providersCfg = bench?.providers && typeof bench.providers === 'object' ? bench.providers : {};

  const result = {};
  for (const prov of PROVIDER_ORDER) {
    const entry = providersCfg?.[prov];
    let enabled = true;
    let models = [];

    if (Array.isArray(entry)) {
      models = entry.map(v => String(v || '').trim()).filter(Boolean);
    } else if (entry && typeof entry === 'object') {
      if (entry.enabled === false) {
        enabled = false;
      }
      if (Array.isArray(entry.models)) {
        models = entry.models.map(v => String(v || '').trim()).filter(Boolean);
      }
    }

    result[prov] = {enabled, models};
  }
  return result;
}

function buildBenchmarkProvidersPayload(benchmarkState) {
  const providers = {};
  for (const prov of PROVIDER_ORDER) {
    const entry = benchmarkState?.[prov] || {};
    const enabled = entry.enabled !== false;
    const models = Array.isArray(entry.models)
      ? entry.models.map(v => String(v || '').trim()).filter(Boolean)
      : [];

    if (!enabled || models.length) {
      providers[prov] = {};
      if (!enabled) providers[prov].enabled = false;
      if (models.length) providers[prov].models = models;
    }
  }
  return providers;
}

function Prompt({label, value = '', placeholder = '', onSubmit, onCancel}) {
  const [draft, setDraft] = useState(value);
  useInput((input, key) => {
    if (key.escape) {
      onCancel?.();
    }
    if (key.return) {
      const next = draft.trim() || value || placeholder || '';
      onSubmit?.(next);
    }
  });
  return h(
    Box,
    {flexDirection: 'column', gap: 1},
    h(Text, null, chalk.magenta(label)),
    h(TextInput, {value: draft, placeholder, onChange: setDraft}),
    h(Text, {dimColor: true}, 'Press Enter to confirm, Esc to cancel.'),
  );
}

export default function ConfigureView({onBack} = {}) {
  const {bootstrap, backend, refreshBootstrap} = useContext(AppContext);
  const hydratedRef = useRef(Boolean(bootstrap));
  const [providersState, setProvidersState] = useState(() => buildInitialProviders(bootstrap));
  const [priorityState, setPriorityState] = useState(() => buildInitialPriority(bootstrap));
  const [step, setStep] = useState('providers');
  const [activeProvider, setActiveProvider] = useState(null);
  const [activeSlot, setActiveSlot] = useState(null);
  const [slotProvider, setSlotProvider] = useState(null);
  const [batchEnabled, setBatchEnabled] = useState(Boolean(bootstrap?.config?.use_batch));
  const [batchModel, setBatchModel] = useState(
    () =>
      bootstrap?.config?.batch_model ||
      buildInitialProviders(bootstrap).openai?.preferred_model ||
      '',
  );
  const [saving, setSaving] = useState(false);
  const savingRef = useRef(false);
  const [providerModels, setProviderModels] = useState(bootstrap?.modelCatalog || {});
  const [loadingModels, setLoadingModels] = useState(false);
  const loadedProvidersRef = useRef(new Set());
  const [benchmarkState, setBenchmarkState] = useState(() => buildInitialBenchmark(bootstrap));
  const [benchmarkProvider, setBenchmarkProvider] = useState(null);
  const [benchMultiCursor, setBenchMultiCursor] = useState(0);
  const [benchMultiSelected, setBenchMultiSelected] = useState([]);

  const ensureModelsLoaded = useCallback(async (provider) => {
    if (loadedProvidersRef.current.has(provider)) {
      return;
    }
    setLoadingModels(true);
    try {
      const result = await backend.listModels([provider]);
      if (result?.modelCatalog) {
        setProviderModels(prev => ({
          ...prev,
          ...result.modelCatalog,
        }));
        loadedProvidersRef.current.add(provider);
      }
    } catch (err) {
      console.error('Failed to load models:', err);
    } finally {
      setLoadingModels(false);
    }
  }, [backend]);

  useEffect(() => {
    if (!bootstrap || hydratedRef.current) {
      return;
    }
    hydratedRef.current = true;
    setProvidersState(buildInitialProviders(bootstrap));
    setPriorityState(buildInitialPriority(bootstrap));
    setBatchEnabled(Boolean(bootstrap?.config?.use_batch));
    setBatchModel(
      bootstrap?.config?.batch_model ||
        buildInitialProviders(bootstrap).openai?.preferred_model ||
        '',
    );
    setProviderModels(bootstrap?.modelCatalog || {});
    setBenchmarkState(buildInitialBenchmark(bootstrap));
  }, [bootstrap]);

  useEffect(() => {
    if (step !== 'benchmark-provider-multi' || !benchmarkProvider) {
      return;
    }
    const models = Array.isArray(benchmarkState?.[benchmarkProvider]?.models)
      ? benchmarkState[benchmarkProvider].models
      : [];
    setBenchMultiSelected(models);
    setBenchMultiCursor(0);
  }, [benchmarkProvider, benchmarkState, step]);

  useEffect(() => {
    const provider = step === 'provider-model'
      ? activeProvider
      : (step === 'benchmark-provider-add' || step === 'benchmark-provider-multi')
        ? benchmarkProvider
        : null;
    if (!provider) {
      return;
    }
    ensureModelsLoaded(provider);
  }, [activeProvider, benchmarkProvider, ensureModelsLoaded, step]);

  const hasApiKey = useCallback(
    provider => {
      const env = providersState[provider]?.api_key_env;
      if (!env) {
        return false;
      }
      const raw = process.env[env];
      return Boolean(raw && String(raw).trim());
    },
    [providersState],
  );

  const applyProviderUpdate = useCallback((provider, updates) => {
    setProvidersState(prev => ({
      ...prev,
      [provider]: {
        ...prev[provider],
        ...updates,
      },
    }));
  }, []);

  const applyPrioritySelection = useCallback((slotIndex, provider, model) => {
    setPriorityState(prev => {
      const next = [...prev];
      for (let i = 0; i < next.length; i += 1) {
        if (i === slotIndex) continue;
        if (next[i] && next[i].provider === provider && next[i].model === model) {
          next[i] = null;
        }
      }
      next[slotIndex] = {provider, model};
      return next;
    });
    setProvidersState(prev => ({
      ...prev,
      [provider]: {
        ...prev[provider],
        preferred_model: model,
      },
    }));
  }, []);

  const applyProviderModel = useCallback(
    (provider, model) => {
      applyProviderUpdate(provider, {preferred_model: model});
      setPriorityState(prev => {
        const next = [...prev];
        let placed = false;
        for (let i = 0; i < next.length; i += 1) {
          if (next[i] && next[i].provider === provider) {
            next[i] = {provider, model};
            placed = true;
          }
        }
        if (!placed) {
          const emptyIdx = next.findIndex(entry => !entry);
          if (emptyIdx !== -1) {
            next[emptyIdx] = {provider, model};
            placed = true;
          }
        }
        if (!placed && next.length) {
          next[next.length - 1] = {provider, model};
        }
        return next;
      });
    },
    [applyProviderUpdate],
  );

  const inferPriority = useCallback(() => {
    const inferred = [];
    for (const prov of PROVIDER_ORDER) {
      const meta = providersState[prov] || {};
      const env = meta.api_key_env || defaultEnv(bootstrap, prov);
      const model = meta.preferred_model || defaultModel(bootstrap, prov);
      if (!env || !model) continue;
      inferred.push({provider: prov, model});
      if (inferred.length >= MAX_PRIORITY) break;
    }
    return inferred;
  }, [bootstrap, providersState]);

  const clearPrioritySlot = useCallback(slotIndex => {
    setPriorityState(prev => {
      const next = [...prev];
      next[slotIndex] = null;
      return next;
    });
  }, []);

  const handleSave = useCallback(() => {
    if (savingRef.current) {
      return;
    }
    let filtered = priorityState.filter(Boolean);
    if (!filtered.length) {
      filtered = inferPriority();
    }
    if (!filtered.length) {
      return;
    }
    const primary = filtered[0];
    const providerSettings = providersState[primary.provider] || {};
    const payload = {
      provider: primary.provider,
      model: primary.model,
      llm_endpoint: providerSettings.endpoint,
      api_key_env: providerSettings.api_key_env,
      providers: providersState,
      model_priority: filtered,
      use_batch: primary.provider === 'openai' ? Boolean(batchEnabled) : false,
      batch_model: primary.provider === 'openai' ? batchModel || primary.model : null,
    };
    savingRef.current = true;
    setSaving(true);
    Promise.resolve(backend.saveConfig(payload))
      .then(() => refreshBootstrap())
      .then(() => onBack())
      .catch(error => {
        console.error(error);
        savingRef.current = false;
        setSaving(false);
      });
  }, [backend, batchEnabled, batchModel, bootstrap, inferPriority, onBack, priorityState, providersState, refreshBootstrap]);

  const applyBenchmarkUpdate = useCallback((provider, updates) => {
    setBenchmarkState(prev => ({
      ...prev,
      [provider]: {
        ...prev[provider],
        ...updates,
      },
    }));
  }, []);

  const addBenchmarkModel = useCallback((provider, modelId) => {
    const cleaned = String(modelId || '').trim();
    if (!cleaned) return;
    setBenchmarkState(prev => {
      const current = prev?.[provider]?.models || [];
      if (current.includes(cleaned)) {
        return prev;
      }
      return {
        ...prev,
        [provider]: {
          ...(prev[provider] || {enabled: true, models: []}),
          models: [...current, cleaned],
        },
      };
    });
  }, []);

  const removeBenchmarkModel = useCallback((provider, modelId) => {
    const cleaned = String(modelId || '').trim();
    if (!cleaned) return;
    setBenchmarkState(prev => {
      const current = Array.isArray(prev?.[provider]?.models) ? prev[provider].models : [];
      const nextModels = current.filter(entry => entry !== cleaned);
      return {
        ...prev,
        [provider]: {
          ...(prev[provider] || {enabled: true, models: []}),
          models: nextModels,
        },
      };
    });
  }, []);

  const clearBenchmarkModels = useCallback((provider) => {
    applyBenchmarkUpdate(provider, {models: []});
  }, [applyBenchmarkUpdate]);

  const handleSaveBenchmark = useCallback(() => {
    if (savingRef.current) {
      return;
    }
    const basePrefs = bootstrap?.preferences && typeof bootstrap.preferences === 'object'
      ? bootstrap.preferences
      : {};
    const providersPayload = buildBenchmarkProvidersPayload(benchmarkState);

    const nextPrefs = {...basePrefs};
    if (Object.keys(providersPayload).length) {
      const bench = nextPrefs.benchmark && typeof nextPrefs.benchmark === 'object' ? nextPrefs.benchmark : {};
      nextPrefs.benchmark = {...bench, providers: providersPayload};
    } else if (nextPrefs.benchmark && typeof nextPrefs.benchmark === 'object') {
      const bench = {...nextPrefs.benchmark};
      delete bench.providers;
      if (Object.keys(bench).length) {
        nextPrefs.benchmark = bench;
      } else {
        delete nextPrefs.benchmark;
      }
    }

    savingRef.current = true;
    setSaving(true);
    Promise.resolve(backend.savePreferences(nextPrefs))
      .then(() => refreshBootstrap())
      .then(() => {
        savingRef.current = false;
        setSaving(false);
        setBenchmarkProvider(null);
        setStep('providers');
      })
      .catch(error => {
        console.error(error);
        savingRef.current = false;
        setSaving(false);
      });
  }, [backend, benchmarkState, bootstrap, refreshBootstrap]);

  useInput((input, key) => {
    if (savingRef.current) {
      return;
    }
    if (key.escape) {
      if (step === 'summary') {
        setStep('priority');
        return;
      }
      if (step === 'providers') {
        onBack();
        return;
      }
      if (step === 'provider-menu') {
        setActiveProvider(null);
        setStep('providers');
        return;
      }
      if (step === 'benchmark') {
        setBenchmarkProvider(null);
        setStep('providers');
        return;
      }
      if (step === 'benchmark-provider') {
        setBenchmarkProvider(null);
        setStep('benchmark');
        return;
      }
      if (
        step === 'benchmark-provider-add' ||
        step === 'benchmark-provider-add-manual' ||
        step === 'benchmark-provider-remove' ||
        step === 'benchmark-provider-multi'
      ) {
        setStep('benchmark-provider');
        return;
      }
      if (
        step === 'provider-endpoint' ||
        step === 'provider-env' ||
        step === 'provider-model' ||
        step === 'provider-model-manual' ||
        step === 'provider-batch-mode' ||
        step === 'provider-batch-model' ||
        step === 'provider-batch-manual'
      ) {
        setStep('provider-menu');
        return;
      }
      if (step === 'priority') {
        setStep('providers');
        return;
      }
      if (step === 'priority-provider' || step === 'priority-model' || step === 'priority-model-manual') {
        setSlotProvider(null);
        setActiveSlot(null);
        setStep('priority');
        return;
      }
    }
    if (step === 'summary' && key.return) {
      handleSave();
    }

    if (step === 'benchmark-provider-multi') {
      const catalogue = Array.isArray(providerModels?.[benchmarkProvider])
        ? providerModels[benchmarkProvider]
        : [];

      const ids = catalogue
        .map(entry => String(entry?.id || '').trim())
        .filter(Boolean);

      const cursorMax = Math.max(0, ids.length - 1);
      if (key.upArrow || String(input || '').toLowerCase() === 'k') {
        setBenchMultiCursor(prev => Math.max(0, prev - 1));
      }
      if (key.downArrow || String(input || '').toLowerCase() === 'j') {
        setBenchMultiCursor(prev => Math.min(cursorMax, prev + 1));
      }

      if (key.return) {
        const selected = new Set(
          Array.isArray(benchMultiSelected) ? benchMultiSelected : [],
        );
        const ordered = [];
        for (const modelId of ids) {
          if (selected.has(modelId)) ordered.push(modelId);
        }
        for (const modelId of selected) {
          if (!ordered.includes(modelId)) ordered.push(modelId);
        }
        applyBenchmarkUpdate(benchmarkProvider, {models: ordered});
        setStep('benchmark-provider');
      }

      if (input === ' ') {
        const currentId = ids[benchMultiCursor];
        if (!currentId) return;
        setBenchMultiSelected(prev => {
          const current = Array.isArray(prev) ? prev : [];
          if (current.includes(currentId)) {
            return current.filter(item => item !== currentId);
          }
          return [...current, currentId];
        });
      }
    }
  });

  if (saving) {
    return h(Box, {flexDirection: 'column'}, h(Text, null, 'Saving configuration…'));
  }

  if (step === 'providers') {
    const items = [
      ...PROVIDER_ORDER.map(prov => {
        const info = providersState[prov];
        const status = hasApiKey(prov) ? '🟢' : '🟡';
        const envValue = looksLikeEnv(info.api_key_env) ? info.api_key_env : '';
        const envLabel = envValue ? ` • env ${envValue}` : '';
        const modelLabel = info.preferred_model ? ` • model ${info.preferred_model}` : '';
        return {
          label: `${status} ${info.name}${envLabel}${modelLabel}`,
          value: prov,
        };
      }),
      {label: '🧪 Benchmark settings', value: '__benchmark__'},
      {label: '📊 Set model priority', value: '__priority__'},
      {label: '✅ Review & Save', value: '__summary__'},
      {label: '↩️  Cancel', value: '__cancel__'},
    ];
    return h(
      Box,
      {flexDirection: 'column', gap: 1},
      h(Text, null, chalk.bold('Configure providers')),
      h(Text, {dimColor: true}, 'Select a provider to edit API key, URL, model, or batch mode.'),
      h(SelectInput, {
        items,
        onSelect: item => {
          if (item.value === '__benchmark__') {
            setStep('benchmark');
            return;
          }
          if (item.value === '__priority__') {
            setStep('priority');
            return;
          }
          if (item.value === '__summary__') {
            setStep('summary');
            return;
          }
          if (item.value === '__cancel__') {
            onBack();
            return;
          }
          setActiveProvider(item.value);
          setStep('provider-menu');
        },
      }),
    );
  }

  if (step === 'benchmark') {
    const items = [
      ...PROVIDER_ORDER.map(prov => {
        const entry = benchmarkState?.[prov] || {enabled: true, models: []};
        const enabled = entry.enabled !== false;
        const models = Array.isArray(entry.models) ? entry.models : [];
        const modelLabel = models.length ? `${models.length} selected` : 'all models';
        return {
          label: `${enabled ? '🟢' : '🚫'} ${PROVIDER_LABELS[prov] || prov} • ${modelLabel}`,
          value: prov,
        };
      }),
      {label: '✅ Save benchmark settings', value: '__save__'},
      {label: '↩️  Back', value: '__back__'},
    ];

    return h(
      Box,
      {flexDirection: 'column', gap: 1},
      h(Text, null, chalk.bold('Benchmark settings')),
      h(Text, {dimColor: true}, 'Enable/disable providers and optionally select models to benchmark.'),
      h(SelectInput, {
        items,
        onSelect: item => {
          if (item.value === '__save__') {
            handleSaveBenchmark();
            return;
          }
          if (item.value === '__back__') {
            setBenchmarkProvider(null);
            setStep('providers');
            return;
          }
          setBenchmarkProvider(item.value);
          setStep('benchmark-provider');
        },
      }),
    );
  }

  if (step === 'benchmark-provider' && benchmarkProvider) {
    const entry = benchmarkState?.[benchmarkProvider] || {enabled: true, models: []};
    const enabled = entry.enabled !== false;
    const models = Array.isArray(entry.models) ? entry.models : [];

    const items = [
      {label: `${enabled ? '✅ Enabled' : '🚫 Disabled'} (toggle)`, value: 'toggle-enabled'},
      {label: '🧠 Select models (multi)…', value: 'multi'},
      {label: '➕ Add model from list…', value: 'add'},
      {label: '✏️  Manual model entry…', value: 'add-manual'},
    ];

    if (models.length) {
      items.push({label: `➖ Remove model (${models.length} selected)…`, value: 'remove'});
      items.push({label: '🧹 Use all models (clear selection)', value: 'clear'});
    }

    items.push({label: '✅ Save benchmark settings', value: '__save__'});
    items.push({label: '↩️  Back', value: '__back__'});

    return h(
      Box,
      {flexDirection: 'column', gap: 1},
      h(Text, null, chalk.bold(`Benchmark: ${PROVIDER_LABELS[benchmarkProvider] || benchmarkProvider}`)),
      h(Text, {dimColor: true}, models.length ? `Selected models: ${models.join(', ')}` : 'Selected models: all'),
      h(SelectInput, {
        items,
        onSelect: item => {
          if (item.value === '__back__') {
            setBenchmarkProvider(null);
            setStep('benchmark');
            return;
          }
          if (item.value === '__save__') {
            handleSaveBenchmark();
            return;
          }
          if (item.value === 'toggle-enabled') {
            applyBenchmarkUpdate(benchmarkProvider, {enabled: !enabled});
            return;
          }
          if (item.value === 'multi') {
            setStep('benchmark-provider-multi');
            return;
          }
          if (item.value === 'add') {
            setStep('benchmark-provider-add');
            return;
          }
          if (item.value === 'add-manual') {
            setStep('benchmark-provider-add-manual');
            return;
          }
          if (item.value === 'remove') {
            setStep('benchmark-provider-remove');
            return;
          }
          if (item.value === 'clear') {
            clearBenchmarkModels(benchmarkProvider);
          }
        },
      }),
    );
  }

  if (step === 'benchmark-provider-multi' && benchmarkProvider) {
    if (loadingModels && !loadedProvidersRef.current.has(benchmarkProvider)) {
      return h(
        Box,
        {flexDirection: 'column', gap: 1},
        h(Text, null, chalk.bold(`Loading models for ${PROVIDER_LABELS[benchmarkProvider] || benchmarkProvider}...`)),
        h(Spinner, {type: 'dots'}),
      );
    }

    const catalogue = Array.isArray(providerModels?.[benchmarkProvider])
      ? providerModels[benchmarkProvider]
      : [];

    const rows = catalogue.map(entry => ({
      id: String(entry?.id || '').trim(),
      quality: entry?.quality ? String(entry.quality) : '',
    })).filter(item => item.id);

    if (!rows.length) {
      return h(
        Box,
        {flexDirection: 'column', gap: 1},
        h(Text, null, chalk.bold(`Select models (${PROVIDER_LABELS[benchmarkProvider] || benchmarkProvider})`)),
        h(Text, {dimColor: true}, 'No models available for selection.'),
        h(Text, {dimColor: true}, 'Press Esc to go back.'),
      );
    }

    const cursor = Math.max(0, Math.min(benchMultiCursor, rows.length - 1));
    const selected = new Set(Array.isArray(benchMultiSelected) ? benchMultiSelected : []);

    const WINDOW = 14;
    const start = Math.max(0, Math.min(cursor - Math.floor(WINDOW / 2), Math.max(0, rows.length - WINDOW)));
    const end = Math.min(rows.length, start + WINDOW);
    const visible = rows.slice(start, end);

    return h(
      Box,
      {flexDirection: 'column', gap: 1},
      h(Text, null, chalk.bold(`Select models (${PROVIDER_LABELS[benchmarkProvider] || benchmarkProvider})`)),
      h(Text, {dimColor: true}, 'Use j/k or arrows • Space toggles • Enter confirms • Esc backs out.'),
      h(
        Box,
        {flexDirection: 'column'},
        ...visible.map((item, idx) => {
          const absoluteIdx = start + idx;
          const isCursor = absoluteIdx === cursor;
          const checked = selected.has(item.id);
          const prefix = isCursor ? chalk.cyan('❯') : ' ';
          const box = checked ? chalk.green('[x]') : '[ ]';
          const quality = item.quality ? chalk.dim(` (${item.quality})`) : '';
          return h(
            Text,
            {key: `bench-multi-${absoluteIdx}`, wrap: 'truncate'},
            `${prefix} ${box} ${item.id}${quality}`,
          );
        }),
      ),
      h(Text, {dimColor: true, wrap: 'truncate'}, `Selected: ${selected.size ? Array.from(selected).join(', ') : 'all models (none explicitly selected)'}`),
    );
  }

  if (step === 'benchmark-provider-add' && benchmarkProvider) {
    if (loadingModels && !loadedProvidersRef.current.has(benchmarkProvider)) {
      return h(
        Box,
        {flexDirection: 'column', gap: 1},
        h(Text, null, chalk.bold(`Loading models for ${PROVIDER_LABELS[benchmarkProvider] || benchmarkProvider}...`)),
        h(Spinner, {type: 'dots'}),
      );
    }

    const catalogue = Array.isArray(providerModels[benchmarkProvider]) ? providerModels[benchmarkProvider] : [];
    const items = catalogue.map(entry => ({
      label: `${entry.id} ${entry.quality ? `(${entry.quality})` : ''}`.trim(),
      value: entry.id,
    }));
    items.push({label: '↩️  Back', value: '__back__'});

    return h(
      Box,
      {flexDirection: 'column', gap: 1},
      h(Text, null, chalk.bold(`Add benchmark model (${PROVIDER_LABELS[benchmarkProvider] || benchmarkProvider})`)),
      h(SelectInput, {
        items,
        onSelect: item => {
          if (item.value === '__back__') {
            setStep('benchmark-provider');
            return;
          }
          addBenchmarkModel(benchmarkProvider, item.value);
          setStep('benchmark-provider');
        },
      }),
    );
  }

  if (step === 'benchmark-provider-add-manual' && benchmarkProvider) {
    return h(Prompt, {
      label: `Manual benchmark model entry (${PROVIDER_LABELS[benchmarkProvider] || benchmarkProvider})`,
      onSubmit: value => {
        addBenchmarkModel(benchmarkProvider, value);
        setStep('benchmark-provider');
      },
      onCancel: () => {
        setStep('benchmark-provider');
      },
    });
  }

  if (step === 'benchmark-provider-remove' && benchmarkProvider) {
    const models = Array.isArray(benchmarkState?.[benchmarkProvider]?.models)
      ? benchmarkState[benchmarkProvider].models
      : [];
    const items = models.map(model => ({label: model, value: model}));
    items.push({label: '↩️  Back', value: '__back__'});

    return h(
      Box,
      {flexDirection: 'column', gap: 1},
      h(Text, null, chalk.bold(`Remove benchmark model (${PROVIDER_LABELS[benchmarkProvider] || benchmarkProvider})`)),
      h(SelectInput, {
        items,
        onSelect: item => {
          if (item.value === '__back__') {
            setStep('benchmark-provider');
            return;
          }
          removeBenchmarkModel(benchmarkProvider, item.value);
          setStep('benchmark-provider');
        },
      }),
    );
  }

  if (step === 'provider-menu' && activeProvider) {
    const info = providersState[activeProvider] || {};
    const envLabelValue = looksLikeEnv(info.api_key_env) ? info.api_key_env : defaultEnv(bootstrap, activeProvider);
    const items = [
      {
        label: `🔑 API Key (${envLabelValue || 'default'})`,
        value: 'api-key',
      },
      {
        label: `🌐 API URL (${info.endpoint || bootstrap?.defaultModels?.[activeProvider]?.endpoint || 'default'})`,
        value: 'endpoint',
      },
      {
        label: `🧠 Model selection (${info.preferred_model || defaultModel(bootstrap, activeProvider) || 'default'})`,
        value: 'model',
      },
    ];
    if (providerSupportsBatch(activeProvider)) {
      items.push({
        label: `📦 Batch mode (${batchEnabled ? 'on' : 'off'})`,
        value: 'batch',
      });
    }
    items.push({label: '✅ Save & exit', value: '__summary__'});
    items.push({label: '↩️  Back', value: '__back__'});

    return h(
      Box,
      {flexDirection: 'column', gap: 1},
      h(Text, null, chalk.bold(`Provider: ${info.name}`)),
      h(Text, {dimColor: true}, 'Choose a field to edit, then save.'),
      h(SelectInput, {
        items,
        onSelect: item => {
          if (item.value === '__back__') {
            setActiveProvider(null);
            setStep('providers');
            return;
          }
          if (item.value === '__summary__') {
            setStep('summary');
            return;
          }
          if (item.value === 'api-key') {
            setStep('provider-env');
            return;
          }
          if (item.value === 'endpoint') {
            setStep('provider-endpoint');
            return;
          }
          if (item.value === 'model') {
            setStep('provider-model');
            return;
          }
          if (item.value === 'batch') {
            setStep('provider-batch-mode');
          }
        },
      }),
    );
  }

  if (step === 'provider-endpoint' && activeProvider) {
    const info = providersState[activeProvider];
    return h(Prompt, {
      label: `Endpoint for ${info.name}`,
      value: info.endpoint,
      placeholder: bootstrap?.defaultModels?.[activeProvider]?.endpoint || info.endpoint,
      onSubmit: value => {
        applyProviderUpdate(activeProvider, {endpoint: value});
        setStep('provider-menu');
      },
      onCancel: () => setStep('provider-menu'),
    });
  }

  if (step === 'provider-env' && activeProvider) {
    const info = providersState[activeProvider];
    const defaultValue = defaultEnv(bootstrap, activeProvider);
    const placeholderEnv = looksLikeEnv(info.api_key_env) ? info.api_key_env : defaultValue;
    return h(Prompt, {
      label: `API key environment variable for ${info.name}`,
      value: '',
      placeholder: placeholderEnv,
      onSubmit: value => {
        applyProviderUpdate(activeProvider, {api_key_env: value});
        setStep('provider-menu');
      },
      onCancel: () => {
        setStep('provider-menu');
      },
    });
  }

  if (step === 'provider-model' && activeProvider) {
    if (loadingModels && !loadedProvidersRef.current.has(activeProvider)) {
      return h(
        Box,
        {flexDirection: 'column', gap: 1},
        h(Text, null, chalk.bold(`Loading models for ${PROVIDER_LABELS[activeProvider] || activeProvider}...`)),
        h(Spinner, {type: 'dots'}),
      );
    }

    const catalogue = Array.isArray(providerModels[activeProvider]) ? providerModels[activeProvider] : [];
    const items = catalogue.map(entry => ({
      label: `${entry.id} ${entry.quality ? `(${entry.quality})` : ''}`.trim(),
      value: entry.id,
    }));
    const defaultChoice =
      providersState[activeProvider]?.preferred_model || defaultModel(bootstrap, activeProvider);
    if (!items.length && defaultChoice) {
      items.push({label: defaultChoice, value: defaultChoice});
    }
    items.push({label: '✏️  Manual entry…', value: '__manual__'});
    items.push({label: '↩️  Back', value: '__back__'});
    return h(
      Box,
      {flexDirection: 'column', gap: 1},
      h(Text, null, chalk.bold(`Model for ${PROVIDER_LABELS[activeProvider] || activeProvider}`)),
      h(SelectInput, {
        items,
        onSelect: item => {
          if (item.value === '__back__') {
            setStep('provider-menu');
            return;
          }
          if (item.value === '__manual__') {
            setStep('provider-model-manual');
            return;
          }
          applyProviderModel(activeProvider, item.value);
          if (providerSupportsBatch(activeProvider)) {
            setBatchModel(prev => prev || item.value);
          }
          setStep('provider-menu');
        },
      }),
    );
  }

  if (step === 'provider-model-manual' && activeProvider) {
    return h(Prompt, {
      label: `Manual model entry for ${PROVIDER_LABELS[activeProvider] || activeProvider}`,
      onSubmit: value => {
        applyProviderModel(activeProvider, value);
        if (providerSupportsBatch(activeProvider)) {
          setBatchModel(prev => prev || value);
        }
        setStep('provider-menu');
      },
      onCancel: () => {
        setStep('provider-model');
      },
    });
  }

  if (step === 'priority') {
    const items = [];
    for (let idx = 0; idx < MAX_PRIORITY; idx += 1) {
      const entry = priorityState[idx];
      const label = entry
        ? `${idx + 1}. ${entry.provider} → ${entry.model}`
        : `${idx + 1}. [empty]`;
      items.push({label, value: `slot-${idx}`});
    }
    items.push({label: '✅ Review & Save', value: '__summary__'});
    items.push({label: '↩️  Back to providers', value: '__providers__'});
    return h(
      Box,
      {flexDirection: 'column', gap: 1},
      h(Text, null, chalk.bold('Model priority (select slot to edit)')),
      h(SelectInput, {
        items,
        onSelect: item => {
          if (item.value === '__summary__') {
            setStep('summary');
            return;
          }
          if (item.value === '__providers__') {
            setStep('providers');
            return;
          }
          const index = Number(item.value.replace('slot-', ''));
          setActiveSlot(index);
          setStep('priority-provider');
        },
      }),
    );
  }

  if (step === 'priority-provider' && typeof activeSlot === 'number') {
    const eligibleProviders = PROVIDER_ORDER.filter(hasApiKey);
    if (!eligibleProviders.length) {
      return h(
        Box,
        {flexDirection: 'column', gap: 1},
        h(Text, null, chalk.yellow('No providers with valid API keys detected. Configure providers first.')),
        h(Text, {dimColor: true}, 'Press Esc to return.'),
      );
    }
    const items = eligibleProviders.map(prov => ({
      label: `${PROVIDER_LABELS[prov] || prov} (${providersState[prov].api_key_env})`,
      value: prov,
    }));
    items.push({label: '🗑️  Clear slot', value: '__clear__'});
    items.push({label: '↩️  Back', value: '__back__'});
    return h(
      Box,
      {flexDirection: 'column', gap: 1},
      h(Text, null, chalk.bold(`Select provider for priority ${activeSlot + 1}`)),
      h(SelectInput, {
        items,
        onSelect: item => {
          if (item.value === '__clear__') {
            clearPrioritySlot(activeSlot);
            setStep('priority');
            setSlotProvider(null);
            return;
          }
          if (item.value === '__back__') {
            setStep('priority');
            setSlotProvider(null);
            return;
          }
          setSlotProvider(item.value);
          setStep('priority-model');
        },
      }),
    );
  }

  if (step === 'priority-model' && typeof activeSlot === 'number' && slotProvider) {
    // Lazy-load models for this provider if not already loaded
    useEffect(() => {
      ensureModelsLoaded(slotProvider);
    }, [slotProvider, ensureModelsLoaded]);

    if (loadingModels && !loadedProvidersRef.current.has(slotProvider)) {
      return h(
        Box,
        {flexDirection: 'column', gap: 1},
        h(Text, null, chalk.bold(`Loading models for ${PROVIDER_LABELS[slotProvider] || slotProvider}...`)),
        h(Spinner, {type: 'dots'}),
      );
    }

    const catalogue = Array.isArray(providerModels[slotProvider]) ? providerModels[slotProvider] : [];
    const items = catalogue.map(entry => ({
      label: `${entry.id} ${entry.quality ? `(${entry.quality})` : ''}`.trim(),
      value: entry.id,
    }));
    if (!items.length) {
      const defaultModel = providersState[slotProvider]?.preferred_model || bootstrap?.defaultModels?.[slotProvider]?.model;
      if (defaultModel) {
        items.push({label: defaultModel, value: defaultModel});
      }
    }
    items.push({label: '✏️  Manual entry…', value: '__manual__'});
    items.push({label: '↩️  Back', value: '__back__'});
    return h(
      Box,
      {flexDirection: 'column', gap: 1},
      h(Text, null, chalk.bold(`Select model for ${PROVIDER_LABELS[slotProvider] || slotProvider}`)),
      h(SelectInput, {
        items,
        onSelect: item => {
          if (item.value === '__back__') {
            setStep('priority-provider');
            return;
          }
          if (item.value === '__manual__') {
            setStep('priority-model-manual');
            return;
          }
          applyPrioritySelection(activeSlot, slotProvider, item.value);
          setSlotProvider(null);
          setActiveSlot(null);
          setStep('priority');
        },
      }),
    );
  }

  if (step === 'priority-model-manual' && typeof activeSlot === 'number' && slotProvider) {
    return h(Prompt, {
      label: `Manual model entry for ${PROVIDER_LABELS[slotProvider] || slotProvider}`,
      onSubmit: value => {
        applyPrioritySelection(activeSlot, slotProvider, value);
        setSlotProvider(null);
        setActiveSlot(null);
        setStep('priority');
      },
      onCancel: () => {
        setStep('priority-model');
      },
    });
  }

  if (step === 'summary') {
    const filtered = priorityState.filter(Boolean);
    const effectivePriority = filtered.length ? filtered : inferPriority();
    const primary = effectivePriority[0];
    const primaryProvider = primary ? providersState[primary.provider] : null;
    const batchLine =
      primary && primary.provider === 'openai'
        ? `Mode: ${batchEnabled ? 'Batch' : 'Standard'}${batchEnabled ? ` (model ${batchModel || primary.model})` : ''}`
        : null;
    return h(
      Box,
      {flexDirection: 'column', gap: 1},
      h(Text, null, chalk.bold('Configuration summary')),
      primary
        ? h(Text, null, `Primary: ${primary.provider} → ${primary.model}`)
        : h(Text, {color: 'red'}, 'No primary model selected'),
      ...effectivePriority.map((entry, idx) =>
        h(Text, {key: `prio-${idx}`, dimColor: idx === 0}, `${idx + 1}. ${entry.provider} → ${entry.model}`),
      ),
      h(Text, null, ''),
      primaryProvider
        ? h(Text, {dimColor: true}, `Endpoint: ${primaryProvider.endpoint}`)
        : null,
      primaryProvider
        ? h(Text, {dimColor: true}, `API key env: ${primaryProvider.api_key_env}`)
        : null,
      batchLine ? h(Text, {dimColor: true}, batchLine) : null,
      h(Text, {dimColor: true}, 'Press Enter to save, Esc to adjust models.'),
    );
  }

  if (step === 'provider-batch-mode' && activeProvider) {
    if (!providerSupportsBatch(activeProvider)) {
      return h(
        Box,
        {flexDirection: 'column', gap: 1},
        h(Text, null, chalk.yellow('Batch mode not available for this provider.')),
        h(Text, {dimColor: true}, 'Press Back to continue.'),
      );
    }
    const modelHint =
      providersState[activeProvider]?.preferred_model ||
      batchModel ||
      defaultModel(bootstrap, activeProvider) ||
      'model';
    const items = [
      {label: 'Standard mode', value: 'standard'},
      {label: 'Batch mode (OpenAI Batch API)', value: 'batch'},
      {label: '↩️  Back', value: '__back__'},
    ];
    return h(
      Box,
      {flexDirection: 'column', gap: 1},
      h(Text, null, chalk.bold(`Mode for ${PROVIDER_LABELS[activeProvider] || activeProvider} (model ${modelHint})`)),
      h(SelectInput, {
        items,
        onSelect: item => {
          if (item.value === '__back__') {
            setStep('provider-menu');
            return;
          }
          if (item.value === 'standard') {
            setBatchEnabled(false);
            setStep('provider-menu');
            return;
          }
          setBatchEnabled(true);
          setBatchModel(prev => prev || modelHint);
          setStep('provider-batch-model');
        },
      }),
    );
  }

  if (step === 'provider-batch-model' && activeProvider) {
    if (!providerSupportsBatch(activeProvider)) {
      return h(
        Box,
        {flexDirection: 'column', gap: 1},
        h(Text, null, chalk.yellow('Batch mode not available for this provider.')),
        h(Text, {dimColor: true}, 'Press Back to continue.'),
      );
    }
    // Lazy-load models for this provider if not already loaded
    useEffect(() => {
      ensureModelsLoaded(activeProvider);
    }, [activeProvider, ensureModelsLoaded]);

    if (loadingModels && !loadedProvidersRef.current.has(activeProvider)) {
      return h(
        Box,
        {flexDirection: 'column', gap: 1},
        h(Text, null, chalk.bold(`Loading models for ${PROVIDER_LABELS[activeProvider] || activeProvider}...`)),
        h(Spinner, {type: 'dots'}),
      );
    }

    const catalogue = Array.isArray(providerModels[activeProvider]) ? providerModels[activeProvider] : [];
    const items = catalogue.map(entry => ({
      label: `${entry.id} ${entry.quality ? `(${entry.quality})` : ''}`.trim(),
      value: entry.id,
    }));
    if (!items.length && batchModel) {
      items.push({label: batchModel, value: batchModel});
    }
    items.push({label: '✏️  Manual entry…', value: '__manual__'});
    items.push({label: '↩️  Back', value: '__back__'});
    return h(
      Box,
      {flexDirection: 'column', gap: 1},
      h(Text, null, chalk.bold('Select batch model')),
      h(SelectInput, {
        items,
        onSelect: item => {
          if (item.value === '__back__') {
            setStep('provider-batch-mode');
            return;
          }
          if (item.value === '__manual__') {
            setStep('provider-batch-manual');
            return;
          }
          setBatchModel(item.value);
          setStep('provider-menu');
        },
      }),
    );
  }

  if (step === 'provider-batch-manual' && activeProvider) {
    if (!providerSupportsBatch(activeProvider)) {
      return h(
        Box,
        {flexDirection: 'column', gap: 1},
        h(Text, null, chalk.yellow('Batch mode not available for this provider.')),
        h(Text, {dimColor: true}, 'Press Back to continue.'),
      );
    }
    return h(Prompt, {
      label: 'Manual batch model',
      value: batchModel,
      onSubmit: value => {
        setBatchModel(value);
        setStep('provider-menu');
      },
      onCancel: () => setStep('provider-batch-model'),
    });
  }

  return h(Box, null, h(Text, null, 'Unsupported configuration state.'));
}
