import React, {useCallback, useContext, useRef, useState} from 'react';
import {Box, Text, useInput} from 'ink';
import SelectInput from 'ink-select-input';
import TextInput from 'ink-text-input';
import chalk from 'chalk';
import {AppContext} from '../app.mjs';

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

export default function ConfigureView({onBack}) {
  const {bootstrap, backend, refreshBootstrap} = useContext(AppContext);
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
