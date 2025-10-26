import React, {useContext, useMemo, useState} from 'react';
import {Box, Text, useInput} from 'ink';
import SelectInput from 'ink-select-input';
import TextInput from 'ink-text-input';
import gradient from 'gradient-string';
import {AppContext} from '../app.mjs';
const h = React.createElement;

const headerGradient = gradient(['#8e2de2', '#4a00e0']);

function ProviderStep({providers, onSelect, detected}) {
  return h(
    Box,
    {flexDirection: 'column', gap: 1},
    h(Text, null, headerGradient('⚙️ Choose your primary provider')),
    h(SelectInput, {
      items: providers.map(provider => ({
        label: `${detected[provider] ? '🟢' : '🟡'} ${provider}`,
        value: provider,
      })),
      onSelect: item => onSelect(item.value),
    }),
  );
}

function ModelStep({models, onSelect}) {
  return h(
    Box,
    {flexDirection: 'column', gap: 1},
    h(Text, null, headerGradient('🧬 Pick the default model')),
    h(SelectInput, {
      items: models.map(model => ({label: `${model.id} (${model.quality || 'n/a'})`, value: model.id})),
      onSelect: item => onSelect(item.value),
    }),
  );
}

function PromptStep({label, value, onSubmit, placeholder}) {
  const [draft, setDraft] = useState(value || '');
  useInput((input, key) => {
    if (key.return) {
      onSubmit(draft.trim() || value || '');
    }
  });
  return h(
    Box,
    {flexDirection: 'column', gap: 1},
    h(Text, null, headerGradient(label)),
    h(TextInput, {value: draft, placeholder, onChange: setDraft}),
    h(Text, {dimColor: true}, 'Press Enter to confirm.'),
  );
}

export default function ConfigureView({onBack, showAdvanced = false}) {
  const {bootstrap, backend, refreshBootstrap} = useContext(AppContext);
  const detected = bootstrap?.providerDetection || {};
  const providerList = useMemo(
    () => Object.keys(bootstrap?.defaultModels || {}),
    [bootstrap]
  );
  const providerModels = bootstrap?.modelCatalog || {};

  const [step, setStep] = useState('provider');
  const [config, setConfig] = useState(() => ({
    provider: bootstrap?.config?.provider || providerList[0] || 'openai',
    model: bootstrap?.config?.model || '',
    llm_endpoint: bootstrap?.config?.llm_endpoint || '',
    api_key_env: bootstrap?.config?.api_key_env || '',
  }));

  useInput((input, key) => {
    if (key.escape) {
      onBack();
    }
    if (key.return && step === 'summary') {
      backend
        .saveConfig(config)
        .then(() => refreshBootstrap())
        .then(() => onBack())
        .catch(error => {
          console.error(error);
        });
    }
  });

  if (step === 'provider') {
    return h(ProviderStep, {
      providers: providerList,
      detected,
      onSelect: provider => {
        setConfig(prev => ({...prev, provider}));
        setStep('model');
      },
    });
  }

  if (step === 'model') {
    const models = providerModels[config.provider] || [];
    const augmented = models.length
      ? models
      : [
          {
            id: bootstrap.defaultModels[config.provider].model,
            quality: 'default',
          },
        ];
    return h(ModelStep, {
      models: augmented,
      onSelect: model => {
        setConfig(prev => ({...prev, model}));
        setStep('endpoint');
      },
    });
  }

  if (step === 'endpoint') {
    const defaultEndpoint =
      bootstrap.defaultModels[config.provider]?.endpoint || config.llm_endpoint;
    return h(PromptStep, {
      label: '🌐 Endpoint URL',
      value: config.llm_endpoint || defaultEndpoint,
      placeholder: defaultEndpoint,
      onSubmit: value => {
        setConfig(prev => ({...prev, llm_endpoint: value || defaultEndpoint}));
        setStep('api');
      },
    });
  }

  if (step === 'api') {
    const defaultKey =
      bootstrap.defaultModels[config.provider]?.api_key_env || config.api_key_env;
    return h(PromptStep, {
      label: '🔐 API key environment variable',
      value: config.api_key_env || defaultKey,
      placeholder: defaultKey,
      onSubmit: value => {
        setConfig(prev => ({...prev, api_key_env: value || defaultKey}));
        setStep('summary');
      },
    });
  }

  return h(
    Box,
    {flexDirection: 'column', gap: 1},
    h(Text, null, headerGradient('✨ Configuration summary')),
    h(Text, null, `Provider: ${config.provider}`),
    h(Text, null, `Model: ${config.model}`),
    h(Text, null, `Endpoint: ${config.llm_endpoint}`),
    h(Text, null, `API key env: ${config.api_key_env}`),
    h(Text, {dimColor: true}, 'Press Enter to save or Esc to cancel.'),
  );
}
