import React, {useContext, useMemo, useState} from 'react';
import {Box, Text, useInput} from 'ink';
import SelectInput from 'ink-select-input';
import TextInput from 'ink-text-input';
import gradient from 'gradient-string';
import {AppContext} from '../app.mjs';

const headerGradient = gradient(['#8e2de2', '#4a00e0']);

function ProviderStep({providers, onSelect, detected}) {
  return (
    <Box flexDirection="column" gap={1}>
      <Text>{headerGradient('⚙️ Choose your primary provider')}</Text>
      <SelectInput
        items={providers.map(provider => ({
          label: `${detected[provider] ? '🟢' : '🟡'} ${provider}`,
          value: provider,
        }))}
        onSelect={item => onSelect(item.value)}
      />
    </Box>
  );
}

function ModelStep({models, onSelect}) {
  return (
    <Box flexDirection="column" gap={1}>
      <Text>{headerGradient('🧬 Pick the default model')}</Text>
      <SelectInput
        items={models.map(model => ({
          label: `${model.id} (${model.quality || 'n/a'})`,
          value: model.id,
        }))}
        onSelect={item => onSelect(item.value)}
      />
    </Box>
  );
}

function PromptStep({label, value, onSubmit, placeholder}) {
  const [draft, setDraft] = useState(value || '');
  useInput((input, key) => {
    if (key.return) {
      onSubmit(draft.trim() || value || '');
    }
  });
  return (
    <Box flexDirection="column" gap={1}>
      <Text>{headerGradient(label)}</Text>
      <TextInput value={draft} placeholder={placeholder} onChange={setDraft} />
      <Text dimColor>Press Enter to confirm.</Text>
    </Box>
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
    return (
      <ProviderStep
        providers={providerList}
        detected={detected}
        onSelect={provider => {
          setConfig(prev => ({...prev, provider}));
          setStep('model');
        }}
      />
    );
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
    return (
      <ModelStep
        models={augmented}
        onSelect={model => {
          setConfig(prev => ({...prev, model}));
          setStep('endpoint');
        }}
      />
    );
  }

  if (step === 'endpoint') {
    const defaultEndpoint =
      bootstrap.defaultModels[config.provider]?.endpoint || config.llm_endpoint;
    return (
      <PromptStep
        label="🌐 Endpoint URL"
        value={config.llm_endpoint || defaultEndpoint}
        placeholder={defaultEndpoint}
        onSubmit={value => {
          setConfig(prev => ({...prev, llm_endpoint: value || defaultEndpoint}));
          setStep('api');
        }}
      />
    );
  }

  if (step === 'api') {
    const defaultKey =
      bootstrap.defaultModels[config.provider]?.api_key_env || config.api_key_env;
    return (
      <PromptStep
        label="🔐 API key environment variable"
        value={config.api_key_env || defaultKey}
        placeholder={defaultKey}
        onSubmit={value => {
          setConfig(prev => ({...prev, api_key_env: value || defaultKey}));
          setStep('summary');
        }}
      />
    );
  }

  return (
    <Box flexDirection="column" gap={1}>
      <Text>{headerGradient('✨ Configuration summary')}</Text>
      <Text>Provider: {config.provider}</Text>
      <Text>Model: {config.model}</Text>
      <Text>Endpoint: {config.llm_endpoint}</Text>
      <Text>API key env: {config.api_key_env}</Text>
      <Text dimColor>Press Enter to save or Esc to cancel.</Text>
    </Box>
  );
}
