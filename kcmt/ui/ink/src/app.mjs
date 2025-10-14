import React, {useCallback, useEffect, useMemo, useState} from 'react';
import {Box, Text} from 'ink';
import Spinner from 'ink-spinner';
import minimist from 'minimist';
import gradient from 'gradient-string';
import {createBackendClient} from './backend-client.mjs';
import MainMenu from './components/main-menu.mjs';
import WorkflowView from './components/workflow-view.mjs';
import BenchmarkView from './components/benchmark-view.mjs';
import ConfigureView from './components/configure-view.mjs';

export const AppContext = React.createContext({
  backend: null,
  bootstrap: null,
  refreshBootstrap: () => Promise.resolve(),
  argv: {},
});

const argv = minimist(process.argv.slice(2));
const initialMode = argv.benchmark
  ? 'benchmark'
  : argv.configure || argv['configure-all']
    ? 'configure'
    : null;

const gradientBanner = gradient(['#4facfe', '#00f2fe']);

const LoadingScreen = ({label = 'Connecting to kcmt magic'} = {}) => (
  <Box flexDirection="column" padding={1} borderStyle="round" borderColor="cyan">
    <Text>{gradientBanner('🚀 kcmt')}</Text>
    <Text>
      <Spinner type="earth" /> {label}
    </Text>
  </Box>
);

const ErrorScreen = ({message}) => (
  <Box flexDirection="column" padding={1} borderStyle="round" borderColor="red">
    <Text color="redBright">✖ {message}</Text>
    <Text dimColor>
      Press Ctrl+C to exit.
    </Text>
  </Box>
);

export default function App() {
  const backend = useMemo(() => createBackendClient(argv), []);
  const [bootstrap, setBootstrap] = useState(null);
  const [status, setStatus] = useState('loading');
  const [error, setError] = useState(null);
  const [view, setView] = useState(initialMode || 'menu');

  const refreshBootstrap = useCallback(async () => {
    setStatus('loading');
    setError(null);
    try {
      const data = await backend.bootstrap();
      setBootstrap(data);
      setStatus('ready');
      return data;
    } catch (err) {
      setError(err instanceof Error ? err : new Error(String(err)));
      setStatus('error');
      throw err;
    }
  }, [backend]);

  useEffect(() => {
    refreshBootstrap().catch(() => null);
  }, [refreshBootstrap]);

  useEffect(() => {
    if (initialMode) {
      setView(initialMode);
    }
  }, []);

  if (status === 'loading') {
    return <LoadingScreen />;
  }

  if (status === 'error' || !bootstrap) {
    return <ErrorScreen message={error?.message || 'Failed to start kcmt backend'} />;
  }

  const contextValue = {
    backend,
    bootstrap,
    refreshBootstrap,
    argv,
    setView,
  };

  if (view === 'benchmark') {
    return (
      <AppContext.Provider value={contextValue}>
        <BenchmarkView onBack={() => setView('menu')} />
      </AppContext.Provider>
    );
  }

  if (view === 'configure') {
    return (
      <AppContext.Provider value={contextValue}>
        <ConfigureView
          onBack={() => setView('menu')}
          showAdvanced={Boolean(argv['configure-all'])}
        />
      </AppContext.Provider>
    );
  }

  if (view === 'workflow') {
    return (
      <AppContext.Provider value={contextValue}>
        <WorkflowView onBack={() => setView('menu')} />
      </AppContext.Provider>
    );
  }

  return (
    <AppContext.Provider value={contextValue}>
      <MainMenu
        onNavigate={mode => {
          if (mode === 'exit') {
            process.exit(0);
          }
          setView(mode);
        }}
      />
    </AppContext.Provider>
  );
}
