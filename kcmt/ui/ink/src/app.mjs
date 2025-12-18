import React, {useCallback, useEffect, useMemo, useState, useRef} from 'react';
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
    : 'workflow';

const h = React.createElement;
const gradientBanner = gradient(['#4facfe', '#00f2fe']);

const LoadingScreen = ({label = 'Connecting to kcmt magic'} = {}) =>
  h(
    Box,
    {flexDirection: 'column', padding: 1, borderStyle: 'round', borderColor: 'cyan'},
    h(Text, null, gradientBanner('🚀 kcmt')),
    h(Text, {dimColor: true}, 'Mode: TUI (Ink)'),
    h(Text, null, h(Spinner, {type: 'earth'}), ' ', label),
  );

const ErrorScreen = ({message}) =>
  h(
    Box,
    {flexDirection: 'column', padding: 1, borderStyle: 'round', borderColor: 'red'},
    h(Text, null, gradientBanner('🚀 kcmt')),
    h(Text, {dimColor: true}, 'Mode: TUI (Ink)'),
    h(Text, {color: 'redBright'}, `✖ ${message}`),
    h(Text, {dimColor: true}, 'Press Ctrl+C to exit.'),
  );

export default function App() {
  const backend = useMemo(() => createBackendClient(argv), []);
  const [bootstrap, setBootstrap] = useState(null);
  const [status, setStatus] = useState('loading');
  const [error, setError] = useState(null);
  const [view, setView] = useState(initialMode || 'workflow');
  const initialisedRef = useRef(false);

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
    if (status === 'ready' && !initialisedRef.current) {
      initialisedRef.current = true;
      const argvFlags = (bootstrap && bootstrap.argv) || {};
      let requestedMode = null;
      if (argvFlags.benchmark) {
        requestedMode = 'benchmark';
      } else if (argvFlags.configure || argvFlags['configure-all']) {
        requestedMode = 'configure';
      } else if (initialMode) {
        requestedMode = initialMode;
      }
      if (requestedMode) {
        setView(requestedMode);
      }
    }
  }, [status, bootstrap]);

  if (status === 'loading') {
    return h(LoadingScreen, null);
  }

  if (status === 'error' || !bootstrap) {
    return h(ErrorScreen, {message: error?.message || 'Failed to start kcmt backend'});
  }

  const contextValue = {
    backend,
    bootstrap,
    refreshBootstrap,
    argv,
    setView,
  };

  if (view === 'benchmark') {
    return h(
      AppContext.Provider,
      {value: contextValue},
      h(BenchmarkView, {onBack: () => setView('menu')}),
    );
  }

  if (view === 'configure') {
    return h(
      AppContext.Provider,
      {value: contextValue},
      h(ConfigureView, {onBack: () => setView('menu'), showAdvanced: Boolean(argv['configure-all'])}),
    );
  }

  if (view === 'workflow') {
    return h(
      AppContext.Provider,
      {value: contextValue},
      h(WorkflowView, {onBack: () => setView('menu')}),
    );
  }

  return h(
    AppContext.Provider,
    {value: contextValue},
    h(MainMenu, {
      onNavigate: mode => {
        if (mode === 'exit') {
          process.exit(0);
        }
        setView(mode);
      },
    }),
  );
}
