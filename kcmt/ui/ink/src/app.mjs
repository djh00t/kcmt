import React, {useCallback, useEffect, useMemo, useState, useRef} from 'react';
import {Box, Text} from 'ink';
import minimist from 'minimist';
import gradient from 'gradient-string';
import {createBackendClient} from './backend-client.mjs';
import WorkflowView from './components/workflow-view.mjs';
import {AppContext} from './app-context.mjs';

// The Python wrapper invokes: `node index.mjs -- <kcmt args...>`.
// `minimist` treats `--` as end-of-options, so we must strip it first.
const rawArgs = process.argv.slice(2);
const dashDashIndex = rawArgs.indexOf('--');
const effectiveArgs = dashDashIndex >= 0 ? rawArgs.slice(dashDashIndex + 1) : rawArgs;
const argv = minimist(effectiveArgs);
const initialMode = argv.benchmark
  ? 'benchmark'
  : argv.configure || argv['configure-all']
    ? 'configure'
    : 'workflow';

const VIEW_LOADERS = {
  menu: () => import('./components/main-menu.mjs'),
  benchmark: () => import('./components/benchmark-view.mjs'),
  configure: () => import('./components/configure-view.mjs'),
};

const h = React.createElement;
const gradientBanner = gradient(['#4facfe', '#00f2fe']);

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
  const [viewComponent, setViewComponent] = useState(() =>
    (initialMode || 'workflow') === 'workflow' ? WorkflowView : null,
  );
  const [viewError, setViewError] = useState(null);
  const initialisedRef = useRef(false);
  const viewCacheRef = useRef(new Map());

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

  useEffect(() => {
    let cancelled = false;
    setViewError(null);

    // Make the default workflow experience instant: render immediately,
    // let bootstrap finish in the background.
    if (view === 'workflow') {
      setViewComponent(() => WorkflowView);
      viewCacheRef.current.set('workflow', WorkflowView);
      return () => {
        cancelled = true;
      };
    }

    const cached = viewCacheRef.current.get(view);
    if (cached) {
      setViewComponent(() => cached);
      return;
    }

    setViewComponent(null);

    const loader = VIEW_LOADERS[view];
    if (!loader) {
      setViewError(new Error(`Unknown view: ${view}`));
      return;
    }

    loader()
      .then(mod => {
        const component = mod && mod.default ? mod.default : null;
        if (!component) {
          throw new Error(`View module missing default export: ${view}`);
        }
        viewCacheRef.current.set(view, component);
        if (!cancelled) {
          setViewComponent(() => component);
        }
      })
      .catch(err => {
        if (!cancelled) {
          setViewError(err instanceof Error ? err : new Error(String(err)));
        }
      });

    return () => {
      cancelled = true;
    };
  }, [view]);

  if (status === 'error') {
    return h(ErrorScreen, {message: error?.message || 'Failed to start kcmt backend'});
  }

  if (viewError) {
    return h(ErrorScreen, {message: viewError.message || 'Failed to load kcmt UI'});
  }

  if (!viewComponent) {
    // Intentionally render nothing while a lazy view module loads.
    return null;
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
      h(viewComponent, {onBack: () => setView('menu')}),
    );
  }

  if (view === 'configure') {
    return h(
      AppContext.Provider,
      {value: contextValue},
      h(viewComponent, {onBack: () => setView('menu'), showAdvanced: Boolean(argv['configure-all'])}),
    );
  }

  if (view === 'workflow') {
    return h(
      AppContext.Provider,
      {value: contextValue},
      h(viewComponent, {onBack: () => setView('menu')}),
    );
  }

  return h(
    AppContext.Provider,
    {value: contextValue},
    h(viewComponent, {
      onNavigate: mode => {
        if (mode === 'exit') {
          process.exit(0);
        }
        setView(mode);
      },
    }),
  );
}
