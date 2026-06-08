import React from 'react';

export const AppContext = React.createContext({
  backend: null,
  bootstrap: null,
  refreshBootstrap: () => Promise.resolve(),
  argv: {},
});
