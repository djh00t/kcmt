import React from 'react';
import {render} from 'ink';
import App from './src/app.mjs';

function primeViewport() {
  try {
    if (!process.stdout || !process.stdout.isTTY) return;
    const rows = Math.max(1, Number(process.stdout.rows) || 24);
    const disable = String(process.env.KCMT_NO_TUI_PRIME || '').toLowerCase();
    if (disable && ['1', 'true', 'yes', 'on'].includes(disable)) return;
    process.stdout.write('\n'.repeat(rows));
    process.stdout.write(`\u001B[${rows}A`);
  } catch {
    // no-op on environments without TTY
  }
}

primeViewport();
render(React.createElement(App));
