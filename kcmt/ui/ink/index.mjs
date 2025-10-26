import React from 'react';
import {render} from 'ink';
import App from './src/app.mjs';

function primeViewport() {
  try {
    if (!process.stdout || !process.stdout.isTTY) return;
    const disable = String(process.env.KCMT_NO_TUI_PRIME || '').toLowerCase();
    if (disable && ['1', 'true', 'yes', 'on'].includes(disable)) return;
    process.stdout.write('\u001B[2J'); // clear screen contents
    process.stdout.write('\u001B[H'); // move cursor to home without using alt screen
  } catch {
    // no-op on environments without TTY
  }
}

primeViewport();
render(React.createElement(App));
