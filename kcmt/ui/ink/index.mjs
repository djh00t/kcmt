import React from 'react';
import {render} from 'ink';
import App from './src/app.mjs';

function primeViewport() {
  try {
    if (!process.stdout || !process.stdout.isTTY) return;
    const disable = String(process.env.KCMT_NO_TUI_PRIME || '').toLowerCase();
    if (disable && ['1', 'true', 'yes', 'on'].includes(disable)) return;
    const useAltScreen = String(process.env.KCMT_NO_ALT_SCREEN || '').toLowerCase();
    const enableAlt = !(useAltScreen && ['1', 'true', 'yes', 'on'].includes(useAltScreen));

    if (enableAlt) {
      process.stdout.write('\u001B[?1049h'); // enter alt screen buffer
    }
    process.stdout.write('\u001B[2J'); // clear screen contents
    process.stdout.write('\u001B[3J'); // clear scrollback
    process.stdout.write('\u001B[H'); // move cursor to home

    if (enableAlt) {
      const restore = () => {
        try {
          process.stdout.write('\u001B[?1049l'); // leave alt screen buffer
        } catch {
          /* no-op */
        }
      };
      process.on('exit', restore);
      process.on('SIGINT', () => {
        restore();
        process.exit(130);
      });
    }
  } catch {
    // no-op on environments without TTY
  }
}

primeViewport();
render(React.createElement(App));
