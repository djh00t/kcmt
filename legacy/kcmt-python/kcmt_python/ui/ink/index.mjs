import React from 'react';
import {render} from 'ink';
import App from './src/app.mjs';

function primeViewport() {
  let restore = () => {};
  try {
    if (!process.stdout || !process.stdout.isTTY) return;
    const disable = String(process.env.KCMT_NO_TUI_PRIME || '').toLowerCase();
    if (disable && ['1', 'true', 'yes', 'on'].includes(disable)) return;
    const useAltScreen = String(process.env.KCMT_NO_ALT_SCREEN || '').toLowerCase();
    const enableAlt = !(useAltScreen && ['1', 'true', 'yes', 'on'].includes(useAltScreen));

    if (enableAlt) {
      process.stdout.write('\u001B[?1049h'); // enter alt screen buffer
      restore = () => {
        try {
          process.stdout.write('\u001B[?1049l'); // leave alt screen buffer
        } catch {
          /* no-op */
        }
      };
    }

    process.stdout.write('\u001B[2J'); // clear screen contents
    // Do not clear scrollback by default; it makes debugging crashes painful.
    // Opt in via KCMT_CLEAR_SCROLLBACK=1.
    const clearScrollback = String(process.env.KCMT_CLEAR_SCROLLBACK || '').toLowerCase();
    if (clearScrollback && ['1', 'true', 'yes', 'on'].includes(clearScrollback)) {
      process.stdout.write('\u001B[3J'); // clear scrollback
    }
    process.stdout.write('\u001B[H'); // move cursor to home

    // Make sure we always restore the user's terminal on exit/crash.
    process.on('exit', () => restore());
    process.on('SIGINT', () => {
      restore();
      process.exit(130);
    });

    const crash = err => {
      try {
        restore();
      } catch {
        /* no-op */
      }
      // Print after restoring so it lands in the normal terminal buffer.
      try {
        // eslint-disable-next-line no-console
        console.error(err && err.stack ? err.stack : err);
      } catch {
        /* no-op */
      }
      process.exit(1);
    };
    process.on('uncaughtException', crash);
    process.on('unhandledRejection', crash);
  } catch {
    // no-op on environments without TTY
  }

  return restore;
}

const restoreViewport = primeViewport() || (() => {});
try {
  render(React.createElement(App));
} catch (err) {
  try {
    restoreViewport();
  } catch {
    /* no-op */
  }
  // eslint-disable-next-line no-console
  console.error(err && err.stack ? err.stack : err);
  process.exit(1);
}
