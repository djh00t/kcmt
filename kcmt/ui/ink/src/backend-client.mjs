import {spawn, spawnSync} from 'child_process';
import EventEmitter from 'events';

const PYTHON_EXECUTABLE = process.env.KCMT_PYTHON_EXECUTABLE || 'python3';
const BACKEND_MODULE = process.env.KCMT_BACKEND_MODULE || 'kcmt.ink_backend';

function parseStdout(stdout) {
  const lines = String(stdout || '')
    .split(/\r?\n/)
    .map(line => line.trim())
    .filter(Boolean);
  let payload = null;
  for (const line of lines) {
    try {
      const message = JSON.parse(line);
      if (message.event === 'error') {
        const detail = message.payload?.message || line;
        const err = new Error(detail);
        err.payload = message.payload;
        throw err;
      }
      if (Object.hasOwn(message, 'payload')) {
        payload = message.payload;
      }
    } catch (err) {
      throw err instanceof Error ? err : new Error(`Failed to parse backend output: ${line}`);
    }
  }
  return payload;
}

export function createBackendClient(argv) {
  const repoPath = argv['repo-path'] || argv.repoPath || process.cwd();

  function run(action, payload = {}) {
    const encoded = JSON.stringify(payload);
    const args = ['-m', BACKEND_MODULE, action, '--repo-path', repoPath, '--payload', encoded];
    const result = spawnSync(PYTHON_EXECUTABLE, args, {encoding: 'utf-8'});
    if (result.error) {
      throw result.error;
    }
    if (result.status !== 0) {
      const stderr = (result.stderr || '').trim();
      const stdout = (result.stdout || '').trim();
      const message = stderr || stdout || `Backend exited with code ${result.status}`;
      throw new Error(message);
    }
    return parseStdout(result.stdout);
  }

  function stream(action, payload = {}) {
    const emitter = new EventEmitter();
    const encoded = JSON.stringify(payload);
    const args = ['-m', BACKEND_MODULE, action, '--repo-path', repoPath, '--payload', encoded];
    const child = spawn(PYTHON_EXECUTABLE, args, {
      stdio: ['ignore', 'pipe', 'pipe'],
    });

    let buffer = '';
    child.stdout.on('data', chunk => {
      buffer += chunk.toString();
      let idx = buffer.indexOf('\n');
      while (idx !== -1) {
        const line = buffer.slice(0, idx).trim();
        buffer = buffer.slice(idx + 1);
        if (line) {
          try {
            const message = JSON.parse(line);
            emitter.emit('event', message);
            if (message.event === 'error') {
              const detail = message.payload?.message || 'Backend error';
              emitter.emit('error', new Error(detail));
            }
          } catch (error) {
            emitter.emit('error', error instanceof Error ? error : new Error(`Failed to parse backend output: ${line}`));
          }
        }
        idx = buffer.indexOf('\n');
      }
    });

    let errBuffer = '';
    child.stderr.on('data', chunk => {
      errBuffer += chunk.toString();
      let idx = errBuffer.indexOf('\n');
      while (idx !== -1) {
        const line = errBuffer.slice(0, idx).trim();
        errBuffer = errBuffer.slice(idx + 1);
        if (line) {
          try {
            const message = JSON.parse(line);
            emitter.emit('event', message);
            if (message.event === 'error') {
              const detail = message.payload?.message || 'Backend error';
              emitter.emit('error', new Error(detail));
            }
          } catch (error) {
            emitter.emit('stderr', line);
          }
        }
        idx = errBuffer.indexOf('\n');
      }
    });

    child.on('close', code => {
      emitter.emit('close', code);
      if (code === 0) {
        emitter.emit('done');
      } else if (!child.killed) {
        emitter.emit('error', new Error(`Backend exited with code ${code}`));
      }
    });

    emitter.cancel = () => {
      if (!child.killed) {
        child.kill();
      }
    };

    return emitter;
  }

  return {
    repoPath,
    run,
    stream,
    bootstrap: () => run('bootstrap', {argv}),
    saveConfig: config => Promise.resolve(run('save-config', {config})),
    savePreferences: preferences => Promise.resolve(run('save-preferences', {preferences})),
    runBenchmark: options => stream('benchmark', options),
    runWorkflow: options => stream('workflow', options),
  };
}
