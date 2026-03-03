# Rust Migration Rollout and Rollback Controls

## Rollout Stages

1. **Shadow mode**: run Rust parity jobs in CI without changing user runtime.
2. **Canary mode**: enable Rust runtime by setting `KCMT_RUNTIME=rust` for selected users/jobs.
3. **Progressive default**: set `KCMT_RUNTIME=auto` and monitor parity/quality/performance gates.
4. **Default cutover**: make Rust runtime the default only after SC-001..SC-006 pass evidence is complete.

## Safety Controls

- Keep Python CLI path available as an immediate fallback.
- Require parity evidence from `tasks-validation.md` before widening rollout.
- Restrict optional TUI behavior to TTY sessions.

## Rollback Procedure

1. Set `KCMT_RUNTIME=python` in runtime environment.
2. Re-run high-usage workflow parity checks.
3. Record rollback reason and affected scenarios in `tasks-validation.md`.
4. Open follow-up task for the root cause before attempting re-cutover.
