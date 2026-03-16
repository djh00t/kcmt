# Rust Migration Rollout and Rollback Controls

## Rollout Stages

1. **Baseline mode**: set `KCMT_RUNTIME=python` and validate expected Python behavior.
2. **Canary mode**: set `KCMT_RUNTIME=auto` and `KCMT_RUST_CANARY=1` for selected users/jobs.
3. **Progressive default planning**: keep canary enabled cohorts expanding only after CI canary + parity gates stay green.
4. **Default cutover (future feature)**: switch defaults only after canary evidence and rollback drills are complete.

## Safety Controls

- Keep Python CLI path available as an immediate fallback using `KCMT_RUNTIME=python`.
- Require canary probe evidence and parity evidence before widening rollout.
- Use `KCMT_RUNTIME_TRACE=1` during canary investigations for machine-readable routing diagnostics.
- Restrict optional TUI behavior to TTY sessions.

## Operator Commands

### Baseline

```bash
KCMT_RUNTIME=python uv run kcmt --help
KCMT_RUNTIME=python uv run kcmt status --help
```

### Canary

```bash
KCMT_RUNTIME=auto KCMT_RUST_CANARY=1 KCMT_RUST_BIN="$(pwd)/rust/target/release/kcmt" KCMT_RUNTIME_TRACE=1 uv run kcmt --help
```

### CI-Equivalent Canary Probe

```bash
uv run python scripts/canary/runtime_canary_probe.py --rust-bin \"$(pwd)/rust/target/release/kcmt\"
```

### Runtime Benchmark Evidence

```bash
python scripts/benchmark/generate_uncommitted_repo.py --file-count 1000 --json
KCMT_RUST_BIN="$(pwd)/rust/target/release/kcmt" uv run kcmt benchmark runtime --repo-path "$REPO" --runtime both --json
```

## Rollback Procedure

1. Set `KCMT_RUNTIME=python` in runtime environment.
2. Optionally keep `KCMT_RUNTIME_TRACE=1` enabled for explicit confirmation.
3. Re-run high-usage workflow parity checks.
4. Record rollback reason and affected scenarios in `tasks-validation.md`.
5. Open follow-up task for the root cause before attempting re-cutover.
