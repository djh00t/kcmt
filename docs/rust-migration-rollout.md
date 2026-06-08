# Rust Migration Rollout and Rollback Controls

## Rollout Stages

1. **Baseline mode**: set `KCMT_RUNTIME=python` and validate expected Python behavior.
2. **Default covered workflows**: with `KCMT_RUNTIME=auto`, covered non-TUI commands route to Rust by default.
3. **Canary mode**: set `KCMT_RUNTIME=auto` and `KCMT_RUST_CANARY=1` to route any remaining unsupported invocation to Rust for investigation.
4. **Widening**: add new default-covered invocations only after CI canary + parity gates stay green.

## Safety Controls

- Keep Python CLI path available as an immediate fallback using `KCMT_RUNTIME=python`.
- Require canary probe evidence and parity evidence before widening rollout.
- Use `KCMT_RUNTIME_TRACE=1` during canary investigations for machine-readable routing diagnostics.
- Restrict optional TUI behavior to TTY sessions.
- Covered default Rust commands currently include non-TTY default workflow
  invocations, `--oneshot`, `--file`, `status`, non-interactive
  `--configure`/`--configure-all` invocations with explicit override flags,
  `--list-models`, `--verify-keys`, `--benchmark`, and `benchmark runtime`.

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

The runtime report includes one baseline row plus five optimization rows with
timings, throughput, quality scores, failures, and the next bottleneck label for
each iteration.

### Rust-Only Live LLM Smoke

Build and run the Rust binary directly when validating production cutover:

```bash
cargo build --release --manifest-path rust/Cargo.toml -p kcmt-cli
./rust/target/release/kcmt --help
```

Run live provider smoke checks only when the relevant key is already present in
the environment. Keep `--no-auto-push` for smoke runs unless the branch is meant
to publish commits.

```bash
test -n "$OPENAI_API_KEY" && \
  ./rust/target/release/kcmt --provider openai --api-key-env OPENAI_API_KEY \
    --model gpt-5.4-mini --file path/to/changed-file --no-auto-push --repo-path .

test -n "$ANTHROPIC_API_KEY" && \
  ./rust/target/release/kcmt --provider anthropic --api-key-env ANTHROPIC_API_KEY \
    --model claude-sonnet-4-20250514 --file path/to/changed-file --no-auto-push --repo-path .

test -n "$OPENAI_API_KEY" && \
  ./rust/target/release/kcmt --provider openai --api-key-env OPENAI_API_KEY \
    --batch --batch-model gpt-5.4-mini --batch-timeout 900 \
    --no-auto-push --repo-path .
```

Expected production-readiness behavior:

- Direct OpenAI and Anthropic calls fail fast for missing keys, invalid models,
  malformed provider output, and provider 4xx/5xx responses.
- OpenAI batch mode uploads every file prompt before committing, maps responses
  by `custom_id`, commits valid responses, and reports invalid or missing
  responses per file.
- Debug/profile output and status snapshots record provider names, model names,
  endpoints, and API key environment variable names, but not API key values.
- To make `kcmt`, `commit`, and `kc` point at Rust without the Python wrapper,
  prepend `$(pwd)/rust/target/release` to `PATH` or install those release
  binaries into the operator's normal bin directory.

## Rollback Procedure

1. Set `KCMT_RUNTIME=python` in runtime environment.
2. Optionally keep `KCMT_RUNTIME_TRACE=1` enabled for explicit confirmation.
3. Re-run high-usage workflow parity checks.
4. Record rollback reason and affected scenarios in `tasks-validation.md`.
5. Open follow-up task for the root cause before attempting re-cutover.
