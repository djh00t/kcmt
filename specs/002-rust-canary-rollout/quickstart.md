# Quickstart: Rust Runtime Canary Rollout

## Prerequisites

- Python/uv dev environment installed
- Rust toolchain installed
- Project dependencies synced:

```bash
uv sync --all-extras --dev
uv pip install -e .
```

## 1) Baseline Verification (Python Runtime)

```bash
KCMT_RUNTIME=python uv run kcmt --help
KCMT_RUNTIME=python uv run kcmt status --repo-path .
```

Expected: commands run via Python path with normal behavior.

## 2) Build Rust Candidate

```bash
cargo build --release --manifest-path rust/Cargo.toml -p kcmt-cli
```

## 3) Canary Verification (Auto + Canary Enabled)

```bash
KCMT_RUNTIME=auto KCMT_RUST_CANARY=1 KCMT_RUST_BIN="$(pwd)/rust/target/release/kcmt" KCMT_RUNTIME_TRACE=1 uv run kcmt --help 1>/tmp/kcmt-help.out 2>/tmp/kcmt-help.trace
```

Expected:
- Exit code is `0`.
- `/tmp/kcmt-help.trace` contains one JSON trace line with
  `selected_runtime` = `"rust"`.

## 4) Fallback Verification (Missing Rust Binary)

```bash
KCMT_RUNTIME=auto KCMT_RUST_CANARY=1 KCMT_RUST_BIN="/tmp/does-not-exist-kcmt" KCMT_RUNTIME_TRACE=1 uv run kcmt --help 1>/tmp/kcmt-help-fallback.out 2>/tmp/kcmt-help-fallback.trace
```

Expected:
- Exit code is `0`.
- Trace shows `selected_runtime` = `"python"` and fallback reason.

## 5) Rollback Verification

```bash
KCMT_RUNTIME=python KCMT_RUST_CANARY=1 KCMT_RUNTIME_TRACE=1 uv run kcmt --help 1>/tmp/kcmt-help-rollback.out 2>/tmp/kcmt-help-rollback.trace
```

Expected:
- Trace shows `selected_runtime` = `"python"`.
- Rollback knob dominates canary settings.

## 6) CI-Equivalent Canary Probe

```bash
uv run python scripts/canary/runtime_canary_probe.py
```

Expected: all scenarios pass and script exits `0`.
