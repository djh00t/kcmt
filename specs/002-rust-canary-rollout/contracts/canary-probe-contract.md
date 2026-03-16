# Canary Probe Contract

## Scope

The canary probe validates runtime routing and exit behavior for wrapper invocations
under explicit environment configurations.

## Inputs

- Scenario list in `scripts/canary/runtime_canary_probe.py`
- Built rust binary at `rust/target/release/kcmt` (or `KCMT_RUST_BIN` override)
- Optional aliases `commit`, `kc` when available

## Required Assertions per Scenario

1. Process exit code matches expected baseline.
2. Runtime trace record parses as JSON matching
   `contracts/runtime-trace.schema.json` required fields.
3. `selected_runtime` matches expected runtime (`python`/`rust`).
4. `decision_reason` matches expected reason.

## Minimum Scenario Set

- `default_python_help`: no canary env, expect Python runtime.
- `auto_canary_missing_bin`: auto+canary enabled with missing binary, expect Python fallback.
- `auto_canary_rust_help`: auto+canary enabled with valid binary, expect Rust runtime.
- `forced_python_rollback`: `KCMT_RUNTIME=python`, expect Python runtime.

## Failure Semantics

- Any scenario mismatch is a hard failure (non-zero probe exit).
- Probe output must list scenario IDs and mismatched fields for triage.
