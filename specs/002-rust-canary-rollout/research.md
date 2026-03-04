# Research: Rust Runtime Canary Rollout and Observability

## Decision 1: Runtime trace records are opt-in JSON lines emitted to stderr

- Decision: Introduce `KCMT_RUNTIME_TRACE=1` to emit one JSON record per invocation to
  stderr. Keep stdout unchanged.
- Rationale: Preserves CLI contract compatibility and script output stability while still
  enabling machine-readable diagnostics in CI and local troubleshooting.
- Alternatives considered:
  - Always-on trace logging: rejected due to output contract risk and noise.
  - Stdout trace output: rejected because it can break automation expecting command output.
  - File-only trace output: rejected as default because it adds file lifecycle complexity.

## Decision 2: Trace schema includes routing reason and safe runtime metadata only

- Decision: Required fields are `selected_runtime`, `decision_reason`, `runtime_mode`,
  `canary_enabled`, `rust_binary`, and `rust_binary_exists`.
- Rationale: These fields fully explain route/fallback outcomes while avoiding sensitive
  environment or provider details.
- Alternatives considered:
  - Include full environment snapshot: rejected for secret leakage risk.
  - Minimal boolean-only traces: rejected because root-cause diagnosis is too weak.

## Decision 3: Canary validation uses a dedicated probe script

- Decision: Add `scripts/canary/runtime_canary_probe.py` that runs deterministic command
  scenarios and validates trace JSON plus exit-code expectations.
- Rationale: Keeps workflow logic maintainable, testable, and reusable across local and CI
  runs.
- Alternatives considered:
  - Inline bash-only workflow probes: rejected due to parsing fragility and readability.
  - Reuse parity workflow only: rejected because canary-specific wrapper behavior needs a
  focused gate.

## Decision 4: Add a dedicated GitHub Actions canary workflow

- Decision: Introduce `.github/workflows/rust-canary-smoke.yml` for pull requests touching
  runtime routing or rollout docs/scripts.
- Rationale: Separates canary signal from broad parity matrix and enables targeted budgeted
  validation.
- Alternatives considered:
  - Extend parity matrix job for canary: rejected to avoid overloading one workflow with
  mixed concerns.
  - Local-only canary checks: rejected because merge gating requires CI evidence.

## Decision 5: Rollout stages are documented without default runtime cutover

- Decision: Keep default runtime unchanged in this feature and document explicit stage
  commands (`baseline`, `canary`, `rollback`) in docs and quickstart.
- Rationale: Reduces risk while building operational confidence through observable canary
  data.
- Alternatives considered:
  - Immediate default switch to `auto`: rejected as too risky before canary evidence.
  - Undocumented ad-hoc rollout: rejected due to operational inconsistency.
