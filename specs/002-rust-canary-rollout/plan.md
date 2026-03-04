# Implementation Plan: Rust Runtime Canary Rollout and Observability

**Branch**: `002-rust-canary-rollout` | **Date**: 2026-03-04 | **Spec**: [`specs/002-rust-canary-rollout/spec.md`](./spec.md)  
**Input**: Feature specification from `/specs/002-rust-canary-rollout/spec.md`

## Summary

Add a dedicated canary rollout layer on top of completed Rust parity work by:
1) introducing opt-in machine-readable runtime decision tracing in the Python wrapper,
2) adding CI canary smoke validation for wrapper-routed Rust execution, and
3) documenting stage/rollback operations. Default runtime behavior remains unchanged.

## Technical Context

**Language/Version**: Python 3.12 (primary wrapper and tests), Rust stable binaries (already implemented)  
**Primary Dependencies**: `pytest`, `uv`, GitHub Actions, existing Rust workspace binaries (`kcmt`, `commit`, `kc`)  
**Storage**: N/A (ephemeral trace output only)  
**Testing**: `pytest`, `make check`, GitHub Actions workflows  
**Target Platform**: `ubuntu-latest`, `macos-latest`, `windows-latest` CI + local macOS/Linux developer environments  
**Project Type**: CLI tool with Python wrapper and Rust runtime candidate  
**Performance Goals**: Canary smoke workflow completes within 15 minutes on Ubuntu; no added latency in default mode path  
**Baseline Corpus**: Existing parity probes + high-usage workflow catalog and exit/error matrix from `001-rust-parity-migration`  
**Constraints**: Preserve CLI/exit compatibility, keep Python fallback safety, no secrets in traces/artifacts, no default cutover in this feature  
**Scale/Scope**: Wrapper runtime routing, CI workflows, tests, rollout docs, feature-specific artifacts under `specs/002-rust-canary-rollout`

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- [x] Principle I: CLI contract compatibility is preserved or explicitly versioned  
  Decision: Trace output is opt-in via env and emitted to stderr-only; default mode behavior remains unchanged.
- [x] Principle II: Git safety/atomic behavior is preserved with parity strategy  
  Decision: No changes to commit staging semantics; canary probes target wrapper routing/CLI contracts only.
- [x] Principle III: Required tests and strict quality gates are defined  
  Decision: Add wrapper trace/fallback unit tests and CI canary smoke workflow; keep `make check` as merge gate.
- [x] Principle IV: Performance claims include baseline corpus and metrics  
  Decision: Scope includes CI runtime budget and deterministic probe matrix rather than new benchmark claims.
- [x] Principle V: Secrets/config precedence/error handling constraints are covered  
  Decision: Trace schema excludes sensitive env values and documents controlled env precedence for rollout.

No constitution violations identified.

## Project Structure

### Documentation (this feature)

```text
specs/002-rust-canary-rollout/
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── contracts/
└── tasks.md
```

### Source Code (repository root)

```text
kcmt/
└── main.py

tests/
└── test_main_entrypoint.py

.github/workflows/
├── rust-canary-smoke.yml
├── rust-parity-matrix.yml
└── keystone-assimilation.yml

docs/
└── rust-migration-rollout.md

scripts/canary/
└── runtime_canary_probe.py
```

**Structure Decision**: Use existing single-project Python/Rust hybrid layout; add one
focused canary probe script and update current CI/docs rather than introducing new service
or package boundaries.

## Post-Design Constitution Check

- [x] Principle I: Trace and canary behavior are opt-in; default CLI contracts unchanged.
- [x] Principle II: No changes to git mutation flows or commit atomicity logic.
- [x] Principle III: Wrapper tests + CI canary probes + `make check` gate are in scope.
- [x] Principle IV: Uses explicit scenario corpus and CI budget target; no unsubstantiated
  performance claims.
- [x] Principle V: Trace schema excludes sensitive values and keeps runtime precedence explicit.

## Complexity Tracking

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| None | N/A | N/A |
