# Implementation Plan: Rust CLI Feature Parity and Runtime Benchmark Mode

**Branch**: `003-bring-rust-cli` | **Date**: 2026-03-15 | **Spec**: [`specs/003-bring-rust-cli/spec.md`](./spec.md)  
**Input**: Feature specification from `/specs/003-bring-rust-cli/spec.md`

## Summary

Close the remaining Rust migration gap by making the Rust CLI a real peer of the
Python CLI for in-scope workflows, then add a separate runtime benchmark mode that
compares Python and Rust on a shared repository corpus without disturbing the
existing provider-quality benchmark. The implementation will keep shell `git` as
the default backend for parity, treat `gitoxide` as an optional future backend
behind explicit gating, and preserve legacy benchmark flags by mapping them to the
provider benchmark mode.

## Technical Context

**Language/Version**: Python 3.12, Rust 1.78 workspace  
**Primary Dependencies**: Python `argparse`, `pytest`, `uv`; Rust `clap`, `serde`,
`serde_json`, `anyhow`, `thiserror`, existing workspace crates (`kcmt-cli`,
`kcmt-core`, `kcmt-bench`, optional `kcmt-tui`)  
**Storage**: Local filesystem only (`~/.config/kcmt`, repo-local git working tree,
benchmark artifacts under repo-scoped state directories)  
**Testing**: `pytest`, `cargo test`, `make check`, GitHub Actions parity/canary
workflows, new runtime benchmark regression fixtures  
**Target Platform**: Local macOS/Linux developer environments plus GitHub Actions on
`ubuntu-latest`, `macos-latest`, and `windows-latest`  
**Project Type**: Hybrid CLI application with Python compatibility wrapper and Rust
runtime candidate  
**Performance Goals**: Produce reproducible Python-vs-Rust benchmark output on a
synthetic 1,000-file uncommitted repo and a realistic checked-in corpus; demonstrate
measurable Rust improvement for at least one core local workflow while preserving
contract compatibility  
**Baseline Corpus**: `specs/003-bring-rust-cli/validation/runtime-corpus.md`
covering a synthetic large-uncommitted repo fixture plus a realistic checked-in
fixture and the workflow parity catalog  
**Constraints**: Preserve CLI contract compatibility, keep provider benchmark
behavior intact, avoid live LLM calls in runtime benchmarking, keep git safety for
file-scoped operations, and gate any `gitoxide` adoption behind explicit parity
validation  
**Scale/Scope**: Rust CLI argument/dispatch parity for in-scope workflows, benchmark
mode separation, runtime benchmark artifacts and schemas, repo corpus fixtures, docs,
and regression tests

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- [x] Principle I: CLI contract compatibility is preserved or explicitly versioned  
  Decision: Rust work is scoped around an explicit workflow contract catalog and
  preserves legacy benchmark flags via compatibility mapping.
- [x] Principle II: Git safety/atomic behavior is preserved with parity strategy  
  Decision: Shell `git` remains the default backend; `gitoxide` is treated as an
  implementation option only after parity validation.
- [x] Principle III: Required tests and strict quality gates are defined  
  Decision: Add Python and Rust contract tests, runtime benchmark schema tests, and
  keep `make check` plus parity validation as required merge gates.
- [x] Principle IV: Performance claims include baseline corpus and metrics  
  Decision: Runtime benchmarking is corpus-driven, offline-capable, and reported via
  a separate versioned schema.
- [x] Principle V: Secrets/config precedence/error handling constraints are covered  
  Decision: Runtime benchmarking excludes live provider calls, preserves config
  precedence, and treats missing-binary or repo errors as explicit reportable outcomes.

No constitution violations identified.

## Project Structure

### Documentation (this feature)

```text
specs/003-bring-rust-cli/
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── contracts/
│   ├── runtime-benchmark.schema.json
│   └── workflow-parity-contract.md
├── validation/
│   ├── workflow-parity-catalog.md
│   ├── runtime-corpus.md
│   └── exit-error-baseline.md
└── tasks.md
```

### Source Code (repository root)

```text
kcmt/
├── benchmark.py
├── cli.py
├── legacy_cli.py
└── main.py

rust/
├── crates/kcmt-cli/
│   └── src/
│       ├── args.rs
│       ├── lib.rs
│       └── commands/
│           ├── benchmark.rs
│           └── status.rs
├── crates/kcmt-bench/
│   └── src/
│       ├── export.rs
│       ├── model.rs
│       └── runner.rs
└── crates/kcmt-core/
    └── src/
        └── git/

scripts/
└── benchmark/
    └── generate_uncommitted_repo.py

tests/
├── test_benchmark.py
├── test_cli.py
├── test_generate_uncommitted_repo.py
└── test_main_entrypoint.py

docs/
├── benchmark.md
└── rust-migration-rollout.md
```

**Structure Decision**: Keep the existing hybrid Python/Rust monorepo layout. Extend
the Rust CLI and benchmark crates directly, preserve Python compatibility surfaces,
and add feature-specific validation docs instead of introducing a new service or
separate benchmarking package.

## Post-Design Constitution Check

- [x] Principle I: Plan defines an explicit workflow parity catalog, preserves legacy
  provider benchmark flags, and adds runtime benchmarking as a distinct mode.
- [x] Principle II: Git backend design keeps shell `git` as the parity baseline and
  forbids silent backend swaps for file-scoped flows.
- [x] Principle III: Unit, contract, parity, schema, and benchmark regression tests
  are all defined in scope, with `make check` retained.
- [x] Principle IV: Runtime performance claims are tied to a defined corpus and
  versioned report schema; provider-quality benchmarking remains distinct.
- [x] Principle V: No new secret persistence is introduced, and runtime benchmark mode
  avoids live provider traffic by design.

## Complexity Tracking

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| None | N/A | N/A |
