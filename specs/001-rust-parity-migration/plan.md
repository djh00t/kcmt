# Implementation Plan: High-Performance Core Migration with Feature Parity

**Branch**: `001-rust-parity-migration` | **Date**: 2026-03-01 | **Spec**: [/Users/djh/work/src/github.com_local/djh00t/kcmt/specs/001-rust-parity-migration/spec.md](/Users/djh/work/src/github.com_local/djh00t/kcmt/specs/001-rust-parity-migration/spec.md)
**Input**: Feature specification from `/specs/001-rust-parity-migration/spec.md`

## Summary

Reimplement `kcmt` in Rust with strict feature parity for CLI behavior, provider/config compatibility, and benchmark reporting, while delivering a measurable performance uplift. The migration uses a compatibility-first rollout with parity gates, keeps Git behavior anchored to the `git` CLI for high-risk flows, and phases in Rust-native UI improvements.

## Technical Context

**Language/Version**: Rust stable (target 1.78+), Python 3.12 retained temporarily for parity harness and transition wrappers  
**Primary Dependencies**: `clap`, `tokio`, `reqwest`, `serde`/`serde_json`, `tracing`, `anyhow`/`thiserror`, optional `ratatui` for interactive TUI phase  
**Storage**: Local filesystem only (`~/.config/kcmt`, environment variables, git working tree/index); no server-side database  
**Testing**: `cargo test`, `trycmd` CLI contract tests, parity regression corpus against Python baseline, benchmark regression suite, existing `pytest` suite during transition  
**Target Platform**: macOS and Linux first-class; Windows supported where `git` CLI is available  
**Project Type**: CLI application with reusable Rust core library crates  
**Performance Goals**: Meet spec SC-001..SC-006, including >=50% faster median commit-generation flow and <=2s local pre-provider benchmark processing for >=95% scenarios  
**Baseline Corpus**: `specs/001-rust-parity-migration/validation/regression-corpus.md` with paired baseline snapshots from the current Python release  
**Constraints**: Preserve aliases (`kcmt`, `commit`, `kc`), exit code semantics, config precedence rules, provider error semantics, and benchmark scoring dimensions; avoid secret leakage in logs  
**Scale/Scope**: Daily developer usage across medium/large repositories, including runs over thousands of changed files and multi-provider benchmark matrices

### Validation Assets

- Workflow catalog: `specs/001-rust-parity-migration/validation/high-usage-workflows.md`
- Exit/error baseline: `specs/001-rust-parity-migration/validation/exit-error-baseline.md`
- Corpus definition: `specs/001-rust-parity-migration/validation/regression-corpus.md`

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

Constitution source: `/Users/djh/work/src/github.com_local/djh00t/kcmt/.specify/memory/constitution.md`.

### Pre-Phase 0 Gate Evaluation

- Principle I (CLI Contract Compatibility) - **PASS**: FR-001/FR-009 and US1 tasks explicitly preserve aliases, command behavior, exit codes, and machine-readable outputs.
- Principle II (Git Safety and Atomic Operations) - **PASS**: Git strategy remains parity-first with CLI semantics; file-scoped atomic commit behavior is covered by FR-003 and US1 tasks.
- Principle III (Quality Gates and Test Discipline) - **PASS**: plan includes contract/integration/parity tasks and strict quality gate expectations (`make check`, parity validation).
- Principle IV (Performance and Benchmark Accountability) - **PASS**: measurable SC targets are tied to explicit corpus/workflow/baseline artifacts.
- Principle V (Security and Configuration Integrity) - **PASS**: config precedence, secret-safe error behavior, and compatibility tests are explicitly included.

Result: **PASS**. No blocking gate violations.

### Post-Phase 1 Gate Re-Check

- Principle I - **PASS**: CLI contract and task plan preserve automation-facing behavior.
- Principle II - **PASS**: architecture and task ordering keep git safety constraints explicit.
- Principle III - **PASS**: tests and checkpoints are included before implementation completion.
- Principle IV - **PASS**: baseline corpus and benchmark validation tasks are present.
- Principle V - **PASS**: security/config requirements and tasks are mapped and testable.

Result: **PASS**. Ready for `/speckit.tasks`.

## Project Structure

### Documentation (this feature)

```text
specs/001-rust-parity-migration/
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── contracts/
│   ├── benchmark-result.schema.json
│   ├── cli-contract.md
│   └── config-contract.md
└── tasks.md                   # created by /speckit.tasks
```

### Source Code (repository root)

```text
kcmt/                          # current Python implementation (compatibility baseline)
├── cli.py
├── core.py
├── commit.py
├── git.py
├── config.py
├── llm.py
└── providers/

tests/                         # current Python tests and parity oracle

docs/

rust/                          # planned addition for migration
├── Cargo.toml
├── crates/
│   ├── kcmt-core/
│   ├── kcmt-cli/
│   ├── kcmt-provider/
│   ├── kcmt-bench/
│   └── kcmt-tui/              # optional phase; Ratatui-based
└── tests/
    ├── parity/
    ├── contract/
    └── integration/
```

**Structure Decision**: Use an incremental dual-runtime structure: keep `kcmt/` + `tests/` as the behavior oracle while building a Rust workspace under `rust/`. This minimizes migration risk and supports side-by-side parity verification before any default runtime switch.

## Phase 0: Outline & Research

Completed research tracks:

1. CLI compatibility strategy (`argparse` to `clap`) with preserved aliases and exit codes
2. Git backend strategy with strict parity constraints
3. `gitoxide` viability for parity-critical operations
4. Provider transport/retry/rate-limit architecture
5. Migration rollout and cutover pattern
6. Ratatui adoption strategy for interactive mode

Output: [/Users/djh/work/src/github.com_local/djh00t/kcmt/specs/001-rust-parity-migration/research.md](/Users/djh/work/src/github.com_local/djh00t/kcmt/specs/001-rust-parity-migration/research.md)

## Phase 1: Design & Contracts

Produced artifacts:

1. Data model: [/Users/djh/work/src/github.com_local/djh00t/kcmt/specs/001-rust-parity-migration/data-model.md](/Users/djh/work/src/github.com_local/djh00t/kcmt/specs/001-rust-parity-migration/data-model.md)
2. Interface contracts:
   - [/Users/djh/work/src/github.com_local/djh00t/kcmt/specs/001-rust-parity-migration/contracts/cli-contract.md](/Users/djh/work/src/github.com_local/djh00t/kcmt/specs/001-rust-parity-migration/contracts/cli-contract.md)
   - [/Users/djh/work/src/github.com_local/djh00t/kcmt/specs/001-rust-parity-migration/contracts/config-contract.md](/Users/djh/work/src/github.com_local/djh00t/kcmt/specs/001-rust-parity-migration/contracts/config-contract.md)
   - [/Users/djh/work/src/github.com_local/djh00t/kcmt/specs/001-rust-parity-migration/contracts/benchmark-result.schema.json](/Users/djh/work/src/github.com_local/djh00t/kcmt/specs/001-rust-parity-migration/contracts/benchmark-result.schema.json)
3. Validation and rollout quickstart: [/Users/djh/work/src/github.com_local/djh00t/kcmt/specs/001-rust-parity-migration/quickstart.md](/Users/djh/work/src/github.com_local/djh00t/kcmt/specs/001-rust-parity-migration/quickstart.md)
4. Agent context refresh via `.specify/scripts/bash/update-agent-context.sh codex`

## Phase 2: Planning Hand-off Status

Task generation has been completed via `/speckit.tasks`, and implementation work is in
progress against the generated checklist in
`specs/001-rust-parity-migration/tasks.md`.

Current hand-off note: implementation is largely complete with final
cross-platform parity execution evidence pending for `T062`.

## Complexity Tracking

No constitution violations or exceptional complexity waivers are required at this stage.
