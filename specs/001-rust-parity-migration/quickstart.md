# Quickstart: Rust Migration Planning Baseline

## 1. Confirm Feature Context

- Branch: `001-rust-parity-migration`
- Spec: `/Users/djh/work/src/github.com_local/djh00t/kcmt/specs/001-rust-parity-migration/spec.md`
- Plan: `/Users/djh/work/src/github.com_local/djh00t/kcmt/specs/001-rust-parity-migration/plan.md`

## 2. Capture Baseline Behavior (Python)

1. Run strict baseline checks:
   ```bash
   make check
   ```
2. Capture CLI contract fixtures for core workflows (`kcmt`, `commit`, `kc`, `status`, `--oneshot`, `--file`).
3. Capture benchmark baseline snapshots (`--benchmark`, `--benchmark-json`, `--benchmark-csv`).

## 3. Scaffold Rust Workspace

1. Create `rust/` workspace and core crates (`kcmt-core`, `kcmt-cli`, `kcmt-provider`, `kcmt-bench`).
2. Implement CLI contract skeleton with aliases and option surface.
3. Implement git command abstraction using `git` CLI backend for parity-critical operations.

## 4. Build Parity Harness

1. Define corpus of representative repositories/diffs.
2. For each scenario, run Python baseline and Rust candidate.
3. Compare:
   - exit code
   - normalized stdout/stderr
   - git side effects
   - config resolution behavior

## 5. Implement Reliability Layer

1. Shared async transport (`tokio` + `reqwest` client reuse).
2. Per-provider adapter modules.
3. Timeout/retry/rate-limit policy with deterministic error mapping.

## 6. Interactive UX Path

1. Keep non-TTY path as baseline.
2. Add optional Ratatui mode for interactive sessions.
3. Gate default TUI enablement behind parity/stability thresholds.

## 7. Readiness Gate for Task Generation

Proceed to `/speckit.tasks` only when:

- No unresolved clarifications remain.
- Contracts and data model are accepted.
- Baseline capture plan is defined for parity and performance checks.

## 8. Validation Gates for Implementation Readiness

Before rollout cutover, record evidence for:

- **FR-008** reliability parity from regression and failure-mode scenarios.
- **SC-001** >=50% median commit generation speedup on the regression corpus.
- **SC-002** >=95% benchmark scenarios with <=2s local preprocessing time.
- **SC-003** 100% parity for high-usage workflow catalog.
- **SC-004** 100% configuration compatibility pass rate.
- **SC-005** benchmark quality score delta within -2 points of baseline.
- **SC-006** 100% exit/error matrix parity against baseline.

Store run metadata and outcomes in `tasks-validation.md`.
