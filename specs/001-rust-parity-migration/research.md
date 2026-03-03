# Phase 0 Research: Rust Migration with Feature Parity

## Decision 1: CLI Compatibility Strategy

**Decision**: Implement Rust CLI parsing with `clap` in compatibility-first mode, preserving aliases (`kcmt`, `commit`, `kc`) and existing exit-code behavior.

**Rationale**: Exact command/flag and exit-code parity is required for user workflows and automation contracts. `clap` provides explicit control for aliasing, parse behavior, and deterministic error code handling.

**Alternatives considered**:

- Keep Python CLI as permanent frontend and call Rust backend (rejected: leaves long-term dual-runtime complexity)
- Big-bang parser rewrite without parity harness (rejected: regression risk too high)

## Decision 2: Git Backend for Parity-Critical Paths

**Decision**: Use `git` CLI subprocess execution as the baseline backend for parity-critical operations (status porcelain, pathspec commit behavior, worktrees, submodules).

**Rationale**: Existing implementation behavior already maps to `git` CLI semantics. Keeping this path minimizes divergence risk during migration while still allowing Rust performance gains in orchestration and data processing.

**Alternatives considered**:

- Full `libgit2`/`git2` replacement immediately (rejected for parity-risk profile)
- Full `gitoxide` replacement immediately (rejected based on current feature gaps for required parity surface)

## Decision 3: gitoxide Adoption Scope

**Decision**: Investigate and adopt `gitoxide` only for non-parity-critical, read-oriented internals in early phases; do not make it the primary backend for parity-critical operations yet.

**Rationale**: `gitoxide` is promising and worth integrating incrementally, but strict parity requirements for worktrees/submodules/pathspec semantics make a full immediate backend swap risky.

**Alternatives considered**:

- Defer `gitoxide` entirely (rejected: misses an optimization opportunity)
- Use `gitoxide` as sole backend from phase 1 (rejected: high compatibility risk)

## Decision 4: Provider Transport and Reliability Stack

**Decision**: Standardize provider I/O on a shared async stack (`tokio` + reusable `reqwest::Client`) with explicit timeout, retry, and rate-limit policies; keep provider-specific adapters for auth and error semantics.

**Rationale**: This yields consistent reliability behavior across OpenAI, Anthropic, xAI, and GitHub Models while preserving provider-specific handling where needed.

**Alternatives considered**:

- Provider-specific SDK per provider with divergent reliability logic (rejected: inconsistent behavior and maintenance overhead)
- Blocking HTTP model with threads (rejected: weaker concurrency control)

## Decision 5: Migration Rollout Pattern

**Decision**: Use a strangler-pattern rollout with parity harness gates: shadow mode in CI, opt-in canary, percentage rollout, then default switch with rapid fallback path.

**Rationale**: Reduces blast radius and allows objective parity/performance validation before broad rollout.

**Alternatives considered**:

- Big-bang runtime switch (rejected: poor rollback safety)
- Indefinite dual implementation without convergence plan (rejected: permanent complexity)

## Decision 6: Ratatui Strategy for Interactive UX

**Decision**: Adopt Ratatui in phases: optional interactive mode first, then default for TTY sessions only after parity/stability gates pass; keep non-TTY and `--no-tui` paths permanently.

**Rationale**: Ratatui is a strong Rust-native TUI option and aligns with long-term stack simplification, but replacing Ink-like UX should be gated to avoid launch regressions.

**Alternatives considered**:

- Ratatui as immediate default on first Rust release (rejected: migration risk)
- Keep Node/Ink interactive path indefinitely (rejected: long-term dual-runtime maintenance)

## Resolved Clarifications

All technical-context clarifications are resolved. No remaining `NEEDS CLARIFICATION` markers.
