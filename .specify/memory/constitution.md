<!--
Sync Impact Report
- Version change: template (unversioned) -> 1.0.0
- Modified principles:
  - Template Principle 1 -> I. CLI Contract Compatibility
  - Template Principle 2 -> II. Git Safety and Atomic Operations
  - Template Principle 3 -> III. Quality Gates and Test Discipline
  - Template Principle 4 -> IV. Performance and Benchmark Accountability
  - Template Principle 5 -> V. Security and Configuration Integrity
- Added sections:
  - Operational Constraints
  - Development Workflow and Review Gates
- Removed sections:
  - None
- Templates requiring updates:
  - .specify/templates/plan-template.md ✅ updated
  - .specify/templates/spec-template.md ✅ updated
  - .specify/templates/tasks-template.md ✅ updated
  - .specify/templates/commands/*.md ⚠ pending (directory not present in repository)
- Runtime guidance updates:
  - AGENTS.md ✅ updated
  - README.md ✅ updated
- Deferred TODOs:
  - None
-->

# kcmt Constitution

## Core Principles

### I. CLI Contract Compatibility
All user-facing CLI entry points (`kcmt`, `commit`, `kc`) MUST preserve behavioral
compatibility unless a change is explicitly documented in spec, plan, and tasks.
Exit codes, machine-readable outputs, and documented flags MUST remain stable for
automation use cases.
Rationale: Migration safety depends on keeping existing developer and CI workflows
working without silent breakage.

### II. Git Safety and Atomic Operations
Git operations MUST remain deterministic and safe: file-scoped commit actions MUST
only affect intended paths, and high-parity operations MUST use semantics equivalent
to the `git` CLI. Experimental Git backends MAY be introduced only behind explicit
feature flags with parity validation.
Rationale: Incorrect git behavior risks data loss and invalid history.

### III. Quality Gates and Test Discipline
Changes MUST include tests that verify the affected behavior at the appropriate layer
(unit, integration, contract, or parity). `make check` (or the equivalent strict gate)
MUST pass before merge. For migration work, parity checks between legacy and new paths
MUST be included where user-visible behavior is touched.
Rationale: High-confidence delivery requires reproducible validation, not ad-hoc checks.

### IV. Performance and Benchmark Accountability
Any performance claim MUST include a reproducible baseline, a defined corpus, and
measurable success criteria. Benchmark output formats and scoring dimensions used for
comparisons MUST stay stable or include a migration path.
Rationale: Performance-driven migrations fail without objective measurement discipline.

### V. Security and Configuration Integrity
Secrets MUST be sourced from environment variables or secure runtime stores and MUST
NOT be persisted in plaintext project files. Configuration precedence (CLI overrides,
environment, persisted config, defaults) MUST remain explicit and tested. Error
messages MUST be actionable and MUST NOT leak secret values.
Rationale: Configuration regressions and secret leakage are critical operational risks.

## Operational Constraints

- Primary implementation language is Python for current runtime; Rust migration work
  MUST maintain feature parity before default cutover.
- Conventional Commit discipline is required for repository history.
- One-feature planning flow is mandatory: `spec -> plan -> tasks -> implement`.
- Documentation and contract artifacts under `specs/<feature>/` are required inputs
  for implementation and review.

## Development Workflow and Review Gates

- Every feature plan MUST include a Constitution Check that evaluates all five
  principles above and records violations in Complexity Tracking.
- Every tasks artifact MUST organize work by user story and include explicit validation
  steps for compatibility, performance, and security requirements when in scope.
- Pull requests MUST reference the governing feature spec/plan/tasks documents and
  include evidence of passing gates (`make check`, parity tests, or benchmark proofs).
- If a principle cannot be met, the exception MUST be documented with rationale,
  alternatives considered, risk, and a follow-up task to close the gap.

## Governance

This constitution is the highest-priority project governance document for planning and
implementation workflows. Conflicts between this document and feature-level artifacts
MUST be resolved in favor of this constitution.

Amendment process:
1. Propose changes in a dedicated update to `.specify/memory/constitution.md`.
2. Include a Sync Impact Report describing downstream template/doc updates.
3. Update dependent templates and guidance docs in the same change set.
4. Record version bump rationale using semantic versioning rules below.

Versioning policy:
- MAJOR: backward-incompatible governance changes or principle removals/redefinitions.
- MINOR: new principle/section or materially expanded mandatory guidance.
- PATCH: clarifications, wording improvements, typo/non-semantic refinements.

Compliance expectations:
- Reviewers MUST verify constitution alignment in plan/task artifacts before approval.
- `/speckit.analyze` findings marked CRITICAL MUST be resolved before
  `/speckit.implement`.

**Version**: 1.0.0 | **Ratified**: 2026-03-01 | **Last Amended**: 2026-03-01
