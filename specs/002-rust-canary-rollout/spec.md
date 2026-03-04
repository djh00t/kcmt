# Feature Specification: Rust Runtime Canary Rollout and Observability

**Feature Branch**: `002-rust-canary-rollout`  
**Created**: 2026-03-04  
**Status**: Draft  
**Input**: User description: "Continue after parity migration by using Spec Kit to define and execute a canary rollout for Rust runtime adoption."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Automated Canary Safety Gate (Priority: P1)

As a release engineer, I can run a deterministic canary smoke workflow that exercises
the Python wrapper in Rust canary mode so regressions are blocked before merge.

**Why this priority**: Canary gating is required before wider rollout and protects core
CLI compatibility commitments.

**Independent Test**: Execute a dedicated CI workflow that builds Rust binaries and
runs canary probes (`KCMT_RUNTIME=auto`, `KCMT_RUST_CANARY=1`) for `kcmt`, `commit`,
`kc`, and error-path commands.

**Acceptance Scenarios**:

1. **Given** a PR that changes runtime routing paths, **When** canary smoke checks run,
   **Then** workflow fails if command contract probes diverge from expected outcomes.
2. **Given** `KCMT_RUNTIME=auto` and `KCMT_RUST_CANARY=1` with a valid Rust binary,
   **When** `kcmt --help` executes, **Then** wrapper routes to Rust runtime and exits
   successfully.

---

### User Story 2 - Runtime Decision Observability (Priority: P2)

As a maintainer diagnosing rollout issues, I can opt-in to machine-readable runtime
decision traces that explain whether Python or Rust was selected and why.

**Why this priority**: Canary failures are expensive to debug without explicit routing
signals, especially for fallback scenarios.

**Independent Test**: With trace mode enabled, run command probes and validate that
decision records parse as JSON and include required fields (`selected_runtime`,
`decision_reason`, `rust_binary`).

**Acceptance Scenarios**:

1. **Given** trace mode is enabled and Rust binary is missing, **When** wrapper runs in
   auto mode, **Then** trace record reports `python` selection with a fallback reason.
2. **Given** trace mode is enabled and canary routing succeeds, **When** wrapper invokes
   Rust, **Then** trace record reports `rust` selection and no secret values.

---

### User Story 3 - Rollout Procedure and Rollback Controls (Priority: P3)

As an operator, I can follow a documented staged rollout and rollback procedure that
uses existing runtime env controls without changing default behavior yet.

**Why this priority**: Operational clarity reduces rollout risk and supports controlled
adoption after CI validation.

**Independent Test**: Follow quickstart rollout steps on a local repo and verify
stages (baseline, canary-on, rollback) all execute with expected runtime selection.

**Acceptance Scenarios**:

1. **Given** rollout stage is canary, **When** operator sets canary env variables,
   **Then** wrapper prefers Rust when binary exists and falls back safely otherwise.
2. **Given** rollback is required, **When** operator sets `KCMT_RUNTIME=python`,
   **Then** wrapper uses Python path deterministically and bypasses Rust runtime.

---

### Edge Cases

- `KCMT_RUNTIME=auto` and `KCMT_RUST_CANARY=1` but Rust binary path does not exist.
- `KCMT_RUNTIME=rust` but Rust binary exits non-zero.
- Trace mode enabled while command is run in scripts expecting clean stdout.
- Trace destination is unavailable or unwritable.
- Unknown `KCMT_RUNTIME` value is provided.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST provide a dedicated canary CI workflow that validates wrapper
  behavior when `KCMT_RUNTIME=auto` and `KCMT_RUST_CANARY=1`.
- **FR-002**: System MUST keep default runtime behavior unchanged for users who do not
  set canary env controls.
- **FR-003**: System MUST preserve existing entry points (`kcmt`, `commit`, `kc`) and
  exit-code behavior during canary checks.
- **FR-004**: System MUST support opt-in runtime decision tracing via environment
  controls without changing stdout command contracts.
- **FR-005**: System MUST include decision reason metadata for Rust selection and Python
  fallback paths.
- **FR-006**: System MUST keep fallback behavior safe: if Rust runtime cannot be used,
  wrapper executes Python CLI path.
- **FR-007**: System MUST document staged canary rollout and rollback procedures in
  repository docs.
- **FR-008**: System MUST add automated tests for runtime decision tracing and fallback
  logic in the Python wrapper.

### Non-Functional Requirements *(mandatory)*

- **NFR-001**: CLI compatibility MUST remain stable: no new required flags or changed
  stdout contract for existing commands in default mode.
- **NFR-002**: Canary and trace artifacts MUST avoid exposing secret values from env or
  provider configuration.
- **NFR-003**: Canary workflow duration MUST complete within 15 minutes on
  `ubuntu-latest` under normal CI conditions.
- **NFR-004**: Runtime decision trace format MUST be machine-parseable JSON with stable
  required fields for automation.

### Key Entities *(include if feature involves data)*

- **Runtime Decision Record**: A trace record describing runtime selection result,
  decision reason, binary path, and canary state for a single invocation.
- **Canary Probe Scenario**: A deterministic command/env test case used in CI to verify
  runtime selection and fallback behavior.
- **Rollout Stage**: Operational state (`baseline`, `canary`, `rollback`) and its
  required environment settings.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Canary workflow passes 100% of defined probe scenarios on PR runs for this
  feature branch.
- **SC-002**: Runtime trace tests cover Rust selection and Python fallback branches with
  no unresolved failures in `make check`.
- **SC-003**: Rollout documentation includes explicit commands for baseline, canary, and
  rollback stages and is validated by quickstart execution notes.
- **SC-004**: Default-mode commands (`kcmt --help`, `commit --help`, `kc --help`) remain
  behaviorally unchanged in local regression checks.
