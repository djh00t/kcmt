# Data Model: Rust Runtime Canary Rollout and Observability

## Entity: Runtime Decision Record

- Purpose: machine-readable record of wrapper runtime selection for one command invocation.
- Fields:
  - `selected_runtime` (`python` | `rust`) - final runtime chosen.
  - `decision_reason` (string enum) - route reason (`runtime_python`, `auto_canary_enabled`,
    `runtime_forced_rust`, `rust_binary_missing`, `invalid_runtime_value`, `fallback_after_rust_error`).
  - `runtime_mode` (string) - normalized `KCMT_RUNTIME` value.
  - `canary_enabled` (boolean) - interpreted state of `KCMT_RUST_CANARY`.
  - `rust_binary` (string) - resolved binary path.
  - `rust_binary_exists` (boolean) - binary existence check result.
- Validation rules:
  - `selected_runtime` MUST be one of `python` or `rust`.
  - `runtime_mode` MUST be lowercased and trimmed.
  - `decision_reason` MUST be non-empty and from supported set.
  - Secret env values MUST NOT be included in any field.
- State transitions:
  - `planned` -> `python-selected` when default/rollback/fallback path is selected.
  - `planned` -> `rust-selected` when canary/forced rust path is selected and binary exists.

## Entity: Canary Probe Scenario

- Purpose: deterministic CI/local scenario defining command, env configuration, and expected result.
- Fields:
  - `id` (string) - stable scenario identifier.
  - `command` (string list) - CLI invocation under test.
  - `env` (map) - runtime env overrides for scenario.
  - `expected_runtime` (`python` | `rust`) - required selection.
  - `expected_exit` (int or set) - expected return code(s).
  - `expected_reason` (string) - expected `decision_reason`.
- Validation rules:
  - `id` MUST be unique in scenario collection.
  - `expected_runtime` MUST match parsed trace record.
  - Scenario MUST assert both runtime and exit behavior.

## Entity: Rollout Stage

- Purpose: operational configuration stage for deployment progression.
- Fields:
  - `name` (`baseline` | `canary` | `rollback`).
  - `runtime_env` (map) - required env values for stage.
  - `entry_criteria` (string list).
  - `exit_criteria` (string list).
- Validation rules:
  - `baseline` MUST leave default behavior unchanged.
  - `rollback` MUST set `KCMT_RUNTIME=python`.
  - `canary` MUST require `KCMT_RUNTIME=auto` and `KCMT_RUST_CANARY=1`.
