# Data Model: Rust CLI Feature Parity and Runtime Benchmark Mode

## Entity: Workflow Contract

- Purpose: defines one user-visible CLI workflow that Rust must match.
- Fields:
  - `id` (string) - stable identifier such as `kcmt-status-repo-path`.
  - `entrypoint` (`kcmt` | `commit` | `kc`) - CLI binary name.
  - `argv` (string list) - arguments that define the contract.
  - `repo_selection_mode` (`cwd` | `repo-path`) - how the workflow selects the repo.
  - `expected_exit_code` (integer or set) - required exit behavior.
  - `expected_stdout_contract` (string) - summary of stdout requirements.
  - `expected_stderr_contract` (string) - summary of stderr requirements.
  - `parity_scope` (`required-now` | `later`) - delivery phase marker.
- Validation rules:
  - `id` MUST be unique.
  - `entrypoint` MUST be one of the documented aliases.
  - `parity_scope=required-now` contracts MUST have automated tests before runtime
    cutover consideration.

## Entity: Repo Corpus Fixture

- Purpose: describes one deterministic repository state used for runtime benchmarking.
- Fields:
  - `id` (string) - stable corpus identifier.
  - `kind` (`synthetic` | `realistic`) - corpus class.
  - `location` (string) - path or generation command.
  - `file_count` (integer) - approximate size.
  - `git_history_state` (`no-commits` | `seeded-history`) - repository history shape.
  - `change_shape` (string list) - categories such as `untracked`, `modified`,
    `deleted`, `nested-paths`, `ignored-files`.
  - `rebuild_steps` (string list) - commands needed to recreate the fixture.
- Validation rules:
  - Every benchmark run MUST record the corpus `id`.
  - Synthetic corpora MUST be reproducible from checked-in tooling.
  - Realistic corpora MUST be small enough to keep CI deterministic.

## Entity: Runtime Benchmark Scenario

- Purpose: one runtime benchmark workload executed across Python and Rust.
- Fields:
  - `id` (string) - stable scenario id.
  - `workflow_contract_id` (string) - linked workflow contract.
  - `corpus_id` (string) - linked repo corpus fixture.
  - `runtime` (`python` | `rust`) - implementation under test.
  - `iterations` (integer) - repeat count.
  - `command_label` (string) - human-readable command name.
  - `status` (`passed` | `failed` | `excluded`) - outcome class.
  - `failure_reason` (string | null) - explanation for failed/excluded runs.
- Validation rules:
  - Python and Rust runs for the same scenario MUST use the same corpus and command
    label.
  - Exclusions MUST include a non-empty `failure_reason`.

## Entity: Runtime Benchmark Run

- Purpose: one exported runtime benchmark report.
- Fields:
  - `schema_version` (string) - report schema version.
  - `timestamp` (ISO 8601 string) - report generation time.
  - `command_set` (string) - named scenario bundle.
  - `corpora` (array of corpus ids) - corpora included in the run.
  - `results` (array of runtime benchmark scenarios) - detailed outcomes.
  - `summary` (object) - aggregate per-runtime medians and counts.
- Validation rules:
  - `schema_version` MUST match the contract schema.
  - Every result MUST refer to a known corpus and workflow contract.
  - Summary statistics MUST be derivable from the underlying results.

## Entity: Git Backend Decision

- Purpose: captures whether a workflow uses shell `git` or an experimental Rust
  backend.
- Fields:
  - `workflow_contract_id` (string) - linked workflow.
  - `backend` (`shell-git` | `gitoxide`) - backend selected.
  - `default_enabled` (boolean) - whether the backend is on by default.
  - `parity_status` (`unvalidated` | `validated`) - contract readiness.
  - `notes` (string) - rationale or guardrail notes.
- Validation rules:
  - `gitoxide` MUST NOT be `default_enabled=true` unless `parity_status=validated`.
  - File-scoped workflows MUST document backend parity evidence before default switch.
