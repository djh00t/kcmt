# Workflow Parity Catalog

## Required-Now Workflows

| ID | Entrypoint(s) | Example Command | Why It Matters |
|----|---------------|-----------------|----------------|
| `help-default` | `kcmt`, `commit`, `kc` | `kcmt --help` | Confirms top-level parser/alias contract |
| `status-repo-path` | `kcmt` | `kcmt status --repo-path <repo>` | Required explicit repo targeting |
| `oneshot-repo-path` | `kcmt`, `commit`, `kc` | `kcmt --oneshot --repo-path <repo>` | Core workflow parity with explicit repo selection |
| `file-repo-path` | `kcmt`, `commit`, `kc` | `kcmt --file README.md --repo-path <repo>` | File-scoped git safety contract |
| `provider-benchmark-legacy` | `kcmt` | `kcmt --benchmark --benchmark-json` | Existing benchmark compatibility |
| `runtime-benchmark` | `kcmt` | `kcmt benchmark runtime --repo-path <repo> --json` | New runtime benchmark feature |
| `parser-error` | `kcmt` | `kcmt --definitely-invalid-flag` | Stable parser exit/error contract |

## Later Workflows

These are relevant but not required to declare basic parity complete for this
feature’s initial implementation slice:

- `--configure`
- `--configure-all`
- `--list-models`
- `--verify-keys`
- `--auto-push`
- `--compact`

## Validation Rule

Parity is considered complete for this feature only when every Required-Now workflow
has automated Python and Rust evidence recorded in tests or validation notes.

## Automated Evidence

| Workflow ID | Python Evidence | Rust Evidence |
|-------------|-----------------|---------------|
| `help-default` | `tests/test_cli.py::test_cli_help_returns_zero`, `tests/test_main_entrypoint.py::test_main_falls_back_to_python_cli` | `rust/crates/kcmt-cli/tests/status_contracts.rs::commit_status_help_uses_commit_branding` |
| `status-repo-path` | `tests/test_cli.py::test_cli_status_without_snapshot`, `tests/test_cli.py::test_cli_status_raw_snapshot` | `rust/crates/kcmt-cli/tests/status_contracts.rs::status_repo_path_without_snapshot_returns_contract_message`, `rust/crates/kcmt-cli/tests/workflow_modes.rs::file_mode_persists_snapshot_for_status_view` |
| `oneshot-repo-path` | `tests/test_cli.py::test_cli_oneshot_happy_path` | `rust/crates/kcmt-cli/tests/workflow_modes.rs::oneshot_mode_commits_first_changed_non_deletion` |
| `file-repo-path` | `tests/test_cli.py::test_cli_executes_workflow_success` and runtime benchmark single-file coverage in `tests/test_benchmark.py::test_run_runtime_benchmark_produces_python_results` | `rust/crates/kcmt-cli/tests/workflow_modes.rs::file_mode_commits_only_requested_path` |
| `provider-benchmark-legacy` | `tests/test_cli.py::test_legacy_cli_benchmark_flag_preserves_provider_dispatch`, `tests/test_benchmark.py::test_render_benchmark_markdown_report_remains_provider_focused` | Parser compatibility in `rust/crates/kcmt-cli/src/args.rs::tests::parses_runtime_benchmark_subcommand` keeps legacy benchmark routing additive |
| `runtime-benchmark` | `tests/test_cli.py::test_cli_runtime_benchmark_json`, `tests/test_benchmark.py::test_run_runtime_benchmark_produces_python_results` | `rust/crates/kcmt-cli/tests/benchmark_contracts.rs::runtime_benchmark_python_emits_passing_results_json`, `rust/crates/kcmt-cli/tests/benchmark_contracts.rs::runtime_benchmark_rust_missing_binary_is_reported_as_excluded_json` |
| `parser-error` | `tests/test_main_entrypoint.py::test_main_returns_rust_runtime_code` covers wrapper routing; Python argparse contract remains in `kcmt/legacy_cli.py` | `rust/crates/kcmt-cli/tests/status_contracts.rs::invalid_flag_returns_non_zero_and_parser_message` |
