# Exit and Error Baseline

## Parser and Contract Errors

| Scenario | Example | Expected Exit | Expected Behavior |
|----------|---------|---------------|-------------------|
| Invalid flag | `kcmt --definitely-invalid-flag` | `2` | Parser error printed to stderr |
| Missing repo path | `kcmt status --repo-path /tmp/does-not-exist` | `1` | Clear repo-selection error on stderr |
| Missing file target | `kcmt --file missing.txt --repo-path <repo>` | `1` | Clear file-scoped workflow error on stderr |

Validated by:

- Python CLI parser contract in `kcmt/legacy_cli.py`
- `rust/crates/kcmt-cli/tests/status_contracts.rs::invalid_flag_returns_non_zero_and_parser_message`
- `rust/crates/kcmt-cli/tests/status_contracts.rs::file_mode_non_git_repo_returns_explicit_repo_error`

## Runtime Benchmark Errors

| Scenario | Example | Expected Exit | Expected Behavior |
|----------|---------|---------------|-------------------|
| Rust binary missing | `kcmt benchmark runtime --runtime rust --repo-path <repo>` | `0` | Rust result recorded as `excluded` with a failure reason |
| Unsupported runtime mode | `kcmt benchmark runtime --runtime nonsense --repo-path <repo>` | `2` | Parser/config error; no silent fallback |
| Runtime scenario failure | benchmarked command returns non-zero | `1` | Report emitted if possible and failed scenario called out |

Validated by:

- `tests/test_benchmark.py::test_run_runtime_benchmark_records_missing_rust_binary_as_excluded`
- `rust/crates/kcmt-cli/tests/benchmark_contracts.rs::runtime_benchmark_rust_missing_binary_is_reported_as_excluded_json`
- `rust/crates/kcmt-cli/src/commands/benchmark.rs` failing the command when any runtime scenario reports `failed`

## Git Safety Errors

| Scenario | Example | Expected Exit | Expected Behavior |
|----------|---------|---------------|-------------------|
| File-scoped command would affect multiple files | `kcmt --file README.md --repo-path <repo>` with broadened stage set | `1` | Workflow aborts with explicit safety failure |
| Experimental backend parity gap | `gitoxide` path diverges from shell `git` | `1` | Backend is rejected or excluded; default shell `git` path remains unchanged |

Validated by:

- `rust/crates/kcmt-cli/tests/workflow_modes.rs::file_mode_commits_only_requested_path`
- `rust/crates/kcmt-core/src/git/gitoxide_readonly.rs` keeping the experimental adapter read-only and out of parity-critical write paths
