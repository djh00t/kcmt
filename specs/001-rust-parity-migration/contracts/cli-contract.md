# CLI Contract

## Scope

This contract defines user-facing and automation-facing behavior that must remain compatible during the Rust migration.

## Entry Points

- `kcmt`
- `commit`
- `kc`

All entry points map to the same command surface and behavior.

## Core Invocation Contract

```text
kcmt [global-options] [subcommand]
```

Supported subcommand:

- `status`

Key option groups (non-exhaustive but compatibility-critical):

- Provider/config: `--configure`, `--configure-all`, `--provider`, `--model`, `--endpoint`, `--api-key-env`, `--github-token`
- Commit workflow: `--oneshot`, `--file`, `--max-commit-length`, `--max-retries`, `--auto-push`, `--no-auto-push`
- Benchmark/modeling: `--list-models`, `--benchmark`, `--benchmark-limit`, `--benchmark-timeout`, `--benchmark-json`, `--benchmark-csv`, `--verify-keys`
- UX/runtime: `--no-progress`, `--workers`, `--limit`, `--verbose`, `--debug`, `--profile-startup`, `--compact`

## Exit Codes

- `0`: Successful completion
- `1`: Runtime/operational failure (provider failure, git operation failure, or benchmark execution failure)
- `2`: Argument parsing / usage error

## Output Compatibility

- Human-readable output remains available for interactive use.
- Structured/machine-consumable benchmark output remains available when JSON/CSV flags are used.
- Error output remains on stderr for operational failures.

## Behavioral Guarantees

- File-targeted commit flow commits only the explicitly requested file path.
- Provider selection/config precedence remains consistent with current behavior.
- Non-interactive invocations remain stable for CI/script use.
