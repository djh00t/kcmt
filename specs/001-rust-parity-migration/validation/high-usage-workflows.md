# High-Usage Workflow Catalog

This catalog enumerates compatibility-critical workflows that MUST pass in parity tests.

## Required Workflows

1. `kcmt` default flow with staged changes.
2. `commit` alias flow with staged changes.
3. `kc` alias flow with staged changes.
4. `kcmt --file <path>` file-scoped commit flow.
5. `kcmt --oneshot` non-interactive flow.
6. `kcmt status` reporting flow.
7. `kcmt --benchmark` table output flow.
8. `kcmt --benchmark-json <file>` JSON export flow.
9. `kcmt --benchmark-csv <file>` CSV export flow.

## Validation Rules

- Each workflow MUST verify exit code, stdout/stderr channel behavior, and git side effects.
- Workflow parity MUST be validated on `macos-latest`, `ubuntu-latest`, and `windows-latest`.
