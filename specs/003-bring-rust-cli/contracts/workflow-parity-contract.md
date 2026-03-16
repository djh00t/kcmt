# Workflow Parity Contract

## Purpose

Define the user-visible CLI workflows that Rust must support with the same command
shape and compatibility guarantees as Python for this feature.

## Required Contracts

| Contract ID | Entrypoint | Command Shape | Compatibility Promise |
|-------------|------------|---------------|-----------------------|
| `help-default` | `kcmt`, `commit`, `kc` | `<entrypoint> --help` | Exit `0`; documented commands and key flags remain visible |
| `status-repo-path` | `kcmt` | `status --repo-path <repo>` | Accept explicit repo selection; preserve exit and output contract |
| `oneshot-repo-path` | `kcmt`, `commit`, `kc` | `--oneshot --repo-path <repo>` | Respect explicit repo selection and file safety rules |
| `file-repo-path` | `kcmt`, `commit`, `kc` | `--file <path> --repo-path <repo>` | Commit or stage only the requested file scope |
| `provider-benchmark-legacy` | `kcmt` | `--benchmark [--benchmark-json] [--benchmark-csv]` | Preserve current provider-quality benchmark meaning |
| `runtime-benchmark` | `kcmt` | `benchmark runtime ...` | Emit separate runtime benchmark artifact; no provider score mixing |
| `parser-error` | `kcmt` | `--definitely-invalid-flag` | Exit `2` with parser error on stderr |

## Notes

- The contract catalog is authoritative for this feature’s parity scope.
- Additional undocumented parser branches MAY exist, but they do not count as parity
  commitments until added to the catalog.
- New benchmark commands MUST distinguish runtime benchmarking from provider-quality
  benchmarking in both help text and output artifacts.
