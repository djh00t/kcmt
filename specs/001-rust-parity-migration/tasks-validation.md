# Task Validation Evidence

## Run Metadata

- Date: 2026-03-03T13:49:13Z (UTC)
- Commit/Branch: `0836003` on `001-rust-parity-migration`
- Runtime mode (`python` / `rust` / `auto`): `python` (baseline), `rust` build probe attempted
- Fixture set revision: `specs/001-rust-parity-migration/validation/*` (current branch working tree)

## FR Validation

| Requirement | Evidence | Status |
|---|---|---|
| FR-007 | `UV_CACHE_DIR=/tmp/uv-cache uv run kcmt --benchmark --benchmark-limit 1 --benchmark-timeout 5 --benchmark-json --benchmark-csv` completed; Rust candidate compile probe failed (`cargo check` DNS failure to `index.crates.io`) | PARTIAL (baseline-only) |
| FR-008 | `UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q` -> `84 passed, 1 skipped` with 81.61% coverage | PASS (baseline) |
| FR-009 | `kcmt --help`, `commit --help`, `kc --help` exit 0; `kcmt status --repo-path .` exit 0; invalid flag probe `kcmt --definitely-invalid-flag` exit 2 | PASS (baseline) |
| NFR-005 | Cross-platform parity matrix passed on `ubuntu-latest`, `windows-latest`, and `macos-latest` (`run 22627397926`) with Python/Rust contract probe artifacts uploaded | PASS (contract probe scope) |

## Success Criteria Evidence

| Criterion | Baseline | Candidate | Status |
|---|---:|---:|---|
| SC-001 median speedup | Not yet established (no prior saved corpus baseline in this run) | Rust candidate unavailable (compile blocked by crates.io DNS) | BLOCKED |
| SC-002 <=2s preprocessing (95%) | Benchmark sample run shows all measured latencies < 2s in this environment | Rust candidate unavailable | PARTIAL (baseline-only) |
| SC-003 workflow parity | Python alias/help/status/invalid-arg probes pass in matrix | Rust contract probes for the same entrypoints pass in matrix (`run 22627397926`) | PARTIAL (core contract probes) |
| SC-004 config compatibility | Existing Python tests pass (`84 passed, 1 skipped`) including config test coverage | Rust candidate unavailable | PARTIAL (baseline-only) |
| SC-005 quality delta | Baseline benchmark quality values observed at `0.0` across sampled providers/models | Rust candidate unavailable | BLOCKED |
| SC-006 exit/error parity | Python invalid-flag parser error returns exit `2`; operational status returns `0` | Rust parity exit-code comparison pending matrix run | IN PROGRESS |

## Command Results (Local)

- `UV_CACHE_DIR=/tmp/uv-cache uv run ruff check kcmt tests` -> PASS
- `UV_CACHE_DIR=/tmp/uv-cache uv run black --check kcmt tests` -> PASS (after formatting `kcmt/main.py`)
- `UV_CACHE_DIR=/tmp/uv-cache uv run isort --check-only kcmt tests` -> PASS
- `UV_CACHE_DIR=/tmp/uv-cache uv run mypy --config-file pyproject.toml` -> PASS
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q` -> PASS (`84 passed, 1 skipped`)
- `CARGO_HOME=/tmp/cargo-home cargo check --manifest-path rust/Cargo.toml` -> FAIL (DNS resolution failure for `index.crates.io`)

## Cross-Platform Matrix Plan (T062)

- Workflow added: `.github/workflows/rust-parity-matrix.yml`
- Matrix targets: `ubuntu-latest`, `macos-latest`, `windows-latest`
- Captures:
  - Python contract probes (`kcmt`, `commit`, `kc`, `status`, invalid-flag exit code)
  - Rust candidate contract probes (`kcmt`, `commit`, `kc`, `status`, invalid-flag exit code)
  - Uploaded artifacts with per-OS exit code evidence

## Cross-Platform Matrix Results

- Latest successful run: `https://github.com/djh00t/kcmt/actions/runs/22627603656`
- Outcome:
  - `Parity (ubuntu-latest)`: success
  - `Parity (windows-latest)`: success
  - `Parity (macos-latest)`: success
- Notes:
  - GitHub cache-service warnings (`Failed to save/restore`) were emitted as annotations but did not fail jobs.

## Notes

- Normalize nondeterministic values (timestamps, temp paths) before comparing outputs.
- Attach command logs and artifact links from `parity-<os>` workflow artifacts once CI matrix runs.
