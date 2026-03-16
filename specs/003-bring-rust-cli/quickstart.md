# Quickstart: Rust CLI Parity and Runtime Benchmark Mode

## Prerequisites

- Python 3.12 + `uv`
- Rust toolchain with Cargo
- Project dependencies installed:

```bash
uv sync --all-extras --dev
uv pip install -e .
cargo build --release --manifest-path rust/Cargo.toml -p kcmt-cli
```

## 1) Build a Synthetic Runtime Corpus

```bash
python scripts/benchmark/generate_uncommitted_repo.py --file-count 1000 --json
```

Capture the returned `repo_path` for the next commands.

## 2) Validate Python Baseline Contracts

```bash
uv run kcmt --help
uv run kcmt status --repo-path "$REPO"
```

Expected:
- Help exits `0`.
- `status --repo-path` accepts the explicit repo path even when no prior kcmt run
  history exists.

## 3) Validate Rust Contract Parity

```bash
./rust/target/release/kcmt --help
./rust/target/release/kcmt status --repo-path "$REPO"
./rust/target/release/kcmt --oneshot --repo-path "$REPO"
```

Expected:
- Rust accepts the same in-scope command shapes as Python.
- Exit codes and stderr behavior match the parity catalog for the scenario.

## 4) Run Provider Benchmark Backward-Compatibility Check

```bash
uv run kcmt --benchmark --benchmark-json
```

Expected:
- Existing provider/model benchmark behavior remains available.
- Output continues to represent provider-quality results, not runtime timing.

## 5) Run Runtime Benchmark Mode

```bash
uv run kcmt benchmark runtime --repo-path "$REPO" --runtime both --json
```

Expected:
- The report includes separate Python and Rust results.
- Output validates against `contracts/runtime-benchmark.schema.json`.
- Missing or unsupported runtimes appear as explicit exclusions, not silent omissions.

## 6) Run Quality Gates

```bash
cargo test --manifest-path rust/Cargo.toml
uv run pytest -q
make check
```

Expected: all tests and quality gates pass before merge.
