# Tasks Validation: Rust CLI Feature Parity and Runtime Benchmark Mode

## Validation Date

- 2026-03-16

## Quality Gates

### Python

Command:

```bash
make check
```

Result:

- Passed
- `117 passed, 1 skipped`
- Coverage: `93.33%`

### Rust

Command:

```bash
cargo test -p kcmt-cli -p kcmt-bench -p kcmt-core
```

Result:

- Passed
- `kcmt-cli`: parser, status, workflow, and benchmark contracts passed
- `kcmt-core`: config and git adapter regressions passed
- `kcmt-bench`: compiled and linked through the runtime benchmark command path

## Runtime Benchmark Evidence

### Synthetic 1,000-file corpus

Command:

```bash
REPO=$(./.venv/bin/python scripts/benchmark/generate_uncommitted_repo.py --file-count 1000 --json | ./.venv/bin/python -c 'import json,sys; print(json.load(sys.stdin)["repo_path"])')
KCMT_RUST_BIN="$PWD/rust/target/debug/kcmt" ./.venv/bin/python -m kcmt.main benchmark runtime --repo-path "$REPO" --runtime both --iterations 1 --json
```

Result:

- Passed
- Corpus: `synthetic-untracked-1000`
- Python summary: `3 passed / 0 failed / 0 excluded`
- Rust summary: `3 passed / 0 failed / 0 excluded`
- Median wall time:
  - Python: `842.477583 ms`
  - Rust: `344.909875 ms`

### Realistic checked-in corpus

Command:

```bash
KCMT_RUST_BIN="$PWD/rust/target/debug/kcmt" ./.venv/bin/python -m kcmt.main benchmark runtime --repo-path tests/fixtures/runtime_corpus/mini_realistic_repo --runtime both --iterations 1 --json
```

Result:

- Passed
- Corpus: `mini-realistic-fixture`
- Python summary: `3 passed / 0 failed / 0 excluded`
- Rust summary: `3 passed / 0 failed / 0 excluded`
- Median wall time:
  - Python: `862.526917 ms`
  - Rust: `356.545625 ms`

## Validation Fixes Landed During Execution

- Rust git porcelain handling now expands nested untracked files using
  `--porcelain=v1 -z --untracked-files=all`, which fixed synthetic-corpus
  `--file`, `--oneshot`, and status snapshot preparation.
- Python runtime benchmark dispatch now preserves the explicit `benchmark runtime
  --repo-path` corpus path instead of widening it to the enclosing repository
  root.
- Ink benchmark UI now rejects runtime benchmark payloads explicitly and points
  operators to the legacy CLI runtime benchmark mode.

## Conclusion

- User Story 1 parity workflows are implemented and validated for the required-now
  catalog.
- User Story 2 runtime benchmark mode is implemented, schema-backed, and proven on
  both synthetic and realistic corpora.
- User Story 3 benchmark UX separation is enforced in CLI routing, Ink backend
  behavior, and user-facing documentation.
