# Tasks Validation: Rust CLI Feature Parity and Runtime Benchmark Mode

## Validation Date

- 2026-06-05 (Australia/Sydney)
- Branch/worktree: `codex/rust-feature-equivalence` in `.worktrees/rust-feature-equivalence`
- Rust binary validated: `rust/target/release/kcmt`

## Quality Gates

### Python, BDD, Rust, and Ink

Command:

```bash
rtk proxy make check
```

Result:

- Passed.
- Python strict test gate passed.
- Rust workspace tests passed.
- Ink model tests passed.

Command:

```bash
rtk proxy make quality-gates
```

Result:

- Passed.
- Python: `149 passed, 1 skipped`.
- Coverage: `93.88%`.
- Rust and Ink checks passed.

### Cross-Platform PR Matrix

PR34 head `bcb997817ac2e1af450e8efac3a490da3dcff888` passed all required
GitHub checks after the Rust parity updates:

- CI / Python 3.12: passed in Actions run `26971474089`.
- CI / Python 3.13: passed in Actions run `26971474089`.
- Rust Canary Smoke / ubuntu: passed in Actions run `26971474090`.
- Keystone Assimilation Watcher: passed in Actions run `26971474102`.
- Rust Parity Matrix / ubuntu-latest: passed in Actions run `26971474093`.
- Rust Parity Matrix / macos-latest: passed in Actions run `26971474093`.
- Rust Parity Matrix / windows-latest: passed in Actions run `26971474093`.

### Focused Rust and Runtime Parity

Commands:

```bash
rtk cargo test --manifest-path rust/Cargo.toml -p kcmt-cli workflow::tests -- --nocapture
rtk cargo test --manifest-path rust/Cargo.toml -p kcmt-cli --test workflow_modes -- --nocapture
rtk cargo test --manifest-path rust/Cargo.toml --workspace --no-fail-fast
rtk uv run pytest -q --no-cov tests/test_main_entrypoint.py tests/test_rust_workflow_parity_bdd.py tests/test_benchmark.py::test_run_runtime_benchmark_produces_python_results tests/test_cli.py::test_cli_runtime_benchmark_json
KCMT_ALLOW_PROVIDER_RESPONSE_FIXTURE=1 KCMT_PROVIDER_RESPONSE=$'fix(core): preserve benchmark quality\n\nMeasure deterministic Rust provider scoring.' OPENAI_API_KEY=test-openai-key rust/target/release/kcmt --benchmark --provider openai --model gpt-bdd --benchmark-limit 1 --benchmark-timeout 0.1 --benchmark-json --repo-path .
```

Result:

- `workflow::tests`: `10 passed`.
- `workflow_modes`: `29 passed`.
- Rust workspace: `85 passed`.
- Focused Python/BDD runtime parity: `48 passed`.
- Release provider quality fixture: `5` runs, quality `100.0`, success rate
  `1.0`; JSON artifact `/tmp/kcmt-provider-benchmark-quality-after-perf.json`.

## Runtime Benchmark Evidence

### Synthetic 1,000-File Corpus

Command:

```bash
uv run python scripts/benchmark/generate_uncommitted_repo.py --file-count 1000 --json
KCMT_PROVIDER_RESPONSE='chore(repo): benchmark fake response' \
KCMT_ALLOW_PROVIDER_RESPONSE_FIXTURE=1 \
KCMT_RUNTIME=python \
KCMT_CONFIG_HOME=/tmp/kcmt-runtime-bench-config-* \
uv run kcmt benchmark runtime \
  --repo-path "$REPO" \
  --runtime both \
  --iterations 1 \
  --rust-bin rust/target/release/kcmt \
  --json
```

Result artifact: `/tmp/kcmt-runtime-current-synthetic-1000-fixed.json`

- Passed.
- Corpus: `synthetic-untracked-1000`.
- Command set: `status-repo-path`, `oneshot-repo-path`, and `file-repo-path`; the documented large synthetic corpus excludes default multi-file commits so runtime mode remains focused on startup, repo scanning, explicit repo-path handling, and file-scoped prompt preparation.
- Python summary: `3 passed / 0 failed / 0 excluded`.
- Rust summary: `3 passed / 0 failed / 0 excluded`.
- Median wall time:
  - Python: `806.140583 ms`.
  - Rust: `227.613875 ms`.
- Speedup: `71.76%`.

### Realistic Checked-In Corpus

Command:

```bash
KCMT_PROVIDER_RESPONSE='chore(repo): benchmark fake response' \
KCMT_ALLOW_PROVIDER_RESPONSE_FIXTURE=1 \
KCMT_RUNTIME=python \
KCMT_CONFIG_HOME=/tmp/kcmt-runtime-bench-config-* \
uv run kcmt benchmark runtime \
  --repo-path tests/fixtures/runtime_corpus/mini_realistic_repo \
  --runtime both \
  --iterations 1 \
  --rust-bin rust/target/release/kcmt \
  --json
```

Result artifact: `/tmp/kcmt-runtime-current-mini-fixed.json`

- Passed.
- Corpus: `mini-realistic-fixture`.
- Command set: `status-repo-path`, `oneshot-repo-path`, `default-repo-path`, and `file-repo-path`.
- Python summary: `4 passed / 0 failed / 0 excluded`.
- Rust summary: `4 passed / 0 failed / 0 excluded`.
- Median wall time:
  - Python: `863.963104 ms`.
  - Rust: `200.194584 ms`.
- Speedup: `76.83%`.
- Rust workflow results include normalized stage rows for `status_scan`,
  `diff_preparation`, `llm_enqueue`, `llm_wait`, `response_validation`,
  `commit`, `push`, and `snapshot`.

## Live Provider Quality Evidence

Command shape:

```bash
env -i PATH="$PATH" HOME="$HOME" USER="$USER" TMPDIR="$TMPDIR" \
  GITHUB_TOKEN="$GITHUB_TOKEN" KCMT_CONFIG_HOME=/tmp/kcmt-live-quality-* \
  <python-or-rust-kcmt> --benchmark \
    --provider github \
    --model openai/gpt-4.1-mini \
    --benchmark-limit 1 \
    --benchmark-timeout 30 \
    --benchmark-json \
    --repo-path .
```

Result:

- Python provider benchmark: quality `92.0`, success `100%`, runs `5`.
- Rust provider benchmark: quality `92.0`, success `100%`, runs `5`.
- Quality delta: `0.0` points.
- Live Anthropic benchmark attempts with `claude-3-5-haiku-latest` returned 404
  in both Python and Rust and were not used for quality-delta evidence.

## Validation Fixes Landed During Execution

- Rust default non-interactive workflow now commits all changed files separately,
  while `--oneshot` selects and commits exactly one file.
- Python auto runtime dispatch covers non-TTY default workflow invocations while
  keeping TTY no-arg interactive behavior on Python.
- Rust workflow snapshots normalize stage timing rows needed by performance
  scoreboards.
- Rust default workflow records per-file prepare/commit failures in the snapshot
  and continues committing other files when at least one file can still succeed.
- Rust auto-push preflight skips repositories with no local `origin` without a
  `git config` subprocess where local `.git/config` is sufficient.
- Rust `--max-retries` is wired through provider and batch retry policies.
- Rust `--github-token` populates `GITHUB_TOKEN` for GitHub Models calls.
- Runtime benchmark command initialization now includes `max_retries` and
  `prepare_workers` overrides so BDD-time Rust binary builds pass.
- Rust production workflow paths ignore `KCMT_PROVIDER_RESPONSE` unless
  `KCMT_ALLOW_PROVIDER_RESPONSE_FIXTURE=1` or runtime benchmark mode is set,
  preventing test fixture output from bypassing provider configuration.
- `.github/workflows/rust-parity-matrix.yml` now runs Rust workspace tests and
  Python wrapper/BDD parity tests across Linux, macOS, and Windows, in addition
  to the existing contract probes.

## Conclusion

- User Story 1 parity workflows are implemented and validated for the required
  command catalog covered by current BDD and Rust integration tests.
- User Story 2 runtime benchmark mode is implemented, schema-backed, and proven
  on both synthetic and realistic corpora with the release Rust binary.
- User Story 3 benchmark UX separation is enforced in CLI routing, Ink backend
  behavior, and documentation.
- Cross-platform proof has been refreshed on PR34, with Linux, macOS, and
  Windows parity matrix jobs passing on the published feature branch.
