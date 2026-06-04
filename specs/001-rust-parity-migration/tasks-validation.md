# Task Validation Evidence

## Run Metadata

- Date: 2026-06-05 (Australia/Sydney)
- Branch/worktree: `codex/rust-feature-equivalence` in `.worktrees/rust-feature-equivalence`
- Runtime modes validated: `python`, `rust`, and Python wrapper `auto`
- Rust binary validated: `rust/target/release/kcmt`
- Fixture set revision: current working tree plus `tests/fixtures/runtime_corpus/mini_realistic_repo/`

## Current Gate Results

| Gate | Command | Result |
|---|---|---|
| Focused Rust workflow unit helpers | `rtk cargo test --manifest-path rust/Cargo.toml -p kcmt-cli workflow::tests -- --nocapture` | PASS, 10 passed |
| Focused Rust workflow integration | `rtk cargo test --manifest-path rust/Cargo.toml -p kcmt-cli --test workflow_modes -- --nocapture` | PASS, 29 passed |
| Focused Python/BDD runtime parity | `rtk uv run pytest -q --no-cov tests/test_main_entrypoint.py tests/test_rust_workflow_parity_bdd.py tests/test_benchmark.py::test_run_runtime_benchmark_produces_python_results tests/test_cli.py::test_cli_runtime_benchmark_json` | PASS, 48 passed |
| Rust workspace | `rtk cargo test --manifest-path rust/Cargo.toml --workspace --no-fail-fast` | PASS, 85 passed |
| Repository check gate | `rtk proxy make check` | PASS, all checks passed |
| Repository quality gate | `rtk proxy make quality-gates` | PASS, 149 Python tests passed, 1 skipped, Rust and Ink tests passed, coverage 93.88% |
| Release build | `rtk cargo build --release --manifest-path rust/Cargo.toml -p kcmt-cli` | PASS |
| Release provider quality fixture | `rust/target/release/kcmt --benchmark --provider openai --model gpt-bdd --benchmark-limit 1 --benchmark-json --repo-path .` with `KCMT_ALLOW_PROVIDER_RESPONSE_FIXTURE=1` | PASS, 5 runs, quality 100.0, success rate 1.0 |

## FR Validation

| Requirement | Evidence | Status |
|---|---|---|
| FR-001 | Rust `kcmt`, `commit`, and `kc` binaries build and route through shared dispatch; BDD covers `kcmt`, `commit`, default non-interactive, `--oneshot`, `--file`, `status`, configure, list-models, verify-keys, provider benchmark, and runtime benchmark flows. | PASS (current local) |
| FR-002 | Rust workflow tests validate conventional commit sanitization, invalid-output rejection, deletion messages, retry limits, fallback provider messages, fixture hook opt-in, file-scoped commits, and per-file failure recovery without blocking successful commits. | PASS (current local) |
| FR-003 | `workflow_modes.rs` and BDD validate `--file tracked.py` commits only the target path and nested untracked file mode commits only the requested path. | PASS (current local) |
| FR-004 | Rust config loader contract tests and BDD validate CLI/config/env overrides, provider entries, batch settings, auto-push, max commit length, max retries, workers, and persisted config compatibility. | PASS (current local) |
| FR-005 | Rust provider tests cover OpenAI-compatible, Anthropic, xAI, GitHub Models, provider fallback, malformed output, retry-limited 500 handling, GitHub token wiring, and OpenAI batch upload/poll/download. | PASS (current local) |
| FR-006 | Provider benchmark and runtime benchmark tests pass; runtime benchmark JSON includes summary, stage timings, and baseline plus five optimization rows. | PASS (current local) |
| FR-007 | Runtime benchmark evidence below shows Rust release binary faster than Python on both checked-in realistic and synthetic 1,000-file corpora for the three scenario set. | PASS for current deterministic corpora |
| FR-008 | `make check` and `make quality-gates` pass with strict Python, Rust, and Ink coverage. | PASS (current local) |
| FR-009 | BDD and contract tests validate parser errors, status raw JSON, runtime trace, missing runtime exclusion, and wrapper fallback/selection behavior. | PASS (current local) |
| NFR-005 | PR34 head `680134f71248635cbc928459a3c9cac30f466c53` passed `.github/workflows/rust-parity-matrix.yml` on `ubuntu-latest`, `macos-latest`, and `windows-latest` in GitHub Actions run `26976496281`. The matrix runs Rust workspace tests plus Python wrapper/BDD parity tests across all three OSes. | PASS (fresh branch CI) |

## Runtime Benchmark Evidence

### Checked-In Realistic Corpus

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

- Corpus: `mini-realistic-fixture`
- Command set: `status-repo-path`, `oneshot-repo-path`, `default-repo-path`, and `file-repo-path`
- Python summary: `4 passed / 0 failed / 0 excluded`, median `863.963104 ms`
- Rust summary: `4 passed / 0 failed / 0 excluded`, median `200.194584 ms`
- Speedup: `76.83%`
- Rust workflow stage rows include `status_scan`, `diff_preparation`, `llm_enqueue`, `llm_wait`, `response_validation`, `commit`, `push`, and `snapshot`.

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

- Corpus: `synthetic-untracked-1000`
- Command set: `status-repo-path`, `oneshot-repo-path`, and `file-repo-path`; the documented large synthetic corpus intentionally excludes default multi-file commits so the benchmark measures startup, repo scanning, explicit repo-path handling, and file-scoped prompt preparation instead of committing all 1,000 untracked files.
- Python summary: `3 passed / 0 failed / 0 excluded`, median `806.140583 ms`
- Rust summary: `3 passed / 0 failed / 0 excluded`, median `227.613875 ms`
- Speedup: `71.76%`

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

Results from 2026-06-05:

- Python provider benchmark: quality `92.0`, success `100%`, runs `5`.
- Rust provider benchmark: quality `92.0`, success `100%`, runs `5`.
- Quality delta: `0.0` points, within SC-005's allowed `-2` point regression threshold.

## Release Smoke Evidence

Command shape:

```bash
env -i PATH="$PATH" HOME="$HOME" USER="$USER" TMPDIR="$TMPDIR" \
  KCMT_ALLOW_LOCAL_SYNTHESIS=1 KCMT_RUNTIME_BENCHMARK=0 \
  KCMT_CONFIG_HOME=/tmp/kcmt-smoke-config-* \
  rust/target/release/kcmt ...
```

Results:

- `--oneshot --no-auto-push` committed `alpha.py` and left `M beta.py`.
- Default non-interactive `--no-auto-push` committed both `alpha.py` and `beta.py` and left a clean worktree.

## Success Criteria Evidence

| Criterion | Evidence | Status |
|---|---|---|
| SC-001 median speedup | Rust release medians are `76.83%` faster than Python on `mini-realistic-fixture` and `71.76%` faster on `synthetic-untracked-1000`, exceeding the >=50% target on both deterministic corpora. | PASS for deterministic corpora |
| SC-002 <=2s preprocessing (95%) | Runtime benchmark scenarios completed under 2s wall time including CLI overhead in current local runs; synthetic `status`, `oneshot`, and `file` scenarios and realistic `status`, `oneshot`, `default`, and `file` scenarios all passed. | PASS for current deterministic corpora |
| SC-003 workflow parity | BDD parity suite covers high-usage workflows, including default multi-file recovery when one file fails preparation, and passed in current local gates. | PASS (current local) |
| SC-004 config compatibility | Rust config contract tests and Python config tests passed in `make quality-gates`. | PASS (current local) |
| SC-005 quality delta | Live GitHub Models provider-quality benchmark scored Python and Rust at `92.0` with `100%` success on the same five-sample benchmark set; deterministic sanitizer and fixture tests also preserve conventional commit validity. | PASS current local |
| SC-006 exit/error parity | BDD and contract tests cover parser/config/provider/git failure classes and passed in current local gates. | PASS (current local) |

## Notes

- Current local validation supersedes the older March 2026 DNS-blocked Rust probe. PR34 head `680134f71248635cbc928459a3c9cac30f466c53` passed the fresh cross-platform Rust parity matrix across Linux, macOS, and Windows in GitHub Actions run `26976496281`.
- Live Anthropic benchmark attempts with `claude-3-5-haiku-latest` returned 404 in both Python and Rust and were not used for quality-delta evidence.
- Runtime benchmark artifacts in `/tmp` are generated evidence and are not committed.
- `KCMT_PROVIDER_RESPONSE` is fixture-only in Rust production workflow paths unless
  `KCMT_ALLOW_PROVIDER_RESPONSE_FIXTURE=1` or runtime benchmark mode is set; BDD
  and Rust tests cover both the accepted and ignored cases.
