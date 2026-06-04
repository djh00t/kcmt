# Rust Fastest Automated Commit Tool Iterations

This report tracks the Rust-only optimization work. Python is retained as the side-by-side comparison baseline, but only the Rust path is optimized.

## Baseline Captured Before Implementation

Environment:

- Worktree: `005-rust-fastest-commit-tool`
- Corpus: `synthetic-untracked-1000`
- Rust build: `rtk cargo build --release`
- Benchmark command: `rtk ./target/release/kcmt benchmark runtime --repo-path <corpus> --runtime rust --iterations 5 --json`
- Static complexity scan: zero Rust findings

Baseline results:

| Scenario | Rust median ms | Notes |
| --- | ---: | --- |
| `status-repo-path` | 35.92 | Local status workflow only |
| `file-repo-path` | 321.26 | Uses Rust heuristic commit message |
| `oneshot-repo-path` | 371.26 | Uses Rust heuristic commit message |

Known limitation: current Rust production workflow still uses `heuristic_commit_message`, so these baseline numbers do not yet measure real Rust LLM-backed commit-message generation.

## Iteration 1 - Telemetry And Score Board Shape

Changes made:

- Added `WorkflowTelemetry`, `StageTiming`, and stage outcome serialization in Rust core.
- Added `WorkflowComparisonScoreboard` and `ScoreboardRow` in Rust benchmark code.
- Added `stage_timings` to runtime benchmark results and top-level `scoreboard` output.
- Added markdown score board rendering for runtime benchmark reports.
- Wrote coarse workflow telemetry into Rust status snapshots for `status_scan`, `commit`, and `snapshot`.

Validation:

- `rtk cargo test -p kcmt-core telemetry::tests`
- `rtk cargo test -p kcmt-bench scoreboard::tests`
- `rtk cargo test -p kcmt-cli --test benchmark_contracts`
- `rtk cargo test -p kcmt-cli --test workflow_modes file_mode_persists_snapshot_for_status_view`

Benchmark command:

`rtk ./target/release/kcmt benchmark runtime --repo-path <synthetic-untracked-1000> --runtime both --iterations 3 --json`

Score board:

| Stage | Python median ms | Rust median ms | Delta ms | Rust change % | Quality impact | Comparable | Notes |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| `status_scan` | 960.34 | 38.55 | -921.79 | -95.99% | neutral | yes | Coarse scenario timing |
| `file_workflow` | 967.24 | 340.45 | -626.79 | -64.80% | neutral | yes | Coarse scenario timing; Rust still heuristic |
| `oneshot_workflow` | 1090.44 | 494.88 | -595.56 | -54.62% | neutral | yes | Coarse scenario timing; Rust still heuristic |

Current bottlenecks:

- Rust benchmark stage timings are still synthetic scenario-level timings, not internal workflow stage timings from every process.
- Rust commit-message generation is still heuristic and therefore not production-quality comparable with Python LLM output.
- Python-vs-Rust rows do not yet include `llm_enqueue`, `llm_wait`, `response_validation`, or `push` as internal stage rows.

Next improvements:

1. Replace the one-scenario synthetic stage with real workflow telemetry ingestion from Rust snapshot files.
2. Add the first deterministic fake-provider path so Rust can measure LLM queue fill and post-response work without live provider variability.
3. Split Rust `file_workflow` timing into status, classification, diff preparation, queue, validation, commit, and snapshot rows.

## Iteration 2 - Reuse One Status Scan In Rust Oneshot

Changes made:

- Refactored `run_oneshot_workflow` so it reuses the status entries gathered for target selection.
- Avoided calling back into the file workflow path for a second `git status --porcelain` scan.
- Preserved file-scoped commit behavior and status snapshot telemetry.

Validation:

- `rtk cargo test -p kcmt-cli --test workflow_modes`

Benchmark command:

`rtk ./target/release/kcmt benchmark runtime --repo-path <synthetic-untracked-1000> --runtime both --iterations 3 --json`

Score board:

| Stage | Python median ms | Rust median ms | Delta ms | Rust change % | Quality impact | Comparable | Notes |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| `status_scan` | 603.89 | 36.08 | -567.80 | -94.02% | neutral | yes | Coarse scenario timing |
| `file_workflow` | 865.59 | 328.76 | -536.83 | -62.02% | neutral | yes | Coarse scenario timing; Rust still heuristic |
| `oneshot_workflow` | 991.86 | 322.22 | -669.64 | -67.51% | neutral | yes | Rust no longer scans status twice |

Rust change from iteration 1:

| Stage | Iteration 1 Rust median ms | Iteration 2 Rust median ms | Delta ms | Change |
| --- | ---: | ---: | ---: | ---: |
| `status_scan` | 38.55 | 36.08 | -2.47 | -6.41% |
| `file_workflow` | 340.45 | 328.76 | -11.69 | -3.43% |
| `oneshot_workflow` | 494.88 | 322.22 | -172.66 | -34.89% |

Current bottlenecks:

- Rust `file_workflow` still pays the full status scan and git commit cost for one selected file.
- Benchmark stage rows are still coarse scenario timings.
- Rust still has no LLM-backed queue, so quality parity remains unmeasured.

Next improvements:

1. Add deterministic fake-provider queue stages for `llm_enqueue`, `llm_wait`, and `response_validation`.
2. Start splitting Rust stage telemetry into internal rows consumed by the benchmark score board.
3. Reduce post-message git overhead only after the quality gate exists.

## Iteration 3 - Add Rust Message Quality Gate

Changes made:

- Added Rust Conventional Commit validation and provider-output sanitization helpers.
- Wired validation into the Rust file workflow before commit.
- Added `response_validation` telemetry to Rust status snapshots.

Validation:

- `rtk cargo test -p kcmt-core quality::tests`
- `rtk cargo test -p kcmt-cli --test workflow_modes file_mode_persists_snapshot_for_status_view`

Benchmark command:

`rtk ./target/release/kcmt benchmark runtime --repo-path <synthetic-untracked-1000> --runtime both --iterations 3 --json`

Score board:

| Stage | Python median ms | Rust median ms | Delta ms | Rust change % | Quality impact | Comparable | Notes |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| `status_scan` | 570.89 | 36.00 | -534.88 | -93.69% | neutral | yes | Coarse scenario timing |
| `file_workflow` | 891.65 | 331.69 | -559.95 | -62.80% | improved | yes | Rust validates heuristic message before commit |
| `oneshot_workflow` | 974.71 | 321.41 | -653.30 | -67.03% | improved | yes | Rust validates heuristic message before commit |

Rust change from iteration 2:

| Stage | Iteration 2 Rust median ms | Iteration 3 Rust median ms | Delta ms | Change |
| --- | ---: | ---: | ---: | ---: |
| `status_scan` | 36.08 | 36.00 | -0.08 | -0.22% |
| `file_workflow` | 328.76 | 331.69 | +2.94 | +0.89% |
| `oneshot_workflow` | 322.22 | 321.41 | -0.81 | -0.25% |

Current bottlenecks:

- Validation exists but is still applied to heuristic messages, not live LLM responses.
- Score board rows remain coarse because benchmark runner does not yet ingest workflow snapshot telemetry.
- The provider queue is not yet implemented in Rust.

Next improvements:

1. Have the runtime benchmark read Rust snapshot telemetry and expose internal stage rows.
2. Add fake-provider queue scaffolding for deterministic `llm_enqueue` and `llm_wait` timing.
3. Expand quality parity fixtures toward Python prompt/postprocessing behavior.

## Iteration 4 - Ingest Rust Workflow Telemetry Into Benchmarks

Changes made:

- Added `KCMT_RUNTIME_TELEMETRY_PATH` export support to Rust workflow runs.
- Updated the benchmark runner to read Rust telemetry JSON after each workflow command.
- Replaced synthetic Rust workflow rows with internal telemetry rows where available.
- Kept synthetic fallback rows for stages with no telemetry, such as the current status command and Python runs.

Validation:

- `rtk cargo test -p kcmt-bench scoreboard::tests`
- `rtk cargo test -p kcmt-cli --test benchmark_contracts`

Benchmark command:

`rtk ./target/release/kcmt benchmark runtime --repo-path <synthetic-untracked-1000> --runtime both --iterations 3 --json`

Score board:

| Stage | Python median ms | Rust median ms | Delta ms | Rust change % | Quality impact | Comparable | Notes |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| `status_scan` | 573.46 | 51.11 | -522.35 | -91.09% | neutral | yes | Python coarse status scenario; Rust internal and status rows combined |
| `commit` | - | 156.86 | - | - | neutral | no | Python stage missing |
| `response_validation` | - | 0.00 | - | - | improved | no | Python stage missing |
| `snapshot` | - | 0.00 | - | - | neutral | no | Python stage missing |
| `file_workflow` | 884.13 | - | - | - | neutral | no | Rust replaced coarse row with internal telemetry |
| `oneshot_workflow` | 975.52 | - | - | - | neutral | no | Rust replaced coarse row with internal telemetry |

Current bottlenecks:

- Internal Rust commit stage is now visible and is the dominant local post-message cost.
- Python internal stage telemetry is not yet exported, so several rows are intentionally non-comparable.
- Rust still does not enqueue real LLM work.

Next improvements:

1. Compact snapshot/report serialization and measure whether snapshot overhead remains negligible.
2. Add fake-provider queue scaffolding after the current five-iteration local workflow report is complete.
3. Add Python internal telemetry export if side-by-side stage comparability becomes more important than avoiding Python changes.

## Iteration 5 - Compact Snapshot Serialization

Changes made:

- Switched persisted Rust workflow snapshots from pretty JSON to compact JSON.
- Kept `kcmt status --raw` pretty-print behavior by formatting after parsing.
- Preserved telemetry export and status summary behavior.

Validation:

- `rtk cargo test -p kcmt-cli --test workflow_modes`

Benchmark command:

`rtk ./target/release/kcmt benchmark runtime --repo-path <synthetic-untracked-1000> --runtime both --iterations 3 --json`

Score board:

| Stage | Python median ms | Rust median ms | Delta ms | Rust change % | Quality impact | Comparable | Notes |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| `status_scan` | 594.66 | 51.78 | -542.88 | -91.29% | neutral | yes | Python coarse status scenario; Rust internal and status rows combined |
| `commit` | - | 155.86 | - | - | neutral | no | Python stage missing |
| `response_validation` | - | 0.00 | - | - | improved | no | Python stage missing |
| `snapshot` | - | 0.00 | - | - | neutral | no | Python stage missing |
| `file_workflow` | 895.77 | - | - | - | neutral | no | Rust reports internal telemetry rows |
| `oneshot_workflow` | 1016.62 | - | - | - | neutral | no | Rust reports internal telemetry rows |

Rust change from iteration 4:

| Stage | Iteration 4 Rust median ms | Iteration 5 Rust median ms | Delta ms | Change |
| --- | ---: | ---: | ---: | ---: |
| `status_scan` | 51.11 | 51.78 | +0.67 | +1.31% |
| `commit` | 156.86 | 155.86 | -1.00 | -0.64% |
| `response_validation` | 0.00 | 0.00 | ~0.00 | stable |
| `snapshot` | 0.00 | 0.00 | ~0.00 | stable |

Final status after five local iterations:

- Rust benchmark output now includes stage timing arrays and Python-vs-Rust score boards.
- Rust workflow snapshots now include telemetry.
- Rust no longer double-scans status in `--oneshot`.
- Rust validates generated commit messages before committing.
- Rust persisted snapshots are compact.
- Rust benchmarks can exercise deterministic fake LLM queue stages through `KCMT_FAKE_LLM_RESPONSE`.

Remaining gaps toward the full end-state:

- Rust now has an OpenAI-compatible provider path behind `KCMT_RUST_LLM=1`; remaining batch work is to fan out multiple file prompts concurrently beyond the current one-file workflow.
- Python internal telemetry is absent, so detailed internal rows are Rust-only or Python-only rather than fully comparable.
- Commit-message quality parity is protected by Rust Conventional Commit validation; broader corpus scoring should be added before removing the Python wrapper entirely.

Additional fake-provider verification after iteration 5:

| Stage | Python median ms | Rust median ms | Comparable | Notes |
| --- | ---: | ---: | --- | --- |
| `llm_enqueue` | - | 0.00 | no | Python stage missing |
| `llm_wait` | - | 0.00 | no | Python stage missing |
| `response_validation` | - | 0.00 | no | Python stage missing |
| `commit` | - | 153.25 | no | Python stage missing |
| `status_scan` | 584.54 | 51.06 | yes | Rust faster |

## Rust LLM Provider Path And Diff Preparation Follow-up

Changes made after the five local iterations:

- Added an OpenAI-compatible Rust chat-completions client with request payload and response parsing tests.
- Added `KCMT_RUST_LLM=1` workflow dispatch so Rust can call the provider path without Python.
- Fixed explicit provider config selection so `KCMT_PROVIDER=openai` uses `OPENAI_API_KEY` unless the API-key env name is explicitly overridden.
- Replaced placeholder provider input with real per-file evidence:
  - modified/deleted/staged files use `git diff --no-ext-diff --text`;
  - untracked files use capped direct file content;
  - all evidence is capped at 24 KB before prompt construction.
- Added `diff_preparation` telemetry before `llm_enqueue`.
- Optimized untracked-file evidence by skipping useless `git diff` calls for `??` paths.
- Added the Rust default batch workflow:
  - one status scan feeds all changed files;
  - evidence preparation runs concurrently;
  - `time_to_first_llm_enqueue` and `time_to_all_llm_enqueued` are first-class telemetry stages;
  - real Rust provider calls fan out concurrently when `KCMT_RUST_LLM=1`;
  - commits remain file-scoped and sequential for git safety.

Validation:

- `rtk cargo test -p kcmt-provider openai::tests`
- `rtk cargo test -p kcmt-core explicit_provider_uses_matching_default_api_key_env`
- `rtk cargo test -p kcmt-cli workflow::tests`
- `rtk cargo test -p kcmt-cli --test workflow_modes`
- `rtk cargo test -p kcmt-cli --test benchmark_contracts`

Fresh benchmark command:

`rtk ./rust/target/release/kcmt benchmark runtime --repo-path <synthetic-untracked-1000> --runtime both --iterations 3 --json`

Current score board:

| Stage | Python median ms | Rust median ms | Delta ms | Rust change % | Comparable | Notes |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| `status_scan` | 312.04 | 92.78 | -219.26 | -70.27% | yes | Rust internal/status rows combined |
| `diff_preparation` | - | 0.41 | - | - | no | Rust direct-read path for untracked files |
| `llm_enqueue` | - | 0.00 | - | - | no | Deterministic fake response path |
| `llm_wait` | - | 0.00 | - | - | no | Deterministic fake response path |
| `response_validation` | - | 0.00 | - | - | no | Rust Conventional Commit gate |
| `commit` | - | 277.69 | - | - | no | Dominant local post-response cost |
| `snapshot` | - | 0.00 | - | - | no | Compact JSON snapshot |
| `file_workflow` | 848.13 | - | - | - | no | Python coarse row only |
| `oneshot_workflow` | 1008.39 | - | - | - | no | Python coarse row only |

Diff-preparation optimization result:

| Stage | Before optimization | After optimization | Delta | Change |
| --- | ---: | ---: | ---: | ---: |
| `diff_preparation` | 172.73 ms | 0.41 ms | -172.32 ms | -99.76% |

Bounded full-batch benchmark command:

`rtk ./rust/target/release/kcmt benchmark runtime --repo-path <synthetic-untracked-20> --runtime both --iterations 2 --json`

Batch score board:

| Stage | Python median ms | Rust median ms | Delta ms | Rust change % | Comparable | Notes |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| `status_scan` | 303.87 | 88.05 | -215.82 | -71.02% | yes | Rust internal/status rows combined |
| `time_to_first_llm_enqueue` | - | 0.55 | - | - | no | Rust batch queue metric |
| `time_to_all_llm_enqueued` | - | 1.14 | - | - | no | Rust batch queue metric |
| `diff_preparation` | - | 0.43 | - | - | no | Concurrent Rust evidence preparation |
| `llm_enqueue` | - | 0.00 | - | - | no | Deterministic fake response path |
| `llm_wait` | - | 0.00 | - | - | no | Deterministic fake response path |
| `response_validation` | - | 0.00 | - | - | no | Rust Conventional Commit gate |
| `commit` | - | 270.37 | - | - | no | Dominant post-response cost |
| `push` | - | 83.65 | - | - | no | No-remote detection path |
| `file_workflow` | 901.25 | - | - | - | no | Python coarse row only |
| `oneshot_workflow` | 1008.24 | - | - | - | no | Python coarse row only |

The attempted 1,000-file full-batch benchmark was stopped because it measures thousands of file-scoped git commits rather than queue-fill speed. The 1,000-file queue/prep benchmark remains valid for local LLM queue preparation; the 20-file benchmark is the bounded full-workflow measurement.

## Top 10 Performance Opportunities

1. **Concurrent Rust batch queue fill**: prepare file evidence in a bounded async worker pool and enqueue provider requests immediately, preserving one-file-one-commit semantics.
2. **Persistent provider runtime**: keep one Tokio runtime and HTTP client alive for the batch instead of constructing them per file.
3. **Single status snapshot**: parse `git status --porcelain=v1 -z` once for the batch and share immutable file metadata across all workers.
4. **Diff mode by status code**: skip expensive diff commands for untracked files, deleted files, and unchanged staged paths when direct evidence is already known.
5. **Git plumbing for commit stage**: evaluate replacing repeated `git add`/`git commit` process setup with lower-level git plumbing or batched index refresh where quality and safety remain intact.
6. **Provider payload reuse**: prebuild static system/developer prompt fragments and serialize only the per-file evidence during queue fill.
7. **Prompt evidence budgeter**: classify generated, binary, vendored, and huge files before prompt construction so tokens go to meaningful hunks first.
8. **Async post-response pipeline**: validate, sanitize, stage, commit, snapshot, and push through separate bounded queues after each LLM response arrives.
9. **Python telemetry bridge**: export Python internal stages to make every score-board row directly comparable during the migration window.
10. **Quality scoring corpus**: add deterministic commit-message scoring fixtures so speed changes are blocked when message quality regresses.
