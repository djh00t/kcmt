# Feature Specification: Rust Fastest Automated Commit Tool

**Feature Branch**: `005-rust-fastest-commit-tool`  
**Created**: 2026-06-04  
**Status**: Draft  
**Input**: User request to optimize the Rust automated commit tool, add stage telemetry, baseline performance, find bottlenecks, identify the top ten speed improvements, and iterate five times without reducing LLM commit-message quality.

## User Scenarios & Testing

### User Story 1 - See Where Time Is Spent (Priority: P1)

As a developer using `kcmt`, `commit`, or `kc`, I can run the Rust workflow and see precise timing for each stage from git discovery through push, so performance regressions and bottlenecks are visible.

**Why this priority**: Optimization without trustworthy telemetry risks improving the wrong stage.

**Independent Test**: Run the Rust workflow against a deterministic corpus and verify the telemetry report contains stage timings, counts, queue latency, provider wait time, commit time, and push time.

**Acceptance Scenarios**:

1. **Given** a repository with changed files, **When** the Rust workflow runs, **Then** telemetry records repo discovery, status scan, diff preparation, LLM enqueue, LLM wait, validation, commit, snapshot, and push stages.
2. **Given** a repository with multiple changed files, **When** batch mode is enabled, **Then** telemetry records time-to-first-LLM-enqueue and time-to-all-LLM-enqueued.
3. **Given** a workflow failure, **When** telemetry is emitted, **Then** the failed stage and elapsed time are preserved without leaking secrets.
4. **Given** equivalent Python and Rust benchmark runs, **When** score board results are rendered, **Then** each comparable stage is shown side by side with absolute timings, deltas, and percent change.

### User Story 2 - Preserve Commit Message Quality (Priority: P1)

As a developer, I get commit messages at least as good as the existing Python LLM workflow while the Rust path becomes faster.

**Why this priority**: Fast low-quality messages are explicitly unacceptable.

**Independent Test**: Run a deterministic commit-quality corpus through the Rust prompt, postprocessor, and validator and compare quality scores and conventional-commit validity against the Python baseline.

**Acceptance Scenarios**:

1. **Given** changed files with staged, unstaged, new, deleted, and renamed paths, **When** Rust generates messages, **Then** all accepted messages validate as Conventional Commits.
2. **Given** malformed provider output, **When** Rust postprocessing runs, **Then** it either recovers a valid message or records a provider-quality failure.
3. **Given** baseline Python prompt behavior, **When** equivalent Rust prompt behavior is measured, **Then** quality does not regress by more than two percentage points.

### User Story 3 - Queue Every File To The LLM Quickly (Priority: P1)

As a developer working in a large repository, I want every changed file prepared and enqueued to the LLM as quickly as possible, with local work parallelized before waiting on provider responses.

**Why this priority**: The user identified batch mode and fast queue fill as the primary performance goal.

**Independent Test**: Run a 1,000-file deterministic corpus and verify local preparation plus enqueue latency improves iteration over iteration while preserving output quality gates.

**Acceptance Scenarios**:

1. **Given** 1,000 changed files, **When** the Rust batch workflow starts, **Then** local file discovery and diff preparation run without per-file repeated full status scans.
2. **Given** a configurable worker count, **When** diff preparation is CPU or I/O bound, **Then** concurrency is bounded and deterministic.
3. **Given** provider rate limits, **When** LLM queueing runs, **Then** the scheduler respects concurrency limits and records wait/backoff time separately from local preparation time.

### User Story 4 - Commit And Push Quickly After LLM Response (Priority: P2)

As a developer, once the LLM returns, Rust should validate, commit, snapshot, and push with minimal overhead while keeping atomic git safety.

**Why this priority**: Post-provider latency remains open for optimization and affects perceived speed.

**Independent Test**: Use a fake provider response set and measure validation-to-commit and commit-to-push timings for one-file and many-file runs.

**Acceptance Scenarios**:

1. **Given** prepared LLM responses, **When** the Rust workflow commits each file, **Then** only intended file paths are committed.
2. **Given** auto-push is enabled, **When** all commits succeed, **Then** Rust pushes once after successful commit completion.
3. **Given** a commit failure, **When** later files remain pending, **Then** git state and telemetry clearly record which file failed.

## Edge Cases

- Empty repository, no changes, only ignored files, and binary-looking files.
- New untracked files with no diff from `git diff`.
- Deleted files where content is unavailable.
- Renamed paths and paths containing spaces.
- Large diffs requiring truncation or summarization.
- Provider timeout, rate limit, malformed response, empty response, and invalid Conventional Commit response.
- Auto-push failure after successful local commits.
- Existing staged files unrelated to the target file.

## Requirements

### Functional Requirements

- **FR-001**: Rust MUST replace `heuristic_commit_message` in production commit flows with LLM-backed commit-message generation.
- **FR-002**: Rust MUST support the current provider/config model needed for OpenAI batch mode and future provider parity.
- **FR-003**: Rust MUST emit machine-readable stage telemetry for every workflow run.
- **FR-004**: Rust MUST expose benchmark output that compares iteration baseline, current run, and percent improvement.
- **FR-005**: Rust MUST preserve file-scoped atomic commit behavior.
- **FR-006**: Rust MUST support deterministic fake-provider benchmark mode for repeatable local measurements.
- **FR-007**: Rust MUST keep Python as reference or wrapper only; no hot-path optimization may depend on Python.
- **FR-008**: Rust MUST validate all accepted LLM responses as Conventional Commits before committing.
- **FR-009**: Rust MUST add executable BDD feature coverage for telemetry, batch queueing, and quality-preserving commit generation.
- **FR-010**: Rust MUST produce a five-iteration performance report under `docs/performance/`.
- **FR-011**: Benchmark score boards MUST compare Python and Rust workflow stages side by side for all comparable stages.
- **FR-012**: Score boards MUST mark non-comparable stages explicitly, such as Rust stages not yet implemented or Python-only wrapper/TUI stages.

### Non-Functional Requirements

- **NFR-001**: The benchmark corpus and commands MUST be deterministic and scriptable.
- **NFR-002**: Time-to-all-LLM-enqueued MUST be a first-class metric.
- **NFR-003**: Local pre-provider processing should target sub-second median time on the 1,000-file synthetic corpus after optimization.
- **NFR-004**: Commit-message quality score MUST NOT regress by more than two percentage points from the Python baseline.
- **NFR-005**: Telemetry MUST NOT persist API keys, prompt secrets, environment secrets, or raw provider credentials.
- **NFR-006**: CLI aliases and automation-facing JSON outputs MUST remain compatible or explicitly versioned.
- **NFR-007**: Python-vs-Rust comparisons MUST use the same corpus, provider mode, fake-provider timing, and iteration count unless the report states the difference.

### Key Entities

- **WorkflowTelemetry**: Per-run timing and count data for workflow stages.
- **StageTiming**: Named stage measurement with start, end, duration, outcome, and optional file path.
- **PreparedChange**: File path, change type, diff or content summary, prompt context, and queue state.
- **LlmQueueItem**: Provider request payload metadata, enqueue time, response time, retry state, and validation result.
- **CommitQualityResult**: Conventional Commit validity, recovery status, and score dimensions.
- **PerformanceIterationReport**: Baseline, changes made, benchmark results, quality results, next bottlenecks, and next actions.
- **WorkflowComparisonScoreboard**: Side-by-side Python and Rust stage timings with deltas, percent change, quality score, and comparability notes.

## Top Ten Optimization Targets

1. Replace the Rust heuristic path with real LLM-backed generation and measure it.
2. Build a single repository status/change scan reused by all files.
3. Prepare diffs/content summaries concurrently with bounded workers.
4. Enqueue all LLM batch items as soon as each file is prepared.
5. Reuse a shared async HTTP client and provider session state.
6. Separate local preparation time from provider wait time and retry/backoff time.
7. Add deterministic fake-provider benchmarks for post-LLM commit throughput.
8. Reduce git subprocess count through batched status, diff, add, and commit strategy where safe.
9. Make snapshot/report writing compact and optionally deferred.
10. Keep prompt/postprocessing quality gates strict while reducing prompt construction overhead.

## Success Criteria

### Measurable Outcomes

- **SC-001**: Baseline report captures current Rust status, file, and oneshot medians on a 1,000-file corpus.
- **SC-002**: Each of five iterations includes a report with benchmark deltas and next bottlenecks.
- **SC-003**: Final Rust local preparation time-to-all-LLM-enqueued improves by at least 50% against the first real LLM/fake-provider queue baseline.
- **SC-004**: Final post-LLM validation-to-local-commit median improves by at least 30% against the first fake-provider baseline.
- **SC-005**: Commit-message quality score remains within two percentage points of Python baseline and all accepted messages validate.
- **SC-006**: `make check` passes before implementation handoff; `make quality-gates` is run before publish/PR.
- **SC-007**: Every iteration report includes a Python-vs-Rust score board for comparable workflow stages.

