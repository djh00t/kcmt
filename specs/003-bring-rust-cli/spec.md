# Feature Specification: Rust CLI Feature Parity and Runtime Benchmark Mode

**Feature Branch**: `003-bring-rust-cli`  
**Created**: 2026-03-15  
**Status**: Draft  
**Input**: User description: "Bring the Rust CLI to feature parity with the Python workflow, add runtime benchmark support for Python vs Rust on a shared repo corpus, and make the benchmark story coherent with existing LLM benchmarking."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Run the Same CLI Workflow on Rust (Priority: P1)

As a developer using `kcmt`, `commit`, or `kc`, I can execute the same documented
workflow on the Rust CLI that I can execute on the Python CLI, including repo
selection and file-scoped flows, so Rust can become a real runtime candidate
instead of a partial shell.

**Why this priority**: Runtime cutover is not credible until Rust can perform the
same user-visible jobs as the current Python CLI.

**Independent Test**: Build the Rust binaries and run the high-usage workflow
catalog against fixture repositories, verifying that Rust accepts the same command
shapes and produces equivalent exit codes, commit behavior, and machine-readable
outputs for the supported workflows.

**Acceptance Scenarios**:

1. **Given** a repository with uncommitted changes, **When** the user runs the Rust
   CLI with `--repo-path`, **Then** the command operates on the selected repository
   instead of requiring a manual `cd`.
2. **Given** the user invokes `kcmt`, `commit`, or `kc` with shared workflow flags
   such as `--oneshot` or `--file`, **When** the Rust runtime is selected,
   **Then** behavior matches the Python workflow contract for the same repository
   state.
3. **Given** a documented error case such as an invalid flag, missing repository,
   or unsupported operation, **When** the Rust CLI handles the request, **Then**
   the exit code and stderr contract match the validated baseline.

---

### User Story 2 - Benchmark Python and Rust on the Same Repo Corpus (Priority: P2)

As a maintainer evaluating the migration, I can run a dedicated runtime benchmark
mode that compares Python and Rust against the same reproducible repo corpus so
performance claims are based on measured evidence.

**Why this priority**: The migration is performance-driven, and the project
constitution requires a reproducible baseline and corpus for any performance claim.

**Independent Test**: Run the runtime benchmark mode on a deterministic synthetic
fixture repo and at least one realistic repo corpus fixture, then verify that the
benchmark report contains both runtimes, stable metrics, and reproducible corpus
metadata.

**Acceptance Scenarios**:

1. **Given** a generated fixture repository with 1,000 uncommitted files, **When**
   the runtime benchmark is executed, **Then** the report includes Python and Rust
   measurements for the same command set and corpus identifier.
2. **Given** a benchmark report consumer, **When** runtime benchmark results are
   exported, **Then** the output includes stable machine-readable fields for
   runtime, command, corpus, timing, and outcome.
3. **Given** a missing Rust binary or unsupported runtime selection, **When** the
   runtime benchmark runs, **Then** the result records the exclusion or failure
   reason without silently dropping the runtime from the report.

---

### User Story 3 - Keep Runtime and LLM Benchmarks Coherent (Priority: P3)

As a maintainer, I can compare runtime performance and LLM quality without mixing
the two into a misleading single score, so benchmark outputs stay interpretable
and historically comparable.

**Why this priority**: The current benchmark feature already measures provider/model
quality; adding runtime timing into the same leaderboard without separation would
degrade clarity and invalidate historical comparisons.

**Independent Test**: Run both benchmark modes and verify that runtime reports and
provider-quality reports are distinct artifacts with explicit dimensions and stable
output schemas.

**Acceptance Scenarios**:

1. **Given** the existing provider benchmark workflow, **When** maintainers run it,
   **Then** provider/model quality scoring remains available and backward compatible.
2. **Given** the new runtime benchmark workflow, **When** maintainers run it,
   **Then** runtime comparisons are emitted separately from provider-quality
   leaderboards.
3. **Given** a maintainer wants to reason about runtime plus provider choice,
   **When** they inspect reports, **Then** shared corpus metadata and command labels
   make the two benchmark modes comparable without combining them into one score.

### Edge Cases

- Rust runtime is selected but the Rust binary is missing, non-executable, or built
  from a stale revision.
- Python and Rust accept the same top-level command but differ in subcommand flags
  or exit-code handling.
- Benchmark corpus includes ignored files, nested directories, deletions, binary-ish
  content, or filenames that stress path handling.
- A repo fixture is created in a temp directory with zero commit history and only
  untracked files.
- Provider-quality benchmark configuration is present but API keys are unavailable;
  runtime benchmarking must still run.
- Experimental Git backends such as `gitoxide` are evaluated; parity or safety gaps
  must not change default git behavior silently.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The Rust CLI MUST support the high-usage workflow catalog currently
  validated for the Python CLI, including `kcmt`, `commit`, `kc`, `--repo-path`,
  `--oneshot`, `--file`, `status`, and benchmark entry points that are documented
  as supported for the migrated runtime.
- **FR-002**: The Rust CLI MUST preserve validated command-line contracts for
  supported workflows, including argument shapes, exit codes, and machine-readable
  outputs used by automation.
- **FR-003**: The system MUST provide a dedicated runtime benchmark mode that
  compares Python and Rust on the same repo corpus and command set.
- **FR-004**: The runtime benchmark mode MUST accept both generated synthetic corpus
  fixtures and checked-in realistic corpus fixtures.
- **FR-005**: Runtime benchmark reports MUST record per-runtime outcome details,
  including runtime identifier, command identifier, corpus identifier, wall-clock
  duration, and failure or exclusion reason when applicable.
- **FR-006**: The existing provider/model benchmark mode MUST remain available and
  MUST NOT be replaced by the runtime benchmark mode.
- **FR-007**: The system MUST document the distinction between runtime benchmarking
  and provider-quality benchmarking, including when to use each mode.
- **FR-008**: The system MUST include a maintained fixture-generation path for large
  uncommitted repositories suitable for repeatable runtime benchmarking.
- **FR-009**: Any Git backend changes introduced for Rust performance work, including
  evaluation of `gitoxide`, MUST remain behind explicit implementation choices or
  flags until parity and safety validation pass.
- **FR-010**: Rust parity work MUST include regression tests that prove feature
  equivalence for every newly supported workflow before the runtime is treated as a
  default candidate.

### Non-Functional Requirements *(mandatory)*

- **NFR-001**: Rust CLI compatibility MUST reach 100% pass on the agreed workflow
  catalog for the workflows declared in scope for this feature.
- **NFR-002**: Runtime benchmark claims MUST be reproducible from repository docs
  using checked-in commands, a defined corpus, and stable output fields.
- **NFR-003**: Runtime benchmark output MUST remain machine-readable and versionable
  so historical comparisons can be automated in CI or local scripts.
- **NFR-004**: Runtime benchmarking MUST function without requiring live LLM API
  calls so performance measurements are not dominated by network variability.
- **NFR-005**: The Rust CLI MUST preserve current safety guarantees for file-scoped
  git operations and MUST NOT silently broaden the set of affected files.
- **NFR-006**: Benchmark documentation and outputs MUST avoid implying that runtime
  timing and provider-quality scores are directly interchangeable metrics.

### Key Entities *(include if feature involves data)*

- **Workflow Contract**: A user-visible command shape, supported flags, exit-code
  semantics, and output expectations for a CLI flow.
- **Runtime Benchmark Run**: A single benchmark execution that records corpus,
  runtime, command set, timings, and outcomes for Python and Rust.
- **Repo Corpus Fixture**: A reproducible repository state used for benchmarking,
  including synthetic generated fixtures and realistic checked-in fixtures.
- **Runtime Benchmark Report**: Structured output artifact for runtime comparison,
  distinct from provider-quality benchmark reports.
- **Git Backend Decision**: An explicit implementation choice describing whether the
  Rust path uses shell `git`, `gitoxide`, or another backend for a workflow.

### Dependencies and Assumptions

- The Python CLI remains the compatibility baseline until this feature’s parity
  validation passes.
- The existing fixture generator for large uncommitted repos can be expanded but
  should not be treated as sufficient by itself for all performance claims.
- Realistic corpus fixtures may need to be reduced or synthesized to avoid shipping
  large copyrighted third-party repositories.
- If `gitoxide` is pursued, it is an implementation option, not a product-level
  requirement, and must satisfy the same compatibility gates as shell `git`.

### Validation Baselines (Normative)

- **Workflow parity catalog**: `specs/003-bring-rust-cli/validation/workflow-parity-catalog.md`
  This catalog MUST enumerate every in-scope command contract and whether parity is
  validated on Python, Rust, or both.
- **Runtime corpus definition**: `specs/003-bring-rust-cli/validation/runtime-corpus.md`
  This definition MUST identify at least one synthetic large-uncommitted fixture and
  one realistic fixture, including creation instructions and command set.
- **Runtime benchmark schema**: `specs/003-bring-rust-cli/contracts/runtime-benchmark.schema.json`
  This schema MUST define stable fields for runtime benchmark JSON export.
- **Exit/error baseline**: `specs/003-bring-rust-cli/validation/exit-error-baseline.md`
  This baseline MUST define expected exit code and stderr behavior for parser
  errors, repo selection failures, missing binary cases, and git-operation errors.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: 100% of in-scope workflows in
  `specs/003-bring-rust-cli/validation/workflow-parity-catalog.md` pass parity
  validation for both Python and Rust.
- **SC-002**: Runtime benchmark mode produces reproducible JSON output for Python
  and Rust across the defined corpora with zero unexplained omissions.
- **SC-003**: On the synthetic 1,000-file uncommitted corpus and the realistic
  corpus, Rust median local runtime is measurably better than Python for at least
  one core workflow while preserving contract compatibility.
- **SC-004**: Existing provider-quality benchmark reports remain backward
  compatible for documented fields and usage patterns.
- **SC-005**: Documentation allows a maintainer to generate a repo corpus fixture,
  run the runtime benchmark, and interpret the output without additional tribal
  knowledge.
