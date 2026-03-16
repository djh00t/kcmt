# Research: Rust CLI Feature Parity and Runtime Benchmark Mode

## Decision 1: Define parity against an explicit workflow catalog, not every parser branch

- Decision: Scope Rust parity against a maintained workflow catalog that lists the
  user-visible commands, flags, exit codes, and outputs that must match Python.
- Rationale: The Python CLI surface is broad; an explicit contract catalog keeps the
  parity target testable and prevents accidental over- or under-scoping.
- Alternatives considered:
  - Aim for undocumented full parser parity immediately: rejected because it obscures
    the compatibility target and makes regression review harder.
  - Only mirror `--help` output: rejected because migration risk lies in workflow
    behavior, not just parser listings.

## Decision 2: Add `--repo-path` as a first-class Rust contract and use it across modes

- Decision: The Rust CLI will accept `--repo-path` for in-scope workflows instead of
  requiring the caller to `cd` into the repository first.
- Rationale: Python already supports repo selection as a stable automation contract,
  and the absence of this flag blocks fair parity and benchmarking.
- Alternatives considered:
  - Require process-level `cwd` changes only: rejected because it diverges from the
    Python contract and complicates automation.
  - Support `--repo-path` only for `status`: rejected because runtime benchmarking and
    commit workflows also need explicit repo targeting.

## Decision 3: Keep shell `git` as the default Rust backend; treat `gitoxide` as opt-in evaluation only

- Decision: Continue using shell `git` semantics as the default compatibility path in
  Rust, while introducing backend seams that allow `gitoxide` experiments behind
  explicit flags or internal switches after parity tests exist.
- Rationale: The constitution prioritizes git safety and atomic behavior over backend
  novelty. `gitoxide` may improve performance, but it cannot become the default path
  until file-scoped and error-path parity are proven.
- Alternatives considered:
  - Replace shell `git` with `gitoxide` immediately: rejected because parity and edge
    case safety for atomic operations are not yet demonstrated.
  - Ignore `gitoxide` completely: rejected because it remains a legitimate future
    performance lever worth preserving in the design.

## Decision 4: Separate runtime benchmarking from provider-quality benchmarking

- Decision: Introduce a dedicated runtime benchmark mode and keep the current provider
  benchmark mode intact. Legacy `--benchmark` and related flags continue to mean
  provider/model benchmarking for backward compatibility.
- Rationale: Runtime timing and provider-quality scores are different dimensions and
  combining them into one leaderboard would make historical comparisons misleading.
- Alternatives considered:
  - Add runtime timing columns to the existing provider leaderboard: rejected because
    it mixes unrelated variables and implies comparability where none exists.
  - Replace provider benchmarking with runtime benchmarking: rejected because it would
    remove an existing supported feature.

## Decision 5: Runtime benchmarks must be offline-capable and corpus-driven

- Decision: The runtime benchmark mode will measure deterministic local operations on
  reproducible repo corpora and must not require live LLM API calls.
- Rationale: Network and provider variance would swamp the runtime signal and violate
  the constitution’s requirement for reproducible performance claims.
- Alternatives considered:
  - Benchmark full end-to-end commit generation with live providers: rejected because
    the timing signal would be dominated by external latency and credentials.
  - Use only microbenchmarks inside Rust crates: rejected because maintainers need
    CLI-level evidence tied to actual repository corpora.

## Decision 6: Use a dual corpus strategy for runtime benchmarking

- Decision: Runtime benchmarking will use both a generated synthetic corpus (large
  uncommitted repo from `scripts/benchmark/generate_uncommitted_repo.py`) and a
  checked-in realistic mini-repository fixture.
- Rationale: Synthetic corpora are easy to scale and reproduce, while realistic
  fixtures catch path-shape and content-shape behavior that synthetic files miss.
- Alternatives considered:
  - Synthetic corpus only: rejected because it is too uniform to represent actual
    repository shapes.
  - Third-party repo clones: rejected because licensing, size, and drift make them
    unsuitable for checked-in deterministic validation.

## Decision 7: Keep Ratatui out of this feature’s critical path

- Decision: Do not couple Rust CLI parity or runtime benchmarking to Ratatui work in
  this feature; keep `kcmt-tui` optional and non-blocking.
- Rationale: The immediate gap is CLI contract parity and local performance evidence,
  not interactive UI capability.
- Alternatives considered:
  - Build runtime benchmarking into a new Ratatui interface first: rejected because it
    delays parity and adds UI complexity before core contracts are stable.

## Decision 8: Version runtime benchmark JSON independently

- Decision: Runtime benchmark JSON exports will use a separate versioned schema from
  the existing provider benchmark exports.
- Rationale: The data model and compatibility promises differ, and versioning them
  independently allows stable automation without coupling unrelated report formats.
- Alternatives considered:
  - Reuse the provider benchmark schema: rejected because the entities and metrics are
    materially different.

## Gitoxide Readiness Note

- Current readiness: `gitoxide` remains explicitly deferred as the default backend.
- Implemented posture: `rust/crates/kcmt-core/src/git/gitoxide_readonly.rs` is limited
  to read-only capability discovery and is not part of the file-scoped commit path.
- Promotion criteria:
  - file-scoped add/commit parity is proven against shell `git`
  - status snapshot parity remains stable under nested `--repo-path` usage
  - runtime benchmark evidence shows no regression against the shell `git` baseline
  - failure and exclusion behavior stays explicit for backend gaps
