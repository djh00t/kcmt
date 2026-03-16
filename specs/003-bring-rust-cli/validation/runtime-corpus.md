# Runtime Corpus Definition

## Corpus A: Synthetic 1,000-File Uncommitted Repository

- **ID**: `synthetic-untracked-1000`
- **Kind**: synthetic
- **Creation**:

```bash
python scripts/benchmark/generate_uncommitted_repo.py --file-count 1000 --json
```

- **Shape**:
  - zero git commits
  - `README.md` plus 1,000 nested source files
  - all files untracked
- **Use**:
  - startup and repo-scanning overhead
  - path enumeration and working-tree traversal
  - explicit repo-path contract validation

## Corpus B: Realistic Mini Repository Fixture

- **ID**: `mini-realistic-fixture`
- **Kind**: realistic
- **Planned Location**: `tests/fixtures/runtime_corpus/mini_realistic_repo/`
- **Shape**:
  - committed baseline history
  - mix of modified, untracked, deleted, and ignored files
  - nested directories and at least one non-Python text asset
- **Use**:
  - file-scoped workflow parity
  - git status/error-path realism
  - benchmark coverage beyond uniform synthetic files

## Benchmark Command Set

- `status-repo-path`
- `oneshot-repo-path`
- `file-repo-path`

## Exclusions

- Live provider/API latency
- Network-dependent benchmark scenarios
- Large third-party repositories checked into the repo
