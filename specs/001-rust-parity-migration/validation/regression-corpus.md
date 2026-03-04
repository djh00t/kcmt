# Regression Corpus

This corpus defines reproducible fixtures for parity and performance baselining.

## Fixture Classes

1. Small text diff: <= 5 changed files, <= 200 lines total.
2. Medium text diff: 6-50 changed files, <= 2,500 lines total.
3. Large text diff: 51-1,500 changed files, <= 200,000 lines total.
4. Binary-like text fixtures: generated/minified diffs that stress token trimming.
5. No-change repositories: clean working tree and index.

## Baseline Capture Requirements

- Baseline command set MUST include `kcmt`, `commit`, `kc`, `status`, and `--file`.
- Baseline captures MUST record wall-clock time and exit code.
- Baseline snapshots MUST include Python release tag and fixture revision hash.
