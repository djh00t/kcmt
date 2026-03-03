# Rust Test Harness Scaffold

This directory hosts migration-era Rust validation suites aligned with
`specs/001-rust-parity-migration/tasks.md`.

## Layout

- `contract/`: CLI/config/schema compatibility checks.
- `integration/`: end-to-end behavior checks for workflows and provider handling.
- `parity/`: Python-vs-Rust parity and performance harnesses.

## Intended Commands

```bash
cargo test --workspace
cargo test --test '*'
```
