# Repository Guidelines

## Project Structure & Modules
- `kcmt/` core package: CLI (`cli.py`, `main.py`), workflow (`core.py`, `commit.py`), VCS utils (`git.py`), config (`config.py`), LLM clients (`llm.py`), and provider drivers (`providers/`).
- `tests/` Pytest suite mirroring package paths.
- `.kcmt/` local config/cache created at runtime.
- Build metadata in `pyproject.toml`; common tasks in `Makefile`.

## Build, Test, and Development
- `make install-dev` — editable install + dev deps via `uv`.
- `make format` — run `isort` and `black`.
- `make lint` — `ruff`, `isort --check`, `black --check`.
- `make typecheck` — `mypy --strict`.
- `make test` / `make test-verbose` / `make test-strict` — run tests.
- `make coverage` — HTML + terminal coverage report.
- `make check` — lint + typecheck + strict tests.
- Run CLI locally: `uv run kcmt ...` (aliases: `commit`, `kc`).

## Coding Style & Naming
- Python 3.12+, 4‑space indent, type hints required.
- Formatting: `black`; imports: `isort` (profile black); lint: `ruff`.
- Names: modules/functions `snake_case`, classes `CapWords`, constants `UPPER_SNAKE`.
- Keep public CLI behavior in `kcmt/cli.py`; isolate I/O from core logic.

## Testing Guidelines
- Framework: `pytest` (with `pytest-cov`; `pytest-bdd` available).
- Tests live in `tests/` and mirror package layout; name files `test_*.py`.
- Focus on `core.py`, `commit.py`, and provider drivers for unit tests.
- Run: `uv run pytest -q` or `make coverage` for reports.

## Commit & Pull Request Guidelines
- Conventional Commits via Commitizen (`pyproject.toml`): `type(scope): subject`.
- Example scopes: `cli`, `core`, `git`, `providers`, `docs`.
- Before PR: `make dev-check`; include description, linked issues, and CLI examples/output.
- Versioning with `make bump-*`; build/release via `make build`, `make release*`.

## Security & Configuration
- API keys via environment variables; see `.env.example`. Do not commit secrets.
- `.kcmt/` is local-only project state; safe to ignore in commits.

## Architecture Notes
- Entrypoints defined in `pyproject.toml`: `kcmt`, `commit`, `kc` -> `kcmt.main:main`.
- Workflow orchestrated by `KlingonCMTWorkflow` in `kcmt/core.py`; provider drivers live in `kcmt/providers/`.

## Governance
- Constitution authority: `.specify/memory/constitution.md` is the top governance source for spec/plan/tasks workflows.
- Required planning flow: `spec -> plan -> tasks -> implement`.
- Constitution gates and `make check` are required before merge for implementation changes.


## Active Technologies
- Rust stable (target 1.78+), Python 3.12 retained temporarily for parity harness and transition wrappers + `clap`, `tokio`, `reqwest`, `serde`/`serde_json`, `tracing`, `anyhow`/`thiserror`, optional `ratatui` for interactive TUI phase (001-rust-parity-migration)
- Local filesystem only (`~/.config/kcmt`, environment variables, git working tree/index); no server-side database (001-rust-parity-migration)
- Python 3.12 (primary wrapper and tests), Rust stable binaries (already implemented) + `pytest`, `uv`, GitHub Actions, existing Rust workspace binaries (`kcmt`, `commit`, `kc`) (002-rust-canary-rollout)
- N/A (ephemeral trace output only) (002-rust-canary-rollout)

## Recent Changes
- 001-rust-parity-migration: Added Rust stable (target 1.78+), Python 3.12 retained temporarily for parity harness and transition wrappers + `clap`, `tokio`, `reqwest`, `serde`/`serde_json`, `tracing`, `anyhow`/`thiserror`, optional `ratatui` for interactive TUI phase
