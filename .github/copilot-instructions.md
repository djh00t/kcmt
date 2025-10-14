# kcmt Copilot Instructions

These guidelines help GitHub Copilot propose changes that respect the repository's standards and workflow.

## Project priorities
- Treat Python 3.12+ as the baseline for all runtime and typing features.
- Keep suggestions focused on the `kcmt/` package and its supporting tests under `tests/`.
- Preserve the CLI contract in `kcmt/cli.py` and leave user-facing behavior unchanged unless explicitly requested.

## Coding conventions
- Use four-space indentation and provide explicit type hints on new or modified functions.
- Match the existing style: format with `black`, order imports with `isort --profile black`, and satisfy `ruff` lint rules.
- Avoid rewriting unrelated code or reformatting entire files; limit edits to the necessary diff.
- Name modules and functions in `snake_case`, classes in `CapWords`, and constants in `UPPER_SNAKE_CASE`.

## Architecture cues
- Core workflow lives in `kcmt/core.py` and `kcmt/commit.py`; provider integrations reside in `kcmt/providers/`.
- Configuration logic is centralized in `kcmt/config.py`; VCS helpers are in `kcmt/git.py`.
- Tests mirror the package structure. Add or update tests in `tests/` whenever behavior changes.

## Build, test, and validation steps
- When code changes, assume these checks should pass:
  - `make format` (runs `isort` + `black`).
  - `make lint` (runs `ruff`, `isort --check`, `black --check`).
  - `make typecheck` (runs `mypy --strict`).
  - `make test` or targeted `uv run pytest` commands.
- Prefer minimal, fast test selections when touching a narrow area, but keep full suite compatibility in mind.

## Documentation and CLI examples
- Update `README.md` or docs under `docs/` when altering user-visible features or commands.
- Provide CLI usage examples with `uv run kcmt ...` (aliases `commit` or `kc`) when documenting workflows.

## Git & release safety
- Follow Conventional Commits (`type(scope): subject`) if suggesting commit messages.
- Run `make dev-check` (aggregate lint, typecheck, and strict tests) before proposing a PR merge.
- Do not touch `.kcmt/` runtime state or commit secrets; rely on `.env.example` for env variable references.

## Requesting clarification
- If requirements are ambiguous, inspect `AGENTS.md`, relevant module docstrings, and existing tests before proceeding.
- When a safe assumption is required, state it and bias toward the least disruptive behavior.
