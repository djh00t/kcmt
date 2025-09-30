# kcmt — AI-powered atomic Git staging and committing

kcmt is a small Python library and CLI that helps you:

- Parse and stage changes atomically (file-by-file).
- Generate clear, conventional-commit messages using an LLM.
- Commit safely with validation, retries, real-time progress, and helpful output.
- (Optional) auto-push successful commits to your remote.

It’s designed to be drop-in for your local repositories and integrates with multiple hosted LLM providers (OpenAI, Anthropic, xAI, GitHub Models).

Key features

- Atomic workflow: stage and commit per-file, with deletions handled first.
- LLM-assisted messages: conventional commit style with validation, retries, and auto-fixes.
- Strict failure on repeated invalid/empty LLM responses (no heuristic commit synthesis).
- Prepare phase aborts after 25 per-file failures/timeouts to avoid wasting additional requests.
- Built-in metrics summary (diff, queue, LLM, commit timings) to diagnose performance quickly.
- Multi-provider support: OpenAI, Anthropic, xAI, and GitHub Models via a guided wizard.
- Parallel preparation: generate per-file commit messages concurrently with live stats.
- Optional automatic push (`--auto-push`) after a successful run.
- Small and composable core: use the CLI or import the library directly.

Supported Python versions

- Python 3.12+

## Installation

Install from the repository with pip or uv:

- Latest from GitHub (subdirectory install):
  - pip:
    - pip install "git+<https://github.com/djh00t/arby#subdirectory=kcmt>"
  - uv:
    - uv pip install "git+<https://github.com/djh00t/arby#subdirectory=kcmt>"

- Local editable install (from the monorepo root):
  - pip:
    - pip install -e ./kcmt
  - uv:
    - uv pip install -e ./kcmt

Dependencies

- openai>=1.108.1 (shared client for OpenAI-compatible providers)
- httpx>=0.25.0 (Anthropic REST client)

## Configuration

Run `kcmt --configure` inside a repository to launch a colourful wizard that:

- Detects available API keys (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `XAI_API_KEY`, `GITHUB_TOKEN`).
- Lets you choose the provider, tweak model/endpoint, and pick the env var to use.
- Persists the selection to `.kcmt/config.json` (commit it if you want teammates to share defaults).

### Provider defaults

| Provider  | Default model             | Default endpoint                         |
|-----------|---------------------------|------------------------------------------|
| OpenAI    | `gpt-5-mini-2025-08-07`   | `https://api.openai.com/v1`              |
| Anthropic | `claude-3-5-haiku-latest` | `https://api.anthropic.com`             |
| xAI       | `grok-code-fast`          | `https://api.x.ai/v1`                   |
| GitHub    | `openai/gpt-4.1-mini`     | `https://models.github.ai/inference`    |

You can still override values at runtime:

```
kcmt --provider openai --model gpt-5-mini-2025-08-07 --endpoint https://api.openai.com/v1 \
     --api-key-env OPENAI_API_KEY --repo-path .
```

Additional environment tweaks remain available:

- `KLINGON_CMT_LLM_MODEL`
- `KLINGON_CMT_LLM_ENDPOINT`
- `KLINGON_CMT_GIT_REPO_PATH`
- `KLINGON_CMT_MAX_COMMIT_LENGTH` (applies to subject line validation; body is no longer truncated)
- `KCMT_PROVIDER` – force provider selection for one-off runs without editing `.kcmt/config.json`
  (useful when CI supplies secrets that differ from the persisted repo defaults)
Deprecated: `KLINGON_CMT_ALLOW_FALLBACK` previously enabled a heuristic
fallback subject after repeated LLM failures. This path has been removed;
kcmt now fails fast with an explicit LLMError so you never get an invented
message. The variable is ignored if set.
  
Additional LLM behaviour environment variables:

- `KCMT_LLM_REQUEST_TIMEOUT` – per-request HTTP timeout (seconds, default 5)
- `KCMT_PREPARE_PER_FILE_TIMEOUT` – per-file generation timeout in atomic workflow
- `KCMT_OPENAI_DISABLE_REASONING` – disable reasoning / chain-of-thought (default on)
- `KCMT_OPENAI_MINIMAL_PROMPT` – force minimal prompt style (adaptive toggle)
- `KCMT_OPENAI_MAX_TOKENS` – max completion tokens for OpenAI-like providers
- `KLINGON_CMT_AUTO_PUSH=1` (enable automatic `git push` after success)

## Quick start (CLI)

```
kcmt --configure              # guided setup -> .kcmt/config.json
kcmt                          # per-file atomic commits with live stats
kcmt --oneshot --verbose      # single best-effort commit
kcmt --file README.md         # commit a specific file
kcmt --provider xai --model grok-code-fast --api-key-env XAI_API_KEY
```

Exit codes

- 0 on success
- 1 on workflow error (input/validation/LLM/Git failures)
- 2 on configuration error (no usable API key)

## CLI reference

`kcmt` accepts the following common options:

- `--configure` – launch the interactive setup wizard.
- `--provider`, `--model`, `--endpoint`, `--api-key-env` – override saved provider details.
- `--repo-path PATH` – target repository (defaults to current working directory).
- `--max-commit-length INT` – validate (not hard truncate) the subject line length (default 72 for legacy compatibility; body is preserved).
- `--auto-push` – push the current branch after successful commits (or set `KLINGON_CMT_AUTO_PUSH=1`).
- `--max-retries INT` – retries when Git rejects (default 3).
- `--oneshot` – stage all changes, pick one file, and commit it once.
- `--file PATH` – stage & commit an explicit file.
- `--no-progress` – disable the live stats bar.
- `--verbose`, `-v` – emit detailed logs and per-file results.

## Conventional commit automation

kcmt now ships with a [Commitizen](https://commitizen-tools.github.io/commitizen/) configuration that mirrors the
LLM-generated commit format. After installing the project you can:

- run `cz check` to validate a message before committing manually;
- run `cz commit` to invoke Commitizen's prompt flow while still benefitting from kcmt's
  validation rules and version tracking (it watches `kcmt/__init__.py`).

The configuration lives in `pyproject.toml` under `[tool.commitizen]`, so any repository that
adopts kcmt inherits the same conventional commit guardrails automatically.

## Library usage examples

Basic: generate a message from staged changes

- from kcmt.config import load_config
- cfg = load_config()
- from kcmt.commit import CommitGenerator
- gen = CommitGenerator(repo_path=cfg.git_repo_path, config=cfg)
- msg = gen.generate_from_staged(context="Refactor widgets", style="conventional")
- print(msg)

Generate from working tree changes

- from kcmt.config import load_config
- cfg = load_config()
- from kcmt.commit import CommitGenerator
- gen = CommitGenerator(repo_path=cfg.git_repo_path, config=cfg)
- msg = gen.generate_from_working(context="Work in progress", style="conventional")
- print(msg)

Run the full atomic workflow

- from kcmt.config import load_config
- cfg = load_config()
- from kcmt.core import KlingonCMTWorkflow
- wf = KlingonCMTWorkflow(repo_path=cfg.git_repo_path, max_retries=3, config=cfg)
- results = wf.execute_workflow()
- print(results["summary"])
- for r in results.get("file_commits", []):
-     print(r.success, r.commit_hash, r.message)

Using GitRepo directly

- from kcmt.config import load_config
- cfg = load_config()
- from kcmt.git import GitRepo
- repo = GitRepo(cfg.git_repo_path, cfg)
- print(repo.get_working_diff())
- if repo.has_working_changes():
-     repo.stage_file("README.md")
-     repo.commit("docs: update readme")

## API documentation (high-level)

Exceptions

- kcmt.exceptions.KlingonCMTError: Base error
- kcmt.exceptions.GitError: Git command errors
- kcmt.exceptions.LLMError: LLM call errors
- kcmt.exceptions.ConfigError: Config/ENV errors
- kcmt.exceptions.ValidationError: Validation failures

Configuration

- kcmt.config.Config
  - Fields: provider, model, llm_endpoint, api_key_env, git_repo_path, max_commit_length, auto_push
- kcmt.config.load_config(overrides=None)
  - Merge repo `.kcmt/config.json`, environment, and optional overrides.
- kcmt.config.save_config(config, repo_root=None)
- kcmt.config.get_active_config() / set_active_config()

LLM

- kcmt.llm.LLMClient
  - generate_commit_message(diff: str, context: str = "", style: str = "conventional") -> str

Git operations

- kcmt.git.GitRepo
  - has_staged_changes() -> bool
  - get_staged_diff() -> str
  - has_working_changes() -> bool
  - get_working_diff() -> str
  - get_commit_diff(commit_hash: str) -> str
  - get_recent_commits(count: int = 5) -> list[str]
  - stage_file(file_path: str) -> None
  - unstage(file_path: str) -> None
  - commit(message: str) -> None
  - process_deletions_first() -> list[str]
  - push(remote: str = "origin", branch: str | None = None) -> str

Commit generation

- kcmt.commit.CommitGenerator
  - generate_from_staged(context: str = "", style: str = "conventional") -> str
  - generate_from_working(context: str = "", style: str = "conventional") -> str
  - generate_from_commit(commit_hash: str, context: str = "", style: str = "conventional") -> str
  - suggest_commit_message(diff: str, context: str = "", style: str = "conventional") -> str
  - validate_conventional_commit(message: str) -> bool
  - validate_and_fix_commit_message(message: str) -> str

Core workflow

- kcmt.core.FileChange
  - file_path: str
  - change_type: str  # "A" | "M" | "D"
  - diff_content: str
- kcmt.core.CommitResult
  - success: bool
  - commit_hash: str | None
  - message: str | None
  - error: str | None
- kcmt.core.KlingonCMTWorkflow(repo_path: str | None = None, max_retries: int = 3)
  - execute_workflow() -> dict
    - deletions_committed: list[CommitResult]
    - file_commits: list[CommitResult]
    - errors: list[str]
    - summary: str

Notes on behavior

- Deletions are grouped and committed first with a generated message.
- Then remaining file changes are parsed from git diff and committed per-file.
- Commit messages are validated and may be LLM-fixed on failure; retries are applied.
- Enrichment pass: for substantial diffs (>=10 changed lines) a second LLM call may add a concise multi-line body explaining what and why while preserving the original header.
- Fast fail: after retry exhaustion (currently 3 attempts) kcmt raises an `LLMError`; no heuristic commit message is generated.
- If `--auto-push` is enabled and at least one commit succeeds, kcmt attempts `git push origin <current-branch>` and records the result (`results['pushed']=True`).

## Development

Prereqs

- Python 3.12+
- A virtual environment
- Git

Set up

- uv venv &amp;&amp; source .venv/bin/activate  # or python -m venv .venv
- uv pip install -e ./kcmt  # or pip install -e ./kcmt
- uv pip install pytest            # or pip install pytest

Run tests (from project root)

- make test                  # quiet pytest run via Makefile
- uv run -m pytest tests     # or: pytest

Lint/format (recommendations)

- Use Black and isort (not enforced by this package directly).
- Keep lines &lt;=80 chars where practical, file size &lt;500 lines for readability.

Publishing

- Version is defined in kcmt/__init__.py (managed by hatch).
- Build and publish via hatch (example):
  - uv pip install hatch
  - hatch build
  - hatch publish  # configure as needed

Security notes

- Your GITHUB_TOKEN is used as the API key for the configured endpoint.
- Treat it like any other secret; avoid checking it into logs or code.
- Prefer passing the token via the CLI or securely provisioned environment.

## Changelog

- 0.1.2 — Update default OpenAI model to `gpt-5-mini-2025-08-07` (automatic
  migration from legacy `gpt-5-mini`), test helper env var
  `KCMT_TEST_DISABLE_OPENAI` to bypass real API calls in isolated tests.
- 0.1.1 — Auto-push option, improved retries, subject-only length enforcement.
- 0.1.0 — Initial release: CLI + atomic workflow + LLM commit generation.

## License

MIT (see repository license)

## Strict testing and CI quick reference

Environment

- uv venv && source .venv/bin/activate
- uv sync --group dev
- uv pip install -e .

Lint/format/type-check

- uv run ruff check kcmt tests
- uv run black --check kcmt tests
- uv run isort --check-only kcmt tests
- uv run mypy kcmt

Run tests

- Basic: uv run pytest -ra -vv tests
- Strict CI-like run (parallel, warnings as errors, coverage):
  uv run pytest -n auto -ra -vv -W default -W error::DeprecationWarning -W error::ResourceWarning --strict-config --strict-markers --cov=kcmt --cov-branch --cov-report=term-missing:skip-covered --cov-fail-under=85 tests
  - On Windows/PowerShell use a single line (no trailing `\`) or replace the
    line continuation with a backtick (`` ` ``); PowerShell treats ``\`` as a
    literal character and will otherwise break up the arguments.

Make targets

- make test
- make test-verbose
- make test-strict
- make check     # lint + typecheck + strict tests
