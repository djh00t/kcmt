# kcmt - Rust-first atomic automated conventional commits

kcmt is the Rust CLI for generating conventional commits from staged or working
tree changes. The legacy Python package still exists under
`legacy/kcmt-python/` for compatibility, but the repo now treats the Rust CLI
as the primary surface.

## What it does

- Parse and stage changes atomically, file by file.
- Generate conventional-commit messages with provider-backed LLMs.
- Validate commit messages, retry invalid output, and fail fast on repeated
  bad responses.
- Support OpenAI, Anthropic, xAI, and GitHub Models through the same config
  and provider model selection flow.
- Auto-push successful commits to `origin` by default.
- Benchmark provider/model quality and compare Python vs Rust runtime behavior.

## Install

Install the Rust CLI from the repo root:

```sh
make install
```

Or build the release binaries directly:

```sh
cargo build --locked --release --manifest-path rust/Cargo.toml -p kcmt-cli
```

The release build produces these binaries:

- `kcmt`
- `commit`
- `kc`

If you are still using the legacy Python package for compatibility work, install
that separately with:

```sh
make install-python
```

## Quick start

```sh
kcmt --help
kcmt --configure
kcmt status --repo-path .
kcmt --oneshot --verbose
kcmt --file README.md
kcmt --provider xai --model grok-code-fast --api-key-env XAI_API_KEY
```

## Configuration

Run `kcmt --configure` inside a repository to write or refresh global config.
Use `kcmt --configure --tui` to open the terminal menu shell when stdin/stdout
are attached to a terminal. The TUI shows the resolved
provider/model/credential summary and saves on `s` or Enter; `q` or Esc exits
without changing files.

`kcmt` keeps its settings in `~/.config/kcmt/config.json` and
`~/.config/kcmt/preferences.json`.

Provider defaults:

| Provider  | Default model             | Default endpoint                      |
|-----------|---------------------------|---------------------------------------|
| OpenAI    | `gpt-5.4-mini`           | `https://api.openai.com/v1`           |
| Anthropic | `claude-3-5-haiku-latest`| `https://api.anthropic.com`           |
| xAI       | `grok-code-fast`         | `https://api.x.ai/v1`                 |
| GitHub    | `openai/gpt-4.1-mini`    | `https://models.github.ai/inference`  |

Common config and provider commands:

- `kcmt --configure-all` - pick which providers to configure and set the API
  key env var for each one.
- `kcmt --verify-keys` - show which env vars are in use and whether they are
  set.
- `kcmt --list-models` - show a pricing-aware model board.

Provider/model benchmarking is available with `--benchmark`.
Runtime benchmarking is a separate command:

```sh
kcmt benchmark runtime --repo-path /path/to/generated/repo --runtime both --json
```

That compares the Python and Rust CLIs on the same repo corpus without mixing
runtime timing into the provider-quality leaderboard.

## Benchmarking

```sh
kcmt --benchmark
kcmt --benchmark --benchmark-limit 5
```

Benchmark output can be emitted as JSON or CSV with:

- `--benchmark-json`
- `--benchmark-csv`

Snapshots are saved under
`~/.config/kcmt/repos/<repo-id>/benchmarks/benchmark-<timestamp>.json`.

## Homebrew

See [docs/homebrew.md](docs/homebrew.md) for the current Homebrew packaging
shape, release requirements, and tap guidance.

## Release

- Rust workspace version source: `rust/Cargo.toml`
- Semver release tags: `vX.Y.Z`
- Release notes generation: `.github/workflows/release-notes.yml`
- Release artifacts: compiled Rust binaries plus source archives and checksums

## Legacy Python package

The legacy Python package lives in `legacy/kcmt-python/kcmt_python/`.
Use `make install-python` if you need editable development or compatibility
workflows around that package.

## Development

```sh
make check
make quality-gates
make test-rust
make test-ink
```

The strict CI surface also includes the Rust release build and the UI tests in
`legacy/kcmt-python/kcmt_python/ui/ink`.

## Testing

For local validation, the repo's current CI-equivalent gates are:

- `make check`
- `make quality-gates`

## License

MIT
