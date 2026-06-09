# Homebrew packaging

kcmt should be registered as a third-party Homebrew tap, not as a binary blob in
`homebrew/core`.

## Recommended shape

- The dedicated tap repository is `djh00t/kcmt-homebrew`.
- The tap exposes a `Formula/kcmt.rb` formula named `kcmt`.
- The formula points at a tagged release source archive from `djh00t/kcmt`.
- Releases stay semver-tagged as `vX.Y.Z`.
- Release notes live on the GitHub release so the tap points at a stable tag.

## What this repo now provides

- `rust/Cargo.toml` is the version source for Rust releases.
- `make release` tags `vX.Y.Z` from that Cargo version.
- `.github/workflows/release-notes.yml` generates GitHub release notes from the
  Conventional Commit history.
- Release assets now include:
  - a compiled Rust binary archive for the runner platform
  - a source archive for formula builds or tap review
  - SHA256 checksums for the archives
- `make homebrew-sync` updates and publishes the tap formula from the current
  release checksums when a `kcmt-homebrew` checkout is present.

## What the tap still needs

1. A bottle strategy if prebuilt binaries are required later.
2. `brew audit --new --strict` validation before publishing formula changes.
3. If bottles are introduced later, platform-specific checksums and bottle
   publication steps.

## Practical install path

For a CLI tool, the normal user-facing path is:

```sh
brew tap djh00t/kcmt-homebrew
brew install kcmt
```

That keeps the release flow conventional and avoids trying to force a binary-only
package into `homebrew/core`.

## Tap scaffold in this repo

This repository now carries a `kcmt-homebrew/` scaffold with:

- `kcmt-homebrew/Formula/kcmt.rb`

The scaffold is the starting point for the real tap repository and keeps the
Homebrew packaging contract aligned with the Rust release flow. The published
tap lives at `https://github.com/djh00t/kcmt-homebrew`.
