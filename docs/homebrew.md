# Homebrew packaging

kcmt should be registered as a third-party Homebrew tap, not as a binary blob in
`homebrew/core`.

## Recommended shape

- Create a dedicated tap repository, for example `kcmt-homebrew`.
- Add a `Formula/kcmt.rb` formula named `kcmt`.
- Point the formula at a tagged release source archive or GitHub release asset.
- Keep releases semver-tagged as `vX.Y.Z`.
- Publish release notes on the GitHub release so the tap points at a stable tag.

## What this repo now provides

- `rust/Cargo.toml` is the version source for Rust releases.
- `make release` tags `vX.Y.Z` from that Cargo version.
- `.github/workflows/release-notes.yml` generates GitHub release notes from the
  Conventional Commit history.
- Release assets now include:
  - a compiled Rust binary archive for the runner platform
  - a source archive for formula builds or tap review
  - SHA256 checksums for the archives

## What the tap still needs

1. A Homebrew formula that builds the CLI from source with Cargo, or a bottle
   strategy if prebuilt binaries are required later.
2. A tap repository with `brew tap djh00t/kcmt-homebrew` style installation docs.
3. `brew audit --new-formula` validation before publishing the tap.
4. If bottles are introduced later, platform-specific checksums and bottle
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
Homebrew packaging contract aligned with the Rust release flow.
