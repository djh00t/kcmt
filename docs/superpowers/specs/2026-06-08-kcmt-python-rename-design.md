# kcmt Python legacy rename design

## Goal

Rename the Python implementation to `kcmt-python`, move it into a dedicated
legacy area in the repository, and rename every Python entrypoint so it is
obviously distinct from the Rust-first surface.

## Scope

Included:

- Python package rename from `kcmt` to `kcmt_python`
- Python project/distribution rename to `kcmt-python`
- Python console scripts renamed to `kcmt-python`, `commit-python`, and
  `kc-python`
- File and import updates needed to keep the renamed package executable
- Test updates for package imports, script names, and CLI entrypoints
- Documentation updates where the Python surface is referenced

Excluded:

- Changes to Rust runtime behavior
- New compatibility shims for the old Python package name
- Broader cleanup of unrelated packaging or workflow code

## Proposed Layout

Move the Python source tree into a legacy directory that makes the retirement
intent obvious, for example:

- `legacy/kcmt-python/kcmt_python/`

That keeps the Python code isolated from the Rust code and makes it clear the
Python package is being preserved only as a named legacy artifact.

## Package Contract

The Python package should be importable as `kcmt_python` and should advertise
itself as `kcmt-python` in packaging metadata.

The package version source should move with the package so version bumps and
wheel/sdist generation continue to work without a secondary alias.

## EntryPoint Contract

The Python console scripts should be renamed as follows:

- `kcmt-python` -> main Python CLI
- `commit-python` -> commit alias
- `kc-python` -> shortest alias

The old Python entrypoint names should not remain as Python distribution
scripts.

## Verification Plan

Add or update tests so they prove:

- the Python project metadata names the distribution `kcmt-python`
- the wheel/sdist packaging points at `kcmt_python`
- the renamed console scripts are exposed
- imports and lazy exports still work from `kcmt_python`
- user-facing tests reference the suffixed Python entrypoints where relevant

Run the repo quality gates after the rename so the package move is validated
alongside the existing Rust migration tests.

## Risks

- Some tests or docs may still assume the old import path `kcmt`
- Packaging metadata may need coordinated updates in more than one place
- The move may expose hidden references in helper scripts or fixtures

## Success Criteria

- The Python implementation lives in its own legacy location
- The active Python package name is `kcmt-python`
- All Python entrypoints use the `-python` suffix
- The repo tests and quality gates pass with the renamed surface
