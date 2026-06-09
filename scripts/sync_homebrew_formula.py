#!/usr/bin/env python3

from __future__ import annotations

import argparse
import re
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sync the kcmt Homebrew formula from release checksums."
    )
    parser.add_argument("--tap-repo", required=True, help="Path to the kcmt-homebrew repo")
    parser.add_argument("--version", required=True, help="Release version without the v prefix")
    parser.add_argument("--sums-file", required=True, help="Path to dist/SHA256SUMS")
    return parser.parse_args()


def _load_source_checksum(sums_file: Path, version: str) -> str:
    target_name = f"kcmt-{version}-source.tar.gz"
    for line in sums_file.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[1].endswith(target_name):
            return parts[0]
    raise SystemExit(f"Could not find checksum for {target_name} in {sums_file}")


def _replace(pattern: str, replacement: str, text: str, label: str) -> str:
    updated, count = re.subn(pattern, replacement, text, count=1, flags=re.MULTILINE)
    if count != 1:
        raise SystemExit(f"Could not update {label} in kcmt formula")
    return updated


def sync_formula(tap_repo: Path, version: str, sums_file: Path) -> Path:
    formula_path = tap_repo / "Formula" / "kcmt.rb"

    if not formula_path.exists():
        raise SystemExit(f"Missing formula file: {formula_path}")
    if not sums_file.exists():
        raise SystemExit(f"Missing checksum file: {sums_file}")

    source_sha256 = _load_source_checksum(sums_file, version)
    formula = formula_path.read_text(encoding="utf-8")

    formula = _replace(
        r'^  version "([^"]+)"$',
        f'  version "{version}"',
        formula,
        "version",
    )
    formula = _replace(
        r'^  sha256 "([^"]+)"$',
        f'  sha256 "{source_sha256}"',
        formula,
        "sha256",
    )

    formula_path.write_text(formula, encoding="utf-8")
    return formula_path


def main() -> int:
    args = _parse_args()
    tap_repo = Path(args.tap_repo)
    sums_file = Path(args.sums_file)
    formula_path = sync_formula(tap_repo, args.version, sums_file)
    source_sha256 = _load_source_checksum(sums_file, args.version)
    print(f"Updated {formula_path} for v{args.version} ({source_sha256})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
