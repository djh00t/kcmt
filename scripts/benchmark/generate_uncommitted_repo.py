#!/usr/bin/env python3
"""Create a temporary git repository with many uncommitted files."""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

CORPUS_METADATA_FILENAME = ".kcmt-runtime-corpus.json"


@dataclass(frozen=True)
class FileSpec:
    """One synthetic file to create inside the generated repository."""

    relative_path: Path
    content: str


def build_file_specs(file_count: int, fanout: int = 25) -> list[FileSpec]:
    """Build a deterministic set of synthetic source files."""

    specs: list[FileSpec] = []
    for index in range(file_count):
        bucket = index % fanout
        relative_path = Path("src") / f"module_{bucket:03d}" / f"file_{index:04d}.py"
        content = "\n".join(
            [
                '"""Synthetic benchmark fixture."""',
                "",
                f"FILE_INDEX = {index}",
                f"MODULE_INDEX = {bucket}",
                "",
                "def compute(value: int) -> int:",
                f"    total = value + {index}",
                "    for step in range(5):",
                "        total += step",
                "    return total",
                "",
            ]
        )
        specs.append(FileSpec(relative_path=relative_path, content=content))
    return specs


def init_git_repo(repo_path: Path) -> None:
    """Initialize a git repository with a stable default branch."""

    subprocess.run(
        ["git", "init", "--initial-branch=main", str(repo_path)],
        check=False,
        capture_output=True,
        text=True,
    )
    if not (repo_path / ".git").exists():
        subprocess.run(["git", "init", str(repo_path)], check=True)
        subprocess.run(
            ["git", "-C", str(repo_path), "branch", "-M", "main"],
            check=True,
        )


def write_repo_files(repo_path: Path, specs: list[FileSpec]) -> None:
    """Materialize the synthetic files in the target repository."""

    for spec in specs:
        target = repo_path / spec.relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(spec.content, encoding="utf-8")

    readme = repo_path / "README.md"
    readme.write_text(
        "# Synthetic Runtime Benchmark Fixture\n\n"
        "This repository was generated for kcmt runtime performance comparisons.\n",
        encoding="utf-8",
    )
    default_target = str(specs[0].relative_path) if specs else "README.md"
    metadata = {
        "id": f"synthetic-untracked-{len(specs)}",
        "kind": "synthetic",
        "file_count": len(specs),
        "git_history_state": "no-commits",
        "change_shape": ["untracked", "nested-paths"],
        "default_file_target": default_target,
    }
    (repo_path / CORPUS_METADATA_FILENAME).write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )


def create_uncommitted_repo(destination: Path | None, file_count: int) -> Path:
    """Create a synthetic repository with untracked, uncommitted files."""

    repo_path = (
        destination
        if destination is not None
        else Path(tempfile.mkdtemp(prefix="kcmt-uncommitted-repo-"))
    )
    repo_path.mkdir(parents=True, exist_ok=True)
    init_git_repo(repo_path)
    write_repo_files(repo_path, build_file_specs(file_count))
    return repo_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a temporary git repository with uncommitted files"
    )
    parser.add_argument(
        "--file-count",
        type=int,
        default=1000,
        help="Number of uncommitted source files to create (default: 1000)",
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=None,
        help="Optional output directory; defaults to a temporary directory",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable output",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_path = create_uncommitted_repo(args.destination, args.file_count)
    payload = {
        "repo_path": str(repo_path),
        "corpus_id": f"synthetic-untracked-{args.file_count}",
        "file_count": args.file_count,
        "tracked_commits": 0,
        "git_dir": str(repo_path / ".git"),
    }
    if args.json:
        print(json.dumps(payload))
    else:
        print(f"repo_path={repo_path}")
        print(f"file_count={args.file_count}")
        print("state=uncommitted")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
