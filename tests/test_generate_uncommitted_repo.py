from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path


def _load_module():
    root = Path(__file__).resolve().parents[1]
    script_path = root / "scripts" / "benchmark" / "generate_uncommitted_repo.py"
    spec = importlib.util.spec_from_file_location(
        "generate_uncommitted_repo",
        script_path,
    )
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_build_file_specs_is_deterministic():
    module = _load_module()

    specs = module.build_file_specs(4, fanout=2)

    assert len(specs) == 4
    assert specs[0].relative_path == Path("src/module_000/file_0000.py")
    assert specs[1].relative_path == Path("src/module_001/file_0001.py")
    assert "FILE_INDEX = 3" in specs[3].content


def test_create_uncommitted_repo_initializes_git_and_files(tmp_path):
    module = _load_module()

    repo_path = module.create_uncommitted_repo(tmp_path / "fixture", 3)

    assert (repo_path / ".git").exists()
    assert (repo_path / "src/module_000/file_0000.py").exists()
    metadata = json.loads(
        (repo_path / module.CORPUS_METADATA_FILENAME).read_text(encoding="utf-8")
    )
    assert metadata["id"] == "synthetic-untracked-3"
    assert metadata["default_file_target"] == "src/module_000/file_0000.py"

    status = subprocess.run(
        ["git", "-C", str(repo_path), "status", "--short"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "?? README.md" in status.stdout
    assert "?? src/" in status.stdout


def test_main_supports_json_output(monkeypatch, capsys, tmp_path):
    module = _load_module()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_uncommitted_repo.py",
            "--destination",
            str(tmp_path / "json-fixture"),
            "--file-count",
            "2",
            "--json",
        ],
    )

    assert module.main() == 0
    out = capsys.readouterr()

    assert '"corpus_id": "synthetic-untracked-2"' in out.out
    assert '"file_count": 2' in out.out
