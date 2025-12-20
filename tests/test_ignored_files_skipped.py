import subprocess
import sys
from pathlib import Path


def _add_repo_to_sys_path(repo_root: Path):
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


def _git(cmd: list[str], cwd: Path) -> str:
    result = subprocess.run(
        ["git"] + cmd,
        cwd=cwd,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def test_ignored_files_are_not_committed(tmp_path, monkeypatch):
    _git(["init", "-q"], tmp_path)
    _git(["config", "user.name", "Tester"], tmp_path)
    _git(["config", "user.email", "tester@example.com"], tmp_path)

    # Create .gitignore ignoring the cache dir
    (tmp_path / ".gitignore").write_text("ignored_dir/\n")

    ignored_dir = tmp_path / "ignored_dir"
    ignored_dir.mkdir()
    (ignored_dir / "skip.txt").write_text("skip me\n")

    kept = tmp_path / "keep.py"
    kept.write_text("print('keep')\n")

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    _add_repo_to_sys_path(Path.cwd())
    from kcmt import commit as commit_module  # type: ignore
    from kcmt.core import KlingonCMTWorkflow  # type: ignore

    def fake_generate(
        _diff: str, context: str = "", style: str = "conventional"
    ) -> str:  # noqa: D401
        _ = style
        if context.startswith("File:") and "keep.py" in context:
            return "feat(core): add keep script"
        return "feat(core): add file"

    monkeypatch.setattr(
        commit_module.LLMClient,
        "generate_commit_message",
        staticmethod(fake_generate),
    )

    wf = KlingonCMTWorkflow(repo_path=str(tmp_path), show_progress=False)
    results = wf.execute_workflow()

    file_commits = [r for r in results.get("file_commits", []) if r.success]
    # .gitignore and kept file should be committed, but not ignored_dir/skip.txt
    assert len(file_commits) == 2, file_commits
    committed_paths = [r.file_path for r in file_commits]
    assert ".gitignore" in committed_paths
    assert any(path.endswith("keep.py") for path in committed_paths)
    # Ensure no files from the ignored directory were committed
    assert not any("ignored_dir" in path for path in committed_paths)
