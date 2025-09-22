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


def test_untracked_directory_files_committed(tmp_path, monkeypatch):
    # init repo
    _git(["init", "-q"], tmp_path)
    _git(["config", "user.name", "Tester"], tmp_path)
    _git(["config", "user.email", "tester@example.com"], tmp_path)

    # create nested untracked directory with files
    nested = tmp_path / "newpkg" / "sub"
    nested.mkdir(parents=True)
    (nested / "alpha.py").write_text("print('alpha')\n")
    (nested / "beta.py").write_text("print('beta')\n")

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    _add_repo_to_sys_path(Path.cwd())
    from kcmt import commit as commit_module  # type: ignore
    from kcmt.core import KlingonCMTWorkflow  # type: ignore

    def fake_generate(
        _diff: str, context: str = "", style: str = "conventional"
    ) -> str:  # noqa: D401
        _ = style
        fname = "file"
        if context.startswith("File:"):
            fname = context.split("File:", 1)[1].strip().split("/")[-1]
        stem = fname.split(".")[0]
        return f"feat(core): add {stem} module"

    monkeypatch.setattr(
        commit_module.LLMClient,
        "generate_commit_message",
        staticmethod(fake_generate),
    )

    wf = KlingonCMTWorkflow(repo_path=str(tmp_path), show_progress=False)
    results = wf.execute_workflow()

    # Expect two file commits (alpha.py, beta.py)
    file_commits = [r for r in results.get("file_commits", []) if r.success]
    committed = sorted(c.file_path for c in file_commits if c.file_path)
    assert any("alpha.py" in p for p in committed), committed
    assert any("beta.py" in p for p in committed), committed
    # Ensure commits are separate
    log_output = _git(["log", "--pretty=%s"], tmp_path)
    lines = [ln for ln in log_output.split("\n") if ln]
    assert len(lines) == 2, lines
