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


def test_two_new_files_committed_separately(tmp_path, monkeypatch):
    # Initialise a git repo
    _git(["init", "-q"], tmp_path)
    _git(["config", "user.name", "Test"], tmp_path)
    _git(["config", "user.email", "test@example.com"], tmp_path)

    # Create two new files
    (tmp_path / "alpha.py").write_text("print('alpha')\n")
    (tmp_path / "beta.py").write_text("print('beta')\n")

    # Ensure provider env so LLMClient initializes, then monkeypatch generation
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    _add_repo_to_sys_path(Path.cwd())
    from kcmt import commit as commit_module  # type: ignore
    from kcmt.core import KlingonCMTWorkflow  # type: ignore

    def fake_generate(
        _diff: str, context: str = "", style: str = "conventional"
    ) -> str:  # noqa: D401
        # Derive filename from context
        _ = style  # avoid unused
        fname = "file"
        if context.startswith("File:"):
            fname = context.split("File:", 1)[1].strip().split("/")[-1]
        base = fname.split(".")[0]
        return f"feat(core): add {base} script"

    monkeypatch.setattr(
        commit_module.LLMClient,
        "generate_commit_message",
        staticmethod(fake_generate),
    )

    # Run workflow with limit 2
    wf = KlingonCMTWorkflow(
        repo_path=str(tmp_path), show_progress=False, file_limit=2
    )
    results = wf.execute_workflow()

    # Collect commits
    log_output = _git(["log", "--pretty=%h %s"], tmp_path)
    lines = [line for line in log_output.split("\n") if line]

    # Expect exactly 2 commits (order newest first)
    assert len(lines) == 2, f"Unexpected commits: {lines}"
    subjects = [" ".join(line.split(" ")[1:]) for line in lines]
    # Both expected subjects present
    assert any("add alpha script" in s for s in subjects)
    assert any("add beta script" in s for s in subjects)
    # No combined multi-file commit message
    assert not any("alpha" in s and "beta" in s for s in subjects)

    # Validate workflow results recorded two successes
    file_commits = results.get("file_commits", [])
    successes = [r for r in file_commits if r.success]
    assert len(successes) == 2, results
