import subprocess
from pathlib import Path
import sys


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


def test_heuristic_fallback_after_invalid_llm(tmp_path, monkeypatch):
    _git(["init", "-q"], tmp_path)
    _git(["config", "user.name", "Tester"], tmp_path)
    _git(["config", "user.email", "tester@example.com"], tmp_path)

    target_file = tmp_path / "demo.py"
    target_file.write_text("print('demo')\n")

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    _add_repo_to_sys_path(Path.cwd())
    from kcmt.config import Config, set_active_config  # type: ignore
    from kcmt.core import KlingonCMTWorkflow  # type: ignore
    from kcmt import commit as commit_module  # type: ignore

    # Sequence of invalid responses (will be retried) -> fallback
    responses = ["feat", "refactor(core):", ""]

    def fake_generate(
        _diff: str, _context: str = "", _style: str = "conventional"
    ) -> str:  # noqa: D401
        return responses.pop(0) if responses else ""

    monkeypatch.setattr(
        commit_module.LLMClient,
        "generate_commit_message",
        staticmethod(fake_generate),
    )

    cfg = Config(
        provider="openai",
        model="gpt-5-mini",
        llm_endpoint="https://api.openai.com/v1",
        api_key_env="OPENAI_API_KEY",
        git_repo_path=str(tmp_path),
        allow_fallback=True,
    )
    set_active_config(cfg)

    wf = KlingonCMTWorkflow(repo_path=str(tmp_path), show_progress=False)
    results = wf.execute_workflow()

    file_commits = [r for r in results.get("file_commits", []) if r.success]
    assert len(file_commits) == 1, results
    msg = file_commits[0].message
    assert msg is not None
    assert msg.startswith(
        (
            "feat(",
            "refactor(",
            "chore(",
            "docs(",
            "test(",
            "style(",
            "build(",
            "ci(",
            "perf(",
            "revert(",
        )
    )
    # Ensure no generic fallback phrase present
    assert "fallback" not in msg.lower()
