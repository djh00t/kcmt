import os

from kcmt.commit import CommitGenerator
from kcmt.config import Config, clear_active_config, set_active_config
from kcmt.core import KlingonCMTWorkflow


def test_push_failure_recorded(tmp_path, monkeypatch):
    os.chdir(tmp_path)
    os.system("git init -q")
    (tmp_path / "file.txt").write_text("hello")
    os.system("git add file.txt")
    os.system('git commit -m "chore(core): init" -q')
    (tmp_path / "file.txt").write_text("hello world")
    os.system("git add file.txt")

    cfg = Config(
        provider="openai",
        model="gpt-test",
        llm_endpoint="https://api.openai.com/v1",
        api_key_env="OPENAI_API_KEY",
        auto_push=True,  # trigger push attempt
    )
    set_active_config(cfg)

    def fake_suggest(self, diff, context, style):  # noqa: D401, ARG001
        return "chore(core): update file.txt"

    monkeypatch.setattr(
        CommitGenerator,
        "suggest_commit_message",
        fake_suggest,
    )

    # Monkeypatch push to raise error
    from kcmt.exceptions import GitError
    from kcmt.git import GitRepo

    def failing_push(self, remote="origin", branch=None):  # noqa: D401, ARG001
        raise GitError("remote not found")

    monkeypatch.setattr(GitRepo, "push", failing_push)

    wf = KlingonCMTWorkflow(show_progress=False, config=cfg)
    res = wf.execute_workflow()
    assert "pushed" not in res or not res.get("pushed")
    error_msgs = res.get("errors", [])
    assert any("Auto-push failed:" in e for e in error_msgs)

    clear_active_config()
