import os

from kcmt.config import Config, clear_active_config, set_active_config
from kcmt.core import KlingonCMTWorkflow


def test_deletion_only_commit(tmp_path, monkeypatch):
    os.chdir(tmp_path)
    os.system("git init -q")
    (tmp_path / "delete_me.txt").write_text("bye")
    os.system("git add delete_me.txt")
    os.system('git commit -m "chore(core): init" -q')

    # Delete the file
    os.remove(tmp_path / "delete_me.txt")

    cfg = Config(
        provider="openai",
        model="gpt-test",
        llm_endpoint=None,
        api_key_env="OPENAI_API_KEY",
        allow_fallback=True,
        auto_push=False,
    )
    set_active_config(cfg)

    # Patch validation to avoid LLM call in validate_and_fix_commit_message
    def identity(self, message):  # noqa: D401, ARG002
        return message

    from kcmt.commit import CommitGenerator as CG

    monkeypatch.setattr(CG, "validate_and_fix_commit_message", identity)

    wf = KlingonCMTWorkflow(show_progress=False, config=cfg)
    res = wf.execute_workflow()
    deletions = res["deletions_committed"]
    assert deletions, "Expected a deletion commit"
    assert deletions[0].success
    assert deletions[0].message.startswith("chore: remove")

    clear_active_config()
