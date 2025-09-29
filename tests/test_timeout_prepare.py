import os
import time

from kcmt.config import Config, clear_active_config, set_active_config
from kcmt.core import KlingonCMTWorkflow


def test_per_file_timeout(monkeypatch, tmp_path):
    repo = tmp_path
    (repo / "file1.txt").write_text("one\n")
    (repo / "file2.txt").write_text("two\n")

    os.system(f"git -C {repo} init -q")
    os.system(f"git -C {repo} config user.name tester")
    os.system(f"git -C {repo} config user.email tester@example.com")
    os.system(f"git -C {repo} add .")
    os.system(f"git -C {repo} commit -m initial -q")

    # Modify files to create diffs
    (repo / "file1.txt").write_text("one modified\n")
    (repo / "file2.txt").write_text("two modified\n")

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("KCMT_PREPARE_PER_FILE_TIMEOUT", "0.1")
    monkeypatch.setenv("KCMT_TEST_DISABLE_OPENAI", "1")

    cfg = Config(
        provider="openai",
        model="gpt-5-mini-2025-08-07",
        llm_endpoint="https://api.openai.com/v1",
        api_key_env="OPENAI_API_KEY",
        git_repo_path=str(repo),
    )
    set_active_config(cfg)

    # Monkeypatch CommitGenerator.suggest_commit_message to sleep
    import kcmt.commit as commit_module  # noqa: WPS433

    real_suggest = commit_module.CommitGenerator.suggest_commit_message

    def slow_suggest(self, *a, **k):  # noqa: D401, ARG002
        time.sleep(0.5)  # exceed the timeout
        return real_suggest(self, *a, **k)

    monkeypatch.setattr(
        commit_module.CommitGenerator,
        "suggest_commit_message",
        slow_suggest,
    )

    wf = KlingonCMTWorkflow(
        repo_path=str(repo),
        max_retries=1,
        show_progress=False,
        config=cfg,
    )
    results = wf.execute_workflow()
    file_commits = results.get("file_commits", [])
    # Expect both to fail due to timeout
    assert any(r for r in file_commits if not r.success)
    clear_active_config()
