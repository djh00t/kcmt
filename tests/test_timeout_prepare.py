import os
import time
from collections import defaultdict
from threading import Lock

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
    monkeypatch.setenv("KCMT_LLM_REQUEST_TIMEOUT", "0.1")
    monkeypatch.setenv("KCMT_TEST_DISABLE_OPENAI", "1")

    cfg = Config(
        provider="openai",
        model="gpt-5-mini-2025-08-07",
        llm_endpoint="https://api.openai.com/v1",
        api_key_env="OPENAI_API_KEY",
        git_repo_path=str(repo),
    )
    set_active_config(cfg)

    import kcmt.commit as commit_module  # noqa: PLC0415

    call_counts = defaultdict(int)
    lock = Lock()

    # Simulate slow commit generation so the per-file timeout triggers.

    def slow_suggest(self, *args, **kwargs):  # noqa: D401, ARG002
        context = kwargs.get("context") or ""
        file_path = (
            context.split("File: ", 1)[-1] if "File: " in context else context
        )
        with lock:
            call_counts[file_path] += 1
        time.sleep(0.2)  # exceed the timeout without heavy LLM work
        return "feat(test): stubbed commit message"

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

    assert all(not r.success for r in file_commits)
    assert all(
        r.error and "attempt 4/4" in r.error for r in file_commits
    ), "Timeout errors should report final attempt counts"
    assert call_counts["file1.txt"] == 4
    assert call_counts["file2.txt"] == 4

    clear_active_config()


def test_prepare_failure_limit(monkeypatch, tmp_path):
    repo = tmp_path
    os.chdir(repo)
    os.system("git init -q")
    os.system("git config user.name tester")
    os.system("git config user.email tester@example.com")

    for idx in range(30):
        path = repo / f"file{idx}.txt"
        path.write_text(f"initial {idx}\n")

    os.system("git add .")
    os.system('git commit -m "chore(core): seed" -q')

    for idx in range(30):
        path = repo / f"file{idx}.txt"
        path.write_text(f"updated {idx}\n")

    os.system("git add .")

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    cfg = Config(
        provider="openai",
        model="gpt-5-mini-2025-08-07",
        llm_endpoint="https://api.openai.com/v1",
        api_key_env="OPENAI_API_KEY",
        git_repo_path=str(repo),
    )
    set_active_config(cfg)

    from kcmt.core import KlingonCMTWorkflow  # noqa: PLC0415
    from kcmt.exceptions import LLMError  # noqa: PLC0415

    def explode_prepare(self, change):  # noqa: D401, ARG002
        raise LLMError("synthetic failure")

    monkeypatch.setattr(
        KlingonCMTWorkflow,
        "_prepare_single_change",
        explode_prepare,
    )

    wf = KlingonCMTWorkflow(
        repo_path=str(repo),
        show_progress=False,
        config=cfg,
    )
    results = wf.execute_workflow()

    file_commits = results.get("file_commits", [])
    assert len(file_commits) == 25
    failure_errors = [res.error for res in file_commits if not res.success]
    assert len(failure_errors) == 25
    summary_errors = results.get("errors", [])
    assert any("failure limit" in err for err in summary_errors)

    clear_active_config()
