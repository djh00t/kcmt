import types

from kcmt.config import clear_active_config, load_config
from kcmt.core import CommitResult, KlingonCMTWorkflow


def _mk_workflow(auto_push: bool):
    # Ensure fresh config each call
    clear_active_config()
    overrides = {"auto_push": "1" if auto_push else "0"}
    cfg = load_config(overrides=overrides)
    # Instantiate workflow (will create real GitRepo but we stub internals)
    wf = KlingonCMTWorkflow(
        repo_path=cfg.git_repo_path,
        max_retries=1,
        config=cfg,
    )
    return wf, cfg


def test_auto_push_triggers_on_success(monkeypatch):
    wf, _cfg = _mk_workflow(auto_push=True)

    # Stub internal processing to simulate one successful file commit
    monkeypatch.setattr(
        wf,
        "_process_deletions_first",
        types.MethodType(lambda self, *_: [], wf),
    )
    monkeypatch.setattr(
        wf,
        "_process_per_file_commits",
        types.MethodType(
            lambda self, *_: [
                CommitResult(
                    success=True,
                    commit_hash="abcd1234",
                    message="chore(core): test auto push",
                    file_path="kcmt/core.py",
                )
            ],
            wf,
        ),
    )

    push_called = {}

    def fake_push():  # noqa: D401 - simple stub
        push_called["called"] = True
        return "pushed"

    monkeypatch.setattr(wf.git_repo, "push", fake_push)

    results = wf.execute_workflow()
    assert results.get("pushed") is True
    assert push_called.get("called") is True


def test_auto_push_not_triggered_when_disabled(monkeypatch):
    wf, _cfg = _mk_workflow(auto_push=False)
    monkeypatch.setattr(
        wf,
        "_process_deletions_first",
        types.MethodType(lambda self, *_: [], wf),
    )
    monkeypatch.setattr(
        wf,
        "_process_per_file_commits",
        types.MethodType(
            lambda self, *_: [
                CommitResult(
                    success=True,
                    commit_hash="deadbeef",
                    message="chore(core): no push",
                    file_path="kcmt/core.py",
                )
            ],
            wf,
        ),
    )

    monkeypatch.setattr(
        wf.git_repo,
        "push",
        lambda: (_ for _ in ()).throw(AssertionError("should not push")),
    )

    results = wf.execute_workflow()
    assert "pushed" not in results


def test_auto_push_not_triggered_on_no_success(monkeypatch):
    wf, _cfg = _mk_workflow(auto_push=True)
    monkeypatch.setattr(
        wf,
        "_process_deletions_first",
        types.MethodType(lambda self, *_: [], wf),
    )
    monkeypatch.setattr(
        wf,
        "_process_per_file_commits",
        types.MethodType(
            lambda self, *_: [
                CommitResult(
                    success=False,
                    commit_hash=None,
                    message=None,
                    error="failure",
                    file_path="kcmt/core.py",
                )
            ],
            wf,
        ),
    )

    monkeypatch.setattr(
        wf.git_repo,
        "push",
        lambda: (_ for _ in ()).throw(AssertionError("should not push")),
    )

    results = wf.execute_workflow()
    assert "pushed" not in results
