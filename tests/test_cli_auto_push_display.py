from kcmt.cli import CLI


def test_cli_auto_push_display(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    # Fake workflow to simulate successful commit + push flag
    class _FakeWorkflow:
        def __init__(self, repo_path=None, **__):
            self.git_repo = type("Repo", (), {"repo_path": repo_path})()

        def execute_workflow(self):  # noqa: D401
            return {
                "deletions_committed": [],
                "file_commits": [],
                "errors": [],
                "summary": "Done",
                "pushed": True,
            }

        def stats_snapshot(self):  # pragma: no cover - stubbed
            return {}

        def commit_subjects(self):  # pragma: no cover - stubbed
            return []

    monkeypatch.setenv("KLINGON_CMT_AUTO_PUSH", "1")
    monkeypatch.setattr("kcmt.cli.KlingonCMTWorkflow", _FakeWorkflow)
    cli = CLI()
    code = cli.run(
        [
            "--provider",
            "openai",
            "--repo-path",
            str(tmp_path),
            "--no-progress",
        ]
    )
    captured = capsys.readouterr()
    assert code in (0, 2)
    assert "Workflow Summary" in captured.out
    assert "Auto-push" in captured.out
    assert "pushed" in captured.out
