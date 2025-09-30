from kcmt.cli import CLI


def test_cli_auto_push_display(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    # Fake workflow to simulate successful commit + push flag
    class _FakeWorkflow:
        def __init__(self, *_, **__):
            pass

        def execute_workflow(self):  # noqa: D401
            return {
                "deletions_committed": [],
                "file_commits": [],
                "errors": [],
                "summary": "Done",
                "pushed": True,
            }

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
