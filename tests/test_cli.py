import json

import kcmt.cli as cli_module


def test_cli_help_returns_zero():
    cli = cli_module.CLI()
    assert cli.run(["--help"]) == 0


def test_cli_executes_workflow_success(monkeypatch, tmp_path):
    class _FakeWorkflow:
        def __init__(
            self, repo_path=None, max_retries=3, config=None, show_progress=True
        ):
            self.repo_path = repo_path
            self.max_retries = max_retries
            self.config = config
            self.show_progress = show_progress

        def execute_workflow(self):
            return {
                "deletions_committed": [],
                "file_commits": [],
                "errors": [],
                "summary": "No commits were made.",
            }

    monkeypatch.setattr(cli_module, "KlingonCMTWorkflow", _FakeWorkflow)

    cli = cli_module.CLI()
    args = [
        "--provider",
        "openai",
        "--model",
        "test-model",
        "--endpoint",
        "http://localhost",
        "--api-key-env",
        "OPENAI_API_KEY",
        "--repo-path",
        str(tmp_path),
        "--no-progress",
    ]
    assert cli.run(args) == 0


def test_cli_handles_workflow_failure(monkeypatch):
    """Test that CLI handles workflow failures by returning exit code 1."""
    from kcmt.exceptions import KlingonCMTError

    class _FakeWorkflow:
        def __init__(self, *_, **__):
            pass

        def execute_workflow(self):
            raise KlingonCMTError("boom")

    # Set a fake API key to pass validation
    monkeypatch.setenv("OPENAI_API_KEY", "fake-key")
    monkeypatch.setattr(cli_module, "KlingonCMTWorkflow", _FakeWorkflow)
    cli = cli_module.CLI()
    assert cli.run(["--provider", "openai"]) == 1


def test_cli_requires_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    cli = cli_module.CLI()
    assert cli.run(["--provider", "openai"]) == 2


def test_cli_oneshot_happy_path(monkeypatch, tmp_path):
    staged = []
    committed = []

    class _Repo:
        def __init__(self, repo_path, config):
            self.repo_path = repo_path
            self.config = config

        def list_changed_files(self):
            return [(" M", "foo.py"), (" M", "bar.py")]

        def stage_file(self, path):
            staged.append(path)

        def get_file_diff(self, path, staged=False):
            return f"diff for {path}"

        def commit(self, msg):
            committed.append(msg)

        def get_recent_commits(self, count=1):
            return ["abc123 latest"]

    class _CommitGenerator:
        def __init__(self, repo_path=None, config=None):
            pass

        def suggest_commit_message(self, diff, context="", style="conventional"):
            return "feat: update"

        def validate_and_fix_commit_message(self, msg):
            return msg

    monkeypatch.setattr(cli_module, "GitRepo", _Repo)
    monkeypatch.setattr(cli_module, "CommitGenerator", _CommitGenerator)

    cli = cli_module.CLI()
    code = cli.run(
        [
            "--provider",
            "openai",
            "--oneshot",
            "--repo-path",
            str(tmp_path),
            "--no-progress",
        ]
    )

    assert code == 0
    assert staged == ["foo.py"]
    assert committed == ["feat: update"]


def test_cli_configure_writes_file(monkeypatch, tmp_path):
    inputs = iter(
        [
            "1",  # choose anthropic
            "y",  # confirm even without detected key
            "my-model",
            "https://anthropic",
            "ANTHROPIC_API_KEY",
        ]
    )
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))

    cli = cli_module.CLI()
    args = ["--configure", "--repo-path", str(tmp_path)]
    assert cli.run(args) == 0

    config_path = tmp_path / ".kcmt" / "config.json"
    assert config_path.exists()
    data = json.loads(config_path.read_text())
    assert data["provider"] == "anthropic"
    assert data["model"] == "my-model"
