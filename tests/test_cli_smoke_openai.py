import subprocess
from pathlib import Path

from kcmt.config import (
    DEFAULT_MODELS,
    Config,
    clear_active_config,
    load_config,
    save_config,
)


def _git(cmd: list[str], cwd: Path) -> str:
    result = subprocess.run(
        ["git"] + cmd,
        cwd=cwd,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def test_cli_smoke_openai(monkeypatch, tmp_path):
    """Exercise the CLI happy path with OpenAI provider using a stubbed LLM.

    Covers: initial ephemeral config creation, workflow execution, commit
    generation, and summary output without hitting the real network.
    """
    _git(["init", "-q"], tmp_path)
    _git(["config", "user.name", "Tester"], tmp_path)
    _git(["config", "user.email", "tester@example.com"], tmp_path)

    target = tmp_path / "example.py"
    target.write_text("print('hello')\n")
    _git(["add", "example.py"], tmp_path)
    _git(["commit", "-m", "chore(core): seed"], tmp_path)
    target.write_text("print('hello world')\n")  # modify file

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "1")  # force non-interactive

    # Stub LLMClient.generate_commit_message to avoid network
    from kcmt import commit as commit_module  # noqa: WPS433

    monkeypatch.setattr(
        commit_module.LLMClient,
        "generate_commit_message",
        staticmethod(lambda *a, **k: "feat(core): update example"),
    )

    # Run CLI via its main entrypoint
    from kcmt.cli import main  # noqa: WPS433

    exit_code = main([
        "--provider",
        "openai",
        "--no-progress",
        "--limit",
        "1",
        "--repo-path",
        str(tmp_path),
    ])

    assert exit_code == 0
    log = _git(["log", "--oneline", "-n", "1"], tmp_path)
    assert "feat(core): update example" in log


def test_config_migration_openai_model_upgrade(tmp_path):
    """Persist an old model alias then ensure it's upgraded on load."""
    # Write legacy config
    legacy = Config(
        provider="openai",
        model="gpt-5-mini",  # old value
        llm_endpoint=DEFAULT_MODELS["openai"]["endpoint"],
        api_key_env=DEFAULT_MODELS["openai"]["api_key_env"],
        git_repo_path=str(tmp_path),
    )
    save_config(legacy, tmp_path)
    clear_active_config()
    loaded = load_config(repo_root=tmp_path)
    assert loaded.model == DEFAULT_MODELS["openai"]["model"]


def test_llm_env_disable_shortcut(monkeypatch):
    """Ensure KCMT_TEST_DISABLE_OPENAI short-circuits network call path."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("KCMT_TEST_DISABLE_OPENAI", "1")
    clear_active_config()
    cfg = load_config(overrides={"provider": "openai"})

    # Avoid patching underlying OpenAI client; rely on early return.
    from kcmt.llm import LLMClient  # noqa: WPS433

    client = LLMClient(config=cfg)
    msg = client.generate_commit_message(
        "diff --git a/x b/x\n@@\n+line one\n+line two\n", "ctx"
    )
    assert msg.startswith("feat(") or msg.startswith("chore(")
