import subprocess

from kcmt import llm as llm_module
from kcmt.cli import CLI
from kcmt.config import load_persisted_config


def _git(cmd, cwd):
    subprocess.run(["git"] + cmd, cwd=cwd, check=True, capture_output=True)


def test_auto_push_persist_env(monkeypatch, tmp_path):
    # Simulate first run with env var enabling auto push
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("KLINGON_CMT_AUTO_PUSH", "1")

    # Need an initialized git repo (previous failure returned code 1)
    _git(["init", "-q"], tmp_path)
    _git(["config", "user.name", "Tester"], tmp_path)
    _git(["config", "user.email", "tester@example.com"], tmp_path)
    (tmp_path / "stub.txt").write_text("hi")
    _git(["add", "stub.txt"], tmp_path)
    _git(["commit", "-m", "chore(core): init"], tmp_path)

    cli = CLI()
    # Monkeypatch LLM to avoid network & ensure valid commit message
    stub_msg = "chore(config): update configuration"

    def stub_generate(diff, context="", style="conventional"):  # noqa: D401, ARG001
        return stub_msg

    monkeypatch.setattr(
        llm_module.LLMClient,
        "generate_commit_message",
        staticmethod(stub_generate),
    )

    code = cli.run(
        [
            "--provider",
            "openai",
            "--repo-path",
            str(tmp_path),
            "--no-progress",
            "--allow-fallback",
        ]
    )
    # Expect success (0) now that repo exists
    assert code == 0

    cfg = load_persisted_config(tmp_path)
    assert cfg is not None
    assert cfg.auto_push is True

    # Unset env and run again; auto_push should remain True due to persistence
    monkeypatch.delenv("KLINGON_CMT_AUTO_PUSH", raising=False)
    cli2 = CLI()
    code2 = cli2.run(
        [
            "--provider",
            "openai",
            "--repo-path",
            str(tmp_path),
            "--no-progress",
            "--allow-fallback",
        ]
    )
    assert code2 == 0
    cfg2 = load_persisted_config(tmp_path)
    assert cfg2.auto_push is True
    assert cfg2.auto_push is True
    assert cfg2.auto_push is True
    assert cfg2.auto_push is True
    assert cfg2.auto_push is True
