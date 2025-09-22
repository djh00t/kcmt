import os
import types
import importlib

import pytest

from kcmt.commit import CommitGenerator
from kcmt.config import Config, set_active_config


def test_retry_and_optional_fallback(monkeypatch, tmp_path):
    # Setup repo structure
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    os.chdir(repo_dir)
    os.system("git init -q")
    (repo_dir / "file.txt").write_text("hello\n")
    os.system("git add file.txt && git commit -m 'chore: init' -q")
    (repo_dir / "file.txt").write_text("hello world\n")

    cfg = Config(
        provider="openai",
        model="gpt-x",
        llm_endpoint="http://local",
        api_key_env="OPENAI_API_KEY",
        git_repo_path=str(repo_dir),
        allow_fallback=True,
    )
    set_active_config(cfg)
    monkeypatch.setenv("OPENAI_API_KEY", "X")

    # Fake client that produces 2 invalid messages then empty
    responses = ["invalid", "still bad", ""]

    class _FakeClient:
        def __init__(self, *_a, **_k):  # noqa: D401
            pass

        def generate_commit_message(self, *_a, **_k):  # noqa: D401
            return responses.pop(0) if responses else ""

    monkeypatch.setitem(
        importlib.sys.modules,
        "kcmt.llm",
        types.SimpleNamespace(LLMClient=lambda *a, **k: _FakeClient()),
    )

    gen = CommitGenerator(repo_path=str(repo_dir), config=cfg)
    diff = gen.git_repo.get_working_diff()
    msg = gen.suggest_commit_message(diff, context="File: file.txt")
    assert msg.startswith(("feat(", "refactor(", "chore(", "docs(", "test("))


def test_retry_exhaustion_without_fallback(monkeypatch, tmp_path):
    repo_dir = tmp_path / "repo2"
    repo_dir.mkdir()
    os.chdir(repo_dir)
    os.system("git init -q")
    (repo_dir / "a.txt").write_text("a\n")
    os.system("git add a.txt && git commit -m 'chore: init' -q")
    (repo_dir / "a.txt").write_text("b\n")

    cfg = Config(
        provider="openai",
        model="gpt-x",
        llm_endpoint="http://local",
        api_key_env="OPENAI_API_KEY",
        git_repo_path=str(repo_dir),
        allow_fallback=False,
    )
    set_active_config(cfg)
    monkeypatch.setenv("OPENAI_API_KEY", "X")

    responses = ["bad1", "bad2", "bad3"]

    class _BadClient:
        def generate_commit_message(self, *_a, **_k):  # noqa: D401
            return responses.pop(0)

    monkeypatch.setitem(
        importlib.sys.modules,
        "kcmt.llm",
        types.SimpleNamespace(LLMClient=lambda *a, **k: _BadClient()),
    )

    gen = CommitGenerator(repo_path=str(repo_dir), config=cfg)
    diff = gen.git_repo.get_working_diff()
    with pytest.raises(Exception):
        gen.suggest_commit_message(diff, context="File: a.txt")
