import importlib
import os
import types

import pytest

from kcmt.commit import CommitGenerator
from kcmt.config import Config, set_active_config
from kcmt.exceptions import LLMError


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
    )
    set_active_config(cfg)
    monkeypatch.setenv("OPENAI_API_KEY", "X")

    responses = ["bad1", "bad2", "bad3"]  # all invalid conventional headers

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
    with pytest.raises(LLMError):
        gen.suggest_commit_message(diff, context="File: a.txt")


def test_empty_responses_raise_llmerror(monkeypatch, tmp_path):
    """Empty strings after retries raise LLMError."""
    repo_dir = tmp_path / "repo3"
    repo_dir.mkdir()
    os.chdir(repo_dir)
    os.system("git init -q")
    (repo_dir / "b.txt").write_text("start\n")
    os.system("git add b.txt && git commit -m 'chore: init' -q")
    (repo_dir / "b.txt").write_text("change\n")

    cfg = Config(
        provider="openai",
        model="gpt-x",
        llm_endpoint="http://local",
        api_key_env="OPENAI_API_KEY",
        git_repo_path=str(repo_dir),
    )
    set_active_config(cfg)
    os.environ["OPENAI_API_KEY"] = "X"

    class _EmptyClient:
        def generate_commit_message(self, *_a, **_k):  # noqa: D401
            return ""  # always empty

    monkeypatch.setitem(
        importlib.sys.modules,
        "kcmt.llm",
        types.SimpleNamespace(LLMClient=lambda *a, **k: _EmptyClient()),
    )

    gen = CommitGenerator(repo_path=str(repo_dir), config=cfg)
    diff = gen.git_repo.get_working_diff()
    with pytest.raises(LLMError):
        gen.suggest_commit_message(diff, context="File: b.txt")
