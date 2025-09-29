import importlib
import sys
import types

import pytest


# Reuse fake OpenAI scaffolding similar to test_llm.py but allow dynamic content per-call
class _FakeCompletionMsg:
    def __init__(self, content):
        self.message = types.SimpleNamespace(content=content)


class _FakeChatCompletions:
    def __init__(self, content=None, error=None):
        self._content = content
        self._error = error

    def create(self, *args, **kwargs):  # noqa: D401
        if self._error is not None:
            raise self._error
        return types.SimpleNamespace(choices=[_FakeCompletionMsg(self._content)])


class _FakeChat:
    def __init__(self, content=None, error=None):
        self.completions = _FakeChatCompletions(content=content, error=error)


class _FakeOpenAI:
    def __init__(self, *args, **kwargs):
        self._content = kwargs.pop("_content", None)
        self._error = kwargs.pop("_error", None)
        self.chat = _FakeChat(content=self._content, error=self._error)


def _reload_llm_with_fake(monkeypatch, content=None, error=None):
    # Choose provider xai (uses OpenAI style path but different env var)
    monkeypatch.setenv("XAI_API_KEY", "fake_api_key_for_testing")
    # Clear config + llm modules to ensure fresh load
    sys.modules.pop("kcmt.config", None)
    importlib.import_module("kcmt.config")

    monkeypatch.setitem(
        sys.modules,
        "openai",
        types.SimpleNamespace(
            OpenAI=lambda **kwargs: _FakeOpenAI(_content=content, _error=error)
        ),
    )
    import kcmt.llm as llm_module

    importlib.reload(llm_module)
    return llm_module


def test_minimal_diff_path(monkeypatch):
    llm_module = _reload_llm_with_fake(
        monkeypatch, content="feat(core): adjust something"
    )
    client = llm_module.LLMClient()
    # diff < 10 chars triggers minimal path
    msg = client.generate_commit_message("x+y")
    # Should follow minimal commit generation fallback (not from LLM content necessarily)
    assert "minor update" in msg or "update" in msg


def test_binary_diff_path(monkeypatch):
    llm_module = _reload_llm_with_fake(
        monkeypatch, content="feat(core): binary change body"
    )
    client = llm_module.LLMClient()
    diff = "Binary files a/image.png and b/image.png differ"  # triggers binary path
    msg = client.generate_commit_message(diff, context="File: assets/image.png")
    assert msg.startswith("feat(assets):") or "binary file" in msg


def test_large_diff_path(monkeypatch):
    llm_module = _reload_llm_with_fake(
        monkeypatch, content="feat(core): large diff message body"
    )
    client = llm_module.LLMClient()
    huge = "line\n" * 9000  # > 8000 chars threshold
    msg = client.generate_commit_message(huge, context="File: src/big_module.py")
    # Expect heuristic message for large python file
    assert msg.startswith("feat(core): add") or msg.startswith("refactor(core):")


def test_subject_wrapping(monkeypatch):
    long_subject = "feat(core): " + "a" * 90
    llm_module = _reload_llm_with_fake(monkeypatch, content=long_subject)
    client = llm_module.LLMClient()
    diff = "changed line one\n" * 6  # ensure not minimal path
    msg = client.generate_commit_message(diff)
    first_line = msg.splitlines()[0]
    # Should have ellipsis if shortened
    assert len(first_line) <= client.config.max_commit_length + 2  # allow …
    if len("feat(core): " + "a" * 90) > client.config.max_commit_length:
        assert first_line.endswith("…")


def test_openai_error_path(monkeypatch):
    llm_module = _reload_llm_with_fake(monkeypatch, error=RuntimeError("down"))
    client = llm_module.LLMClient()
    diff = "changed line one\n" * 6
    with pytest.raises(llm_module.LLMError):
        client.generate_commit_message(diff)
        client.generate_commit_message(diff)
        client.generate_commit_message(diff)
        client.generate_commit_message(diff)
