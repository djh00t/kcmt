import importlib
import sys
import types

import pytest

from kcmt.config import DEFAULT_MODELS, Config
from kcmt.providers import openai_driver


class _FakeCompletionMsg:
    def __init__(self, content):
        self.message = types.SimpleNamespace(content=content)


class _FakeChatCompletions:
    def __init__(self, content=None, error=None):
        self._content = content
        self._error = error

    def create(self, *args, **kwargs):
        if self._error is not None:
            raise self._error
        return types.SimpleNamespace(choices=[_FakeCompletionMsg(self._content)])


class _FakeChat:
    def __init__(self, content=None, error=None):
        self.completions = _FakeChatCompletions(content=content, error=error)


class _FakeOpenAI:
    def __init__(self, *args, **kwargs):
        # Allow injection later
        self._content = kwargs.pop("_content", None)
        self._error = kwargs.pop("_error", None)
        self.chat = _FakeChat(content=self._content, error=self._error)


def _reload_llm_with_fake(monkeypatch, content=None, error=None):
    # Set required API key environment variable
    monkeypatch.setenv("XAI_API_KEY", "fake_api_key_for_testing")

    # Ensure config loaded and llm reloaded so it picks up env and fake OpenAI
    sys.modules.pop("kcmt.config", None)
    importlib.import_module("kcmt.config")

    # Stub the external 'openai' dependency before importing klingon_cmt.llm
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


def test_build_prompt_variants(monkeypatch):
    """Test that different prompt styles generate appropriate content."""
    # Given
    llm_module = _reload_llm_with_fake(monkeypatch, content="ignored")
    client = llm_module.LLMClient()

    # When/Then: conventional includes guidelines
    p1 = client._build_prompt("diff", "ctx", "conventional")
    assert "Generate a conventional commit message" in p1
    assert "DIFF:" in p1 and "diff" in p1
    assert "CONTEXT:" in p1 and "ctx" in p1
    assert "- MUST use format:" in p1

    # When/Then: simple includes hint


def test_generate_commit_message_subject_only_enforcement(monkeypatch):
    llm_module = _reload_llm_with_fake(
        monkeypatch, content="feat: a very long description"
    )
    client = llm_module.LLMClient()
    # ensure length > 10 to avoid minimal path
    long_diff = "diff line one\n" * 5
    msg = client.generate_commit_message(long_diff, "ctx", "conventional")
    # Should not truncate entire message to env length; subject unchanged
    assert msg.startswith("feat:")
    assert "very long description" in msg


def test_generate_commit_message_empty_response_raises(monkeypatch):
    """Test that empty LLM response raises appropriate error."""
    # Given empty content
    llm_module = _reload_llm_with_fake(monkeypatch, content="")
    client = llm_module.LLMClient()

    long_diff = "added line one\nremoved line two\nchanged line three"
    with pytest.raises(llm_module.LLMError):
        client.generate_commit_message(long_diff)


def test_generate_commit_message_backend_error_raises(monkeypatch):
    """Test that backend errors are properly propagated as LLMError."""
    # Given underlying client raises
    llm_module = _reload_llm_with_fake(monkeypatch, error=RuntimeError("boom"))
    client = llm_module.LLMClient()

    long_diff = "added line one\nremoved line two\nchanged line three"
    with pytest.raises(llm_module.LLMError):
        client.generate_commit_message(long_diff)


def test_llm_batch_invocation_uses_driver_batch(monkeypatch):
    calls: dict[str, object] = {}

    def fake_invoke(
        self,
        messages,
        *,
        minimal_ok,
        request_timeout=None,
        use_batch=False,
        batch_model=None,
        batch_timeout=None,
        progress_callback=None,
    ):
        _ = (messages, minimal_ok, request_timeout)
        calls["use_batch"] = use_batch
        calls["batch_model"] = batch_model
        calls["batch_timeout"] = batch_timeout
        if progress_callback:
            progress_callback("batch status: running")
        return "feat(core): batched commit"

    monkeypatch.setattr(openai_driver.OpenAIDriver, "invoke_messages", fake_invoke)

    config = Config(
        provider="openai",
        model="gpt-4.1-mini",
        llm_endpoint=DEFAULT_MODELS["openai"]["endpoint"],
        api_key_env=DEFAULT_MODELS["openai"]["api_key_env"],
        git_repo_path=".",
        use_batch=True,
        batch_model="gpt-4.1-mini",
        batch_timeout_seconds=1000,  # Must be >= BATCH_TIMEOUT_MIN_SECONDS (900)
    )
    from kcmt import llm as llm_module  # noqa: PLC0415

    client = llm_module.LLMClient(config=config)
    statuses: list[str] = []
    msg = client.generate_commit_message(
        "diff --git a b\n@@\n+code", "ctx", progress_callback=statuses.append
    )
    assert msg.startswith("feat(")
    assert calls["use_batch"] is True
    assert calls["batch_model"] == "gpt-4.1-mini"
    assert calls["batch_timeout"] == 1000
    assert statuses == ["batch status: running"]
