import importlib
import sys
import types

import pytest


def _reload_with_anthropic(
    monkeypatch, content="feat(core): add thing", raise_err: bool = False
):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    # Remove cached modules so config re-evaluates env
    for mod in ["kcmt.config", "kcmt.llm"]:
        if mod in sys.modules:
            sys.modules.pop(mod)

    class _Resp:
        def __init__(self, txt):
            self.content = [types.SimpleNamespace(text=txt)]

    class _Client:
        class messages:  # type: ignore
            @staticmethod
            def create(**_kwargs):  # noqa: D401
                if raise_err:
                    raise RuntimeError("anthropic boom")
                return _Resp(content)

    monkeypatch.setitem(
        sys.modules,
        "anthropic",
        types.SimpleNamespace(Anthropic=lambda **kw: _Client()),
    )
    import kcmt.llm as llm
    importlib.reload(llm)
    return llm


def test_anthropic_success(monkeypatch):
    llm = _reload_with_anthropic(
        monkeypatch, content="feat(core): improve speed"
    )
    client = llm.LLMClient()
    msg = client.generate_commit_message("diff --git a b", "ctx")
    assert msg.startswith("feat(")


def test_anthropic_error(monkeypatch):
    llm = _reload_with_anthropic(monkeypatch, raise_err=True)
    client = llm.LLMClient()
    with pytest.raises(llm.LLMError):
        client.generate_commit_message("diff --git a b", "ctx")
