import importlib
import sys
from pathlib import Path

import httpx
import pytest

from kcmt.config import clear_active_config


def _reload_with_anthropic(
    monkeypatch, content="feat(core): add thing", raise_err: bool = False
):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    # Ensure OpenAI key does NOT force provider auto-selection to openai
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("KCMT_PROVIDER", "anthropic")
    # Remove cached modules so config re-evaluates env
    for mod in ["kcmt.config", "kcmt.llm"]:
        if mod in sys.modules:
            sys.modules.pop(mod)
    # Remove persisted config if present so provider auto-select runs fresh
    cfg_path = Path.cwd() / ".kcmt" / "config.json"
    if cfg_path.exists():  # pragma: no cover
        try:
            cfg_path.unlink()
        except OSError:
            pass
    clear_active_config()

    # Patch httpx.post used by _call_anthropic
    class _Resp:
        def __init__(self, status_code=200):
            self.status_code = status_code
            self._text = "ok"

        def json(self):  # noqa: D401
            if raise_err:
                return {"error": "boom"}
            return {"content": [{"type": "text", "text": content}]}

        @property
        def text(self):  # noqa: D401
            if self.status_code >= 400:
                return '{"error": "auth"}'
            return self._text

    def fake_post(
        _url, headers=None, json=None, timeout=60.0, **_kw
    ):  # noqa: D401  # pragma: no cover
        _ = (headers, json, timeout, _kw)
        if raise_err:
            return _Resp(status_code=500)
        return _Resp(status_code=200)

    monkeypatch.setattr(httpx, "post", fake_post)
    import kcmt.llm as llm

    importlib.reload(llm)
    return llm


def test_anthropic_success(monkeypatch):
    llm = _reload_with_anthropic(monkeypatch, content="feat(core): improve speed")
    client = llm.LLMClient()
    msg = client.generate_commit_message("diff --git a b", "ctx")
    assert msg.startswith("feat(")


def test_anthropic_error(monkeypatch):
    llm = _reload_with_anthropic(monkeypatch, raise_err=True)
    client = llm.LLMClient()
    with pytest.raises(llm.LLMError):
        client.generate_commit_message("diff --git a b", "ctx")
