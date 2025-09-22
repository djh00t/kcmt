import types

from kcmt.config import clear_active_config, load_config
from kcmt.llm import LLMClient


def test_llm_openai_basic(monkeypatch):
    # Force provider openai with stubbed OpenAI client
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    clear_active_config()
    cfg = load_config(overrides={"provider": "openai", "model": "gpt-test"})

    class _ChoiceMsg:
        def __init__(self, content):
            self.message = types.SimpleNamespace(content=content)

    class _Resp:
        def __init__(self, content):
            self.choices = [_ChoiceMsg(content)]

    class _ChatCompletions:
        @staticmethod
        def create(**_kwargs):  # noqa: D401
            return _Resp("feat(core): add feature")

    class _Chat:
        completions = _ChatCompletions()

    class _Client:
        chat = _Chat()

    monkeypatch.setattr("kcmt.llm.OpenAI", lambda base_url, api_key: _Client())

    client = LLMClient(config=cfg)
    msg = client.generate_commit_message(
        "diff --git a b\n@@\n+code", "ctx"
    )
    assert msg.startswith("feat(")
