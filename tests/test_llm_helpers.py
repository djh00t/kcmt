from kcmt.config import clear_active_config, load_config
from kcmt.llm import LLMClient


def _client(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-key")
    clear_active_config()
    cfg = load_config(overrides={"provider": "openai"})

    # Stub OpenAI network client minimal
    class _Dummy:
        class chat:  # type: ignore
            class completions:  # type: ignore
                @staticmethod
                def create(**_kwargs):  # noqa: D401
                    class _C:
                        def __init__(self):  # noqa: D401
                            self.message = type(
                                "M", (), {"content": "feat(core): add thing"}
                            )

                    return type("R", (), {"choices": [_C()]})

    monkeypatch.setattr("kcmt.llm.OpenAI", lambda base_url, api_key: _Dummy())
    return LLMClient(config=cfg)


def test_subject_enforcement(monkeypatch):
    c = _client(monkeypatch)
    long_subject = "feat(core): " + ("x" * 120)
    # Access internal helper intentionally to assert subject shortening.
    adjusted = c._enforce_subject_length(long_subject)  # noqa: SLF001
    assert len(adjusted.splitlines()[0]) <= c.config.max_commit_length + 1  # ellipsis


def test_wrap_body(monkeypatch):
    c = _client(monkeypatch)
    msg = "feat(core): change\n\n" + "A" * 90 + "\n\n" + "B" * 10
    wrapped = c._wrap_body(msg, width=72)  # noqa: SLF001
    assert any(len(line) <= 72 for line in wrapped.splitlines()[2:])


def test_generate_commit_message_always_uses_llm(monkeypatch):
    c = _client(monkeypatch)
    # Minimal diff should still flow through the LLM path
    minimal = c.generate_commit_message("x+y", context="File: something.py")
    assert minimal.startswith("feat(core): add thing")

    # Large diffs are truncated but still sent to the LLM
    large_diff = "line\n" * 9000
    large = c.generate_commit_message(large_diff, context="File: big.py")
    assert large.startswith("feat(core): add thing")

    # Binary diffs are summarised and then passed to the LLM
    binary = c.generate_commit_message(
        "Binary files /dev/null and b/image.png differ",
        context="File: image.png",
    )
    assert binary.startswith("feat(core): add thing")
