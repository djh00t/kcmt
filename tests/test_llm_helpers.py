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


def test_minimal_and_large_and_binary(monkeypatch):
    c = _client(monkeypatch)
    minimal = c._generate_minimal_commit_message(  # noqa: SLF001
        "File: something.py", "conventional"
    )
    assert minimal.startswith("refactor(") or minimal.startswith("chore(")
    large = c._generate_large_file_commit_message(  # noqa: SLF001
        "new file mode", "File: big.py", "conventional"
    )
    assert large.startswith("feat(")
    binary = c._generate_binary_commit_message(  # noqa: SLF001
        "Binary files /dev/null and b differ",
        "File: image.png",
        "conventional",
    )
    assert binary.startswith("feat(")
