import sys

import pytest


@pytest.fixture(autouse=True)
def reset_config(monkeypatch, tmp_path):
    # Default to OpenAI provider with a fake key
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("XAI_API_KEY", raising=False)

    # Ensure no persisted config interferes
    sys.modules.pop("kcmt.config", None)

    from kcmt.config import clear_active_config

    clear_active_config()
    yield
    clear_active_config()


# Ensure no real Anthropic network calls escape during tests that don't
# explicitly mock the endpoint. This only intercepts Anthropic's messages API
# and leaves other providers untouched.
@pytest.fixture(autouse=True)
def _mock_anthropic_messages(monkeypatch):
    import httpx

    original_post = httpx.post

    def fake_post(url, *args, **kwargs):  # noqa: D401
        try:
            if (
                isinstance(url, str)
                and "api.anthropic.com" in url
                and "/v1/messages" in url
            ):

                class _Resp:
                    status_code = 200

                    def json(self):  # noqa: D401
                        return {
                            "content": [
                                {
                                    "type": "text",
                                    "text": "feat(test): stubbed anthropic message",
                                }
                            ]
                        }

                    @property
                    def text(self):  # noqa: D401
                        return "ok"

                return _Resp()
        except Exception:
            # Fall through to original on any unexpected condition
            pass
        return original_post(url, *args, **kwargs)

    monkeypatch.setattr(httpx, "post", fake_post)
