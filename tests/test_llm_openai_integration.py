import os

import pytest

from kcmt.config import clear_active_config, load_config
from kcmt.exceptions import LLMError
from kcmt.llm import LLMClient

pytestmark = pytest.mark.integration


def require_real_key():
    key = os.environ.get("OPENAI_API_KEY")
    if not key or key == "sk-test":  # avoid placeholder
        pytest.skip("Real OPENAI_API_KEY not set; skipping integration test")


def test_openai_integration_basic_round_trip(monkeypatch):
    """Real OpenAI call smoke test (only runs when real key present).

    Ensures that the OpenAI path returns a non-empty commit message
    and that post-processing keeps the conventional prefix structure.
    """
    require_real_key()
    clear_active_config()

    # Use small diff to exercise minimal -> but above minimal threshold
    diff = "diff --git a/foo.py b/foo.py\n@@\n+print('hello world')\n"
    cfg = load_config(
        overrides={
            "provider": "openai",
            "model": os.environ.get("KCMT_OPENAI_MODEL", "gpt-4o-mini"),
        }
    )

    client = LLMClient(config=cfg, debug=False)
    msg = client.generate_commit_message(diff, context="Add greeting")
    assert msg, "Expected non-empty message from OpenAI integration"
    # Loose structural check: starts with type( or type:
    assert any(
        msg.lower().startswith(p)
        for p in ("feat", "chore", "fix", "refactor", "docs", "test")
    )


def test_openai_integration_handles_error(monkeypatch):
    """Force error by using bogus endpoint to ensure LLMError is raised."""
    require_real_key()
    clear_active_config()

    # Bogus endpoint to trigger failure fast
    monkeypatch.setenv(
        "KCMT_LLM_ENDPOINT",
        "https://api.openai.com/v1/does-not-exist",
    )
    diff = "diff --git a/bar.py b/bar.py\n@@\n+print('x')\n"
    cfg = load_config(overrides={"provider": "openai"})
    client = LLMClient(config=cfg, debug=False)
    with pytest.raises(LLMError):
        client.generate_commit_message(diff, context="Force error")
        client.generate_commit_message(diff, context="Force error")
        client.generate_commit_message(diff, context="Force error")
        client.generate_commit_message(diff, context="Force error")
        client.generate_commit_message(diff, context="Force error")
