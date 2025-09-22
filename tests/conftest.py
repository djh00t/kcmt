import json
import sys
from pathlib import Path

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
