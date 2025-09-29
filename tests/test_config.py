import json
import os

import pytest

from kcmt.config import (
    DEFAULT_MODELS,
    Config,
    clear_active_config,
    detect_available_providers,
    get_active_config,
    load_config,
    load_persisted_config,
    save_config,
    set_active_config,
)


@pytest.fixture(autouse=True)
def _reset_config():
    clear_active_config()
    yield
    clear_active_config()


def test_load_config_prefers_openai_env(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("KLINGON_CMT_LLM_MODEL", "gpt-5-x")
    monkeypatch.setenv("KLINGON_CMT_LLM_ENDPOINT", "https://api.openai.test")
    cfg = load_config(repo_root=tmp_path)

    assert cfg.provider == "openai"
    assert cfg.model == "gpt-5-x"
    assert cfg.llm_endpoint == "https://api.openai.test"
    assert cfg.api_key_env == "OPENAI_API_KEY"


def test_load_config_auto_detects_anthropic(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("ANTHROPIC_KEY_ALT", "anth-test")
    cfg = load_config(repo_root=tmp_path)

    assert cfg.provider == "anthropic"
    assert cfg.api_key_env in {"ANTHROPIC_API_KEY", "ANTHROPIC_KEY_ALT"}


def test_load_config_invalid_override_falls_back(tmp_path):
    cfg = load_config(repo_root=tmp_path, overrides={"provider": "unknown"})
    assert cfg.provider == "openai"


def test_overrides_take_precedence(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    overrides = {
        "provider": "xai",
        "model": "grok-code-fast",
        "endpoint": "https://custom.x.ai",
        "api_key_env": "XAI_SECRET",
        "max_commit_length": "80",
    }
    cfg = load_config(repo_root=tmp_path, overrides=overrides)

    assert cfg.provider == "xai"
    assert cfg.model == "grok-code-fast"
    assert cfg.llm_endpoint == "https://custom.x.ai"
    assert cfg.api_key_env == "XAI_SECRET"
    assert cfg.max_commit_length == 80


def test_save_and_load_roundtrip(tmp_path, monkeypatch):
    config = Config(
        provider="github",
        model="openai/gpt-4.1-mini",
        llm_endpoint=DEFAULT_MODELS["github"]["endpoint"],
        api_key_env="GITHUB_TOKEN",
        git_repo_path=str(tmp_path),
        max_commit_length=88,
    )
    save_config(config, tmp_path)
    monkeypatch.delenv("KLINGON_CMT_GIT_REPO_PATH", raising=False)
    monkeypatch.delenv("KLINGON_CMT_MAX_COMMIT_LENGTH", raising=False)
    loaded = load_config(repo_root=tmp_path)

    assert loaded.provider == "github"
    assert loaded.max_commit_length == 88
    assert loaded.git_repo_path == str(tmp_path)


def test_load_config_upgrades_relative_repo_path(tmp_path):
    config_dir = tmp_path / ".kcmt"
    config_dir.mkdir()
    legacy = {
        "provider": "openai",
        "model": DEFAULT_MODELS["openai"]["model"],
        "llm_endpoint": DEFAULT_MODELS["openai"]["endpoint"],
        "api_key_env": DEFAULT_MODELS["openai"]["api_key_env"],
        "git_repo_path": ".",
        "max_commit_length": 72,
        "allow_fallback": False,
        "auto_push": False,
    }
    (config_dir / "config.json").write_text(json.dumps(legacy))

    loaded = load_persisted_config(tmp_path)

    assert loaded is not None
    assert loaded.git_repo_path == str(tmp_path)


def test_detect_available_providers_fuzzy(monkeypatch):
    monkeypatch.setenv("XAI_GROK_TOKEN", "xai")
    monkeypatch.setenv("OPENAI_ALT_KEY", "openai")

    detected = detect_available_providers(os.environ)

    assert any("XAI" in var for var in detected["xai"])
    assert any("OPENAI" in var for var in detected["openai"])


def test_active_config_helpers_roundtrip():
    cfg = Config(
        provider="openai",
        model="gpt-5-mini-2025-08-07",
        llm_endpoint="https://api.openai.com/v1",
        api_key_env="OPENAI_API_KEY",
        git_repo_path=".",
        max_commit_length=77,
    )
    set_active_config(cfg)
    assert get_active_config() is cfg
    clear_active_config()
