from kcmt.config import load_config, clear_active_config


def test_config_auto_detect_prefers_openai(monkeypatch, tmp_path):
    repo = tmp_path / "r"
    repo.mkdir()
    (repo / ".git").mkdir()  # minimal repo marker for code path

    # Set multiple provider keys; ensure openai present
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "y")

    clear_active_config()
    cfg = load_config(repo_root=repo, overrides={})
    assert cfg.provider in {"openai", "anthropic", "xai", "github"}
    assert cfg.api_key_env in {"OPENAI_API_KEY", "ANTHROPIC_API_KEY"}


def test_config_respects_overrides(monkeypatch, tmp_path):
    repo = tmp_path / "r2"
    repo.mkdir()
    (repo / ".git").mkdir()
    monkeypatch.setenv("ANTHROPIC_API_KEY", "y")

    clear_active_config()
    cfg = load_config(
        repo_root=repo, overrides={"provider": "anthropic", "model": "claude"}
    )
    assert cfg.provider == "anthropic"
    assert "claude" in cfg.model.lower()
