"""Configuration management for kcmt."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

CONFIG_DIR_NAME = ".kcmt"
CONFIG_FILE_NAME = "config.json"

DEFAULT_MODELS = {
    "openai": {
        "model": "gpt-5-mini",
        "endpoint": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
    },
    "anthropic": {
        "model": "claude-3-5-haiku-latest",
        "endpoint": "https://api.anthropic.com",
        "api_key_env": "ANTHROPIC_API_KEY",
    },
    "xai": {
        "model": "grok-code-fast",
        "endpoint": "https://api.x.ai/v1",
        "api_key_env": "XAI_API_KEY",
    },
    "github": {
        "model": "openai/gpt-4.1-mini",
        "endpoint": "https://models.github.ai/inference",
        "api_key_env": "GITHUB_TOKEN",
    },
}

_FUZZY_ENV_HINTS = {
    "openai": ["OPENAI", "OPENAI_API", "OA_KEY"],
    "anthropic": ["ANTHROPIC", "CLAUDE"],
    "xai": ["XAI", "GROK"],
    "github": ["GITHUB_TOKEN", "GH_TOKEN", "GH_MODELS"],
}


@dataclass
class Config:
    """Runtime configuration for kcmt."""

    provider: str
    model: str
    llm_endpoint: str
    api_key_env: str
    git_repo_path: str = "."
    max_commit_length: int = 72

    def resolve_api_key(self) -> Optional[str]:
        """Return the API key from the configured environment variable."""
        return os.environ.get(self.api_key_env)

    def to_dict(self) -> Dict[str, str]:
        """Serialise configuration to a dict for persistence."""
        return asdict(self)


_ACTIVE_CONFIG: Optional[Config] = None


def _config_dir(repo_root: Optional[Path] = None) -> Path:
    return (repo_root or Path.cwd()) / CONFIG_DIR_NAME


def _config_file(repo_root: Optional[Path] = None) -> Path:
    return _config_dir(repo_root) / CONFIG_FILE_NAME


def save_config(config: Config, repo_root: Optional[Path] = None) -> None:
    """Persist configuration JSON within the repository."""
    cfg_path = _config_file(repo_root)
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(json.dumps(config.to_dict(), indent=2))


def load_persisted_config(repo_root: Optional[Path] = None) -> Optional[Config]:
    cfg_path = _config_file(repo_root)
    if not cfg_path.exists():
        return None
    data = json.loads(cfg_path.read_text())
    return Config(**data)


def detect_available_providers(env: Optional[Dict[str, str]] = None) -> Dict[str, List[str]]:
    """Return mapping of provider -> matching env vars found."""
    env = env or os.environ
    detected: Dict[str, List[str]] = {p: [] for p in DEFAULT_MODELS}

    for provider, defaults in DEFAULT_MODELS.items():
        key_name = defaults["api_key_env"]
        if key_name in env:
            detected[provider].append(key_name)

        hints = _FUZZY_ENV_HINTS.get(provider, [])
        for env_key in env:
            if env_key in detected[provider]:
                continue
            for hint in hints:
                if hint.lower() in env_key.lower():
                    detected[provider].append(env_key)
                    break

    return detected


def load_config(
    *,
    repo_root: Optional[Path] = None,
    overrides: Optional[Dict[str, str]] = None,
) -> Config:
    """Build configuration from config file, environment and overrides."""

    overrides = overrides or {}
    repo_root = repo_root or Path.cwd()
    persisted = load_persisted_config(repo_root)

    provider = (
        overrides.get("provider")
        or (persisted.provider if persisted else None)
        or _auto_select_provider()
    )
    if provider not in DEFAULT_MODELS:
        provider = "openai"

    defaults = DEFAULT_MODELS[provider]

    model = overrides.get("model") or (persisted.model if persisted else None) or os.environ.get(
        "KLINGON_CMT_LLM_MODEL"
    ) or defaults["model"]

    endpoint = (
        overrides.get("endpoint")
        or (persisted.llm_endpoint if persisted else None)
        or os.environ.get("KLINGON_CMT_LLM_ENDPOINT")
        or defaults["endpoint"]
    )

    # If provider is overridden and differs from persisted configuration,
    # do NOT reuse the old provider's api_key_env (e.g. ANTHROPIC_API_KEY when switching to openai).
    persisted_api_key_env = None
    if persisted and persisted.provider == provider:
        persisted_api_key_env = persisted.api_key_env

    api_key_env = (
        overrides.get("api_key_env")
        or persisted_api_key_env
        or _select_env_var_for_provider(provider)
    )

    git_repo_path = overrides.get("repo_path") or os.environ.get(
        "KLINGON_CMT_GIT_REPO_PATH"
    ) or (persisted.git_repo_path if persisted else ".")

    max_commit_length = int(
        overrides.get("max_commit_length")
        or os.environ.get("KLINGON_CMT_MAX_COMMIT_LENGTH")
        or (persisted.max_commit_length if persisted else 72)
    )

    config = Config(
        provider=provider,
        model=model,
        llm_endpoint=endpoint,
        api_key_env=api_key_env or DEFAULT_MODELS[provider]["api_key_env"],
        git_repo_path=git_repo_path,
        max_commit_length=max_commit_length,
    )

    set_active_config(config)
    return config


def _select_env_var_for_provider(provider: str) -> Optional[str]:
    defaults = DEFAULT_MODELS[provider]["api_key_env"]
    env_matches = detect_available_providers().get(provider, [])
    if defaults in env_matches:
        return defaults
    return env_matches[0] if env_matches else defaults


def _auto_select_provider() -> str:
    detected = detect_available_providers()
    for provider in ("openai", "anthropic", "xai", "github"):
        if detected.get(provider):
            return provider
    return "openai"


def set_active_config(config: Config) -> None:
    global _ACTIVE_CONFIG
    _ACTIVE_CONFIG = config


def get_active_config() -> Config:
    global _ACTIVE_CONFIG
    if _ACTIVE_CONFIG is None:
        return load_config()
    return _ACTIVE_CONFIG


def clear_active_config() -> None:
    global _ACTIVE_CONFIG
    _ACTIVE_CONFIG = None


def describe_provider(provider: str) -> str:
    meta = DEFAULT_MODELS.get(provider)
    if not meta:
        return provider
    return f"{provider} (default model: {meta['model']})"
