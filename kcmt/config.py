"""Configuration management for kcmt."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

CONFIG_DIR_NAME = ".kcmt"
CONFIG_FILE_NAME = "config.json"

DEFAULT_MODELS = {
    "openai": {
        "model": "gpt-5-mini-2025-08-07",
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

# Friendly display names for supported providers
PROVIDER_DISPLAY_NAMES: dict[str, str] = {
    "openai": "OpenAI",
    "anthropic": "Anthropic",
    "xai": "X.AI",
    "github": "GitHub Models",
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
    # Optional secondary provider to allow future fallback/experimentation.
    # Not currently used by the workflow, but persisted for future use and
    # exposed via the configuration wizard.
    secondary_provider: str | None = None
    secondary_model: str | None = None
    secondary_llm_endpoint: str | None = None
    secondary_api_key_env: str | None = None
    git_repo_path: str = "."
    max_commit_length: int = 72
    auto_push: bool = True
    # Optional per-provider API key env var mapping to aid tooling like
    # benchmarking and future multi-provider flows. Keys are provider ids
    # (e.g. "openai", "anthropic", "xai").
    provider_env_overrides: dict[str, str] = field(default_factory=dict)
    # Per-provider settings persisted in the config file.
    # Example shape:
    #   {
    #     "openai": {"name": "OpenAI", "endpoint": "https://...", "api_key_env": "OPENAI_API_KEY", "preferred_model": "gpt-4o-mini"},
    #     "anthropic": {...},
    #   }
    providers: dict[str, dict[str, Any]] = field(default_factory=dict)

    def resolve_api_key(self) -> Optional[str]:
        """Return the API key from the configured environment variable."""
        return os.environ.get(self.api_key_env)

    def to_dict(self) -> Dict[str, str]:
        """Serialise configuration to a dict for persistence."""
        return asdict(self)


_CONFIG_STATE: Dict[str, Optional[Config]] = {"active": None}


def _ensure_path(path_like: Optional[Path]) -> Path:
    if path_like is None:
        return Path.cwd().resolve(strict=False)
    return Path(path_like).expanduser().resolve(strict=False)


def _config_dir(repo_root: Optional[Path] = None) -> Path:
    return _ensure_path(repo_root) / CONFIG_DIR_NAME


def _config_file(repo_root: Optional[Path] = None) -> Path:
    return _config_dir(repo_root) / CONFIG_FILE_NAME


def save_config(config: Config, repo_root: Optional[Path] = None) -> None:
    """Persist configuration JSON within the repository."""
    cfg_path = _config_file(repo_root)
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    data = config.to_dict()
    base_root = _ensure_path(repo_root)
    git_path = data.get("git_repo_path")
    if git_path:
        candidate = Path(git_path).expanduser()
        if candidate.is_absolute():
            normalised = candidate.resolve(strict=False)
        else:
            normalised = (base_root / candidate).resolve(strict=False)
    else:
        normalised = base_root
    data["git_repo_path"] = str(normalised)
    config.git_repo_path = data["git_repo_path"]
    cfg_path.write_text(json.dumps(data, indent=2))


def load_persisted_config(
    repo_root: Optional[Path] = None,
) -> Optional[Config]:
    cfg_path = _config_file(repo_root)
    if not cfg_path.exists():
        return None
    data = json.loads(cfg_path.read_text())
    data.pop("allow_fallback", None)
    if "auto_push" not in data:  # backward compat; now default is True
        data["auto_push"] = True
    # Backward compat: ensure providers map exists in persisted data
    if "providers" not in data or not isinstance(data.get("providers"), dict):
        data["providers"] = {}
    resolved_root = _ensure_path(repo_root) if repo_root else cfg_path.parent.parent
    resolved_root = _ensure_path(resolved_root)
    git_path = data.get("git_repo_path")
    if git_path:
        candidate = Path(git_path).expanduser()
        if candidate.is_absolute():
            candidate = candidate.resolve(strict=False)
        else:
            candidate = (resolved_root / candidate).resolve(strict=False)
        data["git_repo_path"] = str(candidate)
    else:
        data["git_repo_path"] = str(resolved_root)
    return Config(**data)


def detect_available_providers(
    env: Optional[Dict[str, str]] = None,
) -> Dict[str, List[str]]:
    """Return mapping of provider -> matching env vars found."""
    env_dict: Dict[str, str] = dict(env or os.environ)
    detected: Dict[str, List[str]] = {p: [] for p in DEFAULT_MODELS}
    for provider, defaults in DEFAULT_MODELS.items():
        key_name = defaults["api_key_env"]
        if key_name in env_dict:
            detected[provider].append(key_name)
        hints = _FUZZY_ENV_HINTS.get(provider, [])
        for env_key in env_dict:
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
    overrides.pop("allow_fallback", None)
    repo_root = _ensure_path(repo_root)
    persisted = load_persisted_config(repo_root)
    detected = detect_available_providers()

    # Allow lightweight environment overrides without touching the persisted config.
    provider_from_env = os.environ.get("KCMT_PROVIDER")
    provider_override = overrides.get("provider") or provider_from_env
    if persisted and not provider_override:
        provider = persisted.provider
    else:
        provider = provider_override or _auto_select_provider(detected)
    if provider not in DEFAULT_MODELS:
        provider = "openai"

    defaults = DEFAULT_MODELS[provider]

    # Build or upgrade the per-provider settings map. Persisted configs
    # might not have this yet, so we synthesise sensible defaults.
    providers_map: dict[str, dict[str, Any]] = {}
    if persisted and isinstance(getattr(persisted, "providers", {}), dict):
        # Shallow copy to avoid mutating the persisted object
        providers_map = dict(getattr(persisted, "providers", {}) or {})
    # Ensure all known providers are present with defaults
    for prov, meta in DEFAULT_MODELS.items():
        entry = providers_map.get(prov, {}) or {}
        if not entry.get("name"):
            entry["name"] = PROVIDER_DISPLAY_NAMES.get(prov, prov)
        if not entry.get("endpoint"):
            # If the previously persisted (legacy) top-level endpoint was for
            # this provider, carry it over; otherwise use provider default.
            if (
                persisted
                and getattr(persisted, "provider", None) == prov
                and getattr(persisted, "llm_endpoint", None)
            ):
                entry["endpoint"] = persisted.llm_endpoint
            else:
                entry["endpoint"] = meta["endpoint"]
        # Prefer any existing override for env var, fall back to defaults
        # or provider_env_overrides mapping (legacy field)
        if not entry.get("api_key_env"):
            legacy_map = (
                getattr(persisted, "provider_env_overrides", {}) if persisted else {}
            ) or {}
            # If the previously persisted (legacy) top-level api_key_env was for
            # this provider, carry it over; otherwise use overrides/defaults.
            if (
                persisted
                and getattr(persisted, "provider", None) == prov
                and getattr(persisted, "api_key_env", None)
            ):
                entry["api_key_env"] = getattr(persisted, "api_key_env")
            else:
                entry["api_key_env"] = legacy_map.get(prov) or meta["api_key_env"]
        # Carry over any previously selected model for this provider via
        # (1) explicit provider entry, (2) legacy top-level model when the
        # active provider matches, or (3) leave unset to indicate "first use".
        if "preferred_model" not in entry or entry["preferred_model"] is None:
            if persisted and getattr(persisted, "provider", None) == prov:
                # Use previous top-level model as the preferred one for this provider
                entry["preferred_model"] = getattr(persisted, "model", None)
        providers_map[prov] = entry

    # Only reuse persisted model if it's for the same provider; otherwise
    # fall back to defaults for the newly selected provider.
    # Prefer any persisted per-provider preferred model first
    provider_pref_model = None
    if providers_map.get(provider):
        provider_pref_model = providers_map[provider].get("preferred_model")
    persisted_model = (
        persisted.model if (persisted and persisted.provider == provider) else None
    )
    model = (
        overrides.get("model")
        or provider_pref_model
        or persisted_model
        or os.environ.get("KLINGON_CMT_LLM_MODEL")
        or defaults["model"]
    )

    # Backward compatibility / migration: transparently upgrade the old
    # short OpenAI model alias to the new dated default if user has not
    # explicitly overridden it this run (so persisted configs keep working).
    if provider == "openai" and model == "gpt-5-mini":
        model = defaults["model"]

    # Only reuse persisted endpoint if it's for the same provider; otherwise
    # select from environment or provider defaults.
    persisted_endpoint = (
        persisted.llm_endpoint
        if (persisted and persisted.provider == provider)
        else None
    )
    provider_endpoint = (
        providers_map.get(provider, {}).get("endpoint") if providers_map else None
    )
    # Endpoint precedence: overrides > env > per-provider map > persisted > default
    endpoint = (
        overrides.get("endpoint")
        or os.environ.get("KLINGON_CMT_LLM_ENDPOINT")
        or provider_endpoint
        or persisted_endpoint
        or defaults["endpoint"]
    )

    # If provider changed, do not reuse previous provider's api_key_env.
    persisted_api_key_env = None
    if persisted and persisted.provider == provider:
        persisted_api_key_env = persisted.api_key_env

    # If user previously configured explicit env var for this provider, prefer it
    mapped_api_key_env = None
    if providers_map.get(provider):
        mapped_api_key_env = providers_map[provider].get("api_key_env")
    elif persisted and getattr(persisted, "provider_env_overrides", None):
        mapped_api_key_env = persisted.provider_env_overrides.get(provider)

    api_key_env = (
        overrides.get("api_key_env")
        or persisted_api_key_env
        or mapped_api_key_env
        or _select_env_var_for_provider(provider)
    )

    git_repo_path_raw = (
        overrides.get("repo_path")
        or os.environ.get("KLINGON_CMT_GIT_REPO_PATH")
        or (persisted.git_repo_path if persisted else str(repo_root))
    )

    git_repo_candidate = Path(git_repo_path_raw).expanduser()
    if git_repo_candidate.is_absolute():
        git_repo_candidate = git_repo_candidate.resolve(strict=False)
    else:
        git_repo_candidate = (repo_root / git_repo_candidate).resolve(strict=False)
    git_repo_path = str(git_repo_candidate)

    max_commit_length = int(
        overrides.get("max_commit_length")
        or os.environ.get("KLINGON_CMT_MAX_COMMIT_LENGTH")
        or (persisted.max_commit_length if persisted else 72)
    )

    auto_push_env = os.environ.get("KLINGON_CMT_AUTO_PUSH")
    if overrides.get("auto_push") is not None:
        auto_push = str(overrides["auto_push"]).lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
    elif persisted is not None and hasattr(persisted, "auto_push"):
        auto_push = bool(getattr(persisted, "auto_push"))
    elif auto_push_env:
        auto_push = auto_push_env.lower() in {"1", "true", "yes", "on"}
    else:
        auto_push = True

    # Keep provider_env_overrides in sync (legacy field) for compatibility
    provider_env_overrides: dict[str, str] = {}
    for prov, entry in providers_map.items():
        env_name = str(entry.get("api_key_env") or DEFAULT_MODELS[prov]["api_key_env"])
        provider_env_overrides[prov] = env_name

    config = Config(
        provider=provider,
        model=model,
        llm_endpoint=endpoint,
        api_key_env=api_key_env or DEFAULT_MODELS[provider]["api_key_env"],
        git_repo_path=git_repo_path,
        max_commit_length=max_commit_length,
        auto_push=bool(auto_push),
        providers=providers_map,
        provider_env_overrides=provider_env_overrides,
    )

    set_active_config(config)
    return config


def _select_env_var_for_provider(provider: str) -> Optional[str]:
    defaults = DEFAULT_MODELS[provider]["api_key_env"]
    env_matches = detect_available_providers().get(provider, [])
    if defaults in env_matches:
        return defaults
    return env_matches[0] if env_matches else defaults


def _auto_select_provider(
    detected: Optional[Dict[str, List[str]]] = None,
) -> str:
    if detected is None:
        detected = detect_available_providers()
    for provider in ("openai", "anthropic", "xai", "github"):
        if detected.get(provider):
            return provider
    return "openai"


def set_active_config(config: Config) -> None:
    _CONFIG_STATE["active"] = config


def get_active_config() -> Config:
    active = _CONFIG_STATE.get("active")
    if active is None:
        return load_config()
    return active


def clear_active_config() -> None:
    _CONFIG_STATE["active"] = None


def describe_provider(provider: str) -> str:
    meta = DEFAULT_MODELS.get(provider)
    if not meta:
        return provider
    return f"{provider} (default model: {meta['model']})"
