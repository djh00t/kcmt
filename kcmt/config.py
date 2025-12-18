"""Configuration management for kcmt."""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Optional

CONFIG_DIR_NAME = "kcmt"
CONFIG_FILE_NAME = "config.json"
PREFERENCES_FILE_NAME = "preferences.json"

# Default batch timeout (seconds). The Batch API is asynchronous and may take
# longer than typical completion calls; we cap waits at five minutes by default.
DEFAULT_BATCH_TIMEOUT_SECONDS = 900
BATCH_TIMEOUT_MIN_SECONDS = 900

DEFAULT_MODELS = {
    "openai": {
        "model": "gpt-5-mini-2025-08-07",
        "endpoint": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
    },
    "anthropic": {
        "model": "claude-3-5-haiku-latest",
        "endpoint": "https://api.anthropic.com/v1",
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
    git_repo_path: str = "."
    max_commit_length: int = 72
    auto_push: bool = True
    # Per-provider settings persisted in the config file.
    # Example shape:
    #   {
    #     "openai": {"name": "OpenAI", "endpoint": "https://...", "api_key_env": "OPENAI_API_KEY", "preferred_model": "gpt-4o-mini"},
    #     "anthropic": {...},
    #   }
    providers: dict[str, dict[str, Any]] = field(default_factory=dict)
    # Ordered list (priority ascending) of preferred provider/model pairs.
    model_priority: list[dict[str, str]] = field(default_factory=list)
    # Optional OpenAI batch settings
    use_batch: bool = False
    batch_model: Optional[str] = None
    batch_timeout_seconds: int = DEFAULT_BATCH_TIMEOUT_SECONDS

    def resolve_api_key(self) -> Optional[str]:
        """Return the API key from the configured environment variable."""
        return os.environ.get(self.api_key_env)

    def to_dict(self) -> Dict[str, str]:
        """Serialise configuration to a dict for persistence."""
        return asdict(self)


_CONFIG_STATE: Dict[str, Optional[Config]] = {"active": None}


def _safe_load_json(path: Path) -> Optional[Any]:
    """Load JSON from ``path`` tolerating encoding issues.

    Falls back to replacement decoding when utf-8 decoding fails so a single
    bad byte does not crash the CLI.
    """

    try:
        raw = path.read_bytes()
    except OSError:
        return None

    try:
        return json.loads(raw.decode("utf-8"))
    except UnicodeDecodeError:
        pass
    except json.JSONDecodeError:
        return None

    try:
        return json.loads(raw.decode("utf-8", errors="replace"))
    except json.JSONDecodeError:
        return None


def _ensure_path(path_like: Optional[Path]) -> Path:
    if path_like is None:
        return Path.cwd().resolve(strict=False)
    return Path(path_like).expanduser().resolve(strict=False)


def _config_home() -> Path:
    """Return the base config directory independent of repo root."""
    env_home = os.environ.get("KCMT_CONFIG_HOME")
    if env_home:
        return Path(env_home).expanduser().resolve(strict=False)
    xdg_home = os.environ.get("XDG_CONFIG_HOME")
    base = Path(xdg_home).expanduser() if xdg_home else Path.home() / ".config"
    return (base / CONFIG_DIR_NAME).resolve(strict=False)


def _repo_namespace(repo_root: Optional[Path]) -> str:
    """Stable namespace for per-repo cached artifacts (benchmarks/snapshots)."""
    root = _ensure_path(repo_root)
    digest = hashlib.sha256(str(root).encode("utf-8", "ignore")).hexdigest()[:8]
    tail = root.name or "repo"
    safe_tail = re.sub(r"[^a-zA-Z0-9_.-]", "-", tail) or "repo"
    return f"{safe_tail}-{digest}"


def _config_dir(_repo_root: Optional[Path] = None) -> Path:
    # Config is global; repo_root is ignored for location but retained in API.
    return _config_home()


def _config_file(repo_root: Optional[Path] = None) -> Path:
    return _config_dir(repo_root) / CONFIG_FILE_NAME


def _preferences_file(repo_root: Optional[Path] = None) -> Path:
    return _config_dir(repo_root) / PREFERENCES_FILE_NAME


def config_dir(repo_root: Optional[Path] = None) -> Path:
    """Expose the config directory path (global)."""
    return _config_dir(repo_root)


def config_file_path(repo_root: Optional[Path] = None) -> Path:
    """Expose the config.json path (global)."""
    return _config_file(repo_root)


def preferences_file_path(repo_root: Optional[Path] = None) -> Path:
    """Expose the preferences.json path (global)."""
    return _preferences_file(repo_root)


def state_dir(repo_root: Optional[Path] = None) -> Path:
    """Directory for repo-scoped cached artifacts (benchmarks, snapshots)."""
    base = _config_home() / "repos" / _repo_namespace(repo_root)
    return base.resolve(strict=False)


def save_config(config: Config, repo_root: Optional[Path] = None) -> None:
    """Persist configuration JSON to the global config directory."""
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
    data = _safe_load_json(cfg_path)
    if not isinstance(data, dict):
        return None
    data.pop("allow_fallback", None)
    if "auto_push" not in data:  # backward compat; now default is True
        data["auto_push"] = True
    legacy_overrides = data.pop("provider_env_overrides", {}) or {}
    if legacy_overrides:
        providers_block = data.setdefault("providers", {})
        if isinstance(providers_block, dict):
            for prov, env_name in legacy_overrides.items():
                entry = providers_block.setdefault(prov, {})
                entry.setdefault("api_key_env", env_name)
    # Backward compat: ensure providers map exists in persisted data
    if "providers" not in data or not isinstance(data.get("providers"), dict):
        data["providers"] = {}
    for legacy_key in [
        "secondary_provider",
        "secondary_model",
        "secondary_llm_endpoint",
        "secondary_api_key_env",
    ]:
        data.pop(legacy_key, None)
    if "model_priority" in data and not isinstance(data["model_priority"], list):
        data.pop("model_priority", None)
    resolved_root = _ensure_path(repo_root) if repo_root else Path.cwd()
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
    allowed_keys = {f.name for f in fields(Config)}
    filtered = {key: value for key, value in data.items() if key in allowed_keys}
    return Config(**filtered)


def load_preferences(repo_root: Optional[Path] = None) -> Dict[str, Any]:
    """Load CLI preferences persisted for the repository."""

    pref_path = _preferences_file(repo_root)
    if not pref_path.exists():
        return {}
    data = _safe_load_json(pref_path)
    if isinstance(data, dict):
        return data
    return {}


def save_preferences(
    preferences: Dict[str, Any], repo_root: Optional[Path] = None
) -> None:
    """Persist CLI preference state for the repository."""

    pref_path = _preferences_file(repo_root)
    pref_path.parent.mkdir(parents=True, exist_ok=True)
    pref_path.write_text(json.dumps(preferences, indent=2, sort_keys=True))


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

    # Build or upgrade the per-provider settings map. Persisted configs
    # might not have this yet, so we synthesise sensible defaults.
    providers_map: dict[str, dict[str, Any]] = {}
    if persisted and isinstance(getattr(persisted, "providers", {}), dict):
        providers_map = {
            prov: dict(entry or {}) for prov, entry in persisted.providers.items()
        }
    for prov, meta in DEFAULT_MODELS.items():
        entry = providers_map.get(prov, {}) or {}
        entry.setdefault("name", PROVIDER_DISPLAY_NAMES.get(prov, prov))
        entry.setdefault("endpoint", meta["endpoint"])
        entry.setdefault("api_key_env", meta["api_key_env"])
        if "preferred_model" not in entry:
            entry["preferred_model"] = None
        providers_map[prov] = entry

    # Build priority list from persisted data (if any), falling back
    # to legacy top-level provider/model and finally defaults.
    priority_list: list[dict[str, str]] = []
    if persisted and isinstance(getattr(persisted, "model_priority", None), list):
        for item in persisted.model_priority:
            if not isinstance(item, dict):
                continue
            prov = str(item.get("provider", "")).strip()
            model_name = str(item.get("model", "")).strip()
            if prov in DEFAULT_MODELS and model_name:
                priority_list.append({"provider": prov, "model": model_name})

    if not priority_list and persisted and getattr(persisted, "provider", None):
        prov = str(persisted.provider)
        model_name = str(getattr(persisted, "model", ""))
        if prov in DEFAULT_MODELS and model_name:
            priority_list.append({"provider": prov, "model": model_name})

    if not priority_list:
        auto_provider = _auto_select_provider(detected)
        default_model = DEFAULT_MODELS[auto_provider]["model"]
        priority_list.append({"provider": auto_provider, "model": default_model})

    # Apply provider/model overrides (CLI/env).
    provider_from_env = os.environ.get("KCMT_PROVIDER")
    provider_override = overrides.get("provider") or provider_from_env
    if provider_override and provider_override in DEFAULT_MODELS:
        model_override = (
            overrides.get("model")
            or os.environ.get("KLINGON_CMT_LLM_MODEL")
            or DEFAULT_MODELS[provider_override]["model"]
        )
        priority_list.insert(
            0, {"provider": provider_override, "model": model_override}
        )

    elif overrides.get("model") or os.environ.get("KLINGON_CMT_LLM_MODEL"):
        model_override = overrides.get("model") or os.environ.get(
            "KLINGON_CMT_LLM_MODEL"
        )  # type: ignore[assignment]
        if priority_list:
            priority_list[0] = {
                "provider": priority_list[0]["provider"],
                "model": str(model_override),
            }

    # Deduplicate while preserving order and cap to five entries.
    seen_pairs: set[tuple[str, str]] = set()
    deduped: list[dict[str, str]] = []
    for pref in priority_list:
        prov = pref.get("provider")  # type: ignore[assignment]
        model_name = pref.get("model")  # type: ignore[assignment]
        if prov not in DEFAULT_MODELS or not model_name:
            continue
        pair = (prov, model_name)
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        deduped.append({"provider": prov, "model": model_name})
        if len(deduped) >= 5:
            break
    priority_list = deduped or [
        {"provider": "openai", "model": DEFAULT_MODELS["openai"]["model"]}
    ]

    provider = priority_list[0]["provider"]
    model = priority_list[0]["model"]
    if provider not in DEFAULT_MODELS:
        provider = "openai"
        model = DEFAULT_MODELS[provider]["model"]
        priority_list[0] = {"provider": provider, "model": model}

    defaults = DEFAULT_MODELS[provider]
    if provider == "openai" and model == "gpt-5-mini":
        model = defaults["model"]
        priority_list[0]["model"] = model

    endpoint_override = overrides.get("endpoint") or os.environ.get(
        "KLINGON_CMT_LLM_ENDPOINT"
    )
    provider_entry = providers_map.get(provider, {})
    endpoint = (
        endpoint_override or provider_entry.get("endpoint") or defaults["endpoint"]
    )
    provider_entry["endpoint"] = endpoint

    api_override = overrides.get("api_key_env")
    if api_override:
        provider_entry["api_key_env"] = api_override
    api_key_env = provider_entry.get("api_key_env") or _select_env_var_for_provider(
        provider
    )
    # Sanity: if persisted value looks like a URL, reset to default env var
    if isinstance(api_key_env, str) and (
        "://" in api_key_env or api_key_env.startswith("http")
    ):
        api_key_env = DEFAULT_MODELS[provider]["api_key_env"]
    provider_entry["api_key_env"] = api_key_env

    # Sync preferred models per provider based on priority list
    first_by_provider: dict[str, str] = {}
    for pref in priority_list:
        prov = pref["provider"]
        first_by_provider.setdefault(prov, pref["model"])
    for prov, entry in providers_map.items():
        entry["preferred_model"] = (
            first_by_provider.get(prov)
            or entry.get("preferred_model")
            or DEFAULT_MODELS[prov]["model"]
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

    batch_env = os.environ.get("KCMT_USE_BATCH")
    batch_override = overrides.get("use_batch")
    if batch_override is not None:
        use_batch = str(batch_override).lower() in {"1", "true", "yes", "on"}
    elif persisted is not None and hasattr(persisted, "use_batch"):
        use_batch = bool(getattr(persisted, "use_batch"))
    elif batch_env:
        use_batch = batch_env.lower() in {"1", "true", "yes", "on"}
    else:
        use_batch = False
    if provider != "openai":
        use_batch = False

    default_batch_model = (
        providers_map.get("openai", {}).get("preferred_model")
        or DEFAULT_MODELS["openai"]["model"]
    )
    batch_model_override = overrides.get("batch_model") or os.environ.get(
        "KCMT_BATCH_MODEL"
    )
    batch_model = (
        batch_model_override
        or (getattr(persisted, "batch_model", None) if persisted else None)
        or default_batch_model
    )
    batch_timeout_raw = (
        overrides.get("batch_timeout_seconds")
        or overrides.get("batch_timeout")
        or os.environ.get("KCMT_BATCH_TIMEOUT")
        or (
            getattr(persisted, "batch_timeout_seconds", None)
            if persisted
            else DEFAULT_BATCH_TIMEOUT_SECONDS
        )
        or DEFAULT_BATCH_TIMEOUT_SECONDS
    )
    try:
        batch_timeout_seconds = int(float(batch_timeout_raw))
    except (TypeError, ValueError):
        batch_timeout_seconds = DEFAULT_BATCH_TIMEOUT_SECONDS
    batch_timeout_seconds = max(BATCH_TIMEOUT_MIN_SECONDS, batch_timeout_seconds)

    config = Config(
        provider=provider,
        model=model,
        llm_endpoint=endpoint,
        api_key_env=api_key_env or DEFAULT_MODELS[provider]["api_key_env"],
        git_repo_path=git_repo_path,
        max_commit_length=max_commit_length,
        auto_push=bool(auto_push),
        providers=providers_map,
        model_priority=priority_list,
        use_batch=use_batch,
        batch_model=str(batch_model) if batch_model is not None else None,
        batch_timeout_seconds=batch_timeout_seconds,
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
