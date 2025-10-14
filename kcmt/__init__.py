"""kcmt - AI-powered atomic Git staging and committing tool."""

from importlib import import_module
from typing import TYPE_CHECKING

__version__ = "0.2.0"

# Public API (lazy-exported to avoid import-time side effects)
__all__ = [
    # Config
    "Config",
    "load_config",
    "get_active_config",
    "set_active_config",
    "save_config",
    # LLM
    "LLMClient",
    # Git
    "GitRepo",
    # Commit generation
    "CommitGenerator",
    # Core workflow
    "KlingonCMTWorkflow",
    "FileChange",
    "CommitResult",
    # Exceptions
    "KlingonCMTError",
    "GitError",
    "LLMError",
    "ConfigError",
    "ValidationError",
]


def __getattr__(name: str) -> object:
    """Lazy attribute loader to avoid importing heavy modules at package import time.

    This prevents environment-dependent modules (e.g., those that read env in
    kcmt.config) from being imported unless explicitly accessed.
    """
    mapping = {
        # Config
        "Config": ("kcmt.config", "Config"),
        "load_config": ("kcmt.config", "load_config"),
        "get_active_config": ("kcmt.config", "get_active_config"),
        "set_active_config": ("kcmt.config", "set_active_config"),
        "save_config": ("kcmt.config", "save_config"),
        # LLM
        "LLMClient": ("kcmt.llm", "LLMClient"),
        # Git
        "GitRepo": ("kcmt.git", "GitRepo"),
        # Commit generation
        "CommitGenerator": ("kcmt.commit", "CommitGenerator"),
        # Core workflow
        "KlingonCMTWorkflow": ("kcmt.core", "KlingonCMTWorkflow"),
        "FileChange": ("kcmt.core", "FileChange"),
        "CommitResult": ("kcmt.core", "CommitResult"),
        # Exceptions
        "KlingonCMTError": ("kcmt.exceptions", "KlingonCMTError"),
        "GitError": ("kcmt.exceptions", "GitError"),
        "LLMError": ("kcmt.exceptions", "LLMError"),
        "ConfigError": ("kcmt.exceptions", "ConfigError"),
        "ValidationError": ("kcmt.exceptions", "ValidationError"),
    }
    if name in mapping:
        mod_name, attr = mapping[name]
        mod = import_module(mod_name)
        value = getattr(mod, attr)
        globals()[name] = value  # cache for future access
        return value
    raise AttributeError(f"module 'kcmt' has no attribute {name!r}")


if TYPE_CHECKING:
    # For type checkers and IDEs, provide direct imports
    from .commit import CommitGenerator
    from .config import (
        Config,
        get_active_config,
        load_config,
        save_config,
        set_active_config,
    )
    from .core import CommitResult, FileChange, KlingonCMTWorkflow
    from .exceptions import (
        ConfigError,
        GitError,
        KlingonCMTError,
        LLMError,
        ValidationError,
    )
    from .git import GitRepo
    from .llm import LLMClient
