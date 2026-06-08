"""kcmt-python - AI-powered atomic Git staging and committing tool."""

from importlib import import_module
from typing import TYPE_CHECKING

__version__ = "0.3.2"

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
    kcmt_python.config) from being imported unless explicitly accessed.
    """
    mapping = {
        # Config
        "Config": ("kcmt_python.config", "Config"),
        "load_config": ("kcmt_python.config", "load_config"),
        "get_active_config": ("kcmt_python.config", "get_active_config"),
        "set_active_config": ("kcmt_python.config", "set_active_config"),
        "save_config": ("kcmt_python.config", "save_config"),
        # LLM
        "LLMClient": ("kcmt_python.llm", "LLMClient"),
        # Git
        "GitRepo": ("kcmt_python.git", "GitRepo"),
        # Commit generation
        "CommitGenerator": ("kcmt_python.commit", "CommitGenerator"),
        # Core workflow
        "KlingonCMTWorkflow": ("kcmt_python.core", "KlingonCMTWorkflow"),
        "FileChange": ("kcmt_python.core", "FileChange"),
        "CommitResult": ("kcmt_python.core", "CommitResult"),
        # Exceptions
        "KlingonCMTError": ("kcmt_python.exceptions", "KlingonCMTError"),
        "GitError": ("kcmt_python.exceptions", "GitError"),
        "LLMError": ("kcmt_python.exceptions", "LLMError"),
        "ConfigError": ("kcmt_python.exceptions", "ConfigError"),
        "ValidationError": ("kcmt_python.exceptions", "ValidationError"),
    }
    if name in mapping:
        mod_name, attr = mapping[name]
        mod = import_module(mod_name)
        value = getattr(mod, attr)
        globals()[name] = value  # cache for future access
        return value
    raise AttributeError(f"module 'kcmt_python' has no attribute {name!r}")


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
