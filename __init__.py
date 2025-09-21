"""kcmt - AI-powered atomic Git staging and committing tool."""

from importlib import import_module
from typing import TYPE_CHECKING

__version__ = "0.1.0"

# Public API (lazy-exported to avoid import-time side effects)
__all__ = [
    # Config
    "Config", "config",
    # LLM
    "LLMClient",
    # Git
    "GitRepo",
    # Commit generation
    "CommitGenerator",
    # Core workflow
    "KlingonCMTWorkflow", "FileChange", "CommitResult",
    # Exceptions
    "KlingonCMTError", "GitError", "LLMError", "ConfigError", "ValidationError",
]


def __getattr__(name: str):
    """Lazy attribute loader to avoid importing heavy modules at package import time.

    This prevents environment-dependent modules (e.g., those that read env in
    kcmt.config) from being imported unless explicitly accessed.
    """
    mapping = {
        # Config
        "Config": ("kcmt.config", "Config"),
        "config": ("kcmt.config", "config"),
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
    from .config import Config, config
    from .llm import LLMClient
    from .git import GitRepo
    from .commit import CommitGenerator
    from .core import KlingonCMTWorkflow, FileChange, CommitResult
    from .exceptions import (
        KlingonCMTError,
        GitError,
        LLMError,
        ConfigError,
        ValidationError,
    )