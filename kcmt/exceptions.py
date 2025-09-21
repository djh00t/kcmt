"""Custom exceptions for the kcmt library."""


class KlingonCMTError(Exception):
    """Base exception for all kcmt related errors."""


class GitError(KlingonCMTError):
    """Exception raised for Git-related operations."""


class LLMError(KlingonCMTError):
    """Exception raised for LLM-related operations."""


class ConfigError(KlingonCMTError):
    """Exception raised for configuration-related errors."""


class ValidationError(KlingonCMTError):
    """Exception raised for validation errors."""
