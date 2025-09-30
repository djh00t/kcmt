"""Commit message generation logic for kcmt."""

import inspect
import re
from typing import Optional

from .config import Config, get_active_config
from .exceptions import LLMError, ValidationError
from .git import GitRepo
from .llm import LLMClient


def _supports_request_timeout(callable_obj) -> bool:
    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return False
    return "request_timeout" in signature.parameters


class CommitGenerator:
    """Generates commit messages using LLM based on Git diffs."""

    def __init__(
        self,
        repo_path: Optional[str] = None,
        config: Optional[Config] = None,
        debug: bool = False,
    ) -> None:
        """Initialize the commit generator.

        Args:
            repo_path: Path to the Git repository. Defaults to config value.
            config: Optional configuration override.
            debug: Whether to enable debug output for LLM requests.
        """

        self._config = config or get_active_config()
        self.git_repo = GitRepo(repo_path, self._config)
        self.llm_client = LLMClient(self._config, debug=debug)
        self.debug = debug

    def generate_from_staged(
        self, context: str = "", style: str = "conventional"
    ) -> str:
        """Generate commit message from staged changes.

        Args:
            context: Additional context about the changes.
            style: Commit message style (conventional, simple, etc.).

        Returns:
            Generated commit message.

        Raises:
            ValidationError: If no staged changes are found.
        """
        if not self.git_repo.has_staged_changes():
            raise ValidationError("No staged changes found. Stage your changes first.")

        diff = self.git_repo.get_staged_diff()
        return self.llm_client.generate_commit_message(diff, context, style)

    def generate_from_working(
        self, context: str = "", style: str = "conventional"
    ) -> str:
        """Generate commit message from working directory changes.

        Args:
            context: Additional context about the changes.
            style: Commit message style (conventional, simple, etc.).

        Returns:
            Generated commit message.

        Raises:
            ValidationError: If no working directory changes are found.
        """
        if not self.git_repo.has_working_changes():
            raise ValidationError("No working directory changes found.")

        diff = self.git_repo.get_working_diff()
        return self.llm_client.generate_commit_message(diff, context, style)

    def generate_from_commit(
        self, commit_hash: str, context: str = "", style: str = "conventional"
    ) -> str:
        """Generate commit message for an existing commit.

        Args:
            commit_hash: Hash of the commit to analyze.
            context: Additional context about the changes.
            style: Commit message style (conventional, simple, etc.).

        Returns:
            Generated commit message.
        """
        diff = self.git_repo.get_commit_diff(commit_hash)
        return self.llm_client.generate_commit_message(diff, context, style)

    def suggest_commit_message(
        self,
        diff: str,
        context: str = "",
        style: str = "conventional",
        request_timeout: float | None = None,
    ) -> str:
        """Generate a commit message from a provided diff.

        Args:
            diff: Git diff content.
            context: Additional context about the changes.
            style: Commit message style (conventional, simple, etc.).

        Returns:
            Generated commit message.

        Raises:
            ValidationError: If diff is empty.
        """
        if not diff or not diff.strip():
            raise ValidationError("Diff content cannot be empty.")

        last_error: Exception | None = None
        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            if self.debug:
                truncated_ctx = context[:120] + "…" if len(context) > 120 else context
                print(
                    "DEBUG: commit.attempt {} diff_len={} context='{}'".format(
                        attempt, len(diff), truncated_ctx
                    )
                )
            try:
                generate_fn = self.llm_client.generate_commit_message
                if request_timeout is not None and _supports_request_timeout(
                    generate_fn
                ):
                    msg = generate_fn(
                        diff,
                        context,
                        style,
                        request_timeout=request_timeout,
                    )
                else:
                    msg = generate_fn(diff, context, style)
                if request_timeout is not None and "request_timeout" not in getattr(
                    getattr(generate_fn, "__code__", {}), "co_varnames", ()
                ):
                    pass
                if not msg or not msg.strip():
                    raise LLMError("LLM returned empty response")
                # Validate format; if invalid, try again (unless final)
                if not self.validate_conventional_commit(msg):
                    if self.debug:
                        invalid_header = msg.splitlines()[0][:120]
                        print(
                            ("DEBUG: commit.invalid_format attempt={} msg='{}'").format(
                                attempt, invalid_header
                            )
                        )
                    if attempt < max_attempts:
                        continue
                    raise LLMError(
                        (
                            "LLM produced invalid commit message after {} attempts"
                        ).format(max_attempts)
                    )
                if self.debug:
                    print(
                        "DEBUG: commit.valid attempt={} header='{}'".format(
                            attempt, msg.splitlines()[0]
                        )
                    )
                return msg
            except LLMError as e:
                last_error = e
                if self.debug:
                    print(
                        "DEBUG: commit.error attempt={} error='{}'".format(
                            attempt, str(e)[:200]
                        )
                    )
                if attempt < max_attempts:
                    continue
        raise LLMError(
            (
                "LLM unavailable or invalid output after {} attempts; commit aborted"
            ).format(max_attempts)
        ) from last_error

    async def suggest_commit_message_async(
        self,
        diff: str,
        context: str = "",
        style: str = "conventional",
        request_timeout: float | None = None,
    ) -> str:
        if not diff or not diff.strip():
            raise ValidationError("Diff content cannot be empty.")

        last_error: Exception | None = None
        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            if self.debug:
                truncated_ctx = context[:120] + "…" if len(context) > 120 else context
                print(
                    "DEBUG: commit.attempt {} diff_len={} context='{}'".format(
                        attempt, len(diff), truncated_ctx
                    )
                )
            try:
                generate_fn = self.llm_client.generate_commit_message_async
                if request_timeout is not None and _supports_request_timeout(
                    generate_fn
                ):
                    msg = await generate_fn(
                        diff,
                        context,
                        style,
                        request_timeout=request_timeout,
                    )
                else:
                    msg = await generate_fn(diff, context, style)
                if not msg or not msg.strip():
                    raise LLMError("LLM returned empty response")
                if not self.validate_conventional_commit(msg):
                    if self.debug:
                        invalid_header = msg.splitlines()[0][:120]
                        print(
                            ("DEBUG: commit.invalid_format attempt={} msg='{}'").format(
                                attempt, invalid_header
                            )
                        )
                    if attempt < max_attempts:
                        continue
                    raise LLMError(
                        (
                            "LLM produced invalid commit message after {} attempts"
                        ).format(max_attempts)
                    )
                if self.debug:
                    print(
                        "DEBUG: commit.valid attempt={} header='{}'".format(
                            attempt, msg.splitlines()[0]
                        )
                    )
                return msg
            except LLMError as e:
                last_error = e
                if self.debug:
                    print(
                        "DEBUG: commit.error attempt={} error='{}'".format(
                            attempt, str(e)[:200]
                        )
                    )
                if attempt < max_attempts:
                    continue
        raise LLMError(
            (
                "LLM unavailable or invalid output after {} attempts; commit aborted"
            ).format(max_attempts)
        ) from last_error

    def validate_conventional_commit(self, message: str) -> bool:
        """Validate if a commit message follows conventional commit format.

        Args:
            message: The commit message to validate.

        Returns:
            True if the message follows conventional format, else False.
        """
        # Conventional commit pattern: type(scope): description
        # Types: feat, fix, docs, style, refactor, test, chore, etc.
        pattern = (
            r"^(feat|fix|docs|style|refactor|test|chore|perf|ci|build|revert)"
            r"(\([a-zA-Z0-9_-]+\))?: .+"
        )
        return bool(re.match(pattern, message.strip()))

    def validate_and_fix_commit_message(self, message: str) -> str:
        """Validate a commit message and attempt to fix it if invalid.

        Args:
            message: The commit message to validate and potentially fix.

        Returns:
            The validated/fixed commit message.

        Raises:
            ValidationError: If the message cannot be validated or fixed.
        """
        if self.validate_conventional_commit(message):
            return message

        # Try to generate a better message using LLM
        try:
            # Use a simple diff placeholder since we don't have actual diff
            fixed_message = self.llm_client.generate_commit_message(
                "Changes made to codebase", "", "conventional"
            )
            if self.validate_conventional_commit(fixed_message):
                return fixed_message
        except LLMError:
            # Upstream LLM failure; fall through to validation error
            pass

        raise ValidationError(
            f"Commit message does not follow conventional commit format: {message}"
        )
