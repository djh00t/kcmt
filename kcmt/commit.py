"""Commit message generation logic for kcmt."""

import re
from typing import Optional

from .config import Config, get_active_config
from .exceptions import LLMError, ValidationError
from .git import GitRepo
from .llm import LLMClient


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
        self, diff: str, context: str = "", style: str = "conventional"
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

        try:
            msg = self.llm_client.generate_commit_message(diff, context, style)
            if msg and msg.strip():
                return msg
        except LLMError:
            # Fall through to heuristic
            pass
        # Heuristic fallback if LLM failed or returned empty
        return self._heuristic_message(diff, context)

    # ------------------------------------------------------------------
    # Fallback heuristics
    # ------------------------------------------------------------------
    def _heuristic_message(self, diff: str, context: str) -> str:
        """Create a conventional commit message without the LLM.

    Rules (simple & deterministic):
                - type: feat for new file; refactor for code; docs/config/style/test
                    inferred by extension
                - scope: inferred from path (tests, docs, config, ui, core default)
                - subject: 'add <file>' if new else 'update <file>'
                - body: include counts if >5 changed lines
        """
        file_path = None
        if context.startswith("File:"):
            file_path = context.split("File:", 1)[1].strip() or None
        elif "File:" in context:
            file_path = context.split("File:", 1)[1].strip() or None

        basename = None
        if file_path:
            basename = file_path.split("/")[-1]
        else:
            basename = "changes"

        lines = diff.splitlines()
        added = sum(
            1 for line in lines
            if line.startswith("+") and not line.startswith("+++")
        )
        removed = sum(
            1 for line in lines
            if line.startswith("-") and not line.startswith("---")
        )
        is_new = any(
            ("new file mode" in line) or line.startswith("--- /dev/null")
            for line in lines
        )

        # Infer scope from path
        scope = "core"
        if file_path:
            lower = file_path.lower()
            if "test" in lower:
                scope = "tests"
            elif lower.endswith(('.md', '.rst', '.txt')):
                scope = "docs"
            elif lower.endswith(('.json', '.yml', '.yaml', '.toml', '.ini')):
                scope = "config"
            elif lower.endswith(('.css', '.scss', '.sass', '.less')):
                scope = "ui"

        # Infer type
        msg_type = "feat" if is_new else "refactor"
        if scope == "tests":
            msg_type = "test"
        elif scope == "docs":
            msg_type = "docs"
        elif scope == "config":
            msg_type = "chore"
        elif scope == "ui" and not is_new:
            msg_type = "style"

        verb = "add" if is_new else "update"
        subject = f"{verb} {basename}" if basename else "update files"
        header = f"{msg_type}({scope}): {subject}"

        # Body if enough churn
        total_changed = added + removed
        if total_changed > 5:
            body_lines = [
                f"{added} additions, {removed} deletions.",
                "Heuristic commit message fallback (LLM unavailable).",
            ]
            return header + "\n\n" + "\n".join(body_lines)
        return header

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
        except Exception:  # noqa: BLE001
            pass

        raise ValidationError(
            "Commit message does not follow conventional commit format: "
            f"{message}"
        )
