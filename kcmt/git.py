"""Git operations for kcmt."""

import os
import subprocess
from pathlib import Path
from typing import Optional

from .config import Config, get_active_config
from .exceptions import GitError


def find_git_repo_root(start_path: Optional[Path] = None) -> Optional[Path]:
    """Return the top-level Git repository directory for ``start_path``.

    Attempts ``git rev-parse --show-toplevel`` first so worktrees and
    submodules are handled correctly. Falls back to walking parent
    directories looking for a ``.git`` directory or file. Returns ``None``
    when no Git repository can be found starting from ``start_path``.
    """

    path = Path(start_path or Path.cwd()).expanduser().resolve(strict=False)
    if path.is_file():
        path = path.parent

    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=path,
            capture_output=True,
            text=True,
            check=True,
        )
        top = result.stdout.strip()
        if top:
            return Path(top)
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    for candidate in (path, *path.parents):
        git_meta = candidate / ".git"
        if git_meta.exists():
            return candidate

    return None


class GitRepo:
    """Handles Git repository operations."""

    def __init__(
        self,
        repo_path: Optional[str] = None,
        config: Optional[Config] = None,
    ) -> None:
        """Initialize Git repository handler."""

        self._config = config or get_active_config()
        self.repo_path = Path(repo_path or self._config.git_repo_path)
        if not self._is_git_repo():
            raise GitError(f"Not a Git repository: {self.repo_path}")

    def _is_git_repo(self) -> bool:
        """Check if the current directory is a Git repository."""
        try:
            self._run_git_command(["rev-parse", "--git-dir"])
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    def _run_git_command(self, args: list[str]) -> str:
        """Run a Git command and return its output."""
        try:
            result = subprocess.run(
                ["git"] + args,
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=True,
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError as e:
            cmd = " ".join(args)
            raise GitError(
                f"Git command failed: {cmd}\n{e.stderr}"
            ) from e
        except FileNotFoundError as exc:
            raise GitError(
                "Git command not found. Please install Git."
            ) from exc

    def is_ignored(self, rel_path: str) -> bool:
        """Return True if path is ignored by gitignore.

        Uses 'git check-ignore -q'. A zero exit status means ignored; 1 means
        not ignored. We treat other return codes as not ignored to avoid
        masking legitimate files.
        """
        try:
            result = subprocess.run(
                ["git", "check-ignore", "-q", rel_path],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=False,
            )
        except FileNotFoundError:
            return False
        return result.returncode == 0
    
    def get_staged_diff(self) -> str:
        """Get the diff of staged changes."""
        return self._run_git_command(["diff", "--cached"])

    def get_working_diff(self) -> str:
        """Get the diff of working directory changes."""
        return self._run_git_command(["diff"])

    def get_file_diff(self, file_path: str, staged: bool = False) -> str:
        """Get the diff for a specific file.

        Args:
            file_path: Path to file relative to repo root.
            staged: True to get staged diff, False for working tree diff.
        """
        args = ["diff"]
        if staged:
            args.append("--cached")
        args += ["--", file_path]
        return self._run_git_command(args)

    def get_file_diff_text(self, file_path: str, staged: bool = False) -> str:
        """Get a diff forcing text mode, overriding binary heuristics.

        Uses `git diff --text` which treats all files as text for the diff.
        """
        args = ["diff", "--text"]
        if staged:
            args.append("--cached")
        args += ["--", file_path]
        return self._run_git_command(args)

    def has_staged_changes(self) -> bool:
        """Check if there are staged changes."""
        diff = self.get_staged_diff()
        return bool(diff.strip())

    def has_working_changes(self) -> bool:
        """Check if there are working directory changes."""
        diff = self.get_working_diff()
        return bool(diff.strip())

    def get_commit_diff(self, commit_hash: str) -> str:
        """Get the diff for a specific commit."""
        return self._run_git_command(
            ["show", "--no-patch", "--format=", commit_hash]
        )

    def get_recent_commits(self, count: int = 5) -> list[str]:
        """Get recent commit messages."""
        # Use a custom pretty format that always includes abbreviated hash
        # followed by a single space and the subject line. Avoid combining
        # --oneline with --format which discards the hash.
        output = self._run_git_command([
            "log",
            f"-{count}",
            "--pretty=%h %s",
        ])
        return output.split("\n") if output else []

    def stage_file(self, file_path: str) -> None:
        """Stage a specific file for commit."""
        self._run_git_command(["add", file_path])

    def stage_all(self) -> None:
        """Stage all changes (including new and deleted files)."""
        self._run_git_command(["add", "-A"])

    def commit(self, message: str) -> None:
        """Create a commit with the given message."""
        self._run_git_command(["commit", "-m", message])

    def commit_file(self, message: str, file_path: str) -> None:
        """Create a commit including ONLY the specified file.

        This uses a pathspec after the message so that even if other files
        are staged (intentionally or accidentally) they are not part of this
        commit. Ensures true per-file atomic commits.
        """
        self._run_git_command(["commit", "-m", message, "--", file_path])

    def push(
        self, remote: str = "origin", branch: Optional[str] = None
    ) -> str:
        """Push current branch to remote.

        If branch is None, determine it via 'git rev-parse --abbrev-ref HEAD'.
        Returns the stdout from git push.
        """
        if branch is None:
            branch = self._run_git_command([
                "rev-parse",
                "--abbrev-ref",
                "HEAD",
            ])
        return self._run_git_command(["push", remote, branch])

    def reset_index(self) -> None:
        """Reset index (soft) to HEAD to clear staged state."""
        try:
            self._run_git_command(["reset"])
        except GitError:
            # Non-fatal; proceed even if reset fails
            pass

    def unstage(self, file_path: str) -> None:
        """Unstage a specific file."""
        self._run_git_command(["reset", "HEAD", file_path])

    def process_deletions_first(self) -> list[str]:
        """Process deletions first by staging all deleted files."""
        status_output = self._run_git_command(["status", "--porcelain"])

        deleted_files = []
        for line in status_output.split("\n"):
            if not line:
                continue
            status = line[:2]
            if "D" not in status:
                continue
            raw_path = (
                line[3:] if len(line) > 3 and line[2] == " " else line[2:]
            )
            file_path = raw_path.strip()
            if not file_path:
                continue
            deleted_files.append(file_path)
            self.stage_file(file_path)

        return deleted_files

    def list_changed_files(self) -> list[tuple[str, str]]:
        """Return porcelain status entries as (status, path)."""
        status_output = self._run_git_command(["status", "--porcelain"])

        entries: list[tuple[str, str]] = []
        for line in status_output.split("\n"):
            if not line:
                continue
            status = line[:2]
            raw_path = (
                line[3:] if len(line) > 3 and line[2] == " " else line[2:]
            )
            path = raw_path.strip()
            if not path:
                continue
            if " -> " in path:
                path = path.split(" -> ", 1)[1].strip()
            if path.startswith('"') and path.endswith('"') and len(path) >= 2:
                path = path[1:-1]
            # Expand untracked directories (Git collapses them in porcelain)
            if path.endswith("/") and status.startswith("??"):
                dir_rel = path.rstrip("/")
                dir_full = self.repo_path / dir_rel
                if dir_full.is_dir():
                    for root, _dirs, files in os.walk(dir_full):
                        for f in files:
                            full_path = Path(root) / f
                            rel_path = str(
                                full_path.relative_to(self.repo_path)
                            )
                            if self.is_ignored(rel_path):
                                continue
                            entries.append((status, rel_path))
                continue
            # Skip ignored standalone paths
            if self.is_ignored(path):
                continue
            entries.append((status, path))

        return entries
