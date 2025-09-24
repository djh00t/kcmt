"""Core workflow logic for kcmt."""

from __future__ import annotations

import os
import re
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .commit import CommitGenerator
from .config import Config, get_active_config
from .exceptions import GitError, KlingonCMTError, LLMError, ValidationError
from .git import GitRepo


@dataclass
class FileChange:
    """Represents a file change with its type and path."""

    file_path: str
    change_type: str  # 'A' | 'M' | 'D'
    diff_content: str = ""


@dataclass
class CommitResult:
    """Result of a commit operation."""

    success: bool
    commit_hash: Optional[str] = None
    message: Optional[str] = None
    error: Optional[str] = None
    file_path: Optional[str] = None


@dataclass
class PreparedCommit:
    """Holds a file change and its pre-generated commit message."""

    change: FileChange
    message: Optional[str]
    error: Optional[str] = None


class WorkflowStats:
    """Tracks workflow progress and renders real-time stats."""

    def __init__(self) -> None:
        self.total_files = 0
        self.prepared = 0
        self.processed = 0
        self.successes = 0
        self.failures = 0
        self._start = time.time()
        self._lock = threading.Lock()

    def set_total(self, total: int) -> None:
        with self._lock:
            self.total_files = total

    def mark_prepared(self) -> None:
        with self._lock:
            self.prepared += 1

    def mark_result(self, success: bool) -> None:
        with self._lock:
            self.processed += 1
            if success:
                self.successes += 1
            else:
                self.failures += 1

    def snapshot(self) -> Dict[str, float]:
        with self._lock:
            elapsed = max(time.time() - self._start, 1e-6)
            return {
                "total_files": self.total_files,
                "prepared": self.prepared,
                "processed": self.processed,
                "successes": self.successes,
                "failures": self.failures,
                "elapsed": elapsed,
                "rate": self.processed / elapsed if self.processed else 0.0,
            }


class KlingonCMTWorkflow:
    """Atomic staging and committing workflow with LLM assistance."""

    def __init__(
        self,
        repo_path: Optional[str] = None,
        max_retries: int = 3,
        config: Optional[Config] = None,
        show_progress: bool = False,
        file_limit: Optional[int] = None,
        debug: bool = False,
    ) -> None:
        """Initialize the workflow."""
        self._config = config or get_active_config()
        self.git_repo = GitRepo(repo_path, self._config)
        self.commit_generator = CommitGenerator(
            repo_path, self._config, debug=debug
        )
        self.max_retries = max_retries
        self._stats = WorkflowStats()
        self._show_progress = show_progress
        self.file_limit = file_limit
        self.debug = debug

    def execute_workflow(self) -> Dict[str, Any]:
        """Execute the complete kcmt workflow."""
        results: Dict[str, Any] = {
            "deletions_committed": [],
            "file_commits": [],
            "errors": [],
            "summary": "",
        }

        try:
            deletion_results = self._process_deletions_first()
            results["deletions_committed"] = deletion_results

            file_results = self._process_per_file_commits()
            results["file_commits"] = file_results

            results["summary"] = self._generate_summary(results)
        except (
            GitError,
            KlingonCMTError,
            ValidationError,
        ) as e:  # pragma: no cover
            results["errors"].append(str(e))
            raise KlingonCMTError(f"Workflow failed: {e}") from e
        finally:
            self._finalize_progress()

        # Auto-push if enabled and we actually committed something
        any_success = (
            any(r.success for r in results.get("file_commits", []))
            or any(
                r.success
                for r in results.get("deletions_committed", [])
            )
        )
        if any_success and getattr(self._config, "auto_push", False):
            try:
                self.git_repo.push()
                results["pushed"] = True
            except GitError as e:  # pragma: no cover - network dependent
                results.setdefault("errors", []).append(
                    f"Auto-push failed: {e}"
                )

        return results

    def _process_deletions_first(self) -> List[CommitResult]:
        """Process all deletions first with a single commit."""
        results: List[CommitResult] = []

        deleted_files = self.git_repo.process_deletions_first()
        if not deleted_files:
            return results

        commit_message = (
            "chore: remove deleted files\n\nRemoved files:\n" + "\n".join(
                f"- {f}" for f in deleted_files
            )
        )

        try:
            validated_message = (
                self.commit_generator.validate_and_fix_commit_message(
                    commit_message
                )
            )
        except ValidationError:
            validated_message = self._generate_deletion_commit_message(
                deleted_files
            )

        result = self._attempt_commit(
            validated_message,
            max_retries=self.max_retries,
        )
        results.append(result)
        return results

    def _generate_deletion_commit_message(
        self, deleted_files: List[str]
    ) -> str:
        """Generate a commit message for deleted files."""
        if len(deleted_files) == 1:
            return f"chore: remove {deleted_files[0]}"
        return f"chore: remove {len(deleted_files)} files"

    def _process_per_file_commits(self) -> List[CommitResult]:
        """Process remaining changes with per-file commits."""
        results: List[CommitResult] = []

        # First, get all changed files from git status (both staged/unstaged)
        all_changed_files = self.git_repo.list_changed_files()
        
        # Filter out deletions (they're handled separately)
        non_deletion_files = [
            entry for entry in all_changed_files if "D" not in entry[0]
        ]

        # Skip control/meta files that shouldn't usually receive their own
        # atomic commits to avoid surprising commits (e.g. tests expecting
        # .gitignore to be ignored). These can still be committed manually
        # via --file if desired.
        META_SKIP = {".gitignore", ".gitattributes", ".gitmodules"}
        non_deletion_files = [
            e for e in non_deletion_files if e[1] not in META_SKIP
        ]
        
        if not non_deletion_files:
            return results

        # Apply file limit if specified
        if self.file_limit and self.file_limit > 0:
            non_deletion_files = non_deletion_files[:self.file_limit]
        # Build per-file diffs WITHOUT staging all files at once. This avoids
        # the previous behaviour where the first commit would include every
        # file that had been pre-staged. We stage each file temporarily just
        # to capture its diff (so new/untracked files produce a diff), then
        # immediately unstage it. Commit generation later will re-stage only
        # that file for its own atomic commit.
        file_changes: List[FileChange] = []
        for _status, file_path in non_deletion_files:
            try:
                # Stage the file to obtain a reliable diff (untracked files
                # won't appear in plain working diff output otherwise)
                self.git_repo.stage_file(file_path)
                single_diff = self.git_repo.get_file_diff(
                    file_path, staged=True
                )
                # Unstage so subsequent commits are atomic
                try:
                    self.git_repo.unstage(file_path)
                except GitError:
                    # Non-fatal; if unstage fails we still proceed
                    pass
                if not single_diff.strip():
                    continue
                parsed = self._parse_git_diff(single_diff)
                if parsed:
                    # _parse_git_diff returns a list; for a single-file diff
                    # we take the first element.
                    file_changes.append(parsed[0])
            except GitError as e:
                results.append(
                    CommitResult(
                        success=False,
                        error=f"Failed to capture diff for {file_path}: {e}",
                        file_path=file_path,
                    )
                )
                continue

        if not file_changes:
            return results

        self._stats.set_total(len(file_changes))
        prepared_commits = self._prepare_commit_messages(file_changes)

        ordered_prepared = sorted(prepared_commits, key=lambda item: item[0])

        for _, prepared in ordered_prepared:
            if prepared.error:
                # Unstage the file since commit preparation failed
                try:
                    self.git_repo.unstage(prepared.change.file_path)
                except GitError:
                    pass  # Don't fail if unstaging fails
                result = CommitResult(success=False, error=prepared.error)
            else:
                result = self._commit_single_file(
                    prepared.change, prepared.message
                )

            results.append(result)
            self._stats.mark_result(result.success)
            self._print_progress(stage="commit")

        return results

    def _prepare_commit_messages(
        self, file_changes: List[FileChange]
    ) -> List[Tuple[int, PreparedCommit]]:
        cpu_hint = os.cpu_count() or 4
        # Limit concurrent threads to be more API-friendly
        workers = max(1, min(len(file_changes), 4, cpu_hint))
        prepared: List[Tuple[int, PreparedCommit]] = []

        print(
            "Using {} concurrent threads for {} files".format(
                workers, len(file_changes)
            )
        )

        per_file_timeout_env = os.environ.get("KCMT_PREPARE_PER_FILE_TIMEOUT")
        try:
            per_file_timeout = (
                float(per_file_timeout_env)
                if per_file_timeout_env
                else 45.0
            )
        except ValueError:
            per_file_timeout = 45.0

        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_map = {
                executor.submit(self._prepare_single_change, change): idx
                for idx, change in enumerate(file_changes)
            }

            # Track start times for each future so we can enforce a wall-clock
            # timeout per submitted job without a brittle global timeout.
            start_times = {f: time.time() for f in future_map}
            remaining = set(future_map.keys())

            # Poll until all futures handled (either completed or timed out)
            while remaining:
                done, not_done = wait(
                    remaining, timeout=0.05, return_when=FIRST_COMPLETED
                )

                # Process any newly completed futures
                for fut in done:
                    idx = future_map[fut]
                    try:
                        prepared_commit = fut.result()
                    # Broad catch: ensure a single file failure does not
                    # abort the entire workflow.
                    except Exception as exc:  # noqa: BLE001
                        prepared_commit = PreparedCommit(
                            change=file_changes[idx],
                            message=None,
                            error=(
                                "Error preparing "
                                f"{file_changes[idx].file_path}: {exc}"
                            ),
                        )
                    prepared.append((idx, prepared_commit))
                    self._stats.mark_prepared()
                    self._print_progress(stage="prepare")
                    remaining.discard(fut)

                # Check for per-file timeout on still running futures
                now = time.time()
                for fut in list(not_done):
                    if now - start_times[fut] > per_file_timeout:
                        idx = future_map[fut]
                        # Attempt to cancel; if it's already running this will
                        # return False, but we still record a timeout result.
                        fut.cancel()
                        prepared_commit = PreparedCommit(
                            change=file_changes[idx],
                            message=None,
                            error=(
                                "Timeout after "
                                f"{per_file_timeout:.1f}s waiting for "
                                f"{file_changes[idx].file_path}"
                            ),
                        )
                        prepared.append((idx, prepared_commit))
                        self._stats.mark_prepared()
                        self._print_progress(stage="prepare")
                        remaining.discard(fut)

        return prepared

    def _prepare_single_change(self, change: FileChange) -> PreparedCommit:
        generator = CommitGenerator(
            repo_path=str(self.git_repo.repo_path),
            config=self._config,
            debug=self.debug,
        )
        if self.debug:
            snippet = change.diff_content.splitlines()[:20]
            preview = "\n".join(snippet)
            print(
                (
                    "DEBUG: prepare.file path={} change_type={} "
                    "diff_preview=\n{}"
                ).format(
                    change.file_path,
                    change.change_type,
                    preview,
                )
            )

        try:
            commit_message = generator.suggest_commit_message(
                change.diff_content,
                context=f"File: {change.file_path}",
                style="conventional",
            )
            validated = generator.validate_and_fix_commit_message(
                commit_message
            )
            self._print_commit_generated(change.file_path, validated)
            if self.debug:
                print(
                    "DEBUG: prepare.success path={} header='{}'".format(
                        change.file_path,
                        validated.splitlines()[0] if validated else "",
                    )
                )
            return PreparedCommit(change=change, message=validated)
        except (ValidationError, LLMError) as exc:
            if self.debug:
                print(
                    "DEBUG: prepare.failure path={} error='{}'".format(
                        change.file_path, str(exc)[:200]
                    )
                )
            return PreparedCommit(
                change=change,
                message=None,
                error=(
                    "Failed to generate valid commit message for "
                    f"{change.file_path}: {exc}"
                ),
            )
        except KlingonCMTError as exc:  # pragma: no cover
            return PreparedCommit(
                change=change,
                message=None,
                error=(
                    "Internal kcmt error preparing commit for "
                    f"{change.file_path}: {exc}"
                ),
            )
        # Defensive: unexpected exceptions outside kcmt domain should not
        # crash the entire preparation; convert to generic error.
        except Exception as exc:  # pragma: no cover  # noqa: BLE001
            return PreparedCommit(
                change=change,
                message=None,
                error=(
                    "Unexpected non-kcmt error preparing commit for "
                    f"{change.file_path}: {exc}"
                ),
            )

    def _print_progress(self, stage: str) -> None:
        if not getattr(self, "_show_progress", False):
            return

        snapshot = self._stats.snapshot()
        total = snapshot["total_files"]
        prepared = snapshot["prepared"]
        processed = snapshot["processed"]
        success = snapshot["successes"]
        failures = snapshot["failures"]
        rate = snapshot["rate"]

        bar = (
            f"[kcmt] stage={stage:<7} | files {processed}/{total} "
            f"| prepared {prepared}/{total} | ok {success} | fail {failures} "
            f"| {rate:.2f} commits/s"
        )

        print("\r" + bar, end="", flush=True)

    def _finalize_progress(self) -> None:
        if not getattr(self, "_show_progress", False):
            return
        self._print_progress(stage="done")
        print()

    def _print_commit_generated(
        self, file_path: str, commit_message: str
    ) -> None:
        """Display the generated commit message for a file."""
        # Colors for consistency with CLI
        CYAN = "\033[96m"
        GREEN = "\033[92m"
        RESET = "\033[0m"
        
        print(f"\n{CYAN}Generated for {file_path}:{RESET}")
        # Show the full commit message without truncation
        print(f"{GREEN}{commit_message}{RESET}")
        # Add separator line for readability
        print("-" * 50)

    def _parse_git_diff(self, diff: str) -> List[FileChange]:
        """Parse git diff output to extract file changes."""
        changes: List[FileChange] = []
        current_file: Optional[str] = None
        current_diff: List[str] = []

        lines = diff.split("\n")
        i = 0
        while i < len(lines):
            line = lines[i]
            if line.startswith("diff --git"):
                if current_file:
                    changes.append(
                        FileChange(
                            file_path=current_file,
                            change_type=self._determine_change_type(
                                current_diff
                            ),
                            diff_content="\n".join(current_diff),
                        )
                    )

                match = re.search(r"diff --git a/(.+) b/(.+)", line)
                if match:
                    current_file = match.group(2)
                    current_diff = [line]
                else:
                    current_file = None
                    current_diff = []

            elif (
                line.startswith("index ")
                or line.startswith("--- ")
                or line.startswith("+++ ")
                or line.startswith("Binary files")
                or line.startswith("new file mode")
                or line.startswith("deleted file mode")
            ):
                current_diff.append(line)

            elif line.startswith("@@"):
                current_diff.append(line)
                i += 1
                while i < len(lines) and not lines[i].startswith("diff --git"):
                    current_diff.append(lines[i])
                    i += 1
                i -= 1
            i += 1

        if current_file:
            changes.append(
                FileChange(
                    file_path=current_file,
                    change_type=self._determine_change_type(current_diff),
                    diff_content="\n".join(current_diff),
                )
            )

        return changes

    def _determine_change_type(self, diff_lines: List[str]) -> str:
        """Determine the change type from diff content."""
        added_markers = ("new file mode", "--- /dev/null")
        deleted_markers = ("deleted file mode", "+++ /dev/null")

        added = any(
            line.startswith(marker)
            for line in diff_lines
            for marker in added_markers
        )
        deleted = any(
            line.startswith(marker)
            for line in diff_lines
            for marker in deleted_markers
        )

        if added and not deleted:
            return "A"
        if deleted and not added:
            return "D"
        return "M"

    def _commit_single_file(
        self, change: FileChange, prepared_message: Optional[str] = None
    ) -> CommitResult:
        """Commit a single file change."""
        # Reset any stray staging (defensive) then stage ONLY this file
        try:
            # Use a soft reset of index (ignore errors; if clean it is cheap)
            self.git_repo.reset_index()
            self.git_repo.stage_file(change.file_path)
        except GitError as e:
            return CommitResult(
                success=False,
                error=f"Failed to stage {change.file_path}: {e}",
                file_path=change.file_path,
            )

        try:
            if prepared_message is not None:
                validated_message = prepared_message
            else:
                commit_message = self._generate_file_commit_message(change)
                validated_message = (
                    self.commit_generator.validate_and_fix_commit_message(
                        commit_message
                    )
                )
        except ValidationError as e:
            # Unstage the file since we failed to generate a commit message
            try:
                self.git_repo.unstage(change.file_path)
            except GitError:
                pass  # Don't fail if unstaging fails
            return CommitResult(
                success=False,
                error=(
                    "Failed to generate valid commit message for "
                    f"{change.file_path}: {e}"
                ),
                file_path=change.file_path,
            )

        result = self._attempt_commit(
            validated_message,
            max_retries=self.max_retries,
            file_path=change.file_path,
        )
        return result

    def _generate_file_commit_message(self, change: FileChange) -> str:
        """Generate a commit message for a single file change."""
        return self.commit_generator.suggest_commit_message(
            change.diff_content,
            context=f"File: {change.file_path}",
            style="conventional",
        )

    def _attempt_commit(
        self,
        message: str,
        max_retries: int = 3,
        file_path: Optional[str] = None,
    ) -> CommitResult:
        """Attempt to create a commit with retries and LLM assistance."""
        last_error: Optional[str] = None

        for attempt in range(max_retries + 1):
            try:
                if file_path:
                    # Atomic per-file commit: only target file pathspec
                    self.git_repo.commit_file(message, file_path)
                else:
                    self.git_repo.commit(message)
                recent_commits = self.git_repo.get_recent_commits(1)
                commit_hash = (
                    recent_commits[0].split()[0] if recent_commits else None
                )
                return CommitResult(
                    success=True,
                    commit_hash=commit_hash,
                    message=message,
                    file_path=file_path,
                )
            except GitError as e:
                last_error = str(e)
                if attempt < max_retries:
                    try:
                        fixed_message = self._fix_commit_message_with_llm(
                            message, str(e)
                        )
                        if fixed_message != message:
                            message = fixed_message
                            continue
                    except (LLMError, ValidationError):  # pragma: no cover
                        pass
                if attempt == max_retries:
                    return CommitResult(
                        success=False,
                        error=(
                            f"Commit failed after {max_retries + 1} "
                            f"attempts: {last_error}"
                        ),
                        file_path=file_path,
                    )

        return CommitResult(
            success=False,
            error=f"Unexpected error: {last_error}",
            file_path=file_path,
        )

    def _fix_commit_message_with_llm(
        self,
        original_message: str,
        error: str,
    ) -> str:
        """Use LLM to fix a commit message that caused an error."""
        prompt_lines = [
            "The following commit message caused a Git error:\n",
            f"Message: {original_message}",
            f"Error: {error}",
            "",
            "Please provide a corrected conventional commit message.",
            "Rules:",
            "- Format: type(scope): subject",
            "- Scope is mandatory",
            "- Subject (first line) <= 50 characters, no period",
            (
                "- If explanation is helpful, add body after blank line; "
                "wrap body at 72 chars"
            ),
            "Return ONLY the commit message.",
        ]
        prompt = "\n".join(prompt_lines)

        # Try LLM first, then fallback to local synthesis if empty/invalid
        try:
            candidate = (
                self.commit_generator.llm_client.generate_commit_message(
                    "Fix commit message", prompt, "conventional"
                )
            )
            if candidate and candidate.strip():
                return candidate.strip()
        except LLMError:
            pass
        # Local fallback synthesis
        return self._synthesize_fixed_commit(original_message)

    def _synthesize_fixed_commit(self, original: str) -> str:
        """Generate a corrected commit message locally (no LLM).

        Rules applied:
        - Ensure conventional format type(scope): subject
        - If missing scope, insert (core)
        - If missing type, default to chore
        - Trim subject to 50 chars (word boundary) no period
        - Preserve body (if any) after a blank line; wrap not handled here
        """
        lines = original.strip().splitlines()
        if not lines:
            return "chore(core): update"
        header = lines[0].strip()
        body = lines[1:]

        # Extract existing type/scope
        type_pattern = (
            r"^(feat|fix|docs|style|refactor|perf|test|build|ci|chore|revert)"
        )
        scope_pattern = (
            r"^(feat|fix|docs|style|refactor|perf|test|build|ci|chore|revert)"
            r"\(([a-zA-Z0-9_-]+)\):\s+(.+)"
        )

        msg_type = "chore"
        scope = None
        subject = header

        m_scope = re.match(scope_pattern, header)
        if m_scope:
            msg_type = m_scope.group(1)
            scope = m_scope.group(2)
            subject = m_scope.group(3)
        else:
            m_type = re.match(type_pattern, header)
            if m_type and ':' in header:
                msg_type = m_type.group(1)
                after_colon = header.split(':', 1)[1].strip()
                subject = after_colon
            else:
                # No valid type prefix; treat whole header as subject
                subject = header

        if not scope:
            scope = "core"

        # Remove trailing period from subject
        if subject.endswith('.'):
            subject = subject[:-1]

        # Enforce length 50 chars on subject
        max_len = 50
        if len(subject) > max_len:
            cut = subject.rfind(' ', 0, max_len)
            if cut == -1 or cut < max_len * 0.6:
                cut = max_len - 1
            subject = subject[:cut].rstrip() + '…'

        rebuilt = f"{msg_type}({scope}): {subject}"

        if body:
            # Clean body: strip leading/trailing blank lines
            while body and not body[0].strip():
                body.pop(0)
            while body and not body[-1].strip():
                body.pop()
            if body:
                return rebuilt + "\n\n" + "\n".join(body)
        return rebuilt

    def _generate_summary(self, results: Dict[str, Any]) -> str:
        """Generate a human-readable summary string."""
        deletions = results.get("deletions_committed", [])
        file_commits = results.get("file_commits", [])
        errors = results.get("errors", [])

        summary_parts: List[str] = []

        if deletions:
            successful_deletions = [r for r in deletions if r.success]
            summary_parts.append(
                f"Committed {len(successful_deletions)} deletion(s)"
            )

        if file_commits:
            successful_commits = [r for r in file_commits if r.success]
            summary_parts.append(
                f"Committed {len(successful_commits)} file change(s)"
            )

        if errors:
            summary_parts.append(f"Encountered {len(errors)} error(s)")

        total_commits = len([r for r in deletions + file_commits if r.success])

        if total_commits > 0:
            summary_parts.insert(
                0, f"Successfully completed {total_commits} commits"
            )
        else:
            summary_parts.insert(0, "No commits were made")

        return ". ".join(summary_parts)
