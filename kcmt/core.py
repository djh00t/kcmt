"""Core workflow logic for kcmt."""

from __future__ import annotations

import os
import re
import threading
import time
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .commit import CommitGenerator
from .config import Config, get_active_config
from .exceptions import GitError, KlingonCMTError, LLMError, ValidationError
from .git import GitRepo

RESET = "\033[0m"
BOLD = "\033[1m"
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
MAGENTA = "\033[95m"
DIM = "\033[2m"
RED = "\033[91m"


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
        profile: bool = False,
        verbose: bool = False,
    ) -> None:
        """Initialize the workflow."""
        self._config = config or get_active_config()
        self.git_repo = GitRepo(repo_path, self._config)
        self.commit_generator = CommitGenerator(repo_path, self._config, debug=debug)
        self.max_retries = max_retries
        self._stats = WorkflowStats()
        self._show_progress = show_progress
        self.file_limit = file_limit
        self.debug = debug
        self.profile = profile
        self.verbose = verbose
        self._thread_local = threading.local()
        self._thread_local.generator = self.commit_generator
        self._progress_snapshots: dict[str, str] = {}
        self._commit_subjects: list[str] = []
        self._last_progress_stage: Optional[str] = None
        self._prepare_failure_limit_hit = False
        self._prepare_failure_limit_message = ""

    def _profile(self, label: str, elapsed_seconds: float, extra: str = "") -> None:
        if not self.profile:
            return
        details = f" {extra}" if extra else ""
        print(f"[kcmt-profile] {label}: {elapsed_seconds * 1000.0:.1f} ms{details}")

    def execute_workflow(self) -> Dict[str, Any]:
        """Execute the complete kcmt workflow."""
        self._prepare_failure_limit_hit = False
        self._prepare_failure_limit_message = ""
        results: Dict[str, Any] = {
            "deletions_committed": [],
            "file_commits": [],
            "errors": [],
            "summary": "",
        }

        workflow_start = time.perf_counter()

        status_entries: Optional[list[tuple[str, str]]] = None
        try:
            status_start = time.perf_counter()
            status_entries = self.git_repo.scan_status()
            self._profile(
                "git-status",
                time.perf_counter() - status_start,
                extra=f"entries={len(status_entries)}",
            )

            deletion_results = self._process_deletions_first(status_entries)
            results["deletions_committed"] = deletion_results

            file_results = self._process_per_file_commits(status_entries)
            results["file_commits"] = file_results

            if self._prepare_failure_limit_hit and self._prepare_failure_limit_message:
                results["errors"].append(self._prepare_failure_limit_message)

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
        any_success = any(r.success for r in results.get("file_commits", [])) or any(
            r.success for r in results.get("deletions_committed", [])
        )
        if any_success and getattr(self._config, "auto_push", False):
            try:
                self.git_repo.push()
                results["pushed"] = True
            except GitError as e:  # pragma: no cover - network dependent
                results.setdefault("errors", []).append(f"Auto-push failed: {e}")

        total_elapsed = time.perf_counter() - workflow_start
        self._profile(
            "workflow-total",
            total_elapsed,
            extra=(
                "files={} deletions={}".format(
                    len(results.get("file_commits", [])),
                    len(results.get("deletions_committed", [])),
                )
            ),
        )

        return results

    def _process_deletions_first(
        self, status_entries: Optional[list[tuple[str, str]]] = None
    ) -> List[CommitResult]:
        """Process all deletions first with a single commit."""
        results: List[CommitResult] = []

        deletions_start = time.perf_counter()
        deleted_files = self.git_repo.process_deletions_first(status_entries)
        self._profile(
            "process-deletions",
            time.perf_counter() - deletions_start,
            extra=f"count={len(deleted_files)}",
        )
        if not deleted_files:
            return results

        commit_message = "chore: remove deleted files\n\nRemoved files:\n" + "\n".join(
            f"- {f}" for f in deleted_files
        )

        try:
            validated_message = self.commit_generator.validate_and_fix_commit_message(
                commit_message
            )
        except ValidationError:
            validated_message = self._generate_deletion_commit_message(deleted_files)

        result = self._attempt_commit(
            validated_message,
            max_retries=self.max_retries,
        )
        results.append(result)
        return results

    def _generate_deletion_commit_message(self, deleted_files: List[str]) -> str:
        """Generate a commit message for deleted files."""
        if len(deleted_files) == 1:
            return f"chore: remove {deleted_files[0]}"
        return f"chore: remove {len(deleted_files)} files"

    def _process_per_file_commits(
        self, status_entries: Optional[list[tuple[str, str]]] = None
    ) -> List[CommitResult]:
        """Process remaining changes with per-file commits."""
        results: List[CommitResult] = []

        # First, get all changed files from git status (both staged/unstaged)
        status_start = time.perf_counter()
        all_changed_files = self.git_repo.list_changed_files(status_entries)
        self._profile(
            "git-status",
            time.perf_counter() - status_start,
            extra=f"entries={len(all_changed_files)}",
        )

        # Filter out deletions (they're handled separately)
        non_deletion_files = [
            entry for entry in all_changed_files if "D" not in entry[0]
        ]

        # Skip control/meta files that shouldn't usually receive their own
        # atomic commits to avoid surprising commits (e.g. tests expecting
        # .gitignore to be ignored). These can still be committed manually
        # via --file if desired.
        META_SKIP = {".gitignore", ".gitattributes", ".gitmodules"}
        non_deletion_files = [e for e in non_deletion_files if e[1] not in META_SKIP]

        if not non_deletion_files:
            return results

        # Apply file limit if specified
        if self.file_limit and self.file_limit > 0:
            non_deletion_files = non_deletion_files[: self.file_limit]
        # Build per-file diffs WITHOUT staging files at all. Instead of the
        # previous add/diff/reset loop (which spawned three git commands per
        # file), we read the worktree diff directly so large repositories do
        # not thrash the index or pay the subprocess overhead. Each commit is
        # staged only at the last moment inside _commit_single_file.
        file_changes: List[FileChange] = []
        collect_start = time.perf_counter()

        for status, file_path in non_deletion_files:
            try:
                single_diff = self.git_repo.get_worktree_diff_for_path(file_path)
                if not single_diff.strip():
                    continue
                change_type = self._change_type_from_status(status)
                change = FileChange(
                    file_path=file_path,
                    change_type=change_type,
                    diff_content=single_diff,
                )
                file_changes.append(change)
            except GitError as e:
                result = CommitResult(
                    success=False,
                    error=f"Failed to capture diff for {file_path}: {e}",
                    file_path=file_path,
                )
                results.append(result)
                continue

        unique_paths = {change.file_path for change in file_changes}
        self._profile(
            "collect-diffs",
            time.perf_counter() - collect_start,
            extra=(
                "candidates={} collected={} unique_paths={}".format(
                    len(non_deletion_files),
                    len(file_changes),
                    len(unique_paths),
                )
            ),
        )

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
                result = self._commit_single_file(prepared.change, prepared.message)

            results.append(result)
            self._stats.mark_result(result.success)
            self._print_progress(stage="commit")

        return results

    def _change_type_from_status(self, status: str) -> str:
        """Map a porcelain status code to FileChange.change_type."""

        trimmed = status.strip()
        if "D" in trimmed:
            return "D"
        if trimmed == "??" or "A" in trimmed:
            return "A"
        return "M"

    def _prepare_commit_messages(
        self, file_changes: List[FileChange]
    ) -> List[Tuple[int, PreparedCommit]]:
        if not file_changes:
            return []

        cpu_hint = os.cpu_count() or 4
        workers = max(1, min(len(file_changes), 8, cpu_hint))

        print(
            f"{MAGENTA}⚙️  Spinning up {workers} worker(s) for "
            f"{len(file_changes)} file(s){RESET}"
        )

        prepared: List[Tuple[int, PreparedCommit]] = []
        log_queue: dict[int, PreparedCommit] = {}
        next_log_index = 0
        completed: set[int] = set()

        failure_limit = 25
        failure_count = 0
        self._prepare_failure_limit_hit = False
        self._prepare_failure_limit_message = ""

        per_file_timeout_env = os.environ.get("KCMT_PREPARE_PER_FILE_TIMEOUT")
        try:
            per_file_timeout = (
                float(per_file_timeout_env) if per_file_timeout_env else 5.0
            )
        except ValueError:
            per_file_timeout = 5.0

        timeout_retry_limit = 3
        timeout_attempt_limit = timeout_retry_limit + 1

        executor = ThreadPoolExecutor(max_workers=workers)
        future_map: dict[Any, int] = {}
        remaining: set[Any] = set()
        retry_queue: deque[int] = deque()

        start_times: dict[int, Optional[float]] = {}
        pending_debug_log: dict[int, float] = {}
        attempt_counts: dict[int, int] = {idx: 1 for idx in range(len(file_changes))}
        changes_by_idx = {idx: change for idx, change in enumerate(file_changes)}

        def submit(idx: int) -> None:
            change = changes_by_idx[idx]

            def task(change=change, idx=idx) -> PreparedCommit:
                start_times[idx] = time.time()
                return self._prepare_single_change(change)

            future = executor.submit(task)
            future_map[future] = idx
            remaining.add(future)
            pending_debug_log[idx] = 0.0

        def flush_log() -> None:
            nonlocal next_log_index
            while next_log_index in log_queue:
                entry = log_queue.pop(next_log_index)
                self._log_prepared_result(entry)
                next_log_index += 1

        def mark_prepared(idx: int, prepared_commit: PreparedCommit) -> None:
            nonlocal failure_count
            if idx in completed:
                return
            prepared.append((idx, prepared_commit))
            self._stats.mark_prepared()
            self._print_progress(stage="prepare")
            log_queue[idx] = prepared_commit
            completed.add(idx)
            start_times.pop(idx, None)
            pending_debug_log.pop(idx, None)
            attempt_counts.pop(idx, None)

            if prepared_commit.error:
                failure_count += 1
                if failure_count >= failure_limit and not self._prepare_failure_limit_hit:
                    self._prepare_failure_limit_hit = True
                    self._prepare_failure_limit_message = (
                        f"Stopped prepare phase after {failure_count} failures (limit {failure_limit})."
                    )

            flush_log()

        def process_done(done_set: set[Any]) -> None:
            for fut in done_set:
                idx = future_map.pop(fut, None)
                remaining.discard(fut)
                if idx is None:
                    continue
                try:
                    prepared_commit = fut.result()
                except Exception as exc:  # noqa: BLE001
                    change = changes_by_idx[idx]
                    prepared_commit = PreparedCommit(
                        change=change,
                        message=None,
                        error=f"Error preparing {change.file_path}: {exc}",
                    )
                mark_prepared(idx, prepared_commit)
                if self._prepare_failure_limit_hit:
                    break

        def process_timeouts(pending_futures: set[Any], now: float) -> None:
            nonlocal failure_count
            for fut in list(pending_futures):
                idx = future_map.get(fut)
                if idx is None:
                    continue
                start_time = start_times.get(idx)
                if start_time is None:
                    continue
                elapsed = now - start_time
                if elapsed <= per_file_timeout:
                    if self.debug:
                        last_logged = pending_debug_log.get(idx, 0.0)
                        if now - last_logged >= 5.0:
                            change = changes_by_idx[idx]
                            diff_len = len(change.diff_content)
                            print(
                                (
                                    "DEBUG: prepare.pending path={} elapsed={:.1f}s diff_len={}"
                                ).format(
                                    change.file_path,
                                    elapsed,
                                    diff_len,
                                )
                            )
                            pending_debug_log[idx] = now
                    continue

                attempts = attempt_counts.get(idx, 1)
                fut.cancel()
                remaining.discard(fut)
                future_map.pop(fut, None)
                start_times.pop(idx, None)
                pending_debug_log.pop(idx, None)

                if attempts < timeout_attempt_limit:
                    attempt_counts[idx] = attempts + 1
                    retry_queue.append(idx)
                    continue

                change = changes_by_idx[idx]
                error_message = (
                    "Timeout after "
                    f"{per_file_timeout:.1f}s waiting for "
                    f"{change.file_path} "
                    f"(attempt {attempts}/{timeout_attempt_limit})"
                )
                prepared_commit = PreparedCommit(
                    change=change,
                    message=None,
                    error=error_message,
                )
                mark_prepared(idx, prepared_commit)
                if self._prepare_failure_limit_hit:
                    break

        def drain_once(timeout: float = 0.05) -> None:
            if not remaining:
                return
            done, not_done = wait(
                remaining,
                timeout=timeout,
                return_when=FIRST_COMPLETED,
            )
            if done:
                process_done(done)
            if self._prepare_failure_limit_hit:
                return
            if not_done:
                process_timeouts(not_done, time.time())

        idx_iter = iter(range(len(file_changes)))
        try:
            for idx in idx_iter:
                while len(remaining) >= workers and not self._prepare_failure_limit_hit:
                    drain_once()
                    if self._prepare_failure_limit_hit:
                        break
                    while (
                        retry_queue
                        and len(remaining) < workers
                        and not self._prepare_failure_limit_hit
                    ):
                        retry_idx = retry_queue.popleft()
                        submit(retry_idx)
                if self._prepare_failure_limit_hit:
                    break
                submit(idx)

            while (remaining or retry_queue) and not self._prepare_failure_limit_hit:
                if retry_queue and len(remaining) < workers:
                    retry_idx = retry_queue.popleft()
                    submit(retry_idx)
                    continue
                drain_once()
        finally:
            for fut in list(remaining):
                fut.cancel()
            executor.shutdown(wait=False, cancel_futures=True)

        if self._prepare_failure_limit_hit:
            outstanding = set(range(len(file_changes))) - completed
            for idx in sorted(outstanding):
                change = changes_by_idx[idx]
                prepared_commit = PreparedCommit(
                    change=change,
                    message=None,
                    error=self._prepare_failure_limit_message,
                )
                mark_prepared(idx, prepared_commit)

        flush_log()
        return prepared
    def _get_thread_commit_generator(self) -> CommitGenerator:
        """Return a per-thread CommitGenerator instance."""

        generator = getattr(self._thread_local, "generator", None)
        if generator is None:
            generator = CommitGenerator(
                repo_path=str(self.git_repo.repo_path),
                config=self._config,
                debug=self.debug,
            )
            self._thread_local.generator = generator
        return generator

    def _prepare_single_change(self, change: FileChange) -> PreparedCommit:
        generator = self._get_thread_commit_generator()
        if self.debug:
            snippet = change.diff_content.splitlines()[:20]
            preview = "\n".join(snippet)
            print(
                ("DEBUG: prepare.file path={} change_type={} diff_preview=\n{}").format(
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
            validated = generator.validate_and_fix_commit_message(commit_message)
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

    def _clear_progress_line(self) -> None:
        if not getattr(self, "_show_progress", False):
            return
        print("\r\033[K", end="", flush=True)

    def _refresh_progress_line(self) -> None:
        if not getattr(self, "_show_progress", False):
            return
        if not self._last_progress_stage:
            return
        snapshot = self._progress_snapshots.get(self._last_progress_stage)
        if snapshot:
            print(f"\r{snapshot}", end="", flush=True)

    def _build_progress_line(self, stage: str) -> str:
        snapshot = self._stats.snapshot()
        total = snapshot["total_files"]
        prepared = snapshot["prepared"]
        processed = snapshot["processed"]
        success = snapshot["successes"]
        failures = snapshot["failures"]
        rate = snapshot["rate"]

        stage_styles = {
            "prepare": ("🧠", CYAN),
            "commit": ("🚀", GREEN),
            "done": ("🏁", YELLOW),
        }
        icon, color = stage_styles.get(stage, ("🔄", CYAN))
        stage_label = stage.upper()

        return (
            f"{BOLD}{icon} kcmt{RESET} "
            f"{color}{stage_label:<7}{RESET} │ "
            f"{GREEN}{processed:>3}{RESET}/{total:>3} files │ "
            f"{CYAN}{prepared:>3}{RESET}/{total:>3} ready │ "
            f"{GREEN}✓ {success:>3}{RESET} │ "
            f"{RED}✗ {failures:>3}{RESET} │ "
            f"{DIM}{rate:5.2f} commits/s{RESET}   "
        )

    def _print_progress(self, stage: str) -> None:
        if not getattr(self, "_show_progress", False):
            return

        status_line = self._build_progress_line(stage)
        self._progress_snapshots[stage] = status_line
        self._last_progress_stage = stage
        print(f"\r{status_line}", end="", flush=True)

    def _finalize_progress(self) -> None:
        if not getattr(self, "_show_progress", False):
            return

        self._progress_snapshots["done"] = self._build_progress_line("done")
        print("\r\033[K", end="")

        ordered_stages = ["prepare", "commit", "done"]
        block_lines = [
            self._progress_snapshots[stage]
            for stage in ordered_stages
            if stage in self._progress_snapshots
        ]

        if block_lines:
            print("\n".join(block_lines))
        else:
            print()

        if self._commit_subjects:
            print()
            for subject in self._commit_subjects:
                print(f"{GREEN}{subject}{RESET}")

        print()

    def _print_commit_generated(self, file_path: str, commit_message: str) -> None:
        """Display the generated commit message for a file."""
        # Colors for consistency with CLI
        CYAN = "\033[96m"
        GREEN = "\033[92m"
        RESET = "\033[0m"

        self._clear_progress_line()

        lines = commit_message.splitlines()
        subject = lines[0] if lines else commit_message
        body = lines[1:] if len(lines) > 1 else []
        subject_display = subject.strip() if subject else commit_message.strip()
        if not subject_display:
            subject_display = "(empty)"

        print(f"\n{CYAN}Generated for {file_path}:{RESET}")
        print(f"{GREEN}{subject_display}{RESET}")

        if self.verbose or self.debug:
            if body:
                print("\n".join(body))
            print("-" * 50)

        self._refresh_progress_line()

    def _print_prepare_error(self, file_path: str, error: str) -> None:
        RED = "\033[91m"
        CYAN = "\033[96m"
        RESET = "\033[0m"

        self._clear_progress_line()
        print(f"\n{CYAN}Failed to prepare {file_path}:{RESET}")
        if self.verbose or self.debug:
            display = error
        else:
            lines = error.splitlines()
            display = lines[0] if lines else error
        print(f"{RED}{display}{RESET}")
        self._refresh_progress_line()

    def _log_prepared_result(self, prepared: PreparedCommit) -> None:
        if prepared.message:
            self._print_commit_generated(prepared.change.file_path, prepared.message)
        elif prepared.error:
            self._print_prepare_error(prepared.change.file_path, prepared.error)

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
                            change_type=self._determine_change_type(current_diff),
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
            line.startswith(marker) for line in diff_lines for marker in added_markers
        )
        deleted = any(
            line.startswith(marker) for line in diff_lines for marker in deleted_markers
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
                commit_hash = recent_commits[0].split()[0] if recent_commits else None
                result = CommitResult(
                    success=True,
                    commit_hash=commit_hash,
                    message=message,
                    file_path=file_path,
                )
                if message:
                    subject = message.splitlines()[0]
                    self._commit_subjects.append(subject)
                return result
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
            candidate = self.commit_generator.llm_client.generate_commit_message(
                "Fix commit message", prompt, "conventional"
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
            if m_type and ":" in header:
                msg_type = m_type.group(1)
                after_colon = header.split(":", 1)[1].strip()
                subject = after_colon
            else:
                # No valid type prefix; treat whole header as subject
                subject = header

        if not scope:
            scope = "core"

        # Remove trailing period from subject
        if subject.endswith("."):
            subject = subject[:-1]

        # Enforce length 50 chars on subject
        max_len = 50
        if len(subject) > max_len:
            cut = subject.rfind(" ", 0, max_len)
            if cut == -1 or cut < max_len * 0.6:
                cut = max_len - 1
            subject = subject[:cut].rstrip() + "…"

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
            summary_parts.append(f"Committed {len(successful_deletions)} deletion(s)")

        if file_commits:
            successful_commits = [r for r in file_commits if r.success]
            summary_parts.append(f"Committed {len(successful_commits)} file change(s)")

        if errors:
            summary_parts.append(f"Encountered {len(errors)} error(s)")

        total_commits = len([r for r in deletions + file_commits if r.success])

        if total_commits > 0:
            summary_parts.insert(0, f"Successfully completed {total_commits} commits")
        else:
            summary_parts.insert(0, "No commits were made")

        return ". ".join(summary_parts)
