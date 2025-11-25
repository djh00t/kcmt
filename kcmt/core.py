"""Core workflow logic for kcmt."""

from __future__ import annotations

import asyncio
import inspect
import os
import re
import threading
import time
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .commit import CommitGenerator
from .config import BATCH_TIMEOUT_MIN_SECONDS, Config, get_active_config
from .exceptions import GitError, KlingonCMTError, LLMError, ValidationError
from .git import GitRepo
from .providers.base import resolve_default_request_timeout

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


@dataclass
class WorkflowMetrics:
    diff_count: int = 0
    diff_total: float = 0.0
    queue_total: float = 0.0
    queue_samples: int = 0
    llm_count: int = 0
    llm_total: float = 0.0
    commit_total: float = 0.0

    def record_diff(self, elapsed: float) -> None:
        self.diff_count += 1
        self.diff_total += elapsed

    def record_queue(self, elapsed: float) -> None:
        self.queue_samples += 1
        self.queue_total += elapsed

    def record_llm(self, elapsed: float) -> None:
        self.llm_count += 1
        self.llm_total += elapsed

    def record_commit(self, elapsed: float) -> None:
        self.commit_total += elapsed

    def summary(self) -> str:
        parts = []
        if self.diff_count:
            parts.append(f"diffs={self.diff_count} ({self.diff_total:.2f}s)")
        if self.queue_samples:
            avg_queue = self.queue_total / max(1, self.queue_samples)
            parts.append(f"avg queue={avg_queue:.2f}s")
        if self.llm_count:
            avg_llm = self.llm_total / self.llm_count
            parts.append(f"LLM avg={avg_llm:.2f}s across {self.llm_count}")
        if self.commit_total:
            parts.append(f"commit total={self.commit_total:.2f}s")
        return "; ".join(parts)


class WorkflowStats:
    """Tracks workflow progress and renders real-time stats."""

    def __init__(self) -> None:
        self.total_files = 0
        self.diffs_built = 0
        self.requested = 0
        self.responded = 0
        self.prepared = 0
        self.processed = 0
        self.successes = 0
        self.failures = 0
        self._start = time.time()
        self._lock = threading.Lock()

    def set_total(self, total: int) -> None:
        with self._lock:
            self.total_files = total

    def mark_diff(self) -> None:
        with self._lock:
            self.diffs_built += 1

    def mark_prepared(self) -> None:
        with self._lock:
            self.prepared += 1

    def mark_request(self) -> None:
        with self._lock:
            self.requested += 1

    def mark_response(self) -> None:
        with self._lock:
            self.responded += 1

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
                "diffs_built": self.diffs_built,
                "requests": self.requested,
                "responses": self.responded,
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
        workers: Optional[int] = None,
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
        self._workers_override = workers
        self._thread_local = threading.local()
        self._thread_local.generator = self.commit_generator
        self._progress_snapshots: dict[str, str] = {}
        self._commit_subjects: list[str] = []
        self._last_progress_stage: Optional[str] = None
        self._prepare_failure_limit_hit = False
        self._prepare_failure_limit_message = ""
        self._metrics = WorkflowMetrics()
        self._progress_history: list[str] = []
        self._progress_header_shown = False
        self._progress_block_height = 0

    # ------------------------------------------------------------------
    # Progress rendering helpers
    # ------------------------------------------------------------------
    def _format_progress_row(
        self,
        *,
        icon: str,
        stage_label: str,
        diffs: object,
        requests: object,
        responses: object,
        prepared: object,
        total: object,
        success: object,
        failures: object,
        rate: object,
        colorize: bool = True,
    ) -> str:
        """Return a single progress row with aligned columns."""

        num_width = 3
        rate_width = 6

        def _num(val: object, width: int = num_width) -> str:
            try:
                return f"{int(val):>{width}d}"
            except Exception:
                return f"{str(val):>{width}}"

        if isinstance(rate, str):
            rate_part = f"{rate:>{rate_width}}"
        else:
            try:
                rate_part = f"{float(rate):>{rate_width}.2f}"
            except Exception:
                rate_part = f"{str(rate):>{rate_width}}"

        stage_color = CYAN if colorize else ""
        reset = RESET if colorize else ""

        return (
            f"{icon} kcmt "
            f"{stage_color}{stage_label:<7}{reset} │ "
            f"Δ {_num(diffs)} │ "
            f"req {_num(requests)} │ "
            f"res {_num(responses)} │ "
            f"ready {_num(prepared)}/{_num(total)} │ "
            f"✓ {_num(success)} │ "
            f"✗ {_num(failures)} │ "
            f"{DIM if colorize else ''}{rate_part} c/s{reset}"
        )

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

        self._metrics = WorkflowMetrics()
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
            if not self._show_progress:
                metrics_summary = self._metrics.summary()
                if metrics_summary:
                    print(f"{DIM}metrics: {metrics_summary}{RESET}")

        # Auto-push if enabled and we actually committed something
        any_success = any(r.success for r in results.get("file_commits", [])) or any(
            r.success for r in results.get("deletions_committed", [])
        )
        if any_success and getattr(self._config, "auto_push", False):
            try:
                self._progress_event("push-start")
                self.git_repo.push()
                results["pushed"] = True
                self._progress_event("push-done")
            except GitError as e:  # pragma: no cover - network dependent
                results.setdefault("errors", []).append(f"Auto-push failed: {e}")
                self._progress_event("push-error", detail=str(e))

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
        # Build diffs with a batched path for tracked modifications to avoid
        # spawning one subprocess per file. Untracked files are handled using
        # the original per-file fallback.
        file_changes: List[FileChange] = []
        collect_start = time.perf_counter()

        # Partition by porcelain status
        mod_paths: list[str] = []
        other_entries: list[tuple[str, str]] = []
        for status, file_path in non_deletion_files:
            trimmed = status.strip()
            if trimmed.startswith("??"):
                other_entries.append((status, file_path))
            elif "M" in trimmed:
                mod_paths.append(file_path)
            else:
                other_entries.append((status, file_path))

        # Batched HEAD diff for modified tracked files
        if mod_paths:
            try:
                diff_start = time.perf_counter()
                combined = self.git_repo.get_head_diff_for_paths(mod_paths)
                diff_elapsed = time.perf_counter() - diff_start
                # Record per-file average for metrics
                per_file = diff_elapsed / max(1, len(mod_paths))
                for _ in mod_paths:
                    self._metrics.record_diff(per_file)
                if combined.strip():
                    parsed_changes = self._parse_git_diff(combined)
                    by_path = {chg.file_path: chg for chg in parsed_changes}
                    for p in mod_paths:
                        chg = by_path.get(p)
                        if chg and chg.diff_content.strip():
                            file_changes.append(chg)
                        else:
                            # Fallback per-file if missing from batch output
                            try:
                                single = self.git_repo.get_worktree_diff_for_path(p)
                                if single.strip():
                                    file_changes.append(
                                        FileChange(
                                            file_path=p,
                                            change_type="M",
                                            diff_content=single,
                                        )
                                    )
                            except GitError as e:
                                results.append(
                                    CommitResult(
                                        success=False,
                                        error=f"Failed to capture diff for {p}: {e}",
                                        file_path=p,
                                    )
                                )
            except GitError:
                # If batched path fails, fall back to per-file for mods
                for p in mod_paths:
                    try:
                        start = time.perf_counter()
                        single = self.git_repo.get_worktree_diff_for_path(p)
                        self._metrics.record_diff(time.perf_counter() - start)
                        if single.strip():
                            file_changes.append(
                                FileChange(
                                    file_path=p,
                                    change_type="M",
                                    diff_content=single,
                                )
                            )
                    except GitError as ge:
                        results.append(
                            CommitResult(
                                success=False,
                                error=f"Failed to capture diff for {p}: {ge}",
                                file_path=p,
                            )
                        )

        # Handle untracked and other entries with the existing per-file path
        for status, file_path in other_entries:
            try:
                diff_start = time.perf_counter()
                single_diff = self.git_repo.get_worktree_diff_for_path(file_path)
                diff_elapsed = time.perf_counter() - diff_start
                self._metrics.record_diff(diff_elapsed)
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
        return asyncio.run(self._prepare_commit_messages_async(file_changes))

    async def _prepare_commit_messages_async(
        self, file_changes: List[FileChange]
    ) -> List[Tuple[int, PreparedCommit]]:
        cpu_hint = os.cpu_count() or 4
        # Environment/CLI override for concurrency
        env_workers = os.environ.get("KCMT_PREPARE_WORKERS")
        try:
            env_workers_val = int(env_workers) if env_workers else None
        except ValueError:
            env_workers_val = None
        desired_workers = self._workers_override or env_workers_val
        max_default = max(1, min(len(file_changes), 8, cpu_hint))
        workers = (
            max(1, min(len(file_changes), desired_workers))
            if desired_workers
            else max_default
        )
        print(
            f"{MAGENTA}⚙️  Spinning up {workers} worker(s) for "
            f"{len(file_changes)} file(s){RESET}"
        )

        per_file_timeout_env = os.environ.get("KCMT_PREPARE_PER_FILE_TIMEOUT")
        provider_default_timeout = resolve_default_request_timeout(
            getattr(self._config, "provider", None)
        )
        try:
            per_file_timeout = (
                float(per_file_timeout_env)
                if per_file_timeout_env
                else provider_default_timeout
            )
        except ValueError:
            per_file_timeout = provider_default_timeout
        # Batch workflows can take longer; align per-file timeout with batch timeout
        if getattr(self._config, "use_batch", False):
            batch_timeout_cfg = getattr(self._config, "batch_timeout_seconds", None)
            try:
                batch_timeout_val = float(batch_timeout_cfg) if batch_timeout_cfg else None
            except (TypeError, ValueError):
                batch_timeout_val = None
            if batch_timeout_val:
                per_file_timeout = max(per_file_timeout, batch_timeout_val, BATCH_TIMEOUT_MIN_SECONDS)
            else:
                per_file_timeout = max(per_file_timeout, BATCH_TIMEOUT_MIN_SECONDS)

        timeout_retry_limit = 3
        timeout_attempt_limit = timeout_retry_limit + 1
        timeout_state = {"value": per_file_timeout}

        semaphore = asyncio.Semaphore(workers)
        abort_event = asyncio.Event()

        prepared: list[tuple[int, PreparedCommit]] = []
        prepared_by_idx: dict[int, PreparedCommit] = {}
        log_queue: dict[int, PreparedCommit] = {}
        next_log_index = 0
        completed: set[int] = set()

        failure_limit = 25
        failure_count = 0
        self._prepare_failure_limit_hit = False
        self._prepare_failure_limit_message = ""

        adaptive_timeouts_enabled = not (
            os.environ.get("KCMT_TEST_DISABLE_OPENAI")
            or os.environ.get("PYTEST_CURRENT_TEST")
        )

        async def worker(idx: int, change: FileChange) -> tuple[int, PreparedCommit]:
            queue_start = time.perf_counter()
            async with semaphore:
                queue_elapsed = time.perf_counter() - queue_start
                self._metrics.record_queue(queue_elapsed)

                if abort_event.is_set():
                    return idx, PreparedCommit(
                        change=change,
                        message=None,
                        error=self._prepare_failure_limit_message
                        or "Stopped prepare phase after failure limit.",
                    )

                timeout_value = timeout_state["value"]
                adaptive_ceiling = max(per_file_timeout * 4.0, 30.0)
                max_timeout = (
                    adaptive_ceiling if adaptive_timeouts_enabled else timeout_value
                )
                attempts = 0
                prepare_fn = self._prepare_single_change
                prepare_signature = inspect.signature(prepare_fn)
                supports_timeout = "request_timeout" in prepare_signature.parameters
                while attempts < timeout_attempt_limit:
                    start = time.perf_counter()
                    try:
                        if getattr(self._config, "use_batch", False):
                            prepared_commit = await asyncio.wait_for(
                                self._prepare_single_change_async(
                                    change,
                                    request_timeout=timeout_value,
                                ),
                                timeout=timeout_value,
                            )
                        elif supports_timeout:
                            prepared_commit = await asyncio.wait_for(
                                asyncio.to_thread(
                                    prepare_fn,
                                    change,
                                    timeout_value,
                                ),
                                timeout=timeout_value,
                            )
                        else:
                            prepared_commit = await asyncio.wait_for(
                                asyncio.to_thread(
                                    prepare_fn,
                                    change,
                                ),
                                timeout=timeout_value,
                            )
                        llm_elapsed = time.perf_counter() - start
                        if prepared_commit.message:
                            self._metrics.record_llm(llm_elapsed)
                            avg_llm = self._metrics.llm_total / max(
                                1, self._metrics.llm_count
                            )
                            timeout_state["value"] = max(
                                per_file_timeout,
                                min(per_file_timeout * 4, avg_llm * 2),
                            )
                        return idx, prepared_commit
                    except asyncio.TimeoutError:
                        attempts += 1
                        if adaptive_timeouts_enabled:
                            timeout_state["value"] = min(
                                max_timeout,
                                max(timeout_state["value"], timeout_value),
                            )
                        if attempts >= timeout_attempt_limit:
                            error_message = (
                                "Timeout after "
                                f"{timeout_value:.1f}s waiting for "
                                f"{change.file_path} "
                                f"(attempt {attempts}/{timeout_attempt_limit})"
                            )
                            return idx, PreparedCommit(
                                change=change,
                                message=None,
                                error=error_message,
                            )
                        if adaptive_timeouts_enabled:
                            timeout_value = min(timeout_value * 1.5, max_timeout)
                            timeout_state["value"] = min(
                                max_timeout,
                                max(timeout_state["value"], timeout_value),
                            )
                    except Exception as exc:  # noqa: BLE001
                        return idx, PreparedCommit(
                            change=change,
                            message=None,
                            error=f"Error preparing {change.file_path}: {exc}",
                        )
                error_message = (
                    "Timeout after "
                    f"{timeout_value:.1f}s waiting for "
                    f"{change.file_path} "
                    f"(attempt {timeout_attempt_limit}/{timeout_attempt_limit})"
                )
                return idx, PreparedCommit(
                    change=change,
                    message=None,
                    error=error_message,
                )

        def flush_log() -> None:
            nonlocal next_log_index
            while next_log_index in log_queue:
                entry = log_queue.pop(next_log_index)
                self._log_prepared_result(entry)
                next_log_index += 1

        def mark_prepared(idx: int, prepared_commit: PreparedCommit) -> None:
            nonlocal failure_count
            existing = prepared_by_idx.get(idx)
            if existing and existing.message and prepared_commit.error:
                return
            if existing and existing.error and prepared_commit.message:
                prepared_by_idx[idx] = prepared_commit
                log_queue[idx] = prepared_commit
                return
            if idx in completed and existing:
                return
            prepared.append((idx, prepared_commit))
            prepared_by_idx[idx] = prepared_commit
            self._stats.mark_prepared()
            self._print_progress(stage="prepare")
            log_queue[idx] = prepared_commit
            completed.add(idx)

            if prepared_commit.error:
                failure_count += 1
                if failure_count >= failure_limit and not abort_event.is_set():
                    self._prepare_failure_limit_hit = True
                    self._prepare_failure_limit_message = (
                        f"Stopped prepare phase after {failure_count} failures "
                        f"(failure limit {failure_limit})."
                    )
                    abort_event.set()
                if (
                    self._prepare_failure_limit_hit
                    and self._prepare_failure_limit_message
                    and failure_count > failure_limit
                ):
                    prepared_commit.error = self._prepare_failure_limit_message

            flush_log()

        tasks = {
            idx: asyncio.create_task(worker(idx, change))
            for idx, change in enumerate(file_changes)
        }

        try:
            while tasks:
                done, pending = await asyncio.wait(
                    tasks.values(), return_when=asyncio.FIRST_COMPLETED
                )
                for finished in done:
                    idx, prepared_commit = await finished
                    tasks.pop(idx, None)
                    mark_prepared(idx, prepared_commit)

                if abort_event.is_set():
                    for pending_task in pending:
                        pending_task.cancel()
                    await asyncio.gather(*pending, return_exceptions=True)
                    for idx in set(range(len(file_changes))) - completed:
                        change = file_changes[idx]
                        prepared_commit = PreparedCommit(
                            change=change,
                            message=None,
                            error=self._prepare_failure_limit_message,
                        )
                        mark_prepared(idx, prepared_commit)
                    break

        finally:
            for task in tasks.values():
                task.cancel()
            if tasks:
                await asyncio.gather(*tasks.values(), return_exceptions=True)

        if self._prepare_failure_limit_hit and self._prepare_failure_limit_message:
            self._clear_progress_line()
            print(f"\n{RED}{self._prepare_failure_limit_message}{RESET}")
            self._refresh_progress_line()

        flush_log()
        ordered_prepared: list[tuple[int, PreparedCommit]] = []
        for idx in sorted(prepared_by_idx):
            ordered_prepared.append((idx, prepared_by_idx[idx]))
        if not ordered_prepared:
            ordered_prepared = sorted(prepared, key=lambda item: item[0])
        return ordered_prepared

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

    def _prepare_single_change(
        self, change: FileChange, request_timeout: float | None = None
    ) -> PreparedCommit:
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

        self._progress_event("diff-ready", file=change.file_path)
        self._stats.mark_diff()

        def _llm_progress(status: str) -> None:
            key = status.strip().lower()
            if key in {"request-sent", "request sent"}:
                self._stats.mark_request()
                self._progress_event("request-sent", file=change.file_path)
            elif key in {"response-received", "response received"}:
                self._stats.mark_response()
                self._progress_event("response", file=change.file_path)
            else:
                self._progress_event(
                    "llm",
                    file=change.file_path,
                    detail=status,
                )

        suggest_fn = generator.suggest_commit_message
        try:
            call_kwargs: dict[str, object] = {
                "context": f"File: {change.file_path}",
                "style": "conventional",
            }
            try:
                sig = inspect.signature(suggest_fn)
            except (TypeError, ValueError):
                sig = None
            if request_timeout is not None and sig and "request_timeout" in sig.parameters:
                call_kwargs["request_timeout"] = request_timeout
            if sig and "progress_callback" in sig.parameters:
                call_kwargs["progress_callback"] = _llm_progress
            commit_message = suggest_fn(change.diff_content, **call_kwargs)
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

    async def _prepare_single_change_async(
        self, change: FileChange, request_timeout: float | None = None
    ) -> PreparedCommit:
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

        self._progress_event("diff-ready", file=change.file_path)

        def _llm_progress(status: str) -> None:
            key = status.strip().lower()
            if key in {"request-sent", "request sent"}:
                self._progress_event("request-sent", file=change.file_path)
            elif key in {"response-received", "response received"}:
                self._progress_event("response", file=change.file_path)
            else:
                self._progress_event(
                    "llm",
                    file=change.file_path,
                    detail=status,
                )

        try:
            commit_message = await generator.suggest_commit_message_async(
                change.diff_content,
                context=f"File: {change.file_path}",
                style="conventional",
                request_timeout=request_timeout,
                progress_callback=_llm_progress,
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
        self._render_progress_block()

    def _build_progress_line(self, stage: str) -> str:
        snapshot = self._stats.snapshot()
        total = snapshot["total_files"]
        diffs = snapshot.get("diffs_built", 0)
        requests = snapshot.get("requests", 0)
        responses = snapshot.get("responses", 0)
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

        return self._format_progress_row(
            icon=f"{BOLD}{icon}{RESET}",
            stage_label=stage_label,
            diffs=diffs,
            requests=requests,
            responses=responses,
            prepared=prepared,
            total=total,
            success=success,
            failures=failures,
            rate=rate,
            colorize=True,
        )

    def _render_progress_block(self) -> None:
        """Render the progress table pinned to the bottom of the output."""

        if not getattr(self, "_show_progress", False):
            return

        header = self._format_progress_row(
            icon=" ",
            stage_label="stage",
            diffs="dif",
            requests="req",
            responses="res",
            prepared="rdy",
            total="tot",
            success="ok",
            failures="err",
            rate="rate",
            colorize=False,
        )
        separator = f"{DIM}{'─' * len(header)}{RESET}"
        ordered_stages = ["prepare", "commit", "done"]
        block_lines = [
            self._progress_snapshots.get(stage) or self._build_progress_line(stage)
            for stage in ordered_stages
        ]
        lines = [header, separator, *block_lines]
        while len(lines) < 6:
            lines.append("")

        if self._progress_block_height:
            sys.stdout.write(f"\x1b[{self._progress_block_height}F")
        for idx, line in enumerate(lines):
            # Clear line then write content; add newline except last line.
            sys.stdout.write("\r\033[K")
            sys.stdout.write(line)
            if idx < len(lines) - 1:
                sys.stdout.write("\n")
        sys.stdout.flush()
        self._progress_block_height = len(lines)

    def _print_progress(self, stage: str) -> None:
        if not getattr(self, "_show_progress", False):
            return

        status_line = self._build_progress_line(stage)
        self._progress_snapshots[stage] = status_line
        self._last_progress_stage = stage
        self._render_progress_block()

    def _finalize_progress(self) -> None:
        if not getattr(self, "_show_progress", False):
            return

        self._progress_snapshots["done"] = self._build_progress_line("done")
        self._render_progress_block()

        # Leave a newline after the block and reset state for the next run.
        print()
        self._progress_header_shown = False
        self._progress_block_height = 0

        if self._commit_subjects:
            print()
            for subject in self._commit_subjects:
                print(f"{GREEN}{subject}{RESET}")

        metrics_summary = self._metrics.summary()
        if metrics_summary:
            print(f"{DIM}metrics: {metrics_summary}{RESET}")

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

        self._render_progress_block()

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
        self._render_progress_block()

    def _log_prepared_result(self, prepared: PreparedCommit) -> None:
        if prepared.message:
            self._print_commit_generated(prepared.change.file_path, prepared.message)
        elif prepared.error:
            self._print_prepare_error(prepared.change.file_path, prepared.error)

    def _format_progress_message(self, kind: str, info: dict[str, object]) -> str | None:
        file_path = str(info.get("file") or "")
        detail = str(info.get("detail") or "")
        provider = getattr(self._config, "provider", "")
        model = getattr(self._config, "model", "")

        if kind == "diff-ready":
            return f"🧠 diff ready: {file_path}"
        if kind == "request-sent":
            target = f"{provider}/{model}".strip("/")
            return f"📤 sent {file_path} → {target}"
        if kind == "response":
            return f"📥 response received: {file_path}"
        if kind == "commit-start":
            return f"📝 committing {file_path}"
        if kind == "commit-done":
            return f"✅ committed {file_path}"
        if kind == "commit-error":
            return f"⚠️ commit failed for {file_path}: {detail}"
        if kind == "push-start":
            return "⏫ pushing to remote…"
        if kind == "push-done":
            return "📡 push complete"
        if kind == "push-error":
            return f"⚠️ push failed: {detail}"
        if kind == "llm":
            label = detail or "LLM status"
            return f"🤖 {label}"
        return None

    def _progress_event(self, kind: str, **info: object) -> None:
        message = self._format_progress_message(kind, info)
        if not message:
            return
        if getattr(self, "_show_progress", False):
            self._clear_progress_line()
            print(message)
            self._refresh_progress_line()
        self._progress_history.append(message)

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

        commit_start = time.perf_counter()
        self._progress_event("commit-start", file=change.file_path)
        result = self._attempt_commit(
            validated_message,
            max_retries=self.max_retries,
            file_path=change.file_path,
        )
        self._metrics.record_commit(time.perf_counter() - commit_start)
        if result.success:
            self._progress_event("commit-done", file=change.file_path)
        else:
            self._progress_event(
                "commit-error",
                file=change.file_path,
                detail=str(result.error or ""),
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

    def stats_snapshot(self) -> Dict[str, float]:
        """Expose workflow statistics collected during execution."""

        return self._stats.snapshot()

    def commit_subjects(self) -> List[str]:
        """Return the list of commit subjects generated this run."""

        return list(self._commit_subjects)
