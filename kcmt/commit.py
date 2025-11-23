"""Commit message generation logic for kcmt."""

import inspect
import os
import re
import sys
import threading
import time
from typing import Any, Callable, Optional
from typing import Callable as _Callable

from .config import DEFAULT_MODELS, Config, get_active_config
from .exceptions import LLMError, ValidationError
from .git import GitRepo
from .llm import LLMClient


def _supports_request_timeout(callable_obj: _Callable[..., Any]) -> bool:
    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return False
    return "request_timeout" in signature.parameters


def _supports_param(callable_obj: _Callable[..., Any], name: str) -> bool:
    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return False
    return name in signature.parameters


def _should_use_spinner() -> bool:
    if "PYTEST_CURRENT_TEST" in os.environ:
        return False
    if os.environ.get("KCMT_USE_INK") or os.environ.get("KCMT_BACKEND_MODULE"):
        return False
    flag = os.environ.get("KCMT_NO_SPINNER", "").lower()
    if flag in {"1", "true", "yes", "on"}:
        return False
    return sys.stderr.isatty()


class _Spinner:
    """Lightweight stdout spinner to signal batch progress."""

    def __init__(self, label: str) -> None:
        self.label = label
        self._enabled = _should_use_spinner()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._frames = ["|", "/", "-", "\\"]
        self._frame_index = 0

    def start(self) -> None:
        if not self._enabled or self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def update(self, text: str) -> None:
        if not self._enabled:
            return
        text = text.strip()
        if len(text) > 80:
            text = text[:77] + "..."
        self.label = text or self.label

    def stop(self) -> None:
        if not self._enabled:
            return
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=0.2)
            self._thread = None
        # Clear line
        sys.stderr.write("\r" + " " * (len(self.label) + 2) + "\r")
        sys.stderr.flush()

    def _run(self) -> None:
        while not self._stop.is_set():
            frame = self._frames[self._frame_index % len(self._frames)]
            self._frame_index += 1
            sys.stderr.write(f"\r{frame} {self.label}")
            sys.stderr.flush()
            time.sleep(0.1)


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
        self._memo: dict[str, str] = {}

    def _config_for_preference(self, provider: str, model: str) -> Config | None:
        if provider not in DEFAULT_MODELS:
            return None
        defaults = DEFAULT_MODELS[provider]
        providers_map = getattr(self._config, "providers", {}) or {}
        entry = providers_map.get(provider, {}) or {}
        endpoint = entry.get("endpoint") or defaults.get("endpoint") or ""
        api_key_env = entry.get("api_key_env") or defaults.get("api_key_env") or ""
        return Config(
            provider=provider,
            model=str(model),
            llm_endpoint=str(endpoint),
            api_key_env=str(api_key_env),
            git_repo_path=self._config.git_repo_path,
            max_commit_length=self._config.max_commit_length,
            auto_push=self._config.auto_push,
            providers=self._config.providers,
            model_priority=self._config.model_priority,
        )

    def _iter_priority_clients(self) -> list[tuple[Config, LLMClient]]:
        chain: list[tuple[Config, LLMClient]] = [(self._config, self.llm_client)]
        priority = getattr(self._config, "model_priority", []) or []
        for pref in priority[1:]:
            prov = pref.get("provider") if isinstance(pref, dict) else None
            model = pref.get("model") if isinstance(pref, dict) else None
            if not prov or not model:
                continue
            cfg = self._config_for_preference(str(prov), str(model))
            if not cfg or not cfg.resolve_api_key():
                continue
            try:
                client = LLMClient(cfg, debug=self.debug)
            except LLMError:
                continue
            chain.append((cfg, client))
        return chain

    def _attempt_with_client(
        self,
        client: LLMClient,
        diff: str,
        context: str,
        style: str,
        request_timeout: float | None,
        progress_callback: Callable[[str], None] | None,
    ) -> str:
        last_error: Exception | None = None
        max_attempts = 3
        spinner: _Spinner | None = None
        progress_cb = progress_callback
        if getattr(client, "uses_batch", False):
            spinner = _Spinner("Submitting OpenAI batch…")
            spinner.start()
            if progress_callback:
                def _combo(msg: str) -> None:
                    label = "Sending…" if msg == "request-sent" else "Waiting…" if msg == "response-received" else msg
                    spinner.update(label)
                    progress_callback(msg)

                progress_cb = _combo
            else:
                progress_cb = spinner.update
        try:
            for attempt in range(1, max_attempts + 1):
                if self.debug:
                    truncated_ctx = (
                        context[:120] + "…" if len(context) > 120 else context
                    )
                    print(
                        "DEBUG: commit.attempt {} diff_len={} context='{}'".format(
                            attempt, len(diff), truncated_ctx
                        )
                    )
                try:
                    generate_fn = client.generate_commit_message
                    call_kwargs: dict[str, object] = {}
                    if request_timeout is not None and _supports_request_timeout(
                        generate_fn
                    ):
                        call_kwargs["request_timeout"] = request_timeout
                    if progress_cb is not None and _supports_param(
                        generate_fn, "progress_callback"
                    ):
                        call_kwargs["progress_callback"] = progress_cb
                    if progress_cb is not None:
                        progress_cb("request-sent")
                    msg = generate_fn(diff, context, style, **call_kwargs)
                    if not msg or not msg.strip():
                        raise LLMError("LLM returned empty response")
                    if progress_cb is not None:
                        progress_cb("response-received")
                    if not self.validate_conventional_commit(msg):
                        if self.debug:
                            invalid_header = msg.splitlines()[0][:120]
                            print(
                                (
                                    "DEBUG: commit.invalid_format attempt={} msg='{}'"
                                ).format(attempt, invalid_header)
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
        finally:
            if spinner:
                spinner.stop()
        raise LLMError(
            (
                "LLM unavailable or invalid output after {} attempts; commit aborted"
            ).format(3)
        ) from last_error

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
        progress_callback: Callable[[str], None] | None = None,
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

        # Optional within-run memoization to avoid duplicate LLM calls.
        disable_memo = str(os.environ.get("KCMT_DISABLE_MEMO", "")).lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        digest: str | None = None
        if not disable_memo:
            try:
                import hashlib

                digest = hashlib.sha256(diff.encode("utf-8", "ignore")).hexdigest()
            except Exception:  # pragma: no cover - memo is best-effort
                digest = None

        # Optional local fast path for tiny diffs when explicitly enabled.
        fast_local = str(
            os.environ.get("KCMT_FAST_LOCAL_FOR_SMALL_DIFFS", "")
        ).lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        if fast_local:
            changed_lines = 0
            for line in diff.splitlines():
                if not line:
                    continue
                if line.startswith("+++") or line.startswith("---"):
                    continue
                if line.startswith("+") or line.startswith("-"):
                    changed_lines += 1
            if changed_lines <= 3:
                file_path = ""
                if context and "File:" in context:
                    file_path = context.split("File:", 1)[1].strip()
                subject = self._synthesize_small_diff_subject(file_path)
                if self.validate_conventional_commit(subject):
                    return subject

        last_error: LLMError | None = None
        for idx, (cfg, client) in enumerate(self._iter_priority_clients()):
            current_key = None
            if digest:
                current_key = f"{cfg.provider}:{cfg.model}:{digest}"
                cached = self._memo.get(current_key)
                if cached:
                    if self.debug:
                        print("DEBUG: commit.memo hit")
                    return cached
            if idx > 0 and self.debug:
                print(
                    "DEBUG: attempting fallback provider '{}' model '{}'".format(
                        cfg.provider, cfg.model
                    )
                )
            try:
                result = self._attempt_with_client(
                    client, diff, context, style, request_timeout, progress_callback
                )
                if current_key and result:
                    self._memo[current_key] = result
                return result
            except LLMError as err:
                last_error = err
                continue

        if last_error:
            raise last_error
        raise LLMError("LLM unavailable; no providers succeeded")

    def _synthesize_small_diff_subject(self, file_path: str) -> str:
        """Heuristic conventional subject for tiny diffs (opt-in).

        - type: docs for markdown/text, otherwise chore
        - scope: directory name or core
        - description: "update <basename>"
        """
        import os as _os

        base = _os.path.basename(file_path) if file_path else "file"
        name, ext = _os.path.splitext(base)
        scope = _os.path.basename(_os.path.dirname(file_path)) or "core"
        scope = re.sub(r"[^a-zA-Z0-9_-]", "-", scope) or "core"
        ctype = "docs" if ext.lower() in {".md", ".rst", ".txt"} else "chore"
        desc = f"update {base}" if base else "update"
        # Enforce 50-char subject without trailing period
        subject = f"{ctype}({scope}): {desc}".rstrip(".")
        if len(subject) > 50:
            cut = subject.rfind(" ", 0, 50)
            if cut == -1 or cut < 25:
                cut = 49
            subject = subject[:cut].rstrip() + "…"
        return subject

    async def suggest_commit_message_async(
        self,
        diff: str,
        context: str = "",
        style: str = "conventional",
        request_timeout: float | None = None,
        progress_callback: Callable[[str], None] | None = None,
    ) -> str:
        if not diff or not diff.strip():
            raise ValidationError("Diff content cannot be empty.")

        # Primary attempt (async)
        async def _attempt_async(client: LLMClient) -> str:
            last_error: Exception | None = None
            max_attempts = 3
            for attempt in range(1, max_attempts + 1):
                if self.debug:
                    truncated_ctx = (
                        context[:120] + "…" if len(context) > 120 else context
                    )
                    print(
                        "DEBUG: commit.attempt {} diff_len={} context='{}'".format(
                            attempt, len(diff), truncated_ctx
                        )
                    )
                try:
                    generate_fn = client.generate_commit_message_async
                    call_kwargs: dict[str, object] = {}
                    if request_timeout is not None and _supports_request_timeout(
                        generate_fn
                    ):
                        call_kwargs["request_timeout"] = request_timeout
                    if progress_callback and _supports_param(
                        generate_fn, "progress_callback"
                    ):
                        call_kwargs["progress_callback"] = progress_callback
                    msg = await generate_fn(diff, context, style, **call_kwargs)
                    if not msg or not msg.strip():
                        raise LLMError("LLM returned empty response")
                    if not self.validate_conventional_commit(msg):
                        if self.debug:
                            invalid_header = msg.splitlines()[0][:120]
                            print(
                                (
                                    "DEBUG: commit.invalid_format attempt={} msg='{}'"
                                ).format(attempt, invalid_header)
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
                ).format(3)
            ) from last_error

        last_error: LLMError | None = None
        for idx, (cfg, client) in enumerate(self._iter_priority_clients()):
            if idx > 0 and self.debug:
                print(
                    "DEBUG: attempting fallback provider '{}' model '{}' (async)".format(
                        cfg.provider, cfg.model
                    )
                )
            try:
                return await _attempt_async(client)
            except LLMError as err:
                last_error = err
                continue

        if last_error:
            raise last_error
        raise LLMError("LLM unavailable; no providers succeeded")

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
