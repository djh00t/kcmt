"""LLM integration for kcmt.

Enhancements (2025-09-22):
 - Added `KCMT_LLM_REQUEST_TIMEOUT` env var for per-request HTTP timeout.
 - Added robust output sanitization `_sanitize_commit_output` to coerce
     verbose / multi-line OpenAI style responses into a valid conventional
     commit header + optional body (emulating successful xai behaviour).
 - Normalizes quotes, code fences, markdown artifacts; extracts the first
     plausible header and raises when a conventional subject cannot be
     recovered.
"""

from __future__ import annotations

import os
import re
from typing import Any, Optional, cast

from ._optional import OpenAIModule, import_openai
from .config import Config, get_active_config
from .exceptions import LLMError
from .providers.anthropic_driver import AnthropicDriver
from .providers.base import BaseDriver
from .providers.openai_driver import OpenAIDriver
from .providers.xai_driver import XAIDriver

# Compatibility shim for older tests that expect kcmt.llm.OpenAI
# to be available for monkeypatching. We avoid importing openai at
# module import time unless necessary.
_openai: OpenAIModule | None = import_openai()


class OpenAI:  # noqa: D401
    """Compatibility wrapper exposed for legacy tests."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:  # noqa: D401, ARG002
        if _openai is not None:
            client = _openai.OpenAI(*args, **kwargs)
            self._client = client
            self.chat = client.chat
        else:
            # Minimal placeholder; tests will monkeypatch this symbol
            self.chat = type("_Chat", (), {"completions": object()})()


class AsyncOpenAI:  # noqa: D401
    """Compatibility wrapper for async OpenAI usage in tests."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:  # noqa: D401, ARG002
        if _openai is not None:
            client_factory = getattr(_openai, "AsyncOpenAI", None)
            if client_factory is None:  # pragma: no cover - older SDKs
                raise RuntimeError("AsyncOpenAI not available in openai package")
            self._client = client_factory(*args, **kwargs)
            self.chat = self._client.chat
            self.responses = getattr(self._client, "responses", None)
        else:
            raise RuntimeError("openai package not available")


# Removed direct OpenAI/httpx usage here (delegated to drivers)


class LLMClient:
    """Provider-aware client for generating commit messages."""

    def __init__(self, config: Optional[Config] = None, debug: bool = False) -> None:
        self.debug = debug
        if debug:
            print(f"DEBUG: LLMClient initialized with debug={debug}")
        self.config = config or get_active_config()
        self.provider = self.config.provider
        self.model = self.config.model
        self.api_key = self.config.resolve_api_key()

        if not self.api_key:
            # Allow tests / CI (non-interactive) to proceed with dummy key
            if "PYTEST_CURRENT_TEST" in os.environ:
                self.api_key = "DUMMY_TEST_KEY"
                if debug:
                    print("DEBUG: Using dummy key for {} (tests)".format(self.provider))
            else:
                raise LLMError(
                    "Environment variable '"
                    f"{self.config.api_key_env}"
                    "' is not set or empty."
                )

        # Per-request timeout (seconds) configurable; default 5s.
        timeout_env = os.environ.get("KCMT_LLM_REQUEST_TIMEOUT")
        try:
            self._request_timeout = float(timeout_env) if timeout_env else 5.0
        except ValueError:
            self._request_timeout = 5.0

        # Provider driver setup (strategy pattern)
        self._driver: BaseDriver
        if self.provider == "anthropic":
            self._mode = "anthropic"
            self._driver = AnthropicDriver(self.config, debug=debug)
        elif self.provider in {"openai", "github"}:
            # Treat github models as OpenAI-compatible
            self._mode = "openai"
            self._driver = OpenAIDriver(self.config, debug=debug)
        elif self.provider == "xai":
            self._mode = "openai"
            self._driver = XAIDriver(self.config, debug=debug)
        else:
            raise LLMError(f"Unsupported provider: {self.provider}")

        # Retain legacy flags (still used in prompt shaping) even with driver
        if self._mode == "openai":
            self._disable_reasoning = os.environ.get(
                "KCMT_OPENAI_DISABLE_REASONING", "1"
            ).lower() in {"1", "true", "yes", "on"}
            self._minimal_prompt = os.environ.get(
                "KCMT_OPENAI_MINIMAL_PROMPT", "1"
            ).lower() in {"1", "true", "yes", "on"}
            max_tokens_env = os.environ.get("KCMT_OPENAI_MAX_TOKENS")
            try:
                self._max_completion_tokens = (
                    int(max_tokens_env) if max_tokens_env else 512
                )
            except ValueError:
                self._max_completion_tokens = 512

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def generate_commit_message(
        self,
        diff: str,
        context: str = "",
        style: str = "conventional",
        request_timeout: float | None = None,
    ) -> str:
        cleaned_diff, prompt = self._prepare_prompt(diff, context, style)
        raw_message = self._call_provider(prompt, request_timeout=request_timeout)
        try:
            return self._postprocess_message(raw_message, cleaned_diff, context)
        except LLMError as e:
            if (
                "missing conventional commit header" in str(e)
                and self._mode == "openai"
                and not getattr(self, "_minimal_prompt", False)
            ):
                if self.debug:
                    print(
                        "DEBUG: sanitize.retry -> enabling minimal prompt "
                        "after header parse failure"
                    )
                if not self.model.startswith("gpt-5"):
                    self._minimal_prompt = True
                elif self.debug:
                    print("DEBUG: minimal_prompt suppressed (gpt-5 model)")
                retry_raw = self._call_provider(
                    prompt, request_timeout=request_timeout
                )
                return self._postprocess_message(retry_raw, cleaned_diff, context)
            raise

    async def generate_commit_message_async(
        self,
        diff: str,
        context: str = "",
        style: str = "conventional",
        request_timeout: float | None = None,
    ) -> str:
        cleaned_diff, prompt = self._prepare_prompt(diff, context, style)
        raw_message = await self._call_provider_async(
            prompt, request_timeout=request_timeout
        )
        try:
            return self._postprocess_message(raw_message, cleaned_diff, context)
        except LLMError as e:
            if (
                "missing conventional commit header" in str(e)
                and self._mode == "openai"
                and not getattr(self, "_minimal_prompt", False)
            ):
                if self.debug:
                    print(
                        "DEBUG: sanitize.retry(async) -> enabling minimal prompt "
                        "after header parse failure"
                    )
                if not self.model.startswith("gpt-5"):
                    self._minimal_prompt = True
                elif self.debug:
                    print("DEBUG: minimal_prompt suppressed (gpt-5 model)")
                retry_raw = await self._call_provider_async(
                    prompt, request_timeout=request_timeout
                )
                return self._postprocess_message(retry_raw, cleaned_diff, context)
            raise


    def _prepare_prompt(
        self, diff: str, context: str, style: str
    ) -> tuple[str, str]:
        if self.debug:
            print("DEBUG: generate_commit_message called")
            print(f"  Diff length: {len(diff)} characters")
            print(f"  Context: {context}")
            print(f"  Provider: {self.provider}")

        file_path_hint = ""
        if context and "File:" in context:
            file_path_hint = context.split("File:", 1)[1].strip()
        diff_for_prompt = diff
        binary_summary = None
        if self._is_binary_diff(diff) and not self._looks_like_text_file(file_path_hint):
            if self.debug:
                print("DEBUG: Detected binary diff; summarising for LLM prompt")
            snippet = diff[:400].strip()
            binary_summary = (
                "Binary diff detected.\n"
                f"File hint: {file_path_hint or 'unknown'}\n"
                "Git reported: " + (snippet or "<no additional details>")
            )

        if len(diff_for_prompt) > 12000:
            head = diff_for_prompt[:8000]
            tail = diff_for_prompt[-2000:]
            diff_for_prompt = head + "\n...\n" + tail

        cleaned_diff = self._clean_diff_for_llm(diff_for_prompt)
        if binary_summary:
            cleaned_diff = f"{binary_summary}\n\n{cleaned_diff}".strip()
        if self.debug:
            print(f"DEBUG: Cleaned diff length: {len(cleaned_diff)} characters")

        prompt = self._build_prompt(cleaned_diff, context, style)
        return cleaned_diff, prompt

    def _postprocess_message(
        self, raw_message: str, cleaned_diff: str, context: str
    ) -> str:
        if not raw_message:
            raise LLMError("Empty response from LLM")
        raw_message = raw_message.strip()
        sanitized = self._sanitize_commit_output(raw_message, context)

        try:
            changed_lines = self._count_changed_lines(cleaned_diff)
        except (ValueError, TypeError):  # pragma: no cover - defensive
            changed_lines = 0
        if changed_lines >= 10 and "\n" not in sanitized.strip():
            if self.debug:
                print(
                    "DEBUG: enrichment.trigger changed_lines={}".format(changed_lines)
                )
                print("DEBUG: enrichment.header '{}...'".format(sanitized[:60]))
            enriched = self._enrich_with_body(
                header=sanitized.splitlines()[0],
                diff=cleaned_diff,
                context=context,
            )
            if (
                enriched
                and enriched.strip()
                and enriched.splitlines()[0].startswith(sanitized.splitlines()[0])
            ):
                if self.debug:
                    print("DEBUG: enrichment.success length={}".format(len(enriched)))
                sanitized = enriched
            elif self.debug:
                print("DEBUG: enrichment.skip (no improvement)")
        if self.debug:
            raw_first = raw_message.splitlines()[0] if raw_message else ""
            san_first = sanitized.splitlines()[0] if sanitized else ""
            if raw_first != san_first:
                print(
                    "DEBUG: sanitize.header changed '{}' -> '{}'".format(
                        raw_first[:120], san_first[:120]
                    )
                )
            else:
                print("DEBUG: sanitize.header unchanged '{}'".format(raw_first[:120]))

        processed = self._enforce_subject_length(sanitized)
        processed = self._wrap_body(processed)

        if self.debug:
            print("DEBUG: Commit message post-processing:")
            print(f"  Raw length: {len(raw_message)}")
            print(f"  Final length: {len(processed)}")
            if raw_message != processed:
                print(("  Differences were applied (subject enforcement / wrapping)."))
            else:
                print("  No changes applied to raw model output.")
            print("  --- RAW MESSAGE START ---")
            print(raw_message)
            print("  --- RAW MESSAGE END ---")
            print("  --- FINAL MESSAGE START ---")
            print(processed)
            print("  --- FINAL MESSAGE END ---")
        return processed

    def _call_provider(
        self, prompt: str, request_timeout: float | None = None
    ) -> str:
        if self._mode == "openai":
            if request_timeout is None:
                return self._call_openai(prompt)
            return self._call_openai(prompt, request_timeout=request_timeout)
        if request_timeout is None:
            return self._call_anthropic(prompt)
        return self._call_anthropic(prompt, request_timeout=request_timeout)

    async def _call_provider_async(
        self, prompt: str, request_timeout: float | None = None
    ) -> str:
        if self._mode == "openai":
            if request_timeout is None:
                return await self._call_openai_async(prompt)
            return await self._call_openai_async(
                prompt, request_timeout=request_timeout
            )
        if request_timeout is None:
            return await self._call_anthropic_async(prompt)
        return await self._call_anthropic_async(
            prompt, request_timeout=request_timeout
        )
    # ------------------------------------------------------------------
    # Provider calls
    # ------------------------------------------------------------------
    def _call_openai(
        self, prompt: str, request_timeout: float | None = None
    ) -> str:
        """Delegate OpenAI-like provider call to the driver.

        Provides backward-compatible debug logging and adaptive minimal
        prompt retry (now signalled via driver sentinel error).
        """
        # Testing shortcut maintained here for suite compatibility
        if os.environ.get("KCMT_TEST_DISABLE_OPENAI"):
            return "feat(test): stubbed commit message"
        if self._mode != "openai":  # defensive
            raise LLMError("_call_openai invoked for non-openai provider")
        # Build messages (system + user) based on current minimal_prompt flag
        messages = self._build_messages(prompt)
        minimal_allowed = not self.model.startswith("gpt-5")
        driver = cast(OpenAIDriver, self._driver)
        invoke_kwargs = {
            "messages": messages,
            "minimal_ok": minimal_allowed,
        }
        if request_timeout is not None:
            invoke_kwargs["request_timeout"] = request_timeout
        try:
            content = driver.invoke_messages(**invoke_kwargs)
        except LLMError as e:
            msg = str(e)
            if "RETRY_MINIMAL_PROMPT" in msg and minimal_allowed:
                if self.debug:
                    print(
                        "DEBUG: driver signalled minimal prompt retry; "
                        "rebuilding messages"
                    )
                # driver already flipped its own minimal flag; mirror here
                if not self.model.startswith("gpt-5"):
                    self._minimal_prompt = True
                messages = self._build_messages(prompt)
                retry_kwargs = {
                    "messages": messages,
                    "minimal_ok": False,
                }
                if request_timeout is not None:
                    retry_kwargs["request_timeout"] = request_timeout
                content = driver.invoke_messages(**retry_kwargs)
            elif "RETRY_SIMPLE_PROMPT" in msg and self.model.startswith("gpt-5"):
                if self.debug:
                    print(
                        "DEBUG: driver signalled simplified prompt retry; "
                        "rebuilding system message for gpt-5"
                    )
                simple_messages = self._build_messages_simple_gpt5(prompt)
                simple_kwargs = {
                    "messages": simple_messages,
                    "minimal_ok": False,
                }
                if request_timeout is not None:
                    simple_kwargs["request_timeout"] = request_timeout
                content = driver.invoke_messages(**simple_kwargs)
            else:
                raise
        if self.debug:
            preview = content[:300].replace("\n", "\\n")
            print("DEBUG: OpenAI-like API Response (driver):")
            print(f"  Length: {len(content)} characters")
            print(f"  Preview: '{preview}'")
        if not content:
            raise LLMError("Empty OpenAI response after driver invocation")
        return content

    async def _call_openai_async(
        self, prompt: str, request_timeout: float | None = None
    ) -> str:
        if os.environ.get("KCMT_TEST_DISABLE_OPENAI"):
            return "feat(test): stubbed commit message"
        if self._mode != "openai":  # defensive
            raise LLMError("_call_openai_async invoked for non-openai provider")
        messages = self._build_messages(prompt)
        minimal_allowed = not self.model.startswith("gpt-5")
        driver = cast(OpenAIDriver, self._driver)
        invoke_kwargs = {
            "messages": messages,
            "minimal_ok": minimal_allowed,
        }
        if request_timeout is not None:
            invoke_kwargs["request_timeout"] = request_timeout
        try:
            content = await driver.invoke_messages_async(**invoke_kwargs)
        except LLMError as e:
            msg = str(e)
            if "RETRY_MINIMAL_PROMPT" in msg and minimal_allowed:
                if self.debug:
                    print(
                        "DEBUG: driver signalled minimal prompt retry (async); "
                        "rebuilding messages"
                    )
                if not self.model.startswith("gpt-5"):
                    self._minimal_prompt = True
                messages = self._build_messages(prompt)
                retry_kwargs = {
                    "messages": messages,
                    "minimal_ok": False,
                }
                if request_timeout is not None:
                    retry_kwargs["request_timeout"] = request_timeout
                content = await driver.invoke_messages_async(**retry_kwargs)
            elif "RETRY_SIMPLE_PROMPT" in msg and self.model.startswith("gpt-5"):
                if self.debug:
                    print(
                        "DEBUG: driver signalled simplified prompt retry (async); "
                        "rebuilding system message for gpt-5"
                    )
                simple_messages = self._build_messages_simple_gpt5(prompt)
                simple_kwargs = {
                    "messages": simple_messages,
                    "minimal_ok": False,
                }
                if request_timeout is not None:
                    simple_kwargs["request_timeout"] = request_timeout
                content = await driver.invoke_messages_async(**simple_kwargs)
            else:
                raise
        if self.debug:
            preview = content[:300].replace("\n", "\\n")
            print("DEBUG: OpenAI-like API Response (driver, async):")
            print(f"  Length: {len(content)} characters")
            print(f"  Preview: '{preview}'")
        if not content:
            raise LLMError("Empty OpenAI response after driver invocation")
        return content

    def _call_anthropic(
        self, prompt: str, request_timeout: float | None = None
    ) -> str:
        if self._mode != "anthropic":  # defensive
            raise LLMError("_call_anthropic invoked for non-anthropic provider")
        driver = cast(AnthropicDriver, self._driver)
        if request_timeout is None:
            return driver.invoke(prompt)
        return driver.invoke(prompt, request_timeout=request_timeout)

    async def _call_anthropic_async(
        self, prompt: str, request_timeout: float | None = None
    ) -> str:
        if self._mode != "anthropic":  # defensive
            raise LLMError("_call_anthropic_async invoked for non-anthropic provider")
        driver = cast(AnthropicDriver, self._driver)
        if request_timeout is None:
            return await driver.invoke_async(prompt)
        return await driver.invoke_async(prompt, request_timeout=request_timeout)

    # ------------------------------------------------------------------
    # Prompt helpers
    # ------------------------------------------------------------------
    def _build_messages(self, prompt: str) -> list[dict[str, str]]:
        system_lines = [
            "You are a strict conventional commit message generator.",
            "Output ONLY: type(scope): description",
            "",  # blank line
            "Rules:",
            "- types: feat fix docs style refactor perf test build ci chore",
            "  revert",
            "- scope REQUIRED (api ui auth db config tests deps build core)",
            "- subject <= 50 chars, no trailing period",
            "- add body if >5 changed lines",
            "- wrap body at 72 chars",
            "- body explains WHAT and WHY",
            "",  # blank line
            "Return only the commit message.",
        ]
        if getattr(self, "_minimal_prompt", False) and not self.model.startswith(
            "gpt-5"
        ):
            system_lines = [
                "You output a single concise conventional commit message.",
                "Format: type(scope): description",
                "No explanation, no code blocks, no reasoning.",
            ]
        system_content = "\n".join(system_lines)
        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": prompt},
        ]

    def _build_messages_simple_gpt5(self, prompt: str) -> list[dict[str, str]]:
        """One-shot simplified system prompt for gpt-5 retry.

        Keeps strict conventional commit requirements, but removes
        extra wording to minimize chance of empty responses when
        finish_reason=length occurs.
        """
        system_lines = [
            "Output a conventional commit message only.",
            "Header format: type(scope): description (scope REQUIRED).",
            "Subject <= 50 chars. No trailing period.",
            "If >5 changed lines, add a body wrapped at 72 chars.",
            "No code fences, no quotes, no explanations.",
        ]
        system_content = "\n".join(system_lines)
        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": prompt},
        ]

    def _build_prompt(self, diff: str, context: str, style: str) -> str:
        """Construct the textual prompt fed into the provider messages."""
        prompt_parts = [
            "Generate a conventional commit message for these changes:",
            "",
            "DIFF:",
            diff,
        ]
        if context:
            prompt_parts.extend(["", "CONTEXT:", context])
        if style == "conventional":
            prompt_parts.extend(
                [
                    "",
                    "STRICT REQUIREMENTS:",
                    "- MUST use format: type(scope): description",
                    "- Scope REQUIRED (api ui auth db config tests etc.)",
                    "- If diff shows substantial changes (>5 lines), add body",
                    "- Body should explain what changed and why",
                    "- Keep subject line under 50 characters",
                    "- Only output the commit message (no backticks / quotes)",
                ]
            )
        elif style == "simple":
            prompt_parts.extend(["", "Keep it simple but include mandatory scope."])
        prompt_parts.append("")
        prompt_parts.append("Analyze the changes carefully and be specific.")
        return "\n".join(prompt_parts)

    def _is_binary_diff(self, diff: str) -> bool:
        """Return True if diff represents a binary file change."""
        lines = diff.strip().split("\n")
        for line in lines:
            if line.startswith("Binary files") and " differ" in line:
                return True
            if "GIT binary patch" in line:
                return True
        return False

    def _looks_like_text_file(self, file_path: str) -> bool:
        """Heuristic: treat common source/docs formats as text, never binary.

        This prevents static binary fallback for files like .py, ensuring we
        always use the git diff to generate the message (except pure deletions
        which are handled elsewhere in the workflow).
        """
        if not file_path:
            return False
        text_exts = {
            ".py",
            ".pyi",
            ".pyx",
            ".pxd",
            ".js",
            ".ts",
            ".jsx",
            ".tsx",
            ".css",
            ".scss",
            ".sass",
            ".less",
            ".html",
            ".htm",
            ".xml",
            ".json",
            ".yaml",
            ".yml",
            ".toml",
            ".ini",
            ".cfg",
            ".md",
            ".rst",
            ".txt",
            ".csv",
            ".tsv",
            ".java",
            ".c",
            ".cpp",
            ".h",
            ".hpp",
            ".go",
            ".rs",
            ".rb",
            ".php",
            ".sh",
            ".bash",
            ".zsh",
            ".ps1",
            ".bat",
            ".cmd",
            ".gradle",
            ".make",
            ".mk",
            ".cmake",
        }
        lower = file_path.lower()
        for ext in text_exts:
            if lower.endswith(ext):
                return True
        # Also treat files without extension but in typical code dirs as text
        code_dirs = ("/src/", "/lib/", "/app/", "/kcmt/")
        return any(d in lower for d in code_dirs)

    def _clean_diff_for_llm(self, diff: str) -> str:
        """Clean up diff to improve XAI API compatibility."""
        lines = diff.strip().split("\n")
        cleaned_lines: list[str] = []
        for line in lines:
            # Skip problematic /dev/null references that some APIs dislike
            if "--- /dev/null" in line:
                cleaned_lines.append("--- (new file)")
            elif "+++ /dev/null" in line:
                cleaned_lines.append("+++ (deleted)")
            else:
                cleaned_lines.append(line)
        return "\n".join(cleaned_lines)

    # ------------------------------------------------------------------
    # Enrichment helpers
    # ------------------------------------------------------------------
    def _count_changed_lines(self, diff: str) -> int:
        """Count added/removed lines ignoring diff metadata lines."""
        total = 0
        for line in diff.splitlines():
            if not line:
                continue
            if line.startswith("+++") or line.startswith("---"):
                continue
            if line.startswith("@@"):
                continue
            if line.startswith("+") or line.startswith("-"):
                total += 1
        return total

    def _enrich_with_body(
        self,
        header: str,
        diff: str,
        context: str,
    ) -> str:
        """Attempt a second call requesting a concise body.

        We intentionally keep diff truncated to avoid token bloat.
        """
        # Temporarily disable minimal prompt for enrichment
        prev_minimal = getattr(self, "_minimal_prompt", False)
        self._minimal_prompt = False
        truncated_diff = self._truncate_diff_for_body(diff)
        body_prompt = (
            header
            + (
                "\n\nThe above is the conventional commit header. Add a body "
                "(no header change) explaining WHAT changed and WHY in 3-6 "
                "short lines. Do not restate the header."
            )
            + "\n\nTRUNCATED_DIFF:\n"
            + truncated_diff
        )
        try:
            enriched_raw = self._call_openai(body_prompt)
        except LLMError:
            self._minimal_prompt = prev_minimal
            return header
        finally:
            # restore
            self._minimal_prompt = prev_minimal
        enriched = self._sanitize_commit_output(enriched_raw, context)
        # Ensure header unchanged and body added
        if enriched.startswith(header) and "\n" in enriched.strip():
            return enriched
        return header

    def _truncate_diff_for_body(self, diff: str, limit: int = 4000) -> str:
        """Rudimentary truncation: keep first N chars and last 500 if huge."""
        if len(diff) <= limit:
            return diff
        head = diff[: limit - 500]
        tail = diff[-500:]
        return head + "\n...\n" + tail

    # ------------------------------------------------------------------
    # Post-processing helpers
    # ------------------------------------------------------------------
    def _enforce_subject_length(self, message: str) -> str:
        """Ensure the first line (subject) does not exceed configured length.
        We DO NOT truncate the entire message; only adjust the subject line
        if it exceeds the max length. We prefer attempting a smart cut at a
        word boundary. If cutting, we append '…' to indicate the subject was
        shortened while preserving the full body content.
        """
        max_len = self.config.max_commit_length
        lines = message.splitlines()
        if not lines:
            return message
        subject = lines[0].strip()
        if len(subject) <= max_len:
            return message
        # Find last space before limit to avoid mid-word cut
        cutoff = subject.rfind(" ", 0, max_len)
        # fall back to hard cut if no good space
        if cutoff == -1 or cutoff < max_len * 0.6:
            cutoff = max_len - 1
        shortened = subject[:cutoff].rstrip() + "…"
        lines[0] = shortened
        return "\n".join(lines)

    def _wrap_body(self, message: str, width: int = 72) -> str:
        """Wrap body lines (after the first blank separator) to given width.
        We keep pre-existing wrapping if already within limits; we do not
        merge lines.
        """
        import textwrap

        lines = message.splitlines()
        if not lines:
            return message
        # Identify body start: first blank line after subject
        body_start = None
        for i in range(1, len(lines)):
            if lines[i].strip() == "":
                body_start = i + 1
                break
        if body_start is None or body_start >= len(lines):
            return message  # no body
        body = lines[body_start:]
        wrapped_body: list[str] = []
        for paragraph in "\n".join(body).split("\n\n"):
            if not paragraph.strip():
                wrapped_body.append("")
                continue
            # Skip wrapping code blocks or diff-like fenced blocks
            if paragraph.strip().startswith("```"):
                wrapped_body.append(paragraph)
                continue
            # Wrap only lines that exceed width
            for wrapped_line in textwrap.wrap(
                paragraph, width=width, drop_whitespace=True
            ):
                wrapped_body.append(wrapped_line)
            wrapped_body.append("")  # preserve paragraph break
        # Remove trailing blank introduced
        if wrapped_body and wrapped_body[-1] == "":
            wrapped_body.pop()
        new_lines = lines[:body_start] + wrapped_body
        return "\n".join(new_lines)

    # ------------------------------------------------------------------
    # Output sanitization / normalization
    # ------------------------------------------------------------------
    _CC_PATTERN = re.compile(
        r"^(feat|fix|docs|style|refactor|test|chore|perf|ci|build|revert)"
        r"(\([a-zA-Z0-9_-]+\))?: "
        r".+"
    )

    def _sanitize_commit_output(self, raw: str, _context: str) -> str:
        """Attempt to coerce arbitrary LLM output into a conventional commit.

        Steps:
          1. Strip leading/trailing whitespace, remove surrounding quotes.
          2. Remove markdown code fences & YAML frontmatter markers.
          3. Take first non-empty line that looks like a header; if no line
             matches, build a heuristic header from file path + diff nature.
          4. Remove leading list markers(-, *, •) and backticks.
          5. Ensure header line length reasonable (subject line only); keep
             remainder (if any) as body separated by blank line.
        """
        text = raw.strip()
        if not text:
            raise LLMError("Empty LLM output (no heuristic fallback)")

        # Remove surrounding quotes/backticks
        if (text.startswith('"') and text.endswith('"')) or (
            text.startswith("'") and text.endswith("'")
        ):
            text = text[1:-1].strip()
        if text.startswith("```"):
            # Remove fenced code blocks crudely
            parts = text.split("```")
            # take first non-empty non-fence segment
            for part in parts:
                candidate = part.strip()
                if candidate and not candidate.startswith(("yaml", "json")):
                    text = candidate
                    break

        # Split into lines and find first plausible header
        lines = [line.rstrip() for line in text.splitlines() if line.strip()]
        header = None
        body_lines: list[str] = []
        for i, line in enumerate(lines):
            cleaned = line.lstrip("-*• ").strip("`").strip()
            # Collapse internal whitespace
            cleaned = re.sub(r"\s+", " ", cleaned)
            if self._CC_PATTERN.match(cleaned):
                header = cleaned
                body_lines = lines[i + 1 :]
                break

        if header is None:
            # Try to detect pattern like "feat: something" missing scope
            for i, line in enumerate(lines):
                cleaned = line.lstrip("-*• ").strip("`").strip()
                cleaned = re.sub(r"\s+", " ", cleaned)
                pattern_simple = (
                    r"^(feat|fix|docs|style|refactor|test|chore|perf|ci|build|"
                    r"revert): "
                )
                if re.match(pattern_simple, cleaned):
                    header = cleaned  # already acceptable (scope optional)
                    body_lines = lines[i + 1 :]
                    break

        if header is None:
            # No recognizable header and fallback removed -> raise
            raise LLMError(
                "LLM output missing conventional commit header (no fallback)"
            )

        # Strip trailing periods from header (common style issue)
        header = header.rstrip(".")

        body: list[str] = []
        for bline in body_lines:
            if bline.strip().startswith(("```", "---", "===")):
                # Skip decorative / fence lines in body
                continue
            if len(body) > 12:
                # Avoid overly verbose body; keep it concise
                break
            body.append(bline)

        if not body:
            return header
        return header + "\n\n" + "\n".join(body)
