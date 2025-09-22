"""LLM integration for kcmt."""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import httpx
from openai import OpenAI

from .config import Config, get_active_config
from .exceptions import LLMError


class LLMClient:
    """Provider-aware client for generating commit messages."""

    def __init__(
        self, config: Optional[Config] = None, debug: bool = False
    ) -> None:
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
                    print(
                        "DEBUG: Using dummy key for {} (tests)".format(
                            self.provider
                        )
                    )
            else:
                raise LLMError(
                    "Environment variable '"
                    f"{self.config.api_key_env}"
                    "' is not set or empty."
                )

        if self.provider in {"openai", "github", "xai"}:
            self._mode = "openai"
            self._client = OpenAI(
                base_url=self.config.llm_endpoint,
                api_key=self.api_key,
            )
        elif self.provider == "anthropic":
            self._mode = "anthropic"
        else:
            raise LLMError(f"Unsupported provider: {self.provider}")

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def generate_commit_message(
        self,
        diff: str,
        context: str = "",
        style: str = "conventional",
    ) -> str:
        if self.debug:
            print("DEBUG: generate_commit_message called")
            print(f"  Diff length: {len(diff)} characters")
            print(f"  Context: {context}")
            print(f"  Provider: {self.provider}")
        
        # Handle binary files
        if self._is_binary_diff(diff):
            if self.debug:
                print(
                    "DEBUG: Detected binary file, using binary commit message"
                )
            return self._generate_binary_commit_message(diff, context, style)
        
        # Handle empty or minimal diffs
        if not diff.strip() or len(diff.strip()) < 10:
            if self.debug:
                print(
                    "DEBUG: Minimal diff detected, using minimal commit"
                    " message"
                )
            return self._generate_minimal_commit_message(context, style)
        
        # Handle very large diffs that might overwhelm the API
        if len(diff) > 8000:  # Rough threshold for very large diffs
            if self.debug:
                print(
                    "DEBUG: Large diff detected, using large file commit"
                    " message"
                )
            return self._generate_large_file_commit_message(
                diff, context, style
            )
        
        # Clean up diff for better XAI compatibility
        cleaned_diff = self._clean_diff_for_llm(diff)
        if self.debug:
            print(
                f"DEBUG: Cleaned diff length: {len(cleaned_diff)} characters"
            )
        
        prompt = self._build_prompt(cleaned_diff, context, style)

        if self._mode == "openai":
            message = self._call_openai(prompt)
        else:
            message = self._call_anthropic(prompt)

        if not message:
            raise LLMError("Empty response from LLM")
        raw_message = message.strip()

    # Post-process: enforce subject line length only, do not truncate body.
        processed = self._enforce_subject_length(raw_message)
        processed = self._wrap_body(processed)

        if self.debug:
            print("DEBUG: Commit message post-processing:")
            print(f"  Raw length: {len(raw_message)}")
            print(f"  Final length: {len(processed)}")
            if raw_message != processed:
                print(
                    (
                        "  Differences were applied (subject enforcement /"
                        " wrapping)."
                    )
                )
            else:
                print("  No changes applied to raw model output.")
            print("  --- RAW MESSAGE START ---")
            print(raw_message)
            print("  --- RAW MESSAGE END ---")
            print("  --- FINAL MESSAGE START ---")
            print(processed)
            print("  --- FINAL MESSAGE END ---")
        return processed

    # ------------------------------------------------------------------
    # Provider calls
    # ------------------------------------------------------------------
    def _call_openai(self, prompt: str) -> str:
        try:
            messages = self._build_messages(prompt)
            if self.debug:
                print("DEBUG: XAI API Request:")
                print(f"  Model: {self.model}")
                print(f"  Messages: {messages}")
                print("  Max tokens: 512")
                print()
                
            response = self._client.chat.completions.create(  # type: ignore[attr-defined]
                messages=messages,  # type: ignore[arg-type]
                model=self.model,
                max_completion_tokens=512,
            )
            content = response.choices[0].message.content or ""
            
            if self.debug:
                print("DEBUG: XAI API Response:")
                print(f"  Length: {len(content)} characters")
                print(f"  Content: '{content}'")
                print()
                
            return content
        except Exception as e:
            raise LLMError(f"OpenAI client error: {e}") from e
        return content

    def _call_anthropic(self, prompt: str) -> str:
        url = self.config.llm_endpoint.rstrip("/") + "/v1/messages"
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        payload = {
            "model": self.model,
            "max_output_tokens": 512,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Generate a conventional commit message "
                                "following strict rules for the provided "
                                "diff.\n"
                                + prompt
                            ),
                        }
                    ],
                }
            ],
            "system": (
                "You are a strict conventional commit message generator. "
                "Always respond with type(scope): description."
            ),
        }

        try:
            response = httpx.post(
                url, headers=headers, json=payload, timeout=60.0
            )
        except Exception as e:
            raise LLMError(f"Anthropic request failed: {e}") from e

        if response.status_code >= 400:
            raise LLMError(
                f"Anthropic error {response.status_code}: {response.text}"
            )

        data = response.json()
        content = data.get("content") or []
        texts = []
        for chunk in content:
            if chunk.get("type") == "text":
                texts.append(chunk.get("text", ""))
        return "\n".join(filter(None, texts))

    # ------------------------------------------------------------------
    # Prompt helpers
    # ------------------------------------------------------------------
    def _build_messages(self, prompt: str) -> List[Dict[str, str]]:
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
        system_content = "\n".join(system_lines)
        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": prompt},
        ]

    def _build_prompt(self, diff: str, context: str, style: str) -> str:
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
                ]
            )
        elif style == "simple":
            prompt_parts.extend(
                ["", "Keep it simple but include mandatory scope."]
            )

        prompt_parts.extend(
            ["", "Analyze the changes carefully and be specific."]
        )
        return "\n".join(prompt_parts)

    def _is_binary_diff(self, diff: str) -> bool:
        """Check if the diff represents binary file changes."""
        # More specific check for binary file indicators
        lines = diff.strip().split('\n')
        for line in lines:
            if line.startswith('Binary files') and ' differ' in line:
                return True
            # Git also uses this pattern for binary files
            if 'GIT binary patch' in line:
                return True
        return False

    def _clean_diff_for_llm(self, diff: str) -> str:
        """Clean up diff to improve XAI API compatibility."""
        lines = diff.strip().split('\n')
        cleaned_lines = []
        
        for line in lines:
            # Skip problematic /dev/null references that XAI seems to dislike
            if '--- /dev/null' in line:
                cleaned_lines.append('--- (new file)')
            elif '+++ /dev/null' in line:
                cleaned_lines.append('+++ (deleted)')
            else:
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)

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
        cutoff = subject.rfind(' ', 0, max_len)
        # fall back to hard cut if no good space
        if cutoff == -1 or cutoff < max_len * 0.6:
            cutoff = max_len - 1
        shortened = subject[:cutoff].rstrip() + '…'
        lines[0] = shortened
        return '\n'.join(lines)

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
            if lines[i].strip() == '':
                body_start = i + 1
                break
        if body_start is None or body_start >= len(lines):
            return message  # no body
        body = lines[body_start:]
        wrapped_body: list[str] = []
        for paragraph in '\n'.join(body).split('\n\n'):
            if not paragraph.strip():
                wrapped_body.append('')
                continue
            # Skip wrapping code blocks or diff-like fenced blocks
            if paragraph.strip().startswith('```'):
                wrapped_body.append(paragraph)
                continue
            # Wrap only lines that exceed width
            for wrapped_line in textwrap.wrap(
                paragraph, width=width, drop_whitespace=True
            ):
                wrapped_body.append(wrapped_line)
            wrapped_body.append('')  # preserve paragraph break
        # Remove trailing blank introduced
        if wrapped_body and wrapped_body[-1] == '':
            wrapped_body.pop()
        new_lines = lines[:body_start] + wrapped_body
        return '\n'.join(new_lines)

    def _generate_large_file_commit_message(
        self, diff: str, context: str, style: str
    ) -> str:
        """Generate commit message for very large files."""
        # Extract file path from context
        file_path = ""
        if context and "File:" in context:
            file_path = context.split("File:", 1)[1].strip()
        
        # Determine appropriate commit message based on file type and size
        if file_path:
            if file_path.endswith(
                ('.py', '.java', '.cpp', '.c', '.js', '.ts')
            ):
                filename = file_path.split('/')[-1]
                return f"feat(core): add {filename} implementation"
            elif file_path.endswith(('.json', '.yaml', '.yml', '.toml')):
                filename = file_path.split('/')[-1]
                return f"chore(config): add {filename} configuration"
            elif file_path.endswith(('.md', '.rst', '.txt')):
                filename = file_path.split('/')[-1]
                return f"docs: add {filename}"
            elif file_path.endswith(('.html', '.css', '.scss')):
                filename = file_path.split('/')[-1]
                return f"feat(ui): add {filename} styles"
            else:
                filename = file_path.split('/')[-1]
                return f"feat: add {filename}"
        
        # Check if it's a new file vs modification
        if "new file mode" in diff:
            return "feat(core): add new implementation file"
        else:
            return "refactor(core): major code restructuring"

    def _generate_binary_commit_message(
        self, diff: str, context: str, style: str
    ) -> str:
        """Generate commit message for binary file changes."""
        # Extract file path from context or diff
        file_path = ""
        if context and "File:" in context:
            file_path = context.split("File:", 1)[1].strip()
        
        # Determine appropriate type and scope based on file
        if file_path:
            if file_path.endswith(('.coverage', '.cov')):
                return "test(coverage): update test coverage data"
            elif file_path.endswith(
                ('.png', '.jpg', '.jpeg', '.gif', '.ico', '.svg')
            ):
                filename = file_path.split('/')[-1]
                return f"feat(assets): add {filename} image file"
            elif file_path.endswith(('.pdf', '.doc', '.docx')):
                filename = file_path.split('/')[-1]
                return f"docs(assets): add {filename} document"
            elif file_path.endswith(
                ('.zip', '.tar.gz', '.tar', '.gz', '.tgz')
            ):
                filename = file_path.split('/')[-1]
                return f"build(deps): add {filename} archive"
            elif file_path.endswith(('.woff', '.woff2', '.ttf', '.eot')):
                filename = file_path.split('/')[-1]
                return f"feat(fonts): add {filename} font file"
            elif file_path.endswith(('.mp4', '.avi', '.mov', '.webm')):
                filename = file_path.split('/')[-1]
                return f"feat(media): add {filename} video file"
            elif file_path.endswith(('.mp3', '.wav', '.ogg', '.flac')):
                filename = file_path.split('/')[-1]
                return f"feat(media): add {filename} audio file"
            else:
                filename = file_path.split('/')[-1]
                return f"chore(assets): add {filename} binary file"
        
        # Fallback for binary files without clear context
        if "Binary files /dev/null and" in diff:
            return "chore(assets): add binary file"
        else:
            return "chore(assets): update binary file"

    def _generate_minimal_commit_message(
        self, context: str, style: str
    ) -> str:
        """Generate commit message when diff is empty or minimal."""
        if context and "File:" in context:
            file_path = context.split("File:", 1)[1].strip()
            filename = file_path.split('/')[-1]
            
            # Determine appropriate scope based on file path
            if 'test' in file_path.lower() or file_path.endswith('.test.'):
                return f"test(core): update {filename}"
            elif file_path.endswith(('.md', '.txt', '.rst')):
                return f"docs(content): update {filename}"
            elif file_path.endswith(
                ('.json', '.yaml', '.yml', '.toml', '.ini')
            ):
                # Use allowed conventional commit type 'chore'
                return f"chore(config): update {filename}"
            elif file_path.endswith(('.css', '.scss', '.sass', '.less')):
                return f"style(ui): update {filename}"
            elif file_path.endswith(('.js', '.ts', '.jsx', '.tsx')):
                return f"refactor(core): update {filename}"
            elif file_path.endswith(
                ('.py', '.java', '.cpp', '.c', '.go', '.rs')
            ):
                return f"refactor(core): update {filename}"
            else:
                return f"chore(misc): update {filename}"
        return "chore(misc): minor update"
