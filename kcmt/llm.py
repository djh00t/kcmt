"""LLM integration for kcmt."""

from __future__ import annotations

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
        self.config = config or get_active_config()
        self.provider = self.config.provider
        self.model = self.config.model
        self.api_key = self.config.resolve_api_key()

        if not self.api_key:
            raise LLMError(
                f"Environment variable '{self.config.api_key_env}' is not set or empty."
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
        # Handle binary files
        if self._is_binary_diff(diff):
            return self._generate_binary_commit_message(diff, context, style)
        
        # Handle empty or minimal diffs
        if not diff.strip() or len(diff.strip()) < 10:
            return self._generate_minimal_commit_message(context, style)
        
        # Handle very large diffs that might overwhelm the API
        if len(diff) > 8000:  # Rough threshold for very large diffs
            return self._generate_large_file_commit_message(diff, context, style)
        
        # Clean up diff for better XAI compatibility
        cleaned_diff = self._clean_diff_for_llm(diff)
        prompt = self._build_prompt(cleaned_diff, context, style)

        if self._mode == "openai":
            message = self._call_openai(prompt)
        else:
            message = self._call_anthropic(prompt)

        if not message:
            raise LLMError("Empty response from LLM")

        message = message.strip()
        max_len = self.config.max_commit_length
        if len(message) > max_len:
            message = message[: max_len - 3] + "..."
        return message

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
                
            response = self._client.chat.completions.create(
                messages=messages,
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
                                "Generate a conventional commit message following strict rules "
                                "for the provided diff.\n" + prompt
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
            response = httpx.post(url, headers=headers, json=payload, timeout=60.0)
        except Exception as e:
            raise LLMError(f"Anthropic request failed: {e}") from e

        if response.status_code >= 400:
            raise LLMError(
                f"Anthropic error {response.status_code}: {response.text}"
            )

        data = response.json()
        content = data.get("content") or []
        texts = [chunk.get("text", "") for chunk in content if chunk.get("type") == "text"]
        return "\n".join(filter(None, texts))

    # ------------------------------------------------------------------
    # Prompt helpers
    # ------------------------------------------------------------------
    def _build_messages(self, prompt: str) -> List[Dict[str, str]]:
        return [
            {
                "role": "system",
                "content": (
                    "You are a strict conventional commit message generator. "
                    "You MUST follow the EXACT format: type(scope): description\\n\\n"
                    "MANDATORY REQUIREMENTS:\\n"
                    "- Type MUST be one of: feat, fix, docs, style, refactor, perf, test, build, ci, chore, revert\\n"
                    "- Scope is MANDATORY and must be specific (e.g., api, ui, auth, db, config, tests, deps, build)\\n"
                    "- Description MUST start with lowercase verb in imperative mood\\n"
                    "- First line MUST be ≤50 characters total\\n"
                    "- NO period at end of subject line\\n"
                    "- If changes are substantial, add a body separated by blank line\\n"
                    "- Body lines should be wrapped at 72 characters\\n"
                    "- Body should explain what and why, not how\\n\\n"
                    "SCOPE GUIDELINES:\\n"
                    "- api: API endpoints, routes, request/response handling\\n"
                    "- ui: User interface components, styling, layouts\\n"
                    "- auth: Authentication, authorization, security\\n"
                    "- db: Database models, migrations, queries\\n"
                    "- config: Configuration files, environment setup\\n"
                    "- tests: Test files, test utilities, coverage\\n"
                    "- docs: Documentation, README, comments\\n"
                    "- build: Build scripts, dependencies, packaging\\n"
                    "- ci: Continuous integration, workflows, deployment\\n"
                    "- core: Core business logic, main algorithms\\n\\n"
                    "EXAMPLES:\\n"
                    "feat(auth): add JWT token validation middleware\\n\\n"
                    "Implement JWT token validation for protected routes.\\n"
                    "Validates token signature and expiration time.\\n\\n"
                    "fix(api): handle null response in user endpoint\\n\\n"
                    "Add null checks for user data before processing.\\n"
                    "Prevents 500 errors when user data is missing.\\n\\n"
                    "Return ONLY the commit message, no explanation or markdown."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
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
                    "- Scope is MANDATORY (api, ui, auth, db, config, tests, etc.)",
                    "- If diff shows substantial changes (>5 lines), add body",
                    "- Body should explain what changed and why",
                    "- Keep subject line under 50 characters",
                ]
            )
        elif style == "simple":
            prompt_parts.extend(["", "Keep it simple but include mandatory scope."])

        prompt_parts.extend(["", "Analyze the changes carefully and be specific."])
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
            if file_path.endswith(('.py', '.java', '.cpp', '.c', '.js', '.ts')):
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
            elif file_path.endswith(('.png', '.jpg', '.jpeg', '.gif', '.ico', '.svg')):
                filename = file_path.split('/')[-1]
                return f"feat(assets): add {filename} image file"
            elif file_path.endswith(('.pdf', '.doc', '.docx')):
                filename = file_path.split('/')[-1]
                return f"docs(assets): add {filename} document"
            elif file_path.endswith(('.zip', '.tar.gz', '.tar', '.gz', '.tgz')):
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
            elif file_path.endswith(('.json', '.yaml', '.yml', '.toml', '.ini')):
                return f"config(setup): update {filename}"
            elif file_path.endswith(('.css', '.scss', '.sass', '.less')):
                return f"style(ui): update {filename}"
            elif file_path.endswith(('.js', '.ts', '.jsx', '.tsx')):
                return f"refactor(core): update {filename}"
            elif file_path.endswith(('.py', '.java', '.cpp', '.c', '.go', '.rs')):
                return f"refactor(core): update {filename}"
            else:
                return f"chore(misc): update {filename}"
        return "chore(misc): minor update"
