import json
import os
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

from pytest_bdd import given, parsers, scenarios, then, when

REPO_ROOT = Path(__file__).resolve().parents[1]
RUST_MANIFEST = REPO_ROOT / "rust" / "Cargo.toml"
FEATURES = REPO_ROOT / "tests" / "features" / "rust_workflow_parity.feature"

scenarios(str(FEATURES))


def _run(command: list[str], cwd: Path, env: dict[str, str] | None = None) -> str:
    result = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return result.stdout.strip()


def _clean_env(config_home: Path) -> dict[str, str]:
    env = {
        key: value
        for key, value in os.environ.items()
        if key
        in {
            "PATH",
            "HOME",
            "USER",
            "TMPDIR",
            "LANG",
            "LC_ALL",
            "SystemRoot",
            "SYSTEMROOT",
            "ComSpec",
            "PATHEXT",
            "TEMP",
            "TMP",
            "USERPROFILE",
            "APPDATA",
            "LOCALAPPDATA",
        }
    }
    env["KCMT_CONFIG_HOME"] = str(config_home)
    env["KCMT_ALLOW_LOCAL_SYNTHESIS"] = "1"
    env["KCMT_DISABLE_KEYCHAIN"] = "1"
    return env


def _git(repo: Path, args: list[str]) -> str:
    return _run(["git", *args], repo)


def _init_repo(repo: Path) -> None:
    _git(repo, ["init", "-q"])
    _git(repo, ["config", "user.name", "Tester"])
    _git(repo, ["config", "user.email", "tester@example.com"])


def _init_bare_remote(remote: Path) -> None:
    _git(remote, ["init", "--bare", "-q"])


class _BatchHandler(BaseHTTPRequestHandler):
    requests: list[tuple[str, str, str]] = []

    def _record(self) -> str:
        length = int(self.headers.get("content-length", "0") or "0")
        body = self.rfile.read(length).decode("utf-8", "ignore") if length else ""
        self.__class__.requests.append((self.command, self.path, body))
        return body

    def _json(self, body: str) -> None:
        payload = body.encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_POST(self) -> None:  # noqa: N802
        self._record()
        if self.path == "/files":
            self._json('{"id":"file_1"}')
        elif self.path == "/batches":
            self._json('{"id":"batch_1","status":"validating"}')
        else:
            self.send_error(404)

    def do_GET(self) -> None:  # noqa: N802
        self._record()
        if self.path == "/batches/batch_1":
            self._json(
                '{"id":"batch_1","status":"completed","output_file_id":"output_1"}'
            )
        elif self.path == "/files/output_1/content":
            self._json(
                '{"custom_id":"alpha.py","response":{"status_code":200,"body":{"choices":[{"message":{"content":"fix(alpha): batch alpha."}}]}}}\n'
                '{"custom_id":"beta.py","response":{"status_code":200,"body":{"choices":[{"message":{"content":"fix(beta): batch beta."}}]}}}\n'
            )
        else:
            self.send_error(404)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        return


def _start_batch_provider() -> tuple[str, type[_BatchHandler], HTTPServer]:
    class Handler(_BatchHandler):
        requests: list[tuple[str, str, str]] = []

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return f"http://127.0.0.1:{server.server_port}", Handler, server


def _start_partial_batch_provider() -> tuple[str, type[_BatchHandler], HTTPServer]:
    class Handler(_BatchHandler):
        requests: list[tuple[str, str, str]] = []

        def do_GET(self) -> None:  # noqa: N802
            self._record()
            if self.path == "/batches/batch_1":
                self._json(
                    '{"id":"batch_1","status":"completed","output_file_id":"output_1"}'
                )
            elif self.path == "/files/output_1/content":
                self._json(
                    '{"custom_id":"alpha.py","response":{"status_code":200,"body":{"choices":[{"message":{"content":"fix(alpha): batch alpha."}}]}}}\n'
                    '{"custom_id":"beta.py","response":{"status_code":200,"body":{"choices":[{"message":{"content":"This is not conventional"}}]}}}\n'
                )
            else:
                self.send_error(404)

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return f"http://127.0.0.1:{server.server_port}", Handler, server


class _XaiBatchHandler(BaseHTTPRequestHandler):
    requests: list[tuple[str, str, str]] = []

    def _record(self) -> str:
        length = int(self.headers.get("content-length", "0") or "0")
        body = self.rfile.read(length).decode("utf-8", "ignore") if length else ""
        self.__class__.requests.append((self.command, self.path, body))
        return body

    def _json(self, body: str) -> None:
        payload = body.encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_POST(self) -> None:  # noqa: N802
        self._record()
        if self.path == "/batches":
            self._json('{"batch_id":"batch_1"}')
        elif self.path == "/batches/batch_1/requests":
            self._json('{"status":"accepted"}')
        else:
            self.send_error(404)

    def do_GET(self) -> None:  # noqa: N802
        self._record()
        if self.path == "/batches/batch_1":
            self._json(
                '{"status":"completed","state":{"num_pending":0,"num_requests":2}}'
            )
        elif self.path == "/batches/batch_1/results?limit=100":
            self._json(
                '{"results":['
                '{"batch_request_id":"alpha.py","batch_result":{"response":{"chat_get_completion":{"choices":[{"message":{"content":"fix(alpha): batch alpha."}}]}}}},'
                '{"batch_request_id":"beta.py","batch_result":{"response":{"chat_get_completion":{"choices":[{"message":{"content":"fix(beta): batch beta."}}]}}}}'
                "]}"
            )
        else:
            self.send_error(404)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        return


def _start_xai_batch_provider() -> tuple[str, type[_XaiBatchHandler], HTTPServer]:
    class Handler(_XaiBatchHandler):
        requests: list[tuple[str, str, str]] = []

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return f"http://127.0.0.1:{server.server_port}", Handler, server


class _FallbackHandler(BaseHTTPRequestHandler):
    requests: list[tuple[str, str, str]] = []

    def _record(self) -> None:
        length = int(self.headers.get("content-length", "0") or "0")
        body = self.rfile.read(length).decode("utf-8", "ignore") if length else ""
        self.__class__.requests.append((self.command, self.path, body))

    def do_POST(self) -> None:  # noqa: N802
        self._record()
        if self.path == "/chat/completions":
            payload = b'{"error":{"message":"primary unavailable"}}'
            self.send_response(500)
        elif self.path == "/v1/messages":
            payload = b'{"content":[{"type":"text","text":"fix(fallback): use secondary provider."}]}'
            self.send_response(200)
        else:
            payload = b"{}"
            self.send_response(404)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        return


def _start_fallback_provider() -> tuple[str, type[_FallbackHandler], HTTPServer]:
    class Handler(_FallbackHandler):
        requests: list[tuple[str, str, str]] = []

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return f"http://127.0.0.1:{server.server_port}", Handler, server


class _GitHubTokenHandler(BaseHTTPRequestHandler):
    requests: list[tuple[str, str, str, str]] = []

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("content-length", "0") or "0")
        body = self.rfile.read(length).decode("utf-8", "ignore") if length else ""
        auth = self.headers.get("authorization", "")
        self.__class__.requests.append((self.command, self.path, auth, body))
        payload = b'{"choices":[{"message":{"content":"fix(github): use cli token."}}]}'
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        return


def _start_github_token_provider() -> tuple[str, type[_GitHubTokenHandler], HTTPServer]:
    class Handler(_GitHubTokenHandler):
        requests: list[tuple[str, str, str, str]] = []

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return f"http://127.0.0.1:{server.server_port}", Handler, server


class _RetryLimitHandler(BaseHTTPRequestHandler):
    requests: list[tuple[str, str, str]] = []

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("content-length", "0") or "0")
        body = self.rfile.read(length).decode("utf-8", "ignore") if length else ""
        self.__class__.requests.append((self.command, self.path, body))
        payload = b'{"error":{"message":"temporary provider failure"}}'
        self.send_response(500)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        return


def _start_retry_limited_provider() -> tuple[str, type[_RetryLimitHandler], HTTPServer]:
    class Handler(_RetryLimitHandler):
        requests: list[tuple[str, str, str]] = []

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return f"http://127.0.0.1:{server.server_port}", Handler, server


class _DirectProviderHandler(BaseHTTPRequestHandler):
    requests: list[tuple[str, str, str]] = []
    responses: list[bytes] = []

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("content-length", "0") or "0")
        body = self.rfile.read(length).decode("utf-8", "ignore") if length else ""
        self.__class__.requests.append((self.command, self.path, body))
        payload = (
            self.__class__.responses.pop(0)
            if self.__class__.responses
            else b'{"choices":[{"message":{"content":"fix(core): default response."}}]}'
        )
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        return


def _start_direct_provider(
    responses: list[bytes],
) -> tuple[str, type[_DirectProviderHandler], HTTPServer]:
    class Handler(_DirectProviderHandler):
        requests: list[tuple[str, str, str]] = []

    Handler.responses = responses.copy()

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return f"http://127.0.0.1:{server.server_port}", Handler, server


def _stop_server(server: HTTPServer) -> None:
    server.shutdown()
    server.server_close()


def _rust_bin(binary: str) -> Path:
    subprocess.run(
        [
            "cargo",
            "build",
            "--locked",
            "--manifest-path",
            str(RUST_MANIFEST),
            "-p",
            "kcmt-cli",
            "--bin",
            binary,
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    binary_name = f"{binary}.exe" if os.name == "nt" else binary
    return REPO_ROOT / "rust" / "target" / "debug" / binary_name


@given(
    "a git repository with two changed tracked files",
    target_fixture="workflow_context",
)
def git_repository_with_two_changed_tracked_files(tmp_path: Path) -> dict[str, Any]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)
    (repo / "alpha.py").write_text("print('alpha')\n")
    (repo / "beta.py").write_text("print('beta')\n")
    _git(repo, ["add", "alpha.py", "beta.py"])
    _git(repo, ["commit", "-m", "chore(repo): seed"])

    (repo / "alpha.py").write_text("print('alpha updated')\n")
    (repo / "beta.py").write_text("print('beta updated')\n")
    return {"repo": repo, "config_home": tmp_path / "config-home"}


@given(
    "a git repository with ignored files and an untracked directory",
    target_fixture="workflow_context",
)
def git_repository_with_ignored_files_and_untracked_directory(
    tmp_path: Path,
) -> dict[str, Any]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)
    (repo / ".gitignore").write_text("ignored_dir/\n*.log\n")
    nested = repo / "newpkg" / "sub"
    nested.mkdir(parents=True)
    (nested / "alpha.py").write_text("print('alpha')\n")
    (nested / "beta.py").write_text("print('beta')\n")
    ignored = repo / "ignored_dir"
    ignored.mkdir()
    (ignored / "skip.txt").write_text("skip\n")
    (repo / "debug.log").write_text("skip\n")
    return {"repo": repo, "config_home": tmp_path / "config-home"}


@given(
    "a git repository with two changed tracked files and a mocked OpenAI batch provider",
    target_fixture="workflow_context",
)
def git_repository_with_two_changed_files_and_mocked_batch_provider(
    tmp_path: Path,
) -> dict[str, Any]:
    context = git_repository_with_two_changed_tracked_files(tmp_path)
    endpoint, handler, server = _start_batch_provider()
    context["endpoint"] = endpoint
    context["batch_handler"] = handler
    context["batch_server"] = server
    return context


@given(
    "a git repository with two changed tracked files and a partially invalid OpenAI batch provider",
    target_fixture="workflow_context",
)
def git_repository_with_two_changed_files_and_partial_batch_provider(
    tmp_path: Path,
) -> dict[str, Any]:
    context = git_repository_with_two_changed_tracked_files(tmp_path)
    endpoint, handler, server = _start_partial_batch_provider()
    context["endpoint"] = endpoint
    context["batch_handler"] = handler
    context["batch_server"] = server
    return context


@given(
    "a git repository with two changed tracked files and a mocked xAI batch provider",
    target_fixture="workflow_context",
)
def git_repository_with_two_changed_files_and_mocked_xai_batch_provider(
    tmp_path: Path,
) -> dict[str, Any]:
    context = git_repository_with_two_changed_tracked_files(tmp_path)
    endpoint, handler, server = _start_xai_batch_provider()
    context["xai_batch_endpoint"] = endpoint
    context["xai_batch_handler"] = handler
    context["xai_batch_server"] = server
    return context


@given(
    "a git repository with one deleted tracked file",
    target_fixture="workflow_context",
)
def git_repository_with_one_deleted_tracked_file(tmp_path: Path) -> dict[str, Any]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)
    (repo / "delete_me.txt").write_text("bye\n")
    _git(repo, ["add", "delete_me.txt"])
    _git(repo, ["commit", "-m", "chore(repo): seed"])

    (repo / "delete_me.txt").unlink()
    return {"repo": repo, "config_home": tmp_path / "config-home"}


@given(
    "a git repository with one deleted tracked file and one changed tracked file",
    target_fixture="workflow_context",
)
def git_repository_with_one_deleted_and_one_changed_tracked_file(
    tmp_path: Path,
) -> dict[str, Any]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)
    (repo / "delete_me.txt").write_text("bye\n")
    (repo / "tracked.py").write_text("print('seed')\n")
    _git(repo, ["add", "delete_me.txt", "tracked.py"])
    _git(repo, ["commit", "-m", "chore(repo): seed"])

    (repo / "delete_me.txt").unlink()
    (repo / "tracked.py").write_text("print('changed')\n")
    return {"repo": repo, "config_home": tmp_path / "config-home"}


@given(
    "a git repository with one changed tracked file",
    target_fixture="workflow_context",
)
def git_repository_with_one_changed_tracked_file(tmp_path: Path) -> dict[str, Any]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)
    (repo / "tracked.py").write_text("print('seed')\n")
    _git(repo, ["add", "tracked.py"])
    _git(repo, ["commit", "-m", "chore(repo): seed"])

    (repo / "tracked.py").write_text("print('changed')\n")
    return {"repo": repo, "config_home": tmp_path / "config-home"}


@given(
    "a git repository with one changed binary file and a mocked OpenAI provider",
    target_fixture="workflow_context",
)
def git_repository_with_one_changed_binary_file_and_mocked_provider(
    tmp_path: Path,
) -> dict[str, Any]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)
    assets = repo / "assets"
    assets.mkdir()
    (assets / "logo.png").write_bytes(b"\0PNG seed")
    _git(repo, ["add", "assets/logo.png"])
    _git(repo, ["commit", "-m", "chore(assets): seed logo"])
    (assets / "logo.png").write_bytes(b"\0PNG changed")
    endpoint, handler, server = _start_direct_provider(
        [b'{"choices":[{"message":{"content":"chore(assets): update logo."}}]}']
    )
    return {
        "repo": repo,
        "config_home": tmp_path / "config-home",
        "endpoint": endpoint,
        "direct_handler": handler,
        "direct_server": server,
    }


@given(
    "a git repository with one changed tracked file and a malformed-then-valid OpenAI provider",
    target_fixture="workflow_context",
)
def git_repository_with_malformed_then_valid_openai_provider(
    tmp_path: Path,
) -> dict[str, Any]:
    context = git_repository_with_one_changed_tracked_file(tmp_path)
    endpoint, handler, server = _start_direct_provider(
        [
            b'{"choices":[{"message":{"content":"This changes a few files"}}]}',
            b'{"choices":[{"message":{"content":"fix(core): retry simplified prompt."}}]}',
        ]
    )
    context["endpoint"] = endpoint
    context["direct_handler"] = handler
    context["direct_server"] = server
    return context


@given(
    "a git repository with one changed tracked file and Anthropic latest Haiku preferences",
    target_fixture="workflow_context",
)
def git_repository_with_anthropic_latest_haiku_preferences(
    tmp_path: Path,
) -> dict[str, Any]:
    context = git_repository_with_one_changed_tracked_file(tmp_path)
    endpoint, handler, server = _start_fallback_provider()
    config_home = context["config_home"]
    config_home.mkdir(parents=True, exist_ok=True)
    (config_home / "config.json").write_text(
        json.dumps(
            {
                "provider": "anthropic",
                "model": "claude-sonnet-4-20250514",
                "llm_endpoint": endpoint,
                "api_key_env": "ANTHROPIC_API_KEY",
                "git_repo_path": str(context["repo"]),
                "max_commit_length": 72,
                "auto_push": False,
                "use_batch": False,
                "batch_model": None,
                "batch_timeout_seconds": 900,
                "providers": {
                    "anthropic": {
                        "endpoint": endpoint,
                        "api_key_env": "ANTHROPIC_API_KEY",
                        "preferred_model": "claude-sonnet-4-20250514",
                    }
                },
                "model_priority": [
                    {"provider": "anthropic", "model": "claude-sonnet-4-20250514"}
                ],
            }
        )
    )
    (config_home / "preferences.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "selection_policy": "fastest_cheap",
                "provider_rules": {
                    "anthropic": {"preset": "latest_haiku", "strict": False}
                },
                "prompt_profiles": [
                    {
                        "id": "conventional",
                        "name": "Conventional Commit",
                        "system_instruction": "You generate strictly valid Conventional Commit messages.",
                        "user_instruction": "Only output the commit message.",
                    }
                ],
                "default_prompt_profile": "conventional",
                "tui": {},
                "model_cache": {"ttl_seconds": 86400},
            }
        )
    )
    context["anthropic_handler"] = handler
    context["anthropic_server"] = server
    return context


@given(
    "a git repository with one changed tracked file and a broken origin",
    target_fixture="workflow_context",
)
def git_repository_with_one_changed_tracked_file_and_broken_origin(
    tmp_path: Path,
) -> dict[str, Any]:
    context = git_repository_with_one_changed_tracked_file(tmp_path)
    _git(
        context["repo"],
        ["remote", "add", "origin", str(tmp_path / "missing-origin.git")],
    )
    return context


@given("an empty runtime configuration home", target_fixture="workflow_context")
def empty_runtime_configuration_home(tmp_path: Path) -> dict[str, Any]:
    config_home = tmp_path / "config-home"
    config_home.mkdir()
    repo = tmp_path / "repo"
    repo.mkdir()
    return {"repo": repo, "config_home": config_home}


@given("an existing OpenAI runtime configuration", target_fixture="workflow_context")
def existing_openai_runtime_configuration(tmp_path: Path) -> dict[str, Any]:
    config_home = tmp_path / "config-home"
    config_home.mkdir()
    repo = tmp_path / "repo"
    repo.mkdir()
    (config_home / "config.json").write_text(
        json.dumps(
            {
                "provider": "openai",
                "model": "gpt-existing",
                "llm_endpoint": "https://openai.existing/v1",
                "api_key_env": "OPENAI_EXISTING_KEY",
                "git_repo_path": str(repo),
                "max_commit_length": 72,
                "auto_push": False,
                "use_batch": False,
                "batch_model": "gpt-existing",
                "batch_timeout_seconds": 900,
                "providers": {
                    "openai": {
                        "name": "OpenAI",
                        "endpoint": "https://openai.existing/v1",
                        "api_key_env": "OPENAI_EXISTING_KEY",
                        "keychain_account": "provider/openai/existing",
                        "preferred_model": "gpt-existing",
                    },
                    "anthropic": {
                        "name": "Anthropic",
                        "endpoint": "https://anthropic.existing",
                        "api_key_env": "ANTHROPIC_EXISTING_KEY",
                        "preferred_model": "claude-existing",
                    },
                },
                "model_priority": [
                    {"provider": "openai", "model": "gpt-existing"},
                    {"provider": "anthropic", "model": "claude-existing"},
                ],
            }
        )
    )
    return {"repo": repo, "config_home": config_home}


@given("a checked-in runtime benchmark corpus", target_fixture="workflow_context")
def checked_in_runtime_benchmark_corpus(tmp_path: Path) -> dict[str, Any]:
    return {
        "repo": REPO_ROOT
        / "tests"
        / "fixtures"
        / "runtime_corpus"
        / "mini_realistic_repo",
        "config_home": tmp_path / "config-home",
    }


@given(
    "a git repository with one changed tracked file and a pushable origin",
    target_fixture="workflow_context",
)
def git_repository_with_one_changed_tracked_file_and_pushable_origin(
    tmp_path: Path,
) -> dict[str, Any]:
    repo = tmp_path / "repo"
    repo.mkdir()
    remote = tmp_path / "origin.git"
    remote.mkdir()
    _init_repo(repo)
    _init_bare_remote(remote)
    _git(repo, ["remote", "add", "origin", str(remote)])

    (repo / "tracked.py").write_text("print('seed')\n")
    _git(repo, ["add", "tracked.py"])
    _git(repo, ["commit", "-m", "chore(repo): seed"])
    _git(repo, ["push", "origin", "HEAD"])

    (repo / "tracked.py").write_text("print('changed')\n")
    return {"repo": repo, "remote": remote, "config_home": tmp_path / "config-home"}


@given(
    "a git repository with one changed tracked file and a fallback provider chain",
    target_fixture="workflow_context",
)
def git_repository_with_one_changed_file_and_fallback_provider_chain(
    tmp_path: Path,
) -> dict[str, Any]:
    context = git_repository_with_one_changed_tracked_file(tmp_path)
    endpoint, handler, server = _start_fallback_provider()
    config_home = context["config_home"]
    config_home.mkdir(parents=True)
    (config_home / "config.json").write_text(
        f"""{{
  "provider": "openai",
  "model": "gpt-primary",
  "llm_endpoint": "{endpoint}",
  "api_key_env": "OPENAI_TEST_KEY",
  "git_repo_path": ".",
  "max_commit_length": 72,
  "auto_push": false,
  "providers": {{
    "openai": {{"endpoint": "{endpoint}", "api_key_env": "OPENAI_TEST_KEY", "preferred_model": "gpt-primary"}},
    "anthropic": {{"endpoint": "{endpoint}", "api_key_env": "ANTHROPIC_TEST_KEY", "preferred_model": "claude-fallback"}}
  }},
  "model_priority": [
    {{"provider": "openai", "model": "gpt-primary"}},
    {{"provider": "anthropic", "model": "claude-fallback"}}
  ]
}}""",
        encoding="utf-8",
    )
    context["fallback_handler"] = handler
    context["fallback_server"] = server
    return context


@given(
    "a git repository with one changed tracked file and a mocked GitHub Models provider",
    target_fixture="workflow_context",
)
def git_repository_with_one_changed_tracked_file_and_mocked_github_provider(
    tmp_path: Path,
) -> dict[str, Any]:
    context = git_repository_with_one_changed_tracked_file(tmp_path)
    endpoint, handler, server = _start_github_token_provider()
    context["github_endpoint"] = endpoint
    context["github_handler"] = handler
    context["github_server"] = server
    return context


@given(
    "a git repository with one changed tracked file and a transiently failing provider",
    target_fixture="workflow_context",
)
def git_repository_with_one_changed_tracked_file_and_transient_provider(
    tmp_path: Path,
) -> dict[str, Any]:
    context = git_repository_with_one_changed_tracked_file(tmp_path)
    endpoint, handler, server = _start_retry_limited_provider()
    context["retry_endpoint"] = endpoint
    context["retry_handler"] = handler
    context["retry_server"] = server
    return context


@when(parsers.parse("the Rust {binary} command runs in oneshot mode"))
def rust_command_runs_in_oneshot_mode(
    workflow_context: dict[str, Any],
    binary: str,
) -> None:
    env = _clean_env(workflow_context["config_home"])
    output = _run(
        [
            str(_rust_bin(binary)),
            "--oneshot",
            "--no-auto-push",
            "--repo-path",
            str(workflow_context["repo"]),
        ],
        REPO_ROOT,
        env=env,
    )
    workflow_context["output"] = output


@when("the Rust kcmt command runs in default workflow mode")
def rust_kcmt_command_runs_in_default_workflow_mode(
    workflow_context: dict[str, Any],
) -> None:
    env = _clean_env(workflow_context["config_home"])
    output = _run(
        [
            str(_rust_bin("kcmt")),
            "--no-auto-push",
            "--repo-path",
            str(workflow_context["repo"]),
        ],
        REPO_ROOT,
        env=env,
    )
    workflow_context["output"] = output


@when("the Rust kcmt command runs in default workflow mode with two workers")
def rust_kcmt_command_runs_in_default_workflow_mode_with_two_workers(
    workflow_context: dict[str, Any],
) -> None:
    env = _clean_env(workflow_context["config_home"])
    output = _run(
        [
            str(_rust_bin("kcmt")),
            "--workers",
            "2",
            "--no-auto-push",
            "--repo-path",
            str(workflow_context["repo"]),
        ],
        REPO_ROOT,
        env=env,
    )
    workflow_context["output"] = output


@when("the Rust kcmt command runs in compact verbose profile mode")
def rust_kcmt_command_runs_in_compact_verbose_profile_mode(
    workflow_context: dict[str, Any],
) -> None:
    env = _clean_env(workflow_context["config_home"])
    output = _run(
        [
            str(_rust_bin("kcmt")),
            "--compact",
            "--verbose",
            "--profile-startup",
            "--no-auto-push",
            "--repo-path",
            str(workflow_context["repo"]),
        ],
        REPO_ROOT,
        env=env,
    )
    workflow_context["output"] = output


@when("the Rust commit command runs in oneshot mode with a file limit")
def rust_commit_command_runs_in_oneshot_mode_with_file_limit(
    workflow_context: dict[str, Any],
) -> None:
    env = _clean_env(workflow_context["config_home"])
    output = _run(
        [
            str(_rust_bin("commit")),
            "--oneshot",
            "--limit",
            "1",
            "--no-auto-push",
            "--repo-path",
            str(workflow_context["repo"]),
        ],
        REPO_ROOT,
        env=env,
    )
    workflow_context["output"] = output


@when("the Rust kcmt command runs in oneshot mode with auto-push")
def rust_kcmt_runs_in_oneshot_mode_with_auto_push(
    workflow_context: dict[str, Any],
) -> None:
    env = _clean_env(workflow_context["config_home"])
    output = _run(
        [
            str(_rust_bin("kcmt")),
            "--oneshot",
            "--auto-push",
            "--repo-path",
            str(workflow_context["repo"]),
        ],
        REPO_ROOT,
        env=env,
    )
    workflow_context["output"] = output


@when("the Rust kcmt command runs in default batch mode")
def rust_kcmt_runs_in_default_batch_mode(workflow_context: dict[str, Any]) -> None:
    env = _clean_env(workflow_context["config_home"])
    env["OPENAI_TEST_KEY"] = "test-key"
    output = _run(
        [
            str(_rust_bin("kcmt")),
            "--provider",
            "openai",
            "--endpoint",
            workflow_context["endpoint"],
            "--api-key-env",
            "OPENAI_TEST_KEY",
            "--model",
            "gpt-direct",
            "--batch",
            "--batch-model",
            "gpt-batch",
            "--batch-timeout",
            "900",
            "--no-auto-push",
            "--repo-path",
            str(workflow_context["repo"]),
        ],
        REPO_ROOT,
        env=env,
    )
    workflow_context["output"] = output
    workflow_context["batch_server"].shutdown()


@when("the Rust kcmt command runs in default xAI batch mode")
def rust_kcmt_runs_in_default_xai_batch_mode(
    workflow_context: dict[str, Any],
) -> None:
    env = _clean_env(workflow_context["config_home"])
    env["XAI_TEST_KEY"] = "test-key"
    output = _run(
        [
            str(_rust_bin("kcmt")),
            "--provider",
            "xai",
            "--endpoint",
            workflow_context["xai_batch_endpoint"],
            "--api-key-env",
            "XAI_TEST_KEY",
            "--model",
            "grok-code-fast",
            "--batch",
            "--batch-model",
            "grok-4.3",
            "--batch-timeout",
            "900",
            "--no-auto-push",
            "--repo-path",
            str(workflow_context["repo"]),
        ],
        REPO_ROOT,
        env=env,
    )
    workflow_context["output"] = output
    workflow_context["xai_batch_server"].shutdown()


@when("the Rust kcmt command commits the file using configured provider fallback")
def rust_kcmt_commits_file_using_configured_provider_fallback(
    workflow_context: dict[str, Any],
) -> None:
    env = _clean_env(workflow_context["config_home"])
    env["OPENAI_TEST_KEY"] = "primary-key"
    env["ANTHROPIC_TEST_KEY"] = "fallback-key"
    output = _run(
        [
            str(_rust_bin("kcmt")),
            "--file",
            "tracked.py",
            "--repo-path",
            str(workflow_context["repo"]),
        ],
        REPO_ROOT,
        env=env,
    )
    workflow_context["output"] = output
    workflow_context["fallback_server"].shutdown()


@when("the Rust kcmt command commits the file using a GitHub token flag")
def rust_kcmt_commits_file_using_github_token_flag(
    workflow_context: dict[str, Any],
) -> None:
    env = _clean_env(workflow_context["config_home"])
    output = _run(
        [
            str(_rust_bin("kcmt")),
            "--file",
            "tracked.py",
            "--provider",
            "github",
            "--endpoint",
            workflow_context["github_endpoint"],
            "--api-key-env",
            "GITHUB_TOKEN",
            "--model",
            "openai/gpt-test",
            "--github-token",
            "bdd-gh-token",
            "--no-auto-push",
            "--repo-path",
            str(workflow_context["repo"]),
        ],
        REPO_ROOT,
        env=env,
    )
    workflow_context["output"] = output
    workflow_context["github_server"].shutdown()


@when("the Rust kcmt command commits the file with max retries set to zero")
def rust_kcmt_commits_file_with_max_retries_zero(
    workflow_context: dict[str, Any],
) -> None:
    env = _clean_env(workflow_context["config_home"])
    env["OPENAI_TEST_KEY"] = "test-key"
    result = subprocess.run(
        [
            str(_rust_bin("kcmt")),
            "--file",
            "tracked.py",
            "--provider",
            "openai",
            "--endpoint",
            workflow_context["retry_endpoint"],
            "--api-key-env",
            "OPENAI_TEST_KEY",
            "--model",
            "gpt-retry",
            "--max-retries",
            "0",
            "--no-auto-push",
            "--repo-path",
            str(workflow_context["repo"]),
        ],
        cwd=REPO_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    workflow_context["result"] = result
    workflow_context["output"] = result.stdout
    workflow_context["stderr"] = result.stderr
    workflow_context["retry_server"].shutdown()


@when("the Python kcmt entrypoint commits the file in auto runtime mode")
def python_kcmt_entrypoint_commits_file_in_auto_runtime_mode(
    workflow_context: dict[str, Any],
) -> None:
    rust_bin = _rust_bin("kcmt")
    env = _clean_env(workflow_context["config_home"])
    env["KCMT_RUST_BIN"] = str(rust_bin)
    env["KCMT_RUNTIME_TRACE"] = "1"
    result = subprocess.run(
        [
            os.environ.get("PYTHON", "python"),
            "-m",
            "kcmt.main",
            "--file",
            "tracked.py",
            "--no-auto-push",
            "--repo-path",
            str(workflow_context["repo"]),
        ],
        cwd=REPO_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    workflow_context["output"] = result.stdout
    workflow_context["stderr"] = result.stderr


@when("the Python kcmt entrypoint runs the default workflow in auto runtime mode")
def python_kcmt_entrypoint_runs_default_workflow_in_auto_runtime_mode(
    workflow_context: dict[str, Any],
) -> None:
    rust_bin = _rust_bin("kcmt")
    env = _clean_env(workflow_context["config_home"])
    env["KCMT_RUST_BIN"] = str(rust_bin)
    env["KCMT_RUNTIME_TRACE"] = "1"
    result = subprocess.run(
        [
            os.environ.get("PYTHON", "python"),
            "-m",
            "kcmt.main",
            "--no-auto-push",
            "--repo-path",
            str(workflow_context["repo"]),
        ],
        cwd=REPO_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    workflow_context["output"] = result.stdout
    workflow_context["stderr"] = result.stderr


@when("the Rust kcmt command configures Anthropic non-interactively")
def rust_kcmt_configures_anthropic_non_interactively(
    workflow_context: dict[str, Any],
) -> None:
    env = _clean_env(workflow_context["config_home"])
    output = _run(
        [
            str(_rust_bin("kcmt")),
            "--configure",
            "--provider",
            "anthropic",
            "--model",
            "claude-test",
            "--endpoint",
            "https://anthropic.test",
            "--api-key-env",
            "ANTHROPIC_TEST_KEY",
            "--no-auto-push",
            "--repo-path",
            str(workflow_context["repo"]),
        ],
        REPO_ROOT,
        env=env,
    )
    workflow_context["output"] = output


@when("the Python kcmt entrypoint configures Anthropic in auto runtime mode")
def python_kcmt_entrypoint_configures_anthropic_in_auto_runtime_mode(
    workflow_context: dict[str, Any],
) -> None:
    rust_bin = _rust_bin("kcmt")
    env = _clean_env(workflow_context["config_home"])
    env["KCMT_RUST_BIN"] = str(rust_bin)
    env["KCMT_RUNTIME_TRACE"] = "1"
    result = subprocess.run(
        [
            os.environ.get("PYTHON", "python"),
            "-m",
            "kcmt.main",
            "--configure",
            "--provider",
            "anthropic",
            "--model",
            "claude-test",
            "--endpoint",
            "https://anthropic.test",
            "--api-key-env",
            "ANTHROPIC_TEST_KEY",
            "--no-auto-push",
            "--repo-path",
            str(workflow_context["repo"]),
        ],
        cwd=REPO_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    workflow_context["output"] = result.stdout
    workflow_context["stderr"] = result.stderr


@when("the Python kcmt entrypoint runs bare configure in auto runtime mode")
def python_kcmt_entrypoint_runs_bare_configure_in_auto_runtime_mode(
    workflow_context: dict[str, Any],
) -> None:
    rust_bin = _rust_bin("kcmt")
    env = _clean_env(workflow_context["config_home"])
    env["KCMT_RUST_BIN"] = str(rust_bin)
    env["KCMT_RUNTIME_TRACE"] = "1"
    result = subprocess.run(
        [
            os.environ.get("PYTHON", "python"),
            "-m",
            "kcmt.main",
            "--configure",
            "--repo-path",
            str(workflow_context["repo"]),
        ],
        cwd=REPO_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    workflow_context["output"] = result.stdout
    workflow_context["stderr"] = result.stderr


@when("the Python kcmt entrypoint configures all providers in auto runtime mode")
def python_kcmt_entrypoint_configures_all_providers_in_auto_runtime_mode(
    workflow_context: dict[str, Any],
) -> None:
    rust_bin = _rust_bin("kcmt")
    env = _clean_env(workflow_context["config_home"])
    env["KCMT_RUST_BIN"] = str(rust_bin)
    env["KCMT_RUNTIME_TRACE"] = "1"
    result = subprocess.run(
        [
            os.environ.get("PYTHON", "python"),
            "-m",
            "kcmt.main",
            "--configure-all",
            "--provider",
            "anthropic",
            "--model",
            "claude-new",
            "--endpoint",
            "https://anthropic.new",
            "--api-key-env",
            "ANTHROPIC_NEW_KEY",
            "--repo-path",
            str(workflow_context["repo"]),
        ],
        cwd=REPO_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    workflow_context["output"] = result.stdout
    workflow_context["stderr"] = result.stderr


@when("the Rust kcmt command configures all providers with Anthropic overrides")
def rust_kcmt_configures_all_providers_with_anthropic_overrides(
    workflow_context: dict[str, Any],
) -> None:
    env = _clean_env(workflow_context["config_home"])
    output = _run(
        [
            str(_rust_bin("kcmt")),
            "--configure-all",
            "--provider",
            "anthropic",
            "--model",
            "claude-new",
            "--endpoint",
            "https://anthropic.new",
            "--api-key-env",
            "ANTHROPIC_NEW_KEY",
            "--repo-path",
            str(workflow_context["repo"]),
        ],
        REPO_ROOT,
        env=env,
    )
    workflow_context["output"] = output


@when("the Rust kcmt command commits the file with Anthropic preferences")
def rust_kcmt_commits_file_with_anthropic_preferences(
    workflow_context: dict[str, Any],
) -> None:
    env = _clean_env(workflow_context["config_home"])
    env["ANTHROPIC_API_KEY"] = "bdd-anthropic-key"
    output = _run(
        [
            str(_rust_bin("kcmt")),
            "--file",
            "tracked.py",
            "--no-auto-push",
            "--repo-path",
            str(workflow_context["repo"]),
        ],
        REPO_ROOT,
        env=env,
    )
    workflow_context["output"] = output


@when("the Rust kcmt command lists models")
def rust_kcmt_lists_models(workflow_context: dict[str, Any]) -> None:
    env = _clean_env(workflow_context["config_home"])
    output = _run([str(_rust_bin("kcmt")), "--list-models"], REPO_ROOT, env=env)
    workflow_context["output"] = output


@when("the Rust kcmt command lists models in debug mode")
def rust_kcmt_lists_models_in_debug_mode(workflow_context: dict[str, Any]) -> None:
    env = _clean_env(workflow_context["config_home"])
    output = _run(
        [str(_rust_bin("kcmt")), "--debug", "--list-models"],
        REPO_ROOT,
        env=env,
    )
    workflow_context["output"] = output


@when("the Rust kcmt command verifies API keys")
def rust_kcmt_verifies_api_keys(workflow_context: dict[str, Any]) -> None:
    env = _clean_env(workflow_context["config_home"])
    env["OPENAI_API_KEY"] = "bdd-key"
    output = _run([str(_rust_bin("kcmt")), "--verify-keys"], REPO_ROOT, env=env)
    workflow_context["output"] = output


@when("the Rust kcmt command benchmarks a provider with structured outputs")
def rust_kcmt_benchmarks_provider_with_structured_outputs(
    workflow_context: dict[str, Any],
) -> None:
    env = _clean_env(workflow_context["config_home"])
    env["OPENAI_API_KEY"] = "bdd-openai-key"
    env["KCMT_PROVIDER_RESPONSE"] = "fix(core): update benchmark"
    env["KCMT_ALLOW_PROVIDER_RESPONSE_FIXTURE"] = "1"
    output = _run(
        [
            str(_rust_bin("kcmt")),
            "--benchmark",
            "--provider",
            "openai",
            "--model",
            "gpt-bdd",
            "--benchmark-limit",
            "1",
            "--benchmark-json",
            "--benchmark-csv",
            "--repo-path",
            str(workflow_context["repo"]),
        ],
        REPO_ROOT,
        env=env,
    )
    workflow_context["output"] = output


@when("the Rust kcmt runtime benchmark runs against the corpus")
def rust_kcmt_runtime_benchmark_runs_against_the_corpus(
    workflow_context: dict[str, Any],
) -> None:
    rust_bin = _rust_bin("kcmt")
    env = _clean_env(workflow_context["config_home"])
    env["KCMT_PROVIDER_RESPONSE"] = "chore(repo): benchmark fake response"
    env["KCMT_ALLOW_PROVIDER_RESPONSE_FIXTURE"] = "1"
    output = _run(
        [
            str(rust_bin),
            "benchmark",
            "runtime",
            "--repo-path",
            str(workflow_context["repo"]),
            "--runtime",
            "both",
            "--iterations",
            "1",
            "--rust-bin",
            str(rust_bin),
            "--json",
        ],
        REPO_ROOT,
        env=env,
    )
    workflow_context["output"] = output


@when("the Rust kcmt command commits the file with config overrides")
def rust_kcmt_commits_the_file_with_config_overrides(
    workflow_context: dict[str, Any],
) -> None:
    env = _clean_env(workflow_context["config_home"])
    output = _run(
        [
            str(_rust_bin("kcmt")),
            "--file",
            "tracked.py",
            "--provider",
            "openai",
            "--model",
            "gpt-test",
            "--endpoint",
            "https://example.test/v1",
            "--api-key-env",
            "OPENAI_TEST_KEY",
            "--batch",
            "--batch-model",
            "gpt-batch-test",
            "--batch-timeout",
            "1000",
            "--no-auto-push",
            "--max-commit-length",
            "68",
            "--repo-path",
            str(workflow_context["repo"]),
        ],
        REPO_ROOT,
        env=env,
    )
    workflow_context["output"] = output


@when("the Rust kcmt command commits the file with wrapped provider output")
def rust_kcmt_commits_the_file_with_wrapped_provider_output(
    workflow_context: dict[str, Any],
) -> None:
    env = _clean_env(workflow_context["config_home"])
    env["KCMT_PROVIDER_RESPONSE"] = (
        "```text\nHere is the commit:\n" "- `fix(core): handle provider output.`\n```"
    )
    env["KCMT_ALLOW_PROVIDER_RESPONSE_FIXTURE"] = "1"
    output = _run(
        [
            str(_rust_bin("kcmt")),
            "--file",
            "tracked.py",
            "--no-auto-push",
            "--repo-path",
            str(workflow_context["repo"]),
        ],
        REPO_ROOT,
        env=env,
    )
    workflow_context["output"] = output


@when("the Rust kcmt command commits the binary file with the mocked provider")
def rust_kcmt_commits_binary_file_with_mocked_provider(
    workflow_context: dict[str, Any],
) -> None:
    env = _clean_env(workflow_context["config_home"])
    env["OPENAI_TEST_KEY"] = "bdd-openai-key"
    try:
        output = _run(
            [
                str(_rust_bin("kcmt")),
                "--file",
                "assets/logo.png",
                "--provider",
                "openai",
                "--endpoint",
                workflow_context["endpoint"],
                "--api-key-env",
                "OPENAI_TEST_KEY",
                "--model",
                "gpt-bdd",
                "--no-auto-push",
                "--repo-path",
                str(workflow_context["repo"]),
            ],
            REPO_ROOT,
            env=env,
        )
        workflow_context["output"] = output
    finally:
        _stop_server(workflow_context["direct_server"])


@when(
    "the Rust kcmt command commits the file with malformed then valid provider output"
)
def rust_kcmt_commits_file_with_malformed_then_valid_provider_output(
    workflow_context: dict[str, Any],
) -> None:
    env = _clean_env(workflow_context["config_home"])
    env["OPENAI_TEST_KEY"] = "bdd-openai-key"
    try:
        output = _run(
            [
                str(_rust_bin("kcmt")),
                "--file",
                "tracked.py",
                "--provider",
                "openai",
                "--endpoint",
                workflow_context["endpoint"],
                "--api-key-env",
                "OPENAI_TEST_KEY",
                "--model",
                "gpt-bdd",
                "--no-auto-push",
                "--repo-path",
                str(workflow_context["repo"]),
            ],
            REPO_ROOT,
            env=env,
        )
        workflow_context["output"] = output
    finally:
        _stop_server(workflow_context["direct_server"])


@when("the Rust kcmt command receives invalid provider output")
def rust_kcmt_receives_invalid_provider_output(
    workflow_context: dict[str, Any],
) -> None:
    env = _clean_env(workflow_context["config_home"])
    env["KCMT_PROVIDER_RESPONSE"] = "This changes a few files"
    env["KCMT_ALLOW_PROVIDER_RESPONSE_FIXTURE"] = "1"
    result = subprocess.run(
        [
            str(_rust_bin("kcmt")),
            "--file",
            "tracked.py",
            "--no-auto-push",
            "--repo-path",
            str(workflow_context["repo"]),
        ],
        cwd=REPO_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    workflow_context["result"] = result


@when("the Rust kcmt command runs default mode with invalid provider output")
def rust_kcmt_runs_default_mode_with_invalid_provider_output(
    workflow_context: dict[str, Any],
) -> None:
    env = _clean_env(workflow_context["config_home"])
    env["KCMT_PROVIDER_RESPONSE"] = "This changes a few files"
    env["KCMT_ALLOW_PROVIDER_RESPONSE_FIXTURE"] = "1"
    result = subprocess.run(
        [
            str(_rust_bin("kcmt")),
            "--no-auto-push",
            "--repo-path",
            str(workflow_context["repo"]),
        ],
        cwd=REPO_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    workflow_context["result"] = result
    workflow_context["output"] = result.stdout


@when("the Rust kcmt command receives fixture provider output without opt in")
def rust_kcmt_receives_fixture_provider_output_without_opt_in(
    workflow_context: dict[str, Any],
) -> None:
    env = _clean_env(workflow_context["config_home"])
    env["KCMT_PROVIDER_RESPONSE"] = "fix(core): should be ignored"
    env.pop("KCMT_ALLOW_PROVIDER_RESPONSE_FIXTURE", None)
    env.pop("KCMT_ALLOW_LOCAL_SYNTHESIS", None)
    result = subprocess.run(
        [
            str(_rust_bin("kcmt")),
            "--file",
            "tracked.py",
            "--no-auto-push",
            "--repo-path",
            str(workflow_context["repo"]),
        ],
        cwd=REPO_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    workflow_context["result"] = result


@then("both changed files are committed separately")
def both_changed_files_are_committed_separately(
    workflow_context: dict[str, Any],
) -> None:
    output = workflow_context["output"]
    assert "✓ alpha.py" in output
    assert "✓ beta.py" in output

    log = _git(workflow_context["repo"], ["log", "--pretty=%s"])
    subjects = log.splitlines()
    assert "chore(repo): update alpha" in subjects
    assert "chore(repo): update beta" in subjects


@then("the untracked directory files are committed separately")
def untracked_directory_files_are_committed_separately(
    workflow_context: dict[str, Any],
) -> None:
    output = workflow_context["output"]
    assert "✓ newpkg/sub/alpha.py" in output
    assert "✓ newpkg/sub/beta.py" in output
    log = _git(workflow_context["repo"], ["log", "--name-only", "--pretty=%s"])
    assert "newpkg/sub/alpha.py" in log
    assert "newpkg/sub/beta.py" in log


@then("ignored files are not committed")
def ignored_files_are_not_committed(workflow_context: dict[str, Any]) -> None:
    output = workflow_context["output"]
    assert "ignored_dir" not in output
    assert "debug.log" not in output
    log = _git(workflow_context["repo"], ["log", "--name-only", "--pretty=%s"])
    assert "ignored_dir/skip.txt" not in log
    assert "debug.log" not in log


@then("the repository worktree is clean")
def repository_worktree_is_clean(workflow_context: dict[str, Any]) -> None:
    status = _git(workflow_context["repo"], ["status", "--short"])
    assert status == ""


@then("exactly one changed file is committed")
def exactly_one_changed_file_is_committed(workflow_context: dict[str, Any]) -> None:
    output = workflow_context["output"]
    committed = [name for name in ("alpha.py", "beta.py") if f"✓ {name}" in output]
    assert len(committed) == 1
    workflow_context["limited_committed_file"] = committed[0]

    log = _git(workflow_context["repo"], ["log", "--pretty=%s"])
    subjects = log.splitlines()
    committed_subjects = [
        subject
        for subject in (
            "chore(repo): update alpha",
            "chore(repo): update beta",
        )
        if subject in subjects
    ]
    assert len(committed_subjects) == 1


@then("one changed file remains uncommitted")
def one_changed_file_remains_uncommitted(workflow_context: dict[str, Any]) -> None:
    committed = workflow_context["limited_committed_file"]
    remaining = "beta.py" if committed == "alpha.py" else "alpha.py"
    status = _git(workflow_context["repo"], ["status", "--short"])
    assert status == f"M {remaining}"


@then("the deleted file is committed with the deletion message")
def deleted_file_is_committed_with_the_deletion_message(
    workflow_context: dict[str, Any],
) -> None:
    output = workflow_context["output"]
    assert "✓ delete_me.txt" in output
    assert "chore(delete_me-txt): file deleted" in output

    log = _git(workflow_context["repo"], ["log", "--oneline", "-1"])
    assert "chore(delete_me-txt): file deleted" in log


@then("the deletion is recorded in the raw status snapshot")
def deletion_is_recorded_in_the_raw_status_snapshot(
    workflow_context: dict[str, Any],
) -> None:
    env = _clean_env(workflow_context["config_home"])
    raw_status = _run(
        [
            str(_rust_bin("kcmt")),
            "status",
            "--raw",
            "--repo-path",
            str(workflow_context["repo"]),
        ],
        REPO_ROOT,
        env=env,
    )

    assert '"deletions_total": 1' in raw_status
    assert '"deletions_success": 1' in raw_status
    assert '"file_path": "delete_me.txt"' in raw_status
    assert '"telemetry"' in raw_status
    assert '"schema_version": 1' in raw_status
    assert '"stage": "status_scan"' in raw_status
    assert '"stage": "commit"' in raw_status


@then("the changed file remains uncommitted with a recorded prepare failure")
def changed_file_remains_uncommitted_with_recorded_prepare_failure(
    workflow_context: dict[str, Any],
) -> None:
    result = workflow_context["result"]
    assert result.returncode == 0
    assert "✗ tracked.py" in result.stdout
    assert "missing conventional commit header" in result.stdout

    status = _git(workflow_context["repo"], ["status", "--short"])
    assert status == "M tracked.py"

    env = _clean_env(workflow_context["config_home"])
    raw_status = _run(
        [
            str(_rust_bin("kcmt")),
            "status",
            "--raw",
            "--repo-path",
            str(workflow_context["repo"]),
        ],
        REPO_ROOT,
        env=env,
    )
    snapshot = json.loads(raw_status)
    assert snapshot["counts"]["files_total"] == 2
    assert snapshot["counts"]["prepared_total"] == 1
    assert snapshot["counts"]["prepared_failures"] == 1
    assert snapshot["counts"]["commit_failure"] == 1
    assert snapshot["counts"]["deletions_success"] == 1
    assert snapshot["counts"]["overall_success"] == 1
    assert snapshot["counts"]["overall_failure"] == 1
    assert snapshot["commits"][0]["success"] is False
    assert snapshot["commits"][0]["file_path"] == "tracked.py"


@then("the raw status snapshot records two prepare workers")
def raw_status_snapshot_records_two_prepare_workers(
    workflow_context: dict[str, Any],
) -> None:
    env = _clean_env(workflow_context["config_home"])
    raw_status = _run(
        [
            str(_rust_bin("kcmt")),
            "status",
            "--raw",
            "--repo-path",
            str(workflow_context["repo"]),
        ],
        REPO_ROOT,
        env=env,
    )
    snapshot = json.loads(raw_status)
    assert snapshot["telemetry"]["prepare_workers"] == 2


@then("the compact workflow output includes summary and commit details")
def compact_workflow_output_includes_summary_and_commit_details(
    workflow_context: dict[str, Any],
) -> None:
    output = workflow_context["output"]
    assert "Run Summary" in output
    assert "Commits 1  Failures 0" in output
    assert "Latest commit: chore(repo): update tracked" in output
    assert "✓ tracked.py" in output


@then("the compact workflow output includes profile timings")
def compact_workflow_output_includes_profile_timings(
    workflow_context: dict[str, Any],
) -> None:
    output = workflow_context["output"]
    assert "[kcmt-profile] status_scan:" in output
    assert "[kcmt-profile] snapshot:" in output


@then("the raw status snapshot records the config overrides")
def raw_status_snapshot_records_the_config_overrides(
    workflow_context: dict[str, Any],
) -> None:
    env = _clean_env(workflow_context["config_home"])
    raw_status = _run(
        [
            str(_rust_bin("kcmt")),
            "status",
            "--raw",
            "--repo-path",
            str(workflow_context["repo"]),
        ],
        REPO_ROOT,
        env=env,
    )

    assert '"provider": "openai"' in raw_status
    assert '"model": "gpt-test"' in raw_status
    assert '"endpoint": "https://example.test/v1"' in raw_status
    assert '"api_key_env": "OPENAI_TEST_KEY"' in raw_status
    assert '"max_commit_length": 68' in raw_status
    assert '"auto_push": false' in raw_status
    assert '"use_batch": true' in raw_status
    assert '"batch_model": "gpt-batch-test"' in raw_status
    assert '"batch_timeout_seconds": 1000' in raw_status
    assert '"telemetry"' in raw_status
    assert '"stage": "snapshot"' in raw_status


@then("the latest commit uses the sanitized provider message")
def latest_commit_uses_the_sanitized_provider_message(
    workflow_context: dict[str, Any],
) -> None:
    assert "fix(core): handle provider output" in workflow_context["output"]
    log = _git(workflow_context["repo"], ["log", "--pretty=%s", "-1"])
    assert log == "fix(core): handle provider output"


@then("the mocked provider prompt includes the binary diff summary")
def mocked_provider_prompt_includes_binary_diff_summary(
    workflow_context: dict[str, Any],
) -> None:
    requests = workflow_context["direct_handler"].requests
    assert len(requests) == 1
    body = requests[0][2]
    assert "Binary diff detected." in body
    assert "File hint: assets/logo.png" in body
    assert "Git reported:" in body


@then("the latest commit uses the binary provider message")
def latest_commit_uses_the_binary_provider_message(
    workflow_context: dict[str, Any],
) -> None:
    assert "chore(assets): update logo" in workflow_context["output"]
    log = _git(workflow_context["repo"], ["log", "--pretty=%s", "-1"])
    assert log == "chore(assets): update logo"


@then("the mocked provider receives a simplified retry prompt")
def mocked_provider_receives_a_simplified_retry_prompt(
    workflow_context: dict[str, Any],
) -> None:
    requests = workflow_context["direct_handler"].requests
    assert len(requests) == 2
    first_body = requests[0][2]
    retry_body = requests[1][2]
    assert "STRICT REQUIREMENTS" in first_body
    assert "Keep it simple but include mandatory scope." in retry_body
    assert "STRICT REQUIREMENTS" not in retry_body


@then("the latest commit uses the simplified retry provider message")
def latest_commit_uses_the_simplified_retry_provider_message(
    workflow_context: dict[str, Any],
) -> None:
    assert "fix(core): retry simplified prompt" in workflow_context["output"]
    log = _git(workflow_context["repo"], ["log", "--pretty=%s", "-1"])
    assert log == "fix(core): retry simplified prompt"


@then("the workflow fails before committing the file")
def workflow_fails_before_committing_the_file(workflow_context: dict[str, Any]) -> None:
    result = workflow_context["result"]
    assert result.returncode == 1
    assert "missing conventional commit header" in result.stderr

    status = _git(workflow_context["repo"], ["status", "--short"])
    assert "tracked.py" in status
    log = _git(workflow_context["repo"], ["log", "--pretty=%s", "-1"])
    assert log == "chore(repo): seed"


@then("the fixture provider output is ignored before committing the file")
def fixture_provider_output_is_ignored_before_committing_the_file(
    workflow_context: dict[str, Any],
) -> None:
    result = workflow_context["result"]
    assert result.returncode == 1
    assert "No API key available" in result.stderr
    assert "should be ignored" not in result.stderr

    status = _git(workflow_context["repo"], ["status", "--short"])
    assert "tracked.py" in status
    log = _git(workflow_context["repo"], ["log", "--pretty=%s", "-1"])
    assert log == "chore(repo): seed"


@then("the latest commit is pushed to origin")
def latest_commit_is_pushed_to_origin(workflow_context: dict[str, Any]) -> None:
    assert "Auto-push: pushed" in workflow_context["output"]
    local_head = _git(workflow_context["repo"], ["rev-parse", "HEAD"])
    branch = _git(workflow_context["repo"], ["rev-parse", "--abbrev-ref", "HEAD"])
    remote_head = _git(
        workflow_context["remote"], ["rev-parse", f"refs/heads/{branch}"]
    )
    assert remote_head == local_head


@then("the raw status snapshot records auto-push success")
def raw_status_snapshot_records_auto_push_success(
    workflow_context: dict[str, Any],
) -> None:
    env = _clean_env(workflow_context["config_home"])
    raw_status = _run(
        [
            str(_rust_bin("kcmt")),
            "status",
            "--raw",
            "--repo-path",
            str(workflow_context["repo"]),
        ],
        REPO_ROOT,
        env=env,
    )

    assert '"pushed": true' in raw_status
    assert '"auto_push_state": "pushed"' in raw_status
    assert '"errors": []' in raw_status


@then("the latest commit remains in the local repository")
def latest_commit_remains_in_the_local_repository(
    workflow_context: dict[str, Any],
) -> None:
    assert "Auto-push: failed" in workflow_context["output"]
    log = _git(workflow_context["repo"], ["log", "--pretty=%s", "-1"])
    assert log == "chore(repo): update tracked"


@then("the latest commit remains in the local repository without auto-push failure")
def latest_commit_remains_without_auto_push_failure(
    workflow_context: dict[str, Any],
) -> None:
    assert "Auto-push: failed" not in workflow_context["output"]
    log = _git(workflow_context["repo"], ["log", "--pretty=%s", "-1"])
    assert log == "chore(repo): update tracked"


@then("the raw status snapshot records auto-push failure")
def raw_status_snapshot_records_auto_push_failure(
    workflow_context: dict[str, Any],
) -> None:
    env = _clean_env(workflow_context["config_home"])
    raw_status = _run(
        [
            str(_rust_bin("kcmt")),
            "status",
            "--raw",
            "--repo-path",
            str(workflow_context["repo"]),
        ],
        REPO_ROOT,
        env=env,
    )

    assert '"pushed": false' in raw_status
    assert '"auto_push_state": "failed"' in raw_status
    assert "Auto-push failed:" in raw_status


@then("the raw status snapshot records auto-push skip")
def raw_status_snapshot_records_auto_push_skip(
    workflow_context: dict[str, Any],
) -> None:
    env = _clean_env(workflow_context["config_home"])
    raw_status = _run(
        [
            str(_rust_bin("kcmt")),
            "status",
            "--raw",
            "--repo-path",
            str(workflow_context["repo"]),
        ],
        REPO_ROOT,
        env=env,
    )

    assert '"pushed": false' in raw_status
    assert '"auto_push_state": "skipped"' in raw_status
    assert '"errors": []' in raw_status


@then("the batch provider receives both file prompts before commits are written")
def batch_provider_receives_both_file_prompts_before_commits_are_written(
    workflow_context: dict[str, Any],
) -> None:
    requests = workflow_context["batch_handler"].requests
    assert [request[1] for request in requests] == [
        "/files",
        "/batches",
        "/batches/batch_1",
        "/files/output_1/content",
    ]
    upload_body = requests[0][2]
    assert "alpha.py" in upload_body
    assert "beta.py" in upload_body
    assert "print('alpha updated')" in upload_body
    assert "print('beta updated')" in upload_body
    assert all(request[1] != "/chat/completions" for request in requests)


@then("the xAI batch provider receives both file prompts before commits are written")
def xai_batch_provider_receives_both_file_prompts_before_commits_are_written(
    workflow_context: dict[str, Any],
) -> None:
    requests = workflow_context["xai_batch_handler"].requests
    assert [request[1] for request in requests] == [
        "/batches",
        "/batches/batch_1/requests",
        "/batches/batch_1",
        "/batches/batch_1/results?limit=100",
    ]
    requests_body = requests[1][2]
    assert "alpha.py" in requests_body
    assert "beta.py" in requests_body
    assert "print('alpha updated')" in requests_body
    assert "print('beta updated')" in requests_body
    assert '"model":"grok-4.3"' in requests_body
    assert all(request[1] != "/chat/completions" for request in requests)


@then("both files are committed with the batch provider messages")
def both_files_are_committed_with_the_batch_provider_messages(
    workflow_context: dict[str, Any],
) -> None:
    output = workflow_context["output"]
    assert "fix(alpha): batch alpha" in output
    assert "fix(beta): batch beta" in output
    log = _git(workflow_context["repo"], ["log", "--pretty=%s"])
    subjects = log.splitlines()
    assert "fix(alpha): batch alpha" in subjects
    assert "fix(beta): batch beta" in subjects


@then("only the valid batch file is committed")
def only_the_valid_batch_file_is_committed(workflow_context: dict[str, Any]) -> None:
    output = workflow_context["output"]
    assert "fix(alpha): batch alpha" in output
    assert "LLM output missing conventional commit header" in output
    log = _git(workflow_context["repo"], ["log", "--pretty=%s"])
    subjects = log.splitlines()
    assert "fix(alpha): batch alpha" in subjects
    assert "This is not conventional" not in subjects
    status = _git(workflow_context["repo"], ["status", "--short"])
    assert "beta.py" in status
    assert "alpha.py" not in status


@then("provider output and status remain secret-free")
def provider_output_and_status_remain_secret_free(
    workflow_context: dict[str, Any],
) -> None:
    output = workflow_context["output"]
    assert "test-key" not in output
    raw_status = _run(
        [
            str(_rust_bin("kcmt")),
            "status",
            "--raw",
            "--repo-path",
            str(workflow_context["repo"]),
        ],
        REPO_ROOT,
        env=_clean_env(workflow_context["config_home"]),
    )
    assert "test-key" not in raw_status
    assert '"api_key_env": "OPENAI_TEST_KEY"' in raw_status
    assert '"commit_success": 1' in raw_status
    assert '"commit_failure": 1' in raw_status


@then("the fallback provider is used after the primary provider fails")
def fallback_provider_is_used_after_the_primary_provider_fails(
    workflow_context: dict[str, Any],
) -> None:
    requests = workflow_context["fallback_handler"].requests
    assert len(requests) == 5
    assert [request[1] for request in requests[:4]] == [
        "/chat/completions",
        "/chat/completions",
        "/chat/completions",
        "/chat/completions",
    ]
    assert requests[4][1] == "/v1/messages"


@then("the latest commit uses the fallback provider message")
def latest_commit_uses_the_fallback_provider_message(
    workflow_context: dict[str, Any],
) -> None:
    assert "fix(fallback): use secondary provider" in workflow_context["output"]
    log = _git(workflow_context["repo"], ["log", "--pretty=%s", "-1"])
    assert log == "fix(fallback): use secondary provider"


@then("the GitHub provider receives the CLI token")
def github_provider_receives_the_cli_token(
    workflow_context: dict[str, Any],
) -> None:
    requests = workflow_context["github_handler"].requests
    assert len(requests) == 1
    method, path, auth, body = requests[0]
    assert method == "POST"
    assert path == "/chat/completions"
    assert auth == "Bearer bdd-gh-token"
    assert "tracked.py" in body


@then("the latest commit uses the GitHub provider message")
def latest_commit_uses_the_github_provider_message(
    workflow_context: dict[str, Any],
) -> None:
    assert "fix(github): use cli token" in workflow_context["output"]
    log = _git(workflow_context["repo"], ["log", "--pretty=%s", "-1"])
    assert log == "fix(github): use cli token"


@then("the provider receives exactly one retry-limited request")
def provider_receives_exactly_one_retry_limited_request(
    workflow_context: dict[str, Any],
) -> None:
    result = workflow_context["result"]
    assert result.returncode != 0
    requests = workflow_context["retry_handler"].requests
    assert len(requests) == 1
    method, path, body = requests[0]
    assert method == "POST"
    assert path == "/chat/completions"
    assert "tracked.py" in body


@then("no commit is written after the retry-limited provider failure")
def no_commit_is_written_after_retry_limited_provider_failure(
    workflow_context: dict[str, Any],
) -> None:
    log = _git(workflow_context["repo"], ["log", "--pretty=%s", "-1"])
    assert log == "chore(repo): seed"
    status = _git(workflow_context["repo"], ["status", "--short"])
    assert status == "M tracked.py"


@then("the latest commit is written by the Rust runtime")
def latest_commit_is_written_by_the_rust_runtime(
    workflow_context: dict[str, Any],
) -> None:
    assert "auto_covered_workflow" in workflow_context["stderr"]
    assert "chore(repo): update tracked" in workflow_context["output"]
    log = _git(workflow_context["repo"], ["log", "--pretty=%s", "-1"])
    assert log == "chore(repo): update tracked"


@then("the Rust configuration file contains the Anthropic provider settings")
def rust_configuration_file_contains_anthropic_provider_settings(
    workflow_context: dict[str, Any],
) -> None:
    config = json.loads((workflow_context["config_home"] / "config.json").read_text())
    assert config["provider"] == "anthropic"
    assert config["model"] == "claude-test"
    assert config["llm_endpoint"] == "https://anthropic.test"
    assert config["api_key_env"] == "ANTHROPIC_TEST_KEY"
    assert config["providers"]["anthropic"]["api_key_env"] == "ANTHROPIC_TEST_KEY"
    assert config["providers"]["anthropic"]["keychain_account"] == (
        "provider/anthropic/default"
    )
    assert "api_key" not in config["providers"]["anthropic"]


@then("the Rust configuration file contains the default OpenAI provider settings")
def rust_configuration_file_contains_default_openai_provider_settings(
    workflow_context: dict[str, Any],
) -> None:
    config = json.loads((workflow_context["config_home"] / "config.json").read_text())
    assert config["provider"] == "openai"
    assert config["model"] == "gpt-5-mini-2025-08-07"
    assert config["llm_endpoint"] == "https://api.openai.com/v1"
    assert config["api_key_env"] == "OPENAI_API_KEY"
    assert config["providers"]["openai"]["api_key_env"] == "OPENAI_API_KEY"
    assert (
        config["providers"]["openai"]["keychain_account"] == "provider/openai/default"
    )
    assert "api_key" not in config["providers"]["openai"]


@then("the Python entrypoint configure run is handled by Rust")
def python_entrypoint_configure_run_is_handled_by_rust(
    workflow_context: dict[str, Any],
) -> None:
    assert "auto_covered_workflow" in workflow_context["stderr"]


@then("the Rust configuration keeps the OpenAI primary provider")
def rust_configuration_keeps_openai_primary_provider(
    workflow_context: dict[str, Any],
) -> None:
    config = json.loads((workflow_context["config_home"] / "config.json").read_text())
    assert config["provider"] == "openai"
    assert config["model"] == "gpt-existing"
    assert config["llm_endpoint"] == "https://openai.existing/v1"
    assert config["api_key_env"] == "OPENAI_EXISTING_KEY"
    assert config["providers"]["openai"]["keychain_account"] == (
        "provider/openai/existing"
    )
    assert config["model_priority"][0]["provider"] == "openai"
    assert config["model_priority"][1]["provider"] == "anthropic"


@then(
    "the Rust configuration file contains the Anthropic configure-all provider settings"
)
def rust_configuration_file_contains_anthropic_configure_all_provider_settings(
    workflow_context: dict[str, Any],
) -> None:
    config = json.loads((workflow_context["config_home"] / "config.json").read_text())
    provider = config["providers"]["anthropic"]
    assert provider["endpoint"] == "https://anthropic.new"
    assert provider["api_key_env"] == "ANTHROPIC_NEW_KEY"
    assert provider["preferred_model"] == "claude-new"
    assert provider["keychain_account"] == "provider/anthropic/default"
    assert "api_key" not in provider


@then("the Rust preferences file contains default selector preferences")
def rust_preferences_file_contains_default_selector_preferences(
    workflow_context: dict[str, Any],
) -> None:
    preferences = json.loads(
        (workflow_context["config_home"] / "preferences.json").read_text()
    )
    assert preferences["selection_policy"] == "fastest_cheap"
    assert preferences["default_prompt_profile"] == "conventional"
    assert preferences["prompt_profiles"][0]["id"] == "conventional"
    assert preferences["provider_rules"]["openai"]["preset"] == "none"
    assert preferences["provider_rules"]["anthropic"]["preset"] == "none"
    assert preferences["provider_rules"]["xai"]["preset"] == "none"
    assert preferences["provider_rules"]["github"]["preset"] == "none"


@then("the Anthropic provider receives the latest Haiku model")
def anthropic_provider_receives_latest_haiku_model(
    workflow_context: dict[str, Any],
) -> None:
    requests = workflow_context["anthropic_handler"].requests
    assert requests
    assert any("claude-3-5-haiku-latest" in body for _method, _path, body in requests)


@then("the latest commit uses the Anthropic provider message")
def latest_commit_uses_anthropic_provider_message(
    workflow_context: dict[str, Any],
) -> None:
    log = _git(workflow_context["repo"], ["log", "-1", "--pretty=%s"])
    assert log == "fix(fallback): use secondary provider"


@then("the Rust stats command reports usage telemetry")
def rust_stats_command_reports_usage_telemetry(
    workflow_context: dict[str, Any],
) -> None:
    env = _clean_env(workflow_context["config_home"])
    output = _run(
        [
            str(_rust_bin("kcmt")),
            "stats",
            "--json",
            "--repo-path",
            str(workflow_context["repo"]),
        ],
        REPO_ROOT,
        env=env,
    )
    payload = json.loads(output)
    assert payload["aggregates"]
    aggregate = payload["aggregates"][0]
    assert aggregate["runs"] >= 1
    assert "provider" in aggregate
    assert "model" in aggregate


@then("the model list includes all supported providers")
def model_list_includes_all_supported_providers(
    workflow_context: dict[str, Any],
) -> None:
    output = workflow_context["output"]
    assert "openai" in output
    assert "gpt-5-mini-2025-08-07" in output
    assert "anthropic" in output
    assert "claude-3-5-haiku-latest" in output
    assert "xai" in output
    assert "github" in output


@then("the debug model list is structured JSON")
def debug_model_list_is_structured_json(workflow_context: dict[str, Any]) -> None:
    payload = json.loads(workflow_context["output"])
    providers = {entry["provider"]: entry for entry in payload}
    assert providers["openai"]["models"][0]["id"] == "gpt-5-mini-2025-08-07"
    assert providers["openai"]["source"] == "static_fallback"
    assert providers["openai"]["error"]
    assert providers["github"]["models"][0]["api_key_env"] == "GITHUB_TOKEN"
    assert providers["anthropic"]["display_name"] == "Anthropic"
    anthropic_model = providers["anthropic"]["models"][0]
    assert anthropic_model["provider"] == "anthropic"
    assert anthropic_model["endpoint"] == "https://api.anthropic.com"
    assert anthropic_model["family"] == "haiku"
    assert anthropic_model["code_capable"] is True
    assert "bdd-key" not in workflow_context["output"]


@then("the key verification output shows present and missing providers")
def key_verification_output_shows_present_and_missing_providers(
    workflow_context: dict[str, Any],
) -> None:
    output = workflow_context["output"]
    assert "API Key Verification" in output
    assert "openai\tOPENAI_API_KEY\tyes\tOPENAI_API_KEY" in output
    assert "anthropic\tANTHROPIC_API_KEY\tno\t-" in output
    assert "github\tGITHUB_TOKEN\tno\t-" in output


@then("the benchmark output includes leaderboard JSON and CSV sections")
def benchmark_output_includes_leaderboard_json_and_csv_sections(
    workflow_context: dict[str, Any],
) -> None:
    output = workflow_context["output"]
    assert "Benchmark Leaderboard" in output
    assert '"overall"' in output
    assert '"provider": "openai"' in output
    assert "provider,model,avg_latency_ms" in output
    assert "openai,gpt-bdd" in output


@then("the provider benchmark snapshot is persisted")
def provider_benchmark_snapshot_is_persisted(
    workflow_context: dict[str, Any],
) -> None:
    repo = workflow_context["repo"].resolve()
    digest = __import__("hashlib").sha256(str(repo).encode("utf-8")).hexdigest()[:8]
    namespace = f"{repo.name}-{digest}"
    benchmark_dir = workflow_context["config_home"] / "repos" / namespace / "benchmarks"
    snapshots = sorted(benchmark_dir.glob("benchmark-*.json"))
    assert len(snapshots) == 1
    payload = json.loads(snapshots[0].read_text())
    assert payload["schema_version"] == 1
    assert payload["results"][0]["provider"] == "openai"
    assert payload["results"][0]["model"] == "gpt-bdd"


@then("the benchmark report includes Rust workflow stage timings")
def benchmark_report_includes_rust_workflow_stage_timings(
    workflow_context: dict[str, Any],
) -> None:
    payload = json.loads(workflow_context["output"])
    file_result = next(
        item
        for item in payload["results"]
        if item["runtime"] == "rust"
        and item["workflow_contract_id"] == "file-repo-path"
    )
    stages = file_result["stage_timings"]
    stage_names = {stage["stage"] for stage in stages}
    assert {
        "status_scan",
        "diff_preparation",
        "llm_enqueue",
        "llm_wait",
        "response_validation",
        "commit",
        "push",
        "snapshot",
    }.issubset(stage_names)
    assert all(isinstance(stage["duration_ms"], int | float) for stage in stages)
    assert all(isinstance(stage["items"], int) for stage in stages)


@then("the benchmark report compares Python and Rust runtime stages")
def benchmark_report_compares_python_and_rust_runtime_stages(
    workflow_context: dict[str, Any],
) -> None:
    payload = json.loads(workflow_context["output"])
    assert payload["snapshot"]["benchmark_kind"] == "runtime"
    assert payload["snapshot"]["provider_benchmark_kind"] == "provider"
    assert payload["scorecard"]["provider_throughput_included"] is False
    assert "pre-LLM Rust heuristic" in payload["scorecard"]["measurement_basis"]
    matrix = payload["scenario_matrix"]
    assert {row["workflow_contract_id"] for row in matrix} == {
        "status-repo-path",
        "oneshot-repo-path",
        "default-repo-path",
        "file-repo-path",
    }
    file_row = next(
        row for row in matrix if row["workflow_contract_id"] == "file-repo-path"
    )
    assert file_row["python"]["status"] == "passed"
    assert file_row["rust"]["status"] == "passed"
    assert file_row["comparison"]["comparable"] is True
    assert any(
        stage["stage"] == "workflow_total"
        for stage in file_row["comparison"]["stage_deltas"]
    )


@then("the runtime benchmark snapshot is persisted")
def runtime_benchmark_snapshot_is_persisted(
    workflow_context: dict[str, Any],
) -> None:
    repo = workflow_context["repo"].resolve()
    digest = __import__("hashlib").sha256(str(repo).encode("utf-8")).hexdigest()[:8]
    namespace = f"{repo.name}-{digest}"
    benchmark_dir = workflow_context["config_home"] / "repos" / namespace / "benchmarks"
    snapshots = sorted(benchmark_dir.glob("runtime-*.json"))
    assert len(snapshots) == 1
    payload = json.loads(snapshots[0].read_text())
    assert payload["snapshot"]["benchmark_kind"] == "runtime"
    output_payload = json.loads(workflow_context["output"])
    assert len(payload["scenario_matrix"]) == len(output_payload["scenario_matrix"])
    assert len(payload["scenario_matrix"]) >= 4
