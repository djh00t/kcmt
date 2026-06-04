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
    )
    return result.stdout.strip()


def _clean_env(config_home: Path) -> dict[str, str]:
    env = {
        key: value
        for key, value in os.environ.items()
        if key in {"PATH", "HOME", "USER", "TMPDIR", "LANG", "LC_ALL"}
    }
    env["KCMT_CONFIG_HOME"] = str(config_home)
    env["KCMT_ALLOW_LOCAL_SYNTHESIS"] = "1"
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


def _rust_bin(binary: str) -> Path:
    subprocess.run(
        [
            "cargo",
            "build",
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
    )
    return REPO_ROOT / "rust" / "target" / "debug" / binary


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


@when("the Rust kcmt command runs in oneshot batch mode")
def rust_kcmt_runs_in_oneshot_batch_mode(workflow_context: dict[str, Any]) -> None:
    env = _clean_env(workflow_context["config_home"])
    env["OPENAI_TEST_KEY"] = "test-key"
    output = _run(
        [
            str(_rust_bin("kcmt")),
            "--oneshot",
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


@when("the Rust kcmt command lists models")
def rust_kcmt_lists_models(workflow_context: dict[str, Any]) -> None:
    env = _clean_env(workflow_context["config_home"])
    output = _run([str(_rust_bin("kcmt")), "--list-models"], REPO_ROOT, env=env)
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
    output = _run(
        [
            str(rust_bin),
            "benchmark",
            "runtime",
            "--repo-path",
            str(workflow_context["repo"]),
            "--runtime",
            "rust",
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


@when("the Rust kcmt command receives invalid provider output")
def rust_kcmt_receives_invalid_provider_output(
    workflow_context: dict[str, Any],
) -> None:
    env = _clean_env(workflow_context["config_home"])
    env["KCMT_PROVIDER_RESPONSE"] = "This changes a few files"
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


@then("the repository worktree is clean")
def repository_worktree_is_clean(workflow_context: dict[str, Any]) -> None:
    status = _git(workflow_context["repo"], ["status", "--short"])
    assert status == ""


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


@then("the workflow fails before committing the file")
def workflow_fails_before_committing_the_file(workflow_context: dict[str, Any]) -> None:
    result = workflow_context["result"]
    assert result.returncode == 1
    assert "missing conventional commit header" in result.stderr

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


@then("the fallback provider is used after the primary provider fails")
def fallback_provider_is_used_after_the_primary_provider_fails(
    workflow_context: dict[str, Any],
) -> None:
    requests = workflow_context["fallback_handler"].requests
    assert len(requests) == 4
    assert [request[1] for request in requests[:3]] == [
        "/chat/completions",
        "/chat/completions",
        "/chat/completions",
    ]
    assert requests[3][1] == "/v1/messages"


@then("the latest commit uses the fallback provider message")
def latest_commit_uses_the_fallback_provider_message(
    workflow_context: dict[str, Any],
) -> None:
    assert "fix(fallback): use secondary provider" in workflow_context["output"]
    log = _git(workflow_context["repo"], ["log", "--pretty=%s", "-1"])
    assert log == "fix(fallback): use secondary provider"


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
        if item["runtime"] == "rust" and item["workflow_contract_id"] == "file-repo-path"
    )
    stages = file_result["stage_timings"]
    stage_names = {stage["stage"] for stage in stages}
    assert {"status_scan", "llm_wait", "commit", "push", "snapshot"}.issubset(
        stage_names
    )
    assert all(isinstance(stage["duration_ms"], int | float) for stage in stages)
    assert all(isinstance(stage["items"], int) for stage in stages)
