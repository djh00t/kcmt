"""Python runtime telemetry coverage for side-by-side benchmarks."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from kcmt.config import Config
from kcmt.core import KlingonCMTWorkflow


def _git(cmd: list[str], cwd: Path) -> str:
    result = subprocess.run(
        ["git"] + cmd,
        cwd=cwd,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def test_python_workflow_exports_runtime_stage_telemetry(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(["init", "-q"], repo)
    _git(["config", "user.name", "Test"], repo)
    _git(["config", "user.email", "test@example.com"], repo)
    (repo / "alpha.py").write_text("print('alpha')\n", encoding="utf-8")

    telemetry_path = tmp_path / "runtime-telemetry.json"
    monkeypatch.setenv("KCMT_RUNTIME_TELEMETRY_PATH", str(telemetry_path))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    from kcmt import commit as commit_module

    def fake_generate(
        self,
        _diff: str,
        context: str = "",
        style: str = "conventional",
        request_timeout: float | None = None,
        progress_callback=None,
    ) -> str:
        _ = self, style, request_timeout
        if progress_callback is not None:
            progress_callback("request-sent")
            progress_callback("response-received")
        file_name = (
            context.split("File:", 1)[1].strip() if "File:" in context else "file.py"
        )
        stem = Path(file_name).stem
        return f"feat(core): add {stem}"

    monkeypatch.setattr(
        commit_module.LLMClient, "generate_commit_message", fake_generate
    )

    cfg = Config(
        provider="openai",
        model="gpt-test",
        llm_endpoint="https://api.openai.com/v1",
        api_key_env="OPENAI_API_KEY",
        git_repo_path=str(repo),
        auto_push=False,
    )
    workflow = KlingonCMTWorkflow(repo_path=str(repo), config=cfg, show_progress=False)

    results = workflow.execute_workflow()

    assert results["file_commits"][0].success
    payload = json.loads(telemetry_path.read_text(encoding="utf-8"))
    stages = {stage["stage"] for stage in payload["stages"]}
    assert {
        "status_scan",
        "diff_preparation",
        "llm_enqueue",
        "llm_wait",
        "response_validation",
        "commit",
        "push",
        "snapshot",
    }.issubset(stages)
