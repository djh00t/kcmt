import subprocess
import sys
from pathlib import Path


def _add_repo_to_sys_path(repo_root: Path):
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


def _git(cmd: list[str], cwd: Path) -> str:
    result = subprocess.run(
        ["git"] + cmd,
        cwd=cwd,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def test_process_per_file_commits_reports_diff_stage_for_untracked_files(
    tmp_path, monkeypatch
):
    _git(["init", "-q"], tmp_path)
    _git(["config", "user.name", "Tester"], tmp_path)
    _git(["config", "user.email", "tester@example.com"], tmp_path)
    (tmp_path / "alpha.py").write_text("print('alpha')\n")

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    _add_repo_to_sys_path(Path.cwd())
    from kcmt_python.core import KlingonCMTWorkflow  # type: ignore

    wf = KlingonCMTWorkflow(repo_path=str(tmp_path), show_progress=True)
    captured_snapshots: list[dict[str, float]] = []

    def capture_progress(stage: str) -> None:
        if stage == "diff":
            captured_snapshots.append(wf.stats_snapshot())

    monkeypatch.setattr(wf, "_print_progress", capture_progress)
    monkeypatch.setattr(wf, "_prepare_commit_messages", lambda file_changes: [])

    wf._process_per_file_commits(wf.git_repo.scan_status())

    assert captured_snapshots
    assert captured_snapshots[-1]["total_files"] == 1
    assert captured_snapshots[-1]["diffs_built"] == 1
