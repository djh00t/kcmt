import subprocess

import pytest

from kcmt.exceptions import GitError
from kcmt.git import GitRepo


def test_gitrepo_init_validates_repo(monkeypatch):
    # Given _is_git_repo returns False
    monkeypatch.setattr(GitRepo, "_is_git_repo", lambda self: False)

    # When/Then
    with pytest.raises(GitError):
        GitRepo(".")


def test_run_git_command_success(monkeypatch, tmp_path):
    # Given subprocess returns stdout
    class _R:
        stdout = "ok\n"

    def fake_run(cmd, cwd=None, capture_output=True, text=True, check=True):
        return _R()

    monkeypatch.setattr(subprocess, "run", fake_run)
    repo = object.__new__(GitRepo)
    repo.repo_path = tmp_path

    # When
    out = GitRepo._run_git_command(repo, ["status"])

    # Then
    assert out == "ok"


def test_run_git_command_called_process_error(monkeypatch, tmp_path):
    def fake_run(*args, **kwargs):
        raise subprocess.CalledProcessError(1, "git", stderr="bad\n")

    monkeypatch.setattr(subprocess, "run", fake_run)
    repo = object.__new__(GitRepo)
    repo.repo_path = tmp_path

    with pytest.raises(GitError) as ei:
        GitRepo._run_git_command(repo, ["x"])
    assert "failed" in str(ei.value)


def test_run_git_command_file_not_found(monkeypatch, tmp_path):
    def fake_run(*args, **kwargs):
        raise FileNotFoundError()

    monkeypatch.setattr(subprocess, "run", fake_run)
    repo = object.__new__(GitRepo)
    repo.repo_path = tmp_path

    with pytest.raises(GitError) as ei:
        GitRepo._run_git_command(repo, ["x"])
    assert "not found" in str(ei.value).lower()


def test_has_staged_and_working_changes(monkeypatch, tmp_path):
    repo = object.__new__(GitRepo)
    repo.repo_path = tmp_path

    monkeypatch.setattr(GitRepo, "get_staged_diff", lambda self: "diff content")
    monkeypatch.setattr(GitRepo, "get_working_diff", lambda self: "")

    assert GitRepo.has_staged_changes(repo) is True
    assert GitRepo.has_working_changes(repo) is False


def test_process_deletions_first_stages_deleted(monkeypatch, tmp_path):
    # Given status contains deleted entries
    repo = object.__new__(GitRepo)
    repo.repo_path = tmp_path

    calls = []

    def fake_run_git(args):
        if args == ["status", "--porcelain"]:
            # Leading "D  " lines to satisfy parser
            return "D  file1.txt\nM  other.txt\nD  dir/file2.py\n"
        raise AssertionError("Unexpected git invocation")

    def fake_stage(file_path):
        calls.append(file_path)

    monkeypatch.setattr(GitRepo, "_run_git_command", lambda self, a: fake_run_git(a))
    monkeypatch.setattr(GitRepo, "stage_file", lambda self, p: fake_stage(p))

    deleted = GitRepo.process_deletions_first(repo)
    assert deleted == ["file1.txt", "dir/file2.py"]
    assert calls == ["file1.txt", "dir/file2.py"]
