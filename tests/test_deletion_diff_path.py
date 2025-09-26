import os
import subprocess

from kcmt.commit import CommitGenerator
from kcmt.config import Config, set_active_config
from kcmt.exceptions import LLMError


def test_deletion_diff_path(tmp_path):
    repo = tmp_path / "repo_del"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    f = repo / "deleteme.txt"
    f.write_text("hello\nworld\n")
    subprocess.run(["git", "add", "deleteme.txt"], cwd=repo, check=True)
    subprocess.run(
        ["git", "commit", "-m", "chore: init"], cwd=repo, check=True
    )
    # delete file
    f.unlink()

    cfg = Config(
        provider="openai",
        model="gpt-x",
        llm_endpoint="http://local",
        api_key_env="OPENAI_API_KEY",
        git_repo_path=str(repo),
        allow_fallback=False,  # deprecated behavior (strict mode)
    )
    set_active_config(cfg)

    os.environ["OPENAI_API_KEY"] = "X"

    gen = CommitGenerator(repo_path=str(repo), config=cfg)
    diff = gen.git_repo.get_working_diff()
    assert "deleteme.txt" in diff
    # In strict mode we either get a valid conventional header or raise.
    try:
        msg = gen.suggest_commit_message(diff, context="File: deleteme.txt")
    except LLMError:
        # Strict failure acceptable if model stub invalid
        return
    assert ": " in msg  # basic convention check
