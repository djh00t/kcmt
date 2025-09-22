import os
import subprocess
from kcmt.config import Config, set_active_config
from kcmt.commit import CommitGenerator


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
        allow_fallback=True,
    )
    set_active_config(cfg)

    os.environ["OPENAI_API_KEY"] = "X"

    gen = CommitGenerator(repo_path=str(repo), config=cfg)
    diff = gen.git_repo.get_working_diff()
    assert "deleteme.txt" in diff
    # Fallback path should produce a refactor/chore style header (not failing)
    from kcmt.exceptions import LLMError

    try:
        msg = gen.suggest_commit_message(diff, context="File: deleteme.txt")
    except LLMError:  # If LLM attempts fail it should still fallback
        msg = gen.heuristic_message(diff, "File: deleteme.txt")
    assert ": " in msg
