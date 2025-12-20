import os

from kcmt.config import Config, clear_active_config, set_active_config
from kcmt.core import KlingonCMTWorkflow


def test_meta_files_committed(tmp_path, monkeypatch):
    os.chdir(tmp_path)
    os.system("git init -q")
    os.system("git config user.name tester")
    os.system("git config user.email tester@example.com")
    (tmp_path / ".gitignore").write_text("*.pyc\n")
    (tmp_path / ".gitattributes").write_text("* text=auto\n")
    (tmp_path / ".gitmodules").write_text('[submodule "x"]\n')
    (tmp_path / "regular.txt").write_text("hello")

    os.system("git add regular.txt .gitignore .gitattributes .gitmodules")
    os.system('git commit -m "chore(core): initial" -q')

    (tmp_path / "regular.txt").write_text("hello world")
    (tmp_path / ".gitignore").write_text("*.pyc\n__pycache__/\n")
    (tmp_path / ".gitattributes").write_text("* text=auto eol=lf\n")
    (tmp_path / ".gitmodules").write_text('[submodule "x"]\npath = x\n')
    os.system("git add .")

    # Provide required Config args; rely on env OPENAI_API_KEY from fixture
    cfg = Config(
        provider="openai",
        model="gpt-test",
        llm_endpoint="https://api.openai.com/v1",
        api_key_env="OPENAI_API_KEY",
        auto_push=False,
    )
    set_active_config(cfg)

    from kcmt.commit import CommitGenerator

    def fake_suggest(self, diff, context, style):  # noqa: D401, ARG001
        # Generate appropriate commit messages for each file
        if "regular.txt" in context:
            return "chore(core): update regular.txt"
        elif ".gitignore" in context:
            return "chore(core): update .gitignore"
        elif ".gitattributes" in context:
            return "chore(core): update .gitattributes"
        elif ".gitmodules" in context:
            return "chore(core): update .gitmodules"
        return "chore(core): update file"

    monkeypatch.setattr(
        CommitGenerator,
        "suggest_commit_message",
        fake_suggest,
    )

    wf = KlingonCMTWorkflow(show_progress=False, config=cfg)
    results = wf.execute_workflow()

    committed_files = [r.file_path for r in results["file_commits"] if r.success]
    assert "regular.txt" in committed_files
    # Meta files should now be committed (no longer skipped)
    assert ".gitignore" in committed_files
    assert ".gitattributes" in committed_files
    assert ".gitmodules" in committed_files

    clear_active_config()
