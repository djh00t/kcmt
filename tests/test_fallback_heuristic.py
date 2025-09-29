import os

from kcmt import llm as llm_module
from kcmt.config import Config, clear_active_config, set_active_config
from kcmt.core import KlingonCMTWorkflow
from kcmt.exceptions import LLMError


def test_allow_fallback_true_returns_heuristic(tmp_path, monkeypatch):
    os.chdir(tmp_path)
    os.system('git init -q')
    (tmp_path / 'file.txt').write_text('hello')
    os.system('git add file.txt')
    os.system('git commit -m "chore(core): init" -q')
    (tmp_path / 'file.txt').write_text('hello world')
    os.system('git add file.txt')

    cfg = Config(
        provider='openai',
        model='gpt-test',
        llm_endpoint=None,
        api_key_env='OPENAI_API_KEY',
        allow_fallback=True,
        auto_push=False,
    )
    set_active_config(cfg)

    def failing_llm(
        diff, context="", style="conventional"
    ):  # noqa: D401, ARG001
        raise LLMError('simulated failure')

    monkeypatch.setattr(
        llm_module.LLMClient,
        'generate_commit_message',
        staticmethod(failing_llm),
    )

    wf = KlingonCMTWorkflow(show_progress=False, config=cfg)
    res = wf.execute_workflow()
    messages = [r.message for r in res['file_commits'] if r.success]
    assert messages, 'Expected a heuristic commit message'
    assert any(
        m.startswith(('feat(', 'refactor(', 'docs(', 'test(', 'chore('))
        for m in messages
    )

    clear_active_config()


def test_allow_fallback_false_raises(tmp_path, monkeypatch):
    os.chdir(tmp_path)
    os.system('git init -q')
    (tmp_path / 'file.txt').write_text('hello')
    os.system('git add file.txt')
    os.system('git commit -m "chore(core): init" -q')
    (tmp_path / 'file.txt').write_text('hello world')
    os.system('git add file.txt')

    cfg = Config(
        provider='openai',
        model='gpt-test',
        llm_endpoint=None,
        api_key_env='OPENAI_API_KEY',
        allow_fallback=False,
        auto_push=False,
    )
    set_active_config(cfg)

    def failing_llm(
        diff, context="", style="conventional"
    ):  # noqa: D401, ARG001
        raise LLMError('simulated failure')

    monkeypatch.setattr(
        llm_module.LLMClient,
        'generate_commit_message',
        staticmethod(failing_llm),
    )

    wf = KlingonCMTWorkflow(show_progress=False, config=cfg)
    res = wf.execute_workflow()
    successes = [r for r in res['file_commits'] if r.success]
    # Expect at least one failure result capturing the LLM error
    failures = [r for r in res['file_commits'] if not r.success]
    assert (
        not successes and failures
    ), 'Expected failures with fallback disabled'

    clear_active_config()
