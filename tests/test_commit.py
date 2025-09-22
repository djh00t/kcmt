import types

import pytest

from kcmt.commit import CommitGenerator
from kcmt.exceptions import ValidationError


def _new_commit_gen(fake_git=None, fake_llm=None):
    # Avoid CommitGenerator.__init__ (which constructs GitRepo/LLM)
    gen = object.__new__(CommitGenerator)
    gen.git_repo = fake_git or types.SimpleNamespace(
        has_staged_changes=lambda: True,
        get_staged_diff=lambda: "diff",
        has_working_changes=lambda: True,
        get_working_diff=lambda: "wdiff",
        get_commit_diff=lambda h: "cdiff",
    )
    gen.llm_client = fake_llm or types.SimpleNamespace(
        generate_commit_message=lambda diff, context="", style="conventional": "feat: msg"
    )
    return gen


def test_generate_from_staged_requires_changes():
    # Given no staged changes
    gen = _new_commit_gen(
        fake_git=types.SimpleNamespace(has_staged_changes=lambda: False)
    )
    with pytest.raises(ValidationError):
        gen.generate_from_staged()


def test_generate_from_staged_success():
    gen = _new_commit_gen()
    assert gen.generate_from_staged() == "feat: msg"


def test_generate_from_working_requires_changes():
    gen = _new_commit_gen(
        fake_git=types.SimpleNamespace(
            has_working_changes=lambda: False,
            get_working_diff=lambda: "",
        )
    )
    with pytest.raises(ValidationError):
        gen.generate_from_working()


def test_generate_from_working_success():
    gen = _new_commit_gen()
    assert gen.generate_from_working() == "feat: msg"


def test_generate_from_commit_success():
    gen = _new_commit_gen()
    assert gen.generate_from_commit("abc123") == "feat: msg"


def test_suggest_commit_message_empty_diff_raises():
    gen = _new_commit_gen()
    with pytest.raises(ValidationError):
        gen.suggest_commit_message("  ")


def test_validate_conventional_commit_true_false():
    gen = _new_commit_gen()
    assert gen.validate_conventional_commit("feat(core): add feature")
    assert not gen.validate_conventional_commit("bad message")


def test_validate_and_fix_commit_message_valid_returns_same():
    gen = _new_commit_gen()
    assert (
        gen.validate_and_fix_commit_message("fix(ui): correct color")
        == "fix(ui): correct color"
    )


def test_validate_and_fix_commit_message_invalid_fixed_by_llm():
    gen = _new_commit_gen(
        fake_llm=types.SimpleNamespace(
            generate_commit_message=lambda diff, context="", style="conventional": "chore: update"
        )
    )
    assert gen.validate_and_fix_commit_message("bad") == "chore: update"


def test_validate_and_fix_commit_message_invalid_raises_when_llm_fails():
    gen = _new_commit_gen(
        fake_llm=types.SimpleNamespace(
            generate_commit_message=lambda *a, **k: "still bad"
        )
    )
    with pytest.raises(ValidationError):
        gen.validate_and_fix_commit_message("nope")
