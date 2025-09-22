from kcmt.exceptions import (
    ConfigError,
    GitError,
    KlingonCMTError,
    LLMError,
    ValidationError,
)


def test_exceptions_hierarchy_and_str():
    # Given exception classes
    # When instantiating
    e = KlingonCMTError("base")
    g = GitError("git")
    llm_err = LLMError("llm")
    c = ConfigError("cfg")
    v = ValidationError("val")

    # Then hierarchy holds
    assert isinstance(e, Exception)
    assert isinstance(g, KlingonCMTError)
    assert isinstance(llm_err, KlingonCMTError)
    assert isinstance(c, KlingonCMTError)
    assert isinstance(v, KlingonCMTError)
    # And messages are retained
    assert "git" in str(g)
    assert "llm" in str(llm_err)
    assert "cfg" in str(c)
    assert "val" in str(v)
