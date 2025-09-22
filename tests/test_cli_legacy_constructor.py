from kcmt.cli import CLI


def test_cli_legacy_constructor(monkeypatch):
    # Ensure instantiation works and parser exists (covers __init__ path)
    monkeypatch.setenv("OPENAI_API_KEY", "dummy")
    cli = CLI()
    assert hasattr(cli, "parser")
