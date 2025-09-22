from kcmt import main as main_mod


def test_main_entrypoint_no_args(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    # main() delegates to cli.main() without args; just ensure it returns int
    rc = main_mod.main()
    assert isinstance(rc, int)
