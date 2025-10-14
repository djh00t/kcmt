from kcmt import main as main_mod


def test_main_entrypoint_no_args(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    # Run in an isolated temp directory so any persisted config is ephemeral
    monkeypatch.chdir(tmp_path)
    # main() delegates to cli.main() without args; just ensure it returns int
    rc = main_mod.main()
    assert isinstance(rc, int)
