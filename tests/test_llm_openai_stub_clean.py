from kcmt.config import clear_active_config, load_config


def test_llm_openai_basic(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    clear_active_config()

    from kcmt import llm as llm_module  # noqa: PLC0415

    # Provide a dummy OpenAI client to prevent real HTTP calls in case
    # monkeypatching of _call_openai fails (defensive against upstream
    # client internals changing).
    class _DummyMessage:  # noqa: D401
        def __init__(self):
            self.content = "feat(core): add feature"

    class _DummyChoice:  # noqa: D401
        def __init__(self):
            self.message = _DummyMessage()

    class _DummyResponse:  # noqa: D401
        def __init__(self):
            self.choices = [_DummyChoice()]

    class _DummyCompletions:  # noqa: D401
        @staticmethod
        def create(*_a, **_kw):  # noqa: D401, ARG002
            return _DummyResponse()

    class _DummyChat:  # noqa: D401
        def __init__(self):
            self.completions = _DummyCompletions()

    class DummyOpenAI:  # noqa: D401
        def __init__(self, *args, **kwargs):  # noqa: D401, ARG002
            self.chat = _DummyChat()

    monkeypatch.setattr(llm_module, "OpenAI", DummyOpenAI)

    def stub_init(self, config=None, debug=False):  # noqa: D401, ARG002
        # Minimal safe init that installs a dummy OpenAI-compatible client
        self.debug = debug
        self.config = config or load_config(overrides={"provider": "openai"})
        self.provider = self.config.provider
        self.model = self.config.model
        self.api_key = "DUMMY"  # ensure non-empty so validation passes
        self._mode = "openai"
        # Use the dummy we injected above so that _call_openai could run safely
        self._client = DummyOpenAI()

    def stub_call(self, prompt):  # noqa: D401, ARG002, unused-argument
        return "feat(core): add feature"

    monkeypatch.setattr(llm_module.LLMClient, "__init__", stub_init)
    monkeypatch.setattr(llm_module.LLMClient, "_call_openai", stub_call)
    monkeypatch.setenv("KCMT_TEST_DISABLE_OPENAI", "1")

    cfg = load_config(overrides={"provider": "openai", "model": "gpt-test"})
    # Use module reference to ensure patched class is used
    client = llm_module.LLMClient(config=cfg)
    msg = client.generate_commit_message("diff --git a b\n@@\n+code", "ctx")
    assert msg.startswith("feat(")
