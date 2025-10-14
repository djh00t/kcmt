"""Command-line interface for kcmt."""

from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Optional, cast

from .commit import CommitGenerator
from .config import (
    DEFAULT_MODELS,
    Config,
    describe_provider,
    detect_available_providers,
    load_config,
    load_persisted_config,
    save_config,
)
from .core import KlingonCMTWorkflow
from .exceptions import GitError, KlingonCMTError, LLMError
from .git import GitRepo, find_git_repo_root

_json_encoder = json.encoder
INFINITY = cast(float, getattr(_json_encoder, "INFINITY"))
encode_basestring = cast(
    Callable[[str], str], getattr(_json_encoder, "encode_basestring")
)
encode_basestring_ascii = cast(
    Callable[[str], str], getattr(_json_encoder, "encode_basestring_ascii")
)
_make_iterencode = cast(
    Callable[..., Callable[[Any, int], Iterator[str]]],
    getattr(_json_encoder, "_make_iterencode"),
)

RESET = "\033[0m"
BOLD = "\033[1m"
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
MAGENTA = "\033[95m"
RED = "\033[91m"


class DecimalFriendlyJSONEncoder(json.JSONEncoder):
    """JSON encoder that renders floats without scientific notation."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:  # noqa: D401
        indent = kwargs.get("indent")
        if isinstance(indent, int):
            kwargs["indent"] = " " * indent
        super().__init__(*args, **kwargs)

    def iterencode(
        self, o: Any, _one_shot: bool = False
    ) -> Iterator[str]:  # noqa: N802 - match json API
        markers: dict[int, Any] | None
        if self.check_circular:
            markers = {}
        else:
            markers = None

        if self.ensure_ascii:
            _encoder = encode_basestring_ascii
        else:
            _encoder = encode_basestring

        def floatstr(
            value: float,
            allow_nan: bool = self.allow_nan,
            _inf: float = INFINITY,
            _neginf: float = -INFINITY,
        ) -> str:
            if value != value:  # NaN check
                text = "NaN"
            elif value == _inf:
                text = "Infinity"
            elif value == _neginf:
                text = "-Infinity"
            else:
                text = format(value, ".6f")
                if "." in text:
                    text = text.rstrip("0").rstrip(".")
                if "." not in text:
                    text = f"{text}.0"
            if not allow_nan and text in {"NaN", "Infinity", "-Infinity"}:
                raise ValueError("Out of range float values are not JSON compliant")
            return text

        _iterencode = _make_iterencode(
            markers,
            self.default,
            _encoder,
            self.indent,
            floatstr,
            self.key_separator,
            self.item_separator,
            self.sort_keys,
            self.skipkeys,
            _one_shot,
        )
        return _iterencode(o, 0)


class CLI:
    """Command-line interface for kcmt."""

    def __init__(self) -> None:
        self.parser = self._create_parser()
        self._profile_enabled = False
        self._compact_mode = False
        self._repo_root: Optional[Path] = None

    # ------------------------------------------------------------------
    # Argument parsing
    # ------------------------------------------------------------------
    def _create_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            prog="kcmt",
            description="AI-powered atomic Git staging and committing tool",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  kcmt                                  # default workflow with live stats
  kcmt --oneshot                        # commit a single auto-selected file
  kcmt --file README.md                 # commit only README.md
  kcmt --configure                      # interactive provider & model setup
    kcmt --provider openai --model gpt-5-mini-2025-08-07
            """,
        )

        subparsers = parser.add_subparsers(dest="command")
        status_parser = subparsers.add_parser(
            "status",
            help="Show a formatted summary of the most recent kcmt run",
        )
        status_parser.add_argument(
            "--repo-path",
            default=".",
            help="Path to the target Git repo (default: current dir)",
        )
        status_parser.add_argument(
            "--raw",
            action="store_true",
            help="Emit the saved run snapshot JSON instead of formatted output",
        )

        parser.add_argument(
            "--configure",
            action="store_true",
            help="Launch interactive configuration wizard",
        )
        parser.add_argument(
            "--configure-all",
            action="store_true",
            help=(
                "Interactively choose provider(s) and set which env var holds each API key"
            ),
        )
        parser.add_argument(
            "--provider",
            choices=sorted(DEFAULT_MODELS.keys()),
            help="Override provider for this run",
        )
        parser.add_argument(
            "--model",
            help="Override LLM model for this run",
        )
        parser.add_argument(
            "--endpoint",
            help="Override LLM endpoint for this run",
        )
        parser.add_argument(
            "--api-key-env",
            help="Environment variable that holds the API key",
        )
        parser.add_argument(
            "--github-token",
            dest="github_token",
            help="Set GITHUB_TOKEN for this run (convenience flag)",
        )
        parser.add_argument(
            "--repo-path",
            default=".",
            help="Path to the target Git repo (default: current dir)",
        )
        parser.add_argument(
            "--max-commit-length",
            type=int,
            help="Override maximum commit message length",
        )
        parser.add_argument(
            "--max-retries",
            type=int,
            default=3,
            help="Maximum retries when git rejects a commit (default: 3)",
        )
        parser.add_argument(
            "--oneshot",
            action="store_true",
            help="Commit a single automatically selected file",
        )
        parser.add_argument(
            "--file",
            dest="single_file",
            help="Commit a single file specified explicitly",
        )
        parser.add_argument(
            "--no-progress",
            action="store_true",
            help="Disable live progress output",
        )
        parser.add_argument(
            "--limit",
            type=int,
            help="Limit the number of files to process per run",
        )
        parser.add_argument(
            "--workers",
            type=int,
            help=(
                "Override number of concurrent LLM preparations (default: smart heuristic)"
            ),
        )
        parser.add_argument(
            "--verbose",
            "-v",
            action="store_true",
            help="Emit verbose diagnostic output",
        )
        parser.add_argument(
            "--debug",
            action="store_true",
            help="Show detailed LLM API requests and responses",
        )
        parser.add_argument(
            "--profile-startup",
            action="store_true",
            help="Print timing diagnostics for startup phases",
        )
        parser.add_argument(
            "--list-models",
            action="store_true",
            help=("List available models for each provider using your API keys"),
        )
        parser.add_argument(
            "--benchmark",
            action="store_true",
            help=("Run a local benchmark across providers/models using sample diffs"),
        )
        parser.add_argument(
            "--benchmark-limit",
            type=int,
            default=5,
            help=(
                "Limit number of models per provider during --benchmark (default: 5)"
            ),
        )
        parser.add_argument(
            "--benchmark-timeout",
            type=float,
            default=None,
            help=("Per-call timeout seconds for --benchmark requests (optional)"),
        )
        parser.add_argument(
            "--verify-keys",
            action="store_true",
            help=("Verify API key environment variables for supported providers"),
        )
        parser.add_argument(
            "--benchmark-json",
            action="store_true",
            help=("Also output benchmark results as JSON (after the leaderboard)"),
        )
        parser.add_argument(
            "--benchmark-csv",
            action="store_true",
            help=("Also output benchmark results as CSV (after the leaderboard)"),
        )
        parser.add_argument(
            "--auto-push",
            action="store_true",
            help=(
                "Automatically git push after successful workflow "
                "(or set KLINGON_CMT_AUTO_PUSH=1)"
            ),
        )
        parser.add_argument(
            "--no-auto-push",
            action="store_true",
            help=(
                "Disable automatic git push for this run "
                "(or set KLINGON_CMT_AUTO_PUSH=0)"
            ),
        )
        parser.add_argument(
            "--compact",
            "--summary",
            dest="compact",
            action="store_true",
            help=("Use condensed output with a summary table and checklist"),
        )

        return parser

    def _profile_print(self, label: str, elapsed_ms: float, extra: str = "") -> None:
        if not self._profile_enabled:
            return
        details = f" {extra}" if extra else ""
        print(f"[kcmt-profile] {label}: {elapsed_ms:.1f} ms{details}")

    @contextmanager
    def _profile_timer(
        self,
        label: str,
        extra: Optional[Callable[[], str]] = None,
    ) -> Iterator[None]:
        if not self._profile_enabled:
            yield
            return
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            details = extra() if callable(extra) else ""
            self._profile_print(label, elapsed_ms, details)

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------
    def run(self, args: Optional[list[str]] = None) -> int:
        try:
            with self._profile_timer("parse-args"):
                parsed_args = self.parser.parse_args(args)

            profile_env = os.environ.get("KCMT_PROFILE_STARTUP", "")
            env_profile = profile_env.lower() in {"1", "true", "yes", "on"}
            self._profile_enabled = bool(
                getattr(parsed_args, "profile_startup", False) or env_profile
            )

            requested_path = (
                Path(parsed_args.repo_path).expanduser().resolve(strict=False)
            )

            detected_root: Optional[Path] = None
            with self._profile_timer(
                "find-git-root",
                extra=lambda: (
                    f"found={detected_root}" if detected_root else "found=<none>"
                ),
            ):
                detected_root = find_git_repo_root(requested_path)

            repo_root = (detected_root or requested_path).resolve(strict=False)
            self._repo_root = repo_root
            non_interactive = (
                bool(os.environ.get("PYTEST_CURRENT_TEST")) or not sys.stdin.isatty()
            )

            self._compact_mode = bool(getattr(parsed_args, "compact", False))
            if self._compact_mode and hasattr(parsed_args, "no_progress"):
                parsed_args.no_progress = True

            if getattr(parsed_args, "command", None) == "status":
                return self._execute_status(parsed_args, repo_root)

            # Allow providing the token via CLI for this run
            if getattr(parsed_args, "github_token", None):
                os.environ["GITHUB_TOKEN"] = parsed_args.github_token

            if parsed_args.list_models and not parsed_args.benchmark:
                return self._execute_list_models(parsed_args)
            if parsed_args.configure_all:
                return self._run_configuration_all(parsed_args, repo_root)
            if parsed_args.verify_keys:
                return self._execute_verify_keys(parsed_args, repo_root)
            if parsed_args.benchmark:
                return self._execute_benchmark(parsed_args)
            if parsed_args.configure:
                return self._run_configuration(parsed_args, repo_root)

            overrides = self._collect_overrides(parsed_args, repo_root)

            persisted_config: Optional[Config] = None
            with self._profile_timer(
                "load-persisted-config",
                extra=lambda: (
                    "result=missing" if persisted_config is None else "result=loaded"
                ),
            ):
                persisted_config = load_persisted_config(repo_root)

            # Check if this is the first time running kcmt in this repo
            config: Optional[Config] = None
            if not persisted_config:
                if non_interactive:
                    with self._profile_timer(
                        "load-config",
                        extra=lambda: (f"provider={config.provider}" if config else ""),
                    ):
                        config = load_config(
                            repo_root=repo_root,
                            overrides=overrides,
                        )
                    with self._profile_timer("persist-config"):
                        save_config(config, repo_root)
                else:
                    self._print_info(
                        "🚀 Welcome to kcmt! This appears to be your first "
                        "time using kcmt in this repository."
                    )
                    self._print_info(
                        "Let's set up your preferred AI provider for "
                        "generating commit messages."
                    )
                    return self._run_configuration(parsed_args, repo_root)
            else:
                with self._profile_timer(
                    "load-config",
                    extra=lambda: (f"provider={config.provider}" if config else ""),
                ):
                    config = load_config(repo_root=repo_root, overrides=overrides)

            # On first use of a provider without a recorded preferred model,
            # prompt the user to choose and persist the selection.
            if (
                not non_interactive
                and not getattr(parsed_args, "model", None)
                and isinstance(getattr(config, "providers", {}), dict)
            ):
                pmap = getattr(config, "providers", {}) or {}
                pentry = pmap.get(config.provider, {}) if isinstance(pmap, dict) else {}
                preferred = (
                    pentry.get("preferred_model") if isinstance(pentry, dict) else None
                )
                if not preferred:
                    self._print_heading(f"Select preferred model for {config.provider}")
                    default_model = DEFAULT_MODELS[config.provider]["model"]
                    chosen = self._prompt_model_with_menu(
                        config.provider, default_model
                    )
                    # Persist the selection
                    pentry["preferred_model"] = chosen
                    pmap[config.provider] = pentry
                    config.model = chosen
                    config.providers = pmap
                    try:
                        save_config(config, repo_root)
                        self._print_success("Saved preferred model selection.")
                    except OSError:
                        pass

            if not persisted_config:
                refreshed_cfg: Optional[Config] = None
                with self._profile_timer("reload-persisted-config"):
                    refreshed_cfg = load_persisted_config(repo_root)
                persisted_cfg = refreshed_cfg
            else:
                persisted_cfg = persisted_config

            # Persist updated boolean feature flags so subsequent plain runs
            # (without explicit flags) retain user preference. This mirrors
            # typical CLI tooling that records config after feature toggles.
            # Persist when flags explicitly overridden OR when env enabled a
            # feature not yet persisted (so subsequent plain runs inherit it)
            should_persist = False
            if "auto_push" in overrides:
                should_persist = True
            elif config.auto_push and (
                not persisted_cfg or not getattr(persisted_cfg, "auto_push", False)
            ):
                should_persist = True
            if should_persist:
                try:  # pragma: no cover - trivial persistence path
                    with self._profile_timer("persist-flags"):
                        save_config(config, repo_root)
                except OSError:  # Narrowed from broad Exception
                    pass

            if not config.resolve_api_key():
                # Allow tests that explicitly pass --api-key-env but don't
                # exercise LLM paths (monkeypatched workflow) to proceed.
                if os.environ.get("PYTEST_CURRENT_TEST") and getattr(
                    parsed_args, "api_key_env", None
                ):
                    self._print_warning(
                        "Proceeding without API key (test mode, explicit "
                        "api-key-env provided)."
                    )
                else:
                    self._print_error(
                        "No API key available. Run 'kcmt --configure' to "
                        "select a provider."
                    )
                    return 2

            self._print_banner(config, parsed_args)

            if parsed_args.single_file:
                return self._execute_single_file(parsed_args, config)
            if parsed_args.oneshot:
                return self._execute_oneshot(parsed_args, config)
            return self._execute_workflow(parsed_args, config)

        except GitError as err:
            self._print_error(str(err))
            return 1
        except LLMError as err:
            self._print_error(
                "LLM failure: {}\nRun 'kcmt --configure' to update your "
                "provider settings.".format(err)
            )
            return 1
        except KlingonCMTError as err:
            self._print_error(str(err))
            return 1
        except SystemExit as exc:  # argparse
            return int(exc.code) if isinstance(exc.code, int) else 0

    # ------------------------------------------------------------------
    # Configuration helpers
    # ------------------------------------------------------------------
    def _collect_overrides(
        self, args: argparse.Namespace, repo_root: Path
    ) -> dict[str, str]:
        overrides: dict[str, str] = {}
        if args.provider:
            overrides["provider"] = args.provider
        if args.model:
            overrides["model"] = args.model
        if args.endpoint:
            overrides["endpoint"] = args.endpoint
        if args.api_key_env:
            overrides["api_key_env"] = args.api_key_env
        if args.max_commit_length is not None:
            overrides["max_commit_length"] = str(args.max_commit_length)
        if args.repo_path:
            overrides["repo_path"] = str(repo_root.expanduser().resolve(strict=False))
        if getattr(args, "auto_push", False):
            overrides["auto_push"] = "1"
        if getattr(args, "no_auto_push", False):
            overrides["auto_push"] = "0"
        return overrides

    def _run_configuration(self, args: argparse.Namespace, repo_root: Path) -> int:
        detected = detect_available_providers()
        provider = args.provider or self._prompt_provider(detected)

        provider_meta = DEFAULT_MODELS[provider]
        # Rich model selection with pricing, but remain compatible with freeform
        # manual entry used by tests and power users.
        model = args.model or self._prompt_model_with_menu(
            provider, provider_meta["model"]
        )
        endpoint = args.endpoint or self._prompt_endpoint(
            provider, provider_meta["endpoint"]
        )
        api_key_env = args.api_key_env or self._prompt_api_key_env(provider, detected)

        # Optional secondary configuration block
        sec_provider: Optional[str] = None
        sec_model: Optional[str] = None
        sec_endpoint: Optional[str] = None
        sec_api_key_env: Optional[str] = None

        is_non_interactive = (
            bool(os.environ.get("PYTEST_CURRENT_TEST")) or not sys.stdin.isatty()
        )
        choice = ""
        if not is_non_interactive:
            try:
                choice = (
                    input(f"{MAGENTA}Configure a secondary provider?{RESET} [y/N]: ")
                    .strip()
                    .lower()
                )
            except EOFError:
                choice = ""
        if choice in {"y", "yes"}:
            # Exclude already chosen provider from the quick list indicator,
            # but still allow selecting it if user insists.
            self._print_heading("Secondary provider")
            sec_provider = self._prompt_provider(detected)
            sec_meta = DEFAULT_MODELS[sec_provider]
            sec_model = self._prompt_model_with_menu(sec_provider, sec_meta["model"])
            sec_endpoint = self._prompt_endpoint(sec_provider, sec_meta["endpoint"])
            sec_api_key_env = self._prompt_api_key_env(sec_provider, detected)

        # Start from existing providers map if present to preserve prior edits
        from .config import DEFAULT_MODELS as _DM
        from .config import PROVIDER_DISPLAY_NAMES

        existing = load_persisted_config(repo_root)
        providers_map: dict[str, dict[str, Any]] = {}
        if existing and isinstance(getattr(existing, "providers", {}), dict):
            providers_map = dict(existing.providers)
        # Ensure all providers have an entry with defaults
        for prov, meta in _DM.items():
            ent = providers_map.get(prov, {}) or {}
            ent.setdefault("name", PROVIDER_DISPLAY_NAMES.get(prov, prov))
            ent.setdefault("endpoint", meta["endpoint"])
            ent.setdefault("api_key_env", meta["api_key_env"])
            providers_map[prov] = ent
        # Apply primary choices
        p_entry = providers_map.get(provider, {})
        p_entry["endpoint"] = endpoint
        p_entry["api_key_env"] = api_key_env
        p_entry["preferred_model"] = model
        providers_map[provider] = p_entry
        # Apply secondary choices if any
        if sec_provider:
            sp_meta = _DM[sec_provider]
            sp_entry = providers_map.get(sec_provider, {})
            sp_entry.setdefault(
                "name", PROVIDER_DISPLAY_NAMES.get(sec_provider, sec_provider)
            )
            sp_entry["endpoint"] = sec_endpoint or sp_meta["endpoint"]
            sp_entry["api_key_env"] = sec_api_key_env or sp_meta["api_key_env"]
            sp_entry["preferred_model"] = sec_model or sp_meta["model"]
            providers_map[sec_provider] = sp_entry

        # Keep legacy mapping in sync for compatibility
        provider_env_overrides = {
            prov: str(ent.get("api_key_env") or _DM[prov]["api_key_env"])
            for prov, ent in providers_map.items()
        }

        config = Config(
            provider=provider,
            model=model,
            llm_endpoint=endpoint,
            api_key_env=api_key_env,
            secondary_provider=sec_provider,
            secondary_model=sec_model,
            secondary_llm_endpoint=sec_endpoint,
            secondary_api_key_env=sec_api_key_env,
            git_repo_path=str(repo_root.expanduser().resolve(strict=False)),
            providers=providers_map,
            provider_env_overrides=provider_env_overrides,
        )
        save_config(config, repo_root)

        self._print_success(
            "Configuration saved to {}".format(
                (repo_root / ".kcmt" / "config.json").relative_to(repo_root)
            )
        )
        return 0

    def _run_configuration_all(self, args: argparse.Namespace, repo_root: Path) -> int:
        """Let the user choose which providers to configure API key env vars for.

        - Does NOT change the primary/secondary provider.
        - Does NOT prompt for endpoints or models; defaults are fine.
        - Only records the env var name to use for each chosen provider.
        """
        repo_root = Path(repo_root).expanduser().resolve(strict=False)
        det = detect_available_providers()
        providers = sorted(DEFAULT_MODELS.keys())

        self._print_heading("Select providers to configure")
        for idx, p in enumerate(providers, start=1):
            badge = GREEN + "●" + RESET if det.get(p) else YELLOW + "○" + RESET
            print(f"  {idx}. {badge} {describe_provider(p)}")
        try:
            raw = input(
                f"{MAGENTA}Enter number(s) (comma-separated) or 'all'{RESET} [all]: "
            ).strip()
        except EOFError:
            raw = "all"
        if not raw:
            raw = "all"
        selected: list[str] = []
        if raw.lower() in {"all", "*"}:
            selected = providers
        else:
            for token in raw.split(","):
                token = token.strip()
                if not token:
                    continue
                if token.isdigit():
                    i = int(token)
                    if 1 <= i <= len(providers):
                        selected.append(providers[i - 1])
                elif token in providers:
                    selected.append(token)
        if not selected:
            self._print_warning("No valid providers selected; nothing to do.")
            return 1

        # Load existing config or build one; we won't change provider/model here
        try:
            cfg = load_config(repo_root=repo_root)
        except Exception:
            cfg = Config(
                provider="openai",
                model=DEFAULT_MODELS["openai"]["model"],
                llm_endpoint=DEFAULT_MODELS["openai"]["endpoint"],
                api_key_env=DEFAULT_MODELS["openai"]["api_key_env"],
                git_repo_path=str(repo_root),
            )

        # Build/extend the mapping
        overrides = dict(getattr(cfg, "provider_env_overrides", {}) or {})
        providers_map = dict(getattr(cfg, "providers", {}) or {})
        for prov in selected:
            self._print_heading(f"API key for {prov}")
            env_key = self._prompt_api_key_env(prov, det)
            overrides[prov] = env_key
            # Ensure providers map entry exists and is updated
            entry = providers_map.get(prov, {})
            entry["api_key_env"] = env_key
            entry.setdefault("name", describe_provider(prov).split(" (")[0])
            entry.setdefault("endpoint", DEFAULT_MODELS[prov]["endpoint"])
            providers_map[prov] = entry

        cfg.provider_env_overrides = overrides
        cfg.providers = providers_map
        save_config(cfg, repo_root)
        self._print_success(
            "Saved API key env var mapping for: {}".format(", ".join(selected))
        )
        return 0

    def _prompt_provider(self, detected: dict[str, list[str]]) -> str:
        self._print_heading("Select provider")
        for idx, name in enumerate(sorted(DEFAULT_MODELS.keys()), start=1):
            badge = GREEN + "●" + RESET if detected.get(name) else YELLOW + "○" + RESET
            print(f"  {idx}. {badge} {describe_provider(name)}")

        while True:
            choice = input(
                f"{MAGENTA}Provider [1-{len(DEFAULT_MODELS)}]{RESET}: "
            ).strip()
            if not choice:
                choice = "1"
            if choice.isdigit() and 1 <= int(choice) <= len(DEFAULT_MODELS):
                provider = sorted(DEFAULT_MODELS.keys())[int(choice) - 1]
                if not detected.get(provider):
                    confirm = (
                        input(
                            f"{YELLOW}No keys detected for {provider}. Continue?"
                            f" [y/N]{RESET} "
                        )
                        .strip()
                        .lower()
                    )
                    if confirm not in {"y", "yes"}:
                        continue
                return provider
            self._print_warning("Invalid selection. Please choose again.")

    def _prompt_model(self, provider: str, default_model: str) -> str:
        prompt = f"{MAGENTA}Model for {provider}{RESET} [{default_model}]: "
        response = input(prompt).strip()
        return response or default_model

    def _prompt_model_with_menu(self, provider: str, default_model: str) -> str:
        """Display a short model list with pricing; accept manual entry.

        Compatible with tests: if user types a freeform value (e.g. 'my-model'),
        it is accepted directly.
        """
        # Try to fetch models (enriched with pricing). Fall back to a simple
        # prompt if anything goes sideways.
        try:
            models = self._list_enriched_models_for_provider(provider)
        except Exception:  # noqa: BLE001 - Non-fatal; revert to simple prompt
            models = []

        if not models:
            return self._prompt_model(provider, default_model)

        # Prefer cheapest completion token price, then cheapest prompt price
        def _price_key(m: dict[str, Any]) -> tuple[float, float, str]:
            outp = float(m.get("output_price_per_mtok") or 1e12)
            inp = float(m.get("input_price_per_mtok") or 1e12)
            return (outp, inp, str(m.get("id", "")))

        models_sorted = sorted(models, key=_price_key)

        self._print_heading(f"Models for {provider} (per 1M tokens)")
        # Render a compact board (top N entries) to keep it readable.
        preview = models_sorted[:25]
        self._print_models_table(provider, preview, default_model)

        # Accept number or free-form entry
        try:
            choice = input(
                f"{MAGENTA}Select model number or type id{RESET} "
                f"[{default_model}]: "
            ).strip()
        except EOFError:
            choice = ""
        if not choice:
            return default_model
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(preview):
                return str(preview[idx].get("id", default_model))
        # Treat as explicit model id
        return choice

    def _prompt_endpoint(self, provider: str, default_endpoint: str) -> str:
        prompt = f"{MAGENTA}Endpoint for {provider}{RESET} [{default_endpoint}]: "
        response = input(prompt).strip()
        return response or default_endpoint

    def _prompt_api_key_env(self, provider: str, detected: dict[str, list[str]]) -> str:
        matches = detected.get(provider, [])
        if not matches:
            default_env = DEFAULT_MODELS[provider]["api_key_env"]
            prompt = (
                f"{MAGENTA}Environment variable with API key{RESET} [{default_env}]: "
            )
            response = input(prompt).strip()
            return response or default_env

        self._print_heading("Select API key environment variable")
        for idx, env_key in enumerate(matches, start=1):
            marker = GREEN + "●" + RESET if env_key in os.environ else RED + "●" + RESET
            suffix = " (missing)" if env_key not in os.environ else ""
            print(f"  {idx}. {marker} {env_key}{suffix}")
        print(f"  {len(matches) + 1}. {CYAN}Enter a different variable{RESET}")

        while True:
            choice = input(
                f"{MAGENTA}API key variable [1-{len(matches) + 1}]{RESET}: "
            ).strip()
            if not choice:
                choice = "1"
            if choice.isdigit():
                num = int(choice)
                if 1 <= num <= len(matches):
                    return matches[num - 1]
                if num == len(matches) + 1:
                    break
            self._print_warning("Invalid selection. Try again.")

        entered = input(f"{MAGENTA}Enter environment variable name{RESET}: ").strip()
        return entered or DEFAULT_MODELS[provider]["api_key_env"]

    # ------------------------------------------------------------------
    # Execution modes
    # ------------------------------------------------------------------
    def _execute_workflow(self, args: argparse.Namespace, config: Config) -> int:
        if self._compact_mode:
            self._print_compact_header(config, args)
        else:
            self._print_info(f"Provider: {config.provider}")
            self._print_info(f"Model: {config.model}")
            self._print_info(f"Endpoint: {config.llm_endpoint}")
            self._print_info(f"Max retries: {args.max_retries}")
            if hasattr(args, "limit") and args.limit:
                self._print_info(f"File limit: {args.limit}")
            self._print_info("")

        workflow = None
        run_started = time.perf_counter()

        def _wf_extra() -> str:
            if workflow is None:
                return ""
            return f"repo={workflow.git_repo.repo_path}"

        raw_kwargs = {
            "repo_path": config.git_repo_path,
            "max_retries": args.max_retries,
            "config": config,
            "show_progress": not args.no_progress,
            "file_limit": getattr(args, "limit", None),
            "debug": getattr(args, "debug", False),
            "profile": self._profile_enabled,
            "verbose": getattr(args, "verbose", False),
            "workers": getattr(args, "workers", None),
        }
        signature = inspect.signature(KlingonCMTWorkflow)
        filtered_kwargs = {
            key: value
            for key, value in raw_kwargs.items()
            if key in signature.parameters
            and (value is not None or key != "file_limit")
        }

        with self._profile_timer("init-workflow", extra=_wf_extra):
            workflow = KlingonCMTWorkflow(**filtered_kwargs)

        results: dict[str, Any] = {}

        with self._profile_timer(
            "execute-workflow",
            extra=lambda: (
                "files={}".format(
                    len(results.get("file_commits", []))
                    if isinstance(results, dict)
                    else 0
                )
            ),
        ):
            results = workflow.execute_workflow()
        duration = time.perf_counter() - run_started
        snapshot = self._build_run_snapshot(args, config, results, duration, workflow)
        self._display_results(results, args.verbose, snapshot)
        self._persist_run_snapshot(snapshot)
        return 0

    def _execute_oneshot(self, args: argparse.Namespace, config: Config) -> int:
        repo = GitRepo(config.git_repo_path, config)
        entries = repo.list_changed_files()

        if not entries:
            self._print_info("No changes to commit.")
            return 0

        non_deletions = [entry for entry in entries if "D" not in entry[0]]
        target_path = (non_deletions[0] if non_deletions else entries[0])[1]

        self._print_info(f"One-shot targeting file: {target_path}")
        args.single_file = target_path
        return self._execute_single_file(args, config)

    def _execute_list_models(self, args: argparse.Namespace) -> int:
        """Query each provider for models. Drivers include enrichment."""

        from .providers.anthropic_driver import AnthropicDriver
        from .providers.base import BaseDriver
        from .providers.openai_driver import OpenAIDriver
        from .providers.xai_driver import XAIDriver

        # Build per-provider configs using active env
        configs: dict[str, Config] = {}
        for prov in ("openai", "anthropic", "xai", "github"):
            overrides: dict[str, str] = {"provider": prov}
            try:
                cfg = load_config(overrides=overrides)
                configs[prov] = cfg
            except (ValueError, OSError, RuntimeError, TypeError, KeyError):
                continue

        out: dict[str, Any] = {}
        for prov, cfg in configs.items():
            try:
                driver: BaseDriver
                if prov in {"openai", "github"}:
                    driver = OpenAIDriver(
                        cfg,
                        debug=getattr(args, "debug", False),
                    )
                elif prov == "xai":
                    driver = XAIDriver(
                        cfg,
                        debug=getattr(args, "debug", False),
                    )
                else:
                    driver = AnthropicDriver(
                        cfg,
                        debug=getattr(args, "debug", False),
                    )
                out[prov] = driver.list_models()
            except (ValueError, RuntimeError, TypeError, KeyError) as e:
                # Fallback: use dataset-derived listing so users still
                # see models for this provider
                try:
                    from .providers.pricing import build_enrichment_context as _bctx
                    from .providers.pricing import enrich_ids as _enrich

                    alias_lut, _ctx, _mx = _bctx()
                    ids: list[str] = []
                    seen: set[str] = set()
                    for (p, mid), canon in alias_lut.items():
                        if p != prov:
                            continue
                        for candidate in (str(canon), str(mid)):
                            if candidate and candidate not in seen:
                                ids.append(candidate)
                                seen.add(candidate)
                    # Apply provider-specific filters for CLI fallback
                    if prov in {"openai", "github"}:
                        ids = [mm for mm in ids if OpenAIDriver.is_allowed_model_id(mm)]
                    elif prov == "xai":
                        ids = [mm for mm in ids if XAIDriver.is_allowed_model_id(mm)]
                    elif prov == "anthropic":
                        ids = [
                            mm for mm in ids if AnthropicDriver.is_allowed_model_id(mm)
                        ]
                    try:
                        emap = _enrich(prov, ids)
                    except (
                        ValueError,
                        RuntimeError,
                        KeyError,
                        TypeError,
                    ):
                        emap = {}
                    owned_by = prov
                    out_list: list[dict[str, Any]] = []
                    for mid in ids:
                        em = emap.get(mid) or {}
                        if not em or not em.get("_has_pricing", False):
                            if getattr(args, "debug", False):
                                print(
                                    "DEBUG(CLI:list-models): skipping %s/%s "
                                    "due to missing pricing" % (prov, mid)
                                )
                            continue
                        payload = dict(em)
                        payload.pop("_has_pricing", None)
                        out_list.append(
                            {
                                "id": mid,
                                "owned_by": owned_by,
                                **payload,
                            }
                        )
                    out[prov] = out_list
                except (
                    ImportError,
                    ValueError,
                    KeyError,
                    TypeError,
                    RuntimeError,
                    AttributeError,
                ):
                    out[prov] = {"error": str(e)}

        # If debug flag is set, preserve the raw JSON output for tooling.
        if getattr(args, "debug", False):
            print(
                json.dumps(
                    out,
                    indent=2,
                    ensure_ascii=False,
                    cls=DecimalFriendlyJSONEncoder,
                )
            )
            return 0

        # Otherwise render a pricing comparison board across providers
        self._print_heading("Model Pricing (per 1M tokens)")
        self._print_pricing_board(out)
        return 0

    def _execute_benchmark(self, args: argparse.Namespace) -> int:
        """Run a benchmark across providers/models using sample diffs.

        Uses the same discovery as --list-models and estimates cost using
        published per-1M token prices. Only models for providers with available
        API keys are exercised.
        """
        # Build per-provider model lists (same enrichment/filters as list-models)
        providers = ("openai", "anthropic", "xai", "github")
        models_map: dict[str, list[dict[str, Any]]] = {}
        for prov in providers:
            try:
                models = self._list_enriched_models_for_provider(prov)
            except Exception:
                models = []
            # Fallback: if enrichment returned nothing, try raw provider listing
            if not models:
                try:
                    models = self._list_models_raw_for_provider(prov)
                except Exception:
                    models = []
            models_map[prov] = models

        # Limit models per provider
        limit = getattr(args, "benchmark_limit", None)
        if isinstance(limit, int) and limit > 0:
            models_map = {k: (v[:limit] if v else []) for k, v in models_map.items()}

        # Run benchmark
        try:
            from .benchmark import run_benchmark
        except Exception as e:  # pragma: no cover - import protection
            self._print_error(f"Benchmark module unavailable: {e}")
            return 1

        timeout = getattr(args, "benchmark_timeout", None)
        debug_flag = bool(getattr(args, "debug", False))
        # Live progress rendering
        is_tty = sys.stdout.isatty()

        def _make_bar(done: int, total: int, width: int = 40) -> str:
            if total <= 0:
                total = 1
            frac = max(0.0, min(1.0, done / float(total)))
            filled = int(frac * width)
            return "[{}{}] {:>3}%".format(
                "#" * filled, "-" * (width - filled), int(frac * 100)
            )

        # Shared state for progress callback
        _pstate: dict[str, int | str] = {"done": 0, "total": 0, "label": ""}

        from typing import SupportsInt

        def _progress(stage: str, info: dict[str, object]) -> None:
            try:
                if stage == "init":
                    total_raw = info.get("total_runs", 0)
                    try:
                        _pstate["total"] = int(
                            cast(SupportsInt | str | bytes | bytearray, total_raw)
                        )
                    except Exception:
                        _pstate["total"] = 0
                    _pstate["done"] = 0
                    _pstate["label"] = ""
                    if is_tty and _pstate["total"]:
                        bar = _make_bar(0, int(_pstate["total"]))
                        print(f"Benchmarking {bar} ", end="\r", flush=True)
                elif stage == "provider":
                    prov = str(info.get("provider", ""))
                    if is_tty and _pstate.get("total", 0):
                        # Light provider hint in label
                        _pstate["label"] = prov
                elif stage == "model_start":
                    prov = str(info.get("provider", _pstate.get("label", "")))
                    mid = str(info.get("model", ""))
                    if is_tty and _pstate.get("total", 0):
                        _pstate["label"] = f"{prov} / {mid}"
                elif stage == "tick":
                    done_raw = info.get("done", _pstate.get("done", 0))
                    try:
                        _pstate["done"] = int(
                            cast(SupportsInt | str | bytes | bytearray, done_raw)
                        )
                    except Exception:
                        _pstate["done"] = int(_pstate.get("done", 0))
                    prov = str(info.get("provider", ""))
                    mid = str(info.get("model", ""))
                    samp = str(info.get("sample", ""))
                    if prov and mid:
                        _pstate["label"] = f"{prov} / {mid} ({samp})"
                    if is_tty and _pstate.get("total", 0):
                        bar = _make_bar(int(_pstate["done"]), int(_pstate["total"]))
                        msg = f"Benchmarking {bar} {_pstate.get('label','')}"
                        # Ensure line overwrite
                        print(msg.ljust(120), end="\r", flush=True)
                elif stage == "done":
                    if is_tty:
                        bar = _make_bar(
                            int(_pstate.get("total", 0) or 1),
                            int(_pstate.get("total", 0) or 1),
                        )
                        print(f"Benchmarking {bar} done".ljust(120))
            except Exception:
                # Never let progress updates break the benchmark
                pass

        results = run_benchmark(
            models_map,
            per_provider_limit=limit,
            request_timeout=timeout,
            debug=debug_flag,
            progress=_progress,
        )

        if not results:
            self._print_warning(
                "No benchmarkable models found. Ensure API keys are set for providers."
            )
            return 1

        # Build leaderboards
        def _fmt_money(v: float) -> str:
            if v < 1.0:
                return f"${v:.4f}"
            return f"${v:.2f}"

        # Fastest by avg latency
        fastest = sorted(results, key=lambda r: (r.avg_latency_ms, r.avg_cost_usd))[:10]
        # Cheapest by cost
        cheapest = sorted(results, key=lambda r: (r.avg_cost_usd, r.avg_latency_ms))[
            :10
        ]
        # Best quality by quality score desc
        best_quality = sorted(results, key=lambda r: (-r.quality, r.avg_latency_ms))[
            :10
        ]
        # Stability by success rate desc
        most_stable = sorted(
            results, key=lambda r: (-r.success_rate, r.avg_latency_ms)
        )[:10]

        # Overall: normalized mix (quality 0.4, cost 0.3, time 0.3)
        min_lat = min(r.avg_latency_ms for r in results)
        max_lat = max(r.avg_latency_ms for r in results)
        min_cost = min(r.avg_cost_usd for r in results)
        max_cost = max(r.avg_cost_usd for r in results)

        def _norm(val: float, lo: float, hi: float) -> float:
            if hi <= lo:
                return 1.0
            return (val - lo) / (hi - lo)

        overall: list[tuple[float, Any]] = []
        for r in results:
            # Higher quality is better, lower cost/time is better
            q = r.quality / 100.0
            c = 1.0 - _norm(r.avg_cost_usd, min_cost, max_cost)
            t = 1.0 - _norm(r.avg_latency_ms, min_lat, max_lat)
            score = 0.4 * q + 0.3 * c + 0.3 * t
            overall.append((score, r))
        overall_sorted = [r for (_s, r) in sorted(overall, key=lambda kv: -kv[0])[:10]]

        # Persist snapshot for later comparison
        try:
            snapshot = {
                "schema_version": 1,
                "timestamp": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "repo_path": str(self._repo_root) if self._repo_root else ".",
                "params": {
                    "limit": limit,
                    "timeout": timeout,
                },
                "results": [
                    {
                        "provider": r.provider,
                        "model": r.model,
                        "avg_latency_ms": r.avg_latency_ms,
                        "avg_cost_usd": r.avg_cost_usd,
                        "quality": r.quality,
                        "success_rate": r.success_rate,
                        "runs": r.runs,
                    }
                    for r in results
                ],
            }
            self._persist_benchmark_snapshot(snapshot)
        except Exception:  # pragma: no cover - best-effort persistence
            pass

        # Render
        self._print_heading("Benchmark Leaderboard")

        def _print_board(title: str, rows: list[Any]) -> None:
            print(f"{BOLD}{CYAN}{title}{RESET}")
            # widths
            w_prov = max((len(r.provider) for r in rows), default=6)
            w_mid = max((len(r.model) for r in rows), default=10)
            header = (
                f"  {'provider':<{w_prov}}  {'model':<{w_mid}}  "
                f"{'latency(ms)':>12}  {'cost':>10}  {'quality':>8}  {'success':>8}"
            )
            print(CYAN + header + RESET)
            for r in rows:
                print(
                    "  {prov:<{wp}}  {mid:<{wm}}  {lat:>12.1f}  {cost:>10}  {q:>8.1f}  {sr:>8.0%}".format(
                        prov=r.provider,
                        mid=r.model,
                        lat=r.avg_latency_ms,
                        cost=_fmt_money(r.avg_cost_usd),
                        q=r.quality,
                        sr=r.success_rate,
                        wp=w_prov,
                        wm=w_mid,
                    )
                )

        _print_board("Overall", overall_sorted)
        print()
        _print_board("Fastest", fastest)
        print()
        _print_board("Cheapest", cheapest)
        print()
        _print_board("Best Quality", best_quality)
        print()
        _print_board("Most Stable", most_stable)

        # Grouped results: by provider -> models
        print()
        self._print_heading("Results by Provider")
        by_prov: dict[str, list[Any]] = {}
        for r in results:
            by_prov.setdefault(r.provider, []).append(r)
        for prov in sorted(by_prov.keys()):
            rows = by_prov[prov]
            # Sort by latency then cost for readability
            rows_sorted = sorted(rows, key=lambda r: (r.avg_latency_ms, r.avg_cost_usd))
            print(f"{BOLD}{prov}{RESET}")
            w_mid = max((len(r.model) for r in rows_sorted), default=10)
            header = f"  {'model':<{w_mid}}  {'latency(ms)':>12}  {'cost':>10}  {'quality':>8}  {'success':>8}"
            print(CYAN + header + RESET)
            for r in rows_sorted:
                print(
                    "  {mid:<{wm}}  {lat:>12.1f}  {cost:>10}  {q:>8.1f}  {sr:>8.0%}".format(
                        mid=r.model,
                        lat=r.avg_latency_ms,
                        cost=_fmt_money(r.avg_cost_usd),
                        q=r.quality,
                        sr=r.success_rate,
                        wm=w_mid,
                    )
                )

        # Optional JSON / CSV outputs after leaderboard
        if bool(getattr(args, "benchmark_json", False)):
            blob = {
                "overall": [
                    {
                        "provider": r.provider,
                        "model": r.model,
                        "avg_latency_ms": r.avg_latency_ms,
                        "avg_cost_usd": r.avg_cost_usd,
                        "quality": r.quality,
                        "success_rate": r.success_rate,
                        "runs": r.runs,
                    }
                    for r in overall_sorted
                ],
                "fastest": [
                    {
                        "provider": r.provider,
                        "model": r.model,
                        "avg_latency_ms": r.avg_latency_ms,
                        "avg_cost_usd": r.avg_cost_usd,
                        "quality": r.quality,
                        "success_rate": r.success_rate,
                        "runs": r.runs,
                    }
                    for r in fastest
                ],
                "cheapest": [
                    {
                        "provider": r.provider,
                        "model": r.model,
                        "avg_latency_ms": r.avg_latency_ms,
                        "avg_cost_usd": r.avg_cost_usd,
                        "quality": r.quality,
                        "success_rate": r.success_rate,
                        "runs": r.runs,
                    }
                    for r in cheapest
                ],
                "best_quality": [
                    {
                        "provider": r.provider,
                        "model": r.model,
                        "avg_latency_ms": r.avg_latency_ms,
                        "avg_cost_usd": r.avg_cost_usd,
                        "quality": r.quality,
                        "success_rate": r.success_rate,
                        "runs": r.runs,
                    }
                    for r in best_quality
                ],
                "most_stable": [
                    {
                        "provider": r.provider,
                        "model": r.model,
                        "avg_latency_ms": r.avg_latency_ms,
                        "avg_cost_usd": r.avg_cost_usd,
                        "quality": r.quality,
                        "success_rate": r.success_rate,
                        "runs": r.runs,
                    }
                    for r in most_stable
                ],
            }
            print(
                json.dumps(
                    blob, indent=2, ensure_ascii=False, cls=DecimalFriendlyJSONEncoder
                )
            )

        if bool(getattr(args, "benchmark_csv", False)):
            # Write header + rows; CSV for the raw results
            import csv
            from io import StringIO

            buf = StringIO()
            writer = csv.writer(buf)
            writer.writerow(
                [
                    "provider",
                    "model",
                    "avg_latency_ms",
                    "avg_cost_usd",
                    "quality",
                    "success_rate",
                    "runs",
                ]
            )
            for r in results:
                writer.writerow(
                    [
                        r.provider,
                        r.model,
                        f"{r.avg_latency_ms:.3f}",
                        f"{r.avg_cost_usd:.6f}",
                        f"{r.quality:.2f}",
                        f"{r.success_rate:.2f}",
                        r.runs,
                    ]
                )
            print(buf.getvalue().rstrip("\n"))

        return 0

    def _execute_verify_keys(self, args: argparse.Namespace, repo_root: Path) -> int:
        detected = detect_available_providers()
        providers = sorted(DEFAULT_MODELS.keys())

        from .config import load_config

        rows: list[tuple[str, str, str, str]] = []
        for prov in providers:
            try:
                cfg = load_config(repo_root=repo_root, overrides={"provider": prov})
                env_var = cfg.api_key_env
            except Exception:
                env_var = DEFAULT_MODELS[prov]["api_key_env"]
            present = (
                "yes" if env_var in os.environ and os.environ.get(env_var) else "no"
            )
            detected_list = detected.get(prov, [])
            hint = ",".join(detected_list[:3]) if detected_list else "-"
            rows.append((prov, env_var, present, hint))

        # Render table
        self._print_heading("API Key Verification")
        w_p = max(max((len(r[0]) for r in rows), default=7), len("provider"))
        w_e = max(max((len(r[1]) for r in rows), default=8), len("env_var"))
        w_s = len("present")
        w_h = max(max((len(r[3]) for r in rows), default=5), len("detected"))
        header = f"  {'provider':<{w_p}}  {'env_var':<{w_e}}  {'present':<{w_s}}  {'detected':<{w_h}}"
        print(CYAN + header + RESET)
        for prov, env_var, present, hint in rows:
            line = f"  {prov:<{w_p}}  {env_var:<{w_e}}  {present:<{w_s}}  {hint:<{w_h}}"
            print(line)
        return 0

    def _persist_benchmark_snapshot(self, snapshot: dict[str, Any]) -> None:
        try:
            base = self._repo_root or Path.cwd()
            root = Path(base).resolve(strict=False)
            bdir = root / ".kcmt" / "benchmarks"
            bdir.mkdir(parents=True, exist_ok=True)
            fname = f"benchmark-{snapshot.get('timestamp','').replace(':','').replace('Z','Z')}.json"
            path = bdir / fname
            with path.open("w", encoding="utf-8") as handle:
                json.dump(
                    snapshot,
                    handle,
                    indent=2,
                    ensure_ascii=False,
                    cls=DecimalFriendlyJSONEncoder,
                )
        except OSError:  # pragma: no cover - best effort
            pass

    # ------------------------------------------------------------------
    # Discovery/formatting helpers for models & pricing
    # ------------------------------------------------------------------
    def _list_enriched_models_for_provider(self, prov: str) -> list[dict[str, Any]]:
        """Return enriched models for one provider, tolerating missing keys.

        Uses driver.list_models() when possible, with dataset fallback mirroring
        the --list-models path, and includes pricing enrichment.
        """
        from .providers.anthropic_driver import AnthropicDriver
        from .providers.base import BaseDriver
        from .providers.openai_driver import OpenAIDriver
        from .providers.xai_driver import XAIDriver

        overrides: dict[str, str] = {"provider": prov}
        try:
            cfg = load_config(overrides=overrides)
        except Exception:  # noqa: BLE001
            # Minimal config if config load has issues
            meta = DEFAULT_MODELS.get(prov, {})
            cfg = Config(
                provider=prov,
                model=str(meta.get("model", "")),
                llm_endpoint=str(meta.get("endpoint", "")),
                api_key_env=str(meta.get("api_key_env", "")),
            )

        driver: BaseDriver
        if prov in {"openai", "github"}:
            driver = OpenAIDriver(cfg, debug=False)
        elif prov == "xai":
            driver = XAIDriver(cfg, debug=False)
        else:
            driver = AnthropicDriver(cfg, debug=False)

        try:
            models = driver.list_models()
        except Exception:  # noqa: BLE001
            # Fallback matching CLI --list-models semantics
            try:
                from .providers.pricing import build_enrichment_context as _bctx
                from .providers.pricing import enrich_ids as _enrich

                alias_lut, _ctx, _mx = _bctx()
                ids: list[str] = []
                seen: set[str] = set()
                for (p, mid), canon in alias_lut.items():
                    if p != prov:
                        continue
                    for candidate in (str(canon), str(mid)):
                        if candidate and candidate not in seen:
                            ids.append(candidate)
                            seen.add(candidate)
                if prov in {"openai", "github"}:
                    ids = [mm for mm in ids if OpenAIDriver.is_allowed_model_id(mm)]
                elif prov == "xai":
                    ids = [mm for mm in ids if XAIDriver.is_allowed_model_id(mm)]
                elif prov == "anthropic":
                    ids = [mm for mm in ids if AnthropicDriver.is_allowed_model_id(mm)]
                emap = _enrich(prov, ids)
                out_list: list[dict[str, Any]] = []
                for mid in ids:
                    em = emap.get(mid) or {}
                    if not em or not em.get("_has_pricing", False):
                        continue
                    payload = dict(em)
                    payload.pop("_has_pricing", None)
                    out_list.append({"id": mid, "owned_by": prov, **payload})
                models = out_list
            except Exception:  # noqa: BLE001
                models = []

        # Ensure pricing presence
        filtered: list[dict[str, Any]] = []
        for m in models:
            em_flag = m.get("input_price_per_mtok") or m.get("output_price_per_mtok")
            if em_flag is None:
                continue
            filtered.append(m)
        return filtered

    def _list_models_raw_for_provider(self, prov: str) -> list[dict[str, Any]]:
        """Return raw provider model list without requiring pricing enrichment."""
        from .providers.anthropic_driver import AnthropicDriver
        from .providers.base import BaseDriver
        from .providers.openai_driver import OpenAIDriver
        from .providers.xai_driver import XAIDriver

        overrides: dict[str, str] = {"provider": prov}
        try:
            cfg = load_config(overrides=overrides)
        except Exception:  # noqa: BLE001
            meta = DEFAULT_MODELS.get(prov, {})
            cfg = Config(
                provider=prov,
                model=str(meta.get("model", "")),
                llm_endpoint=str(meta.get("endpoint", "")),
                api_key_env=str(meta.get("api_key_env", "")),
            )

        driver: BaseDriver
        if prov in {"openai", "github"}:
            driver = OpenAIDriver(cfg, debug=False)
        elif prov == "xai":
            driver = XAIDriver(cfg, debug=False)
        else:
            driver = AnthropicDriver(cfg, debug=False)

        try:
            return driver.list_models()
        except Exception:  # noqa: BLE001
            return []

    def _fmt_money(self, value: Any) -> str:
        try:
            v = float(value)
        except (TypeError, ValueError):
            return "-"
        # For small values, keep four decimals; otherwise two.
        if v < 1.0:
            return f"${v:.4f}"
        return f"${v:.2f}"

    def _print_models_table(
        self, provider: str, models: list[dict[str, Any]], default_model: str
    ) -> None:
        rows: list[tuple[str, str, str, str, str]] = []
        for idx, m in enumerate(models, start=1):
            mid = str(m.get("id", ""))
            inp = self._fmt_money(m.get("input_price_per_mtok"))
            outp = self._fmt_money(m.get("output_price_per_mtok"))
            ctx = m.get("total_context") or m.get("context") or m.get("context_window")
            ctxs = f"{int(ctx):,}" if isinstance(ctx, int) else "-"
            marker = "*" if mid == default_model else " "
            rows.append((str(idx), marker, mid, f"{inp}/{outp}", ctxs))

        # Compute widths
        w_idx = max((len(r[0]) for r in rows), default=1)
        w_mrk = 1
        w_mid = max((len(r[2]) for r in rows), default=5)
        w_price = max((len(r[3]) for r in rows), default=11)
        w_ctx = max((len(r[4]) for r in rows), default=7)

        header = (
            f"  {'#':>{w_idx}}  {'':{w_mrk}}  {'model':<{w_mid}}  "
            f"{'in/out per MTok':<{w_price}}  {'ctx':>{w_ctx}}"
        )
        print(CYAN + header + RESET)
        for r in rows:
            line = (
                f"  {r[0]:>{w_idx}}  {r[1]:{w_mrk}}  {r[2]:<{w_mid}}  "
                f"{r[3]:<{w_price}}  {r[4]:>{w_ctx}}"
            )
            print(line)

    def _print_pricing_board(self, data: dict[str, Any]) -> None:
        # Build combined rows across providers
        combined: list[
            tuple[str, str, float | None, float | None, int | None, int | None]
        ] = []
        for prov, items in data.items():
            if not isinstance(items, list):
                continue
            for m in items:
                if not isinstance(m, dict):
                    continue
                try:
                    mid = str(m.get("id", ""))
                    inp = m.get("input_price_per_mtok")
                    outp = m.get("output_price_per_mtok")
                    if inp is None and outp is None:
                        continue
                    ctx = (
                        m.get("total_context")
                        or m.get("context")
                        or m.get("context_window")
                    )
                    ctxi = int(ctx) if isinstance(ctx, int) else None
                    mx = m.get("max_output")
                    mxi = int(mx) if isinstance(mx, int) else None
                    combined.append((prov, mid, inp, outp, ctxi, mxi))
                except Exception:  # noqa: BLE001
                    continue

        # Sort by output price then input price
        def _key(row: tuple[str, str, Any, Any, Any, Any]) -> tuple[float, float, str]:
            inp = float(row[2]) if isinstance(row[2], (int, float)) else 1e12
            outp = float(row[3]) if isinstance(row[3], (int, float)) else 1e12
            return (outp, inp, row[1])

        combined.sort(key=_key)

        # Limit output to reasonable number
        view = combined[:60]

        # Compute dynamic widths
        w_prov = max((len(r[0]) for r in view), default=8)
        w_mid = max((len(r[1]) for r in view), default=10)
        w_in = max((len(self._fmt_money(r[2])) for r in view), default=4)
        w_out = max((len(self._fmt_money(r[3])) for r in view), default=4)
        w_ctx = max((len(f"{r[4]:,}") if r[4] else 1 for r in view), default=3)
        w_mx = max((len(str(r[5])) if r[5] else 1 for r in view), default=2)

        header = (
            f"  {'provider':<{w_prov}}  {'model':<{w_mid}}  "
            f"{'input/MTok':>{w_in}}  {'output/MTok':>{w_out}}  "
            f"{'ctx':>{w_ctx}}  {'max_out':>{w_mx}}"
        )
        print(CYAN + header + RESET)
        for r in view:
            prov, mid, inp, outp, ctx, mx = r
            ctxs = f"{ctx:,}" if isinstance(ctx, int) else "-"
            mxs = f"{mx}" if isinstance(mx, int) else "-"
            line = (
                f"  {prov:<{w_prov}}  {mid:<{w_mid}}  "
                f"{self._fmt_money(inp):>{w_in}}  {self._fmt_money(outp):>{w_out}}  "
                f"{ctxs:>{w_ctx}}  {mxs:>{w_mx}}"
            )
            print(line)

    def _execute_single_file(self, args: argparse.Namespace, config: Config) -> int:
        file_path = args.single_file
        repo = GitRepo(config.git_repo_path, config)

        repo.stage_file(file_path)
        diff = repo.get_file_diff(file_path, staged=True)
        if not diff.strip():
            self._print_warning(f"No staged changes for file: {file_path}")
            wdiff = repo.get_file_diff(file_path, staged=False)
            if not wdiff.strip():
                self._print_info("No changes detected in the specified file.")
                return 0
            diff = wdiff

        gen = CommitGenerator(repo_path=config.git_repo_path, config=config)
        msg = gen.suggest_commit_message(
            diff,
            context=f"File: {file_path}",
            style="conventional",
        )
        msg = gen.validate_and_fix_commit_message(msg)

        repo.commit(msg)
        recent = repo.get_recent_commits(1)
        commit_hash = recent[0].split()[0] if recent else None

        # Simple success message for oneshot/single file mode
        self._print_success(f"✓ {file_path}")
        self._print_info(f"  {msg}")
        if commit_hash:
            self._print_info(f"  {commit_hash[:8]}")

        return 0

    # ------------------------------------------------------------------
    # Run snapshot & formatting helpers
    # ------------------------------------------------------------------
    def _print_compact_header(self, config: Config, args: argparse.Namespace) -> None:
        parts = [
            f"{CYAN}provider{RESET} {config.provider}",
            f"{CYAN}model{RESET} {config.model}",
            f"{CYAN}retries{RESET} {args.max_retries}",
        ]
        limit_value = getattr(args, "limit", None)
        if limit_value:
            parts.append(f"{CYAN}limit{RESET} {limit_value}")
        print("  ".join(parts))
        print()

    def _safe_stats_snapshot(self, workflow: Any) -> dict[str, Any]:
        if workflow and hasattr(workflow, "stats_snapshot"):
            try:
                snapshot = workflow.stats_snapshot()
            except Exception:  # pragma: no cover - defensive
                return {}
            if isinstance(snapshot, dict):
                return snapshot
        return {}

    def _safe_commit_subjects(self, workflow: Any) -> list[str]:
        if workflow and hasattr(workflow, "commit_subjects"):
            try:
                subjects = workflow.commit_subjects()
            except Exception:  # pragma: no cover - defensive
                return []
            if isinstance(subjects, list):
                return list(subjects)
        return []

    def _result_to_dict(self, result: Any) -> dict[str, Any]:
        if is_dataclass(result) and not isinstance(result, type):
            return asdict(result)
        if isinstance(result, dict):
            return dict(result)
        payload: dict[str, Any] = {}
        for key in ("success", "commit_hash", "message", "error", "file_path"):
            if hasattr(result, key):
                payload[key] = getattr(result, key)
        return payload

    def _build_run_snapshot(
        self,
        args: argparse.Namespace,
        config: Config,
        results: dict[str, Any],
        duration: float,
        workflow: KlingonCMTWorkflow,
    ) -> dict[str, Any]:
        stats = self._safe_stats_snapshot(workflow)
        commit_subjects = self._safe_commit_subjects(workflow)

        file_commits = list(results.get("file_commits", []) or [])
        deletions = list(results.get("deletions_committed", []) or [])
        errors = [str(err) for err in (results.get("errors", []) or []) if err]

        commit_success = sum(
            1 for item in file_commits if getattr(item, "success", False)
        )
        commit_failure = len(file_commits) - commit_success
        deletion_success = sum(
            1 for item in deletions if getattr(item, "success", False)
        )
        deletion_failure = len(deletions) - deletion_success

        total_files = int(stats.get("total_files", len(file_commits)) or 0)
        prepared_total = int(stats.get("prepared", len(file_commits)) or 0)
        processed_total = int(stats.get("processed", len(file_commits)) or 0)
        prepared_failures = max(total_files - prepared_total, 0)

        safe_duration = max(float(duration or 0.0), 0.0)
        rate_value = stats.get("rate", 0.0)
        if isinstance(rate_value, (int, float)):
            rate = float(rate_value)
        else:
            rate = 0.0
        if rate <= 0.0 and safe_duration > 0.0:
            overall_success = commit_success + deletion_success
            if overall_success:
                rate = overall_success / safe_duration

        repo_display = str(self._repo_root) if self._repo_root else config.git_repo_path

        snapshot = {
            "schema_version": 1,
            "timestamp": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "repo_path": repo_display,
            "provider": config.provider,
            "model": config.model,
            "endpoint": config.llm_endpoint,
            "max_retries": args.max_retries,
            "file_limit": getattr(args, "limit", None),
            "compact": self._compact_mode,
            "duration_seconds": safe_duration,
            "rate_commits_per_sec": rate,
            "counts": {
                "files_total": total_files,
                "prepared_total": prepared_total,
                "processed_total": processed_total,
                "prepared_failures": prepared_failures,
                "commit_success": commit_success,
                "commit_failure": commit_failure,
                "deletions_total": len(deletions),
                "deletions_success": deletion_success,
                "deletions_failure": deletion_failure,
                "overall_success": commit_success + deletion_success,
                "overall_failure": commit_failure + deletion_failure,
                "errors": len(errors),
            },
            "pushed": results.get("pushed"),
            "summary": results.get("summary", ""),
            "errors": errors,
            "commits": [self._result_to_dict(entry) for entry in file_commits],
            "deletions": [self._result_to_dict(entry) for entry in deletions],
            "subjects": commit_subjects,
            "stats": stats,
        }
        snapshot["auto_push_state"] = self._describe_auto_push(snapshot["pushed"])
        return snapshot

    def _describe_auto_push(self, pushed: Any) -> str:
        if pushed is True:
            return "pushed"
        if pushed is False:
            return "not triggered"
        return ""

    def _persist_run_snapshot(self, snapshot: dict[str, Any]) -> None:
        if not self._repo_root:
            return
        try:
            history_dir = self._repo_root / ".kcmt"
            history_dir.mkdir(parents=True, exist_ok=True)
            path = history_dir / "last_run.json"
            with path.open("w", encoding="utf-8") as handle:
                json.dump(
                    snapshot,
                    handle,
                    indent=2,
                    ensure_ascii=False,
                    cls=DecimalFriendlyJSONEncoder,
                )
        except OSError:  # pragma: no cover - best effort persistence
            pass

    def _load_run_snapshot(self, repo_root: Path) -> Optional[dict[str, Any]]:
        path = repo_root / ".kcmt" / "last_run.json"
        try:
            with path.open("r", encoding="utf-8") as handle:
                loaded = json.load(handle)
                if isinstance(loaded, dict):
                    return cast(dict[str, Any], loaded)
                return None
        except FileNotFoundError:
            return None
        except (OSError, json.JSONDecodeError):  # pragma: no cover - corrupt snapshot
            return None

    def _fmt_column(self, value: Optional[int], color: str) -> str:
        raw = "-" if value is None else str(int(value))
        padded = f"{raw:>6}"
        return f"{color}{padded}{RESET}"

    def _fmt_rate(self, value: Optional[float]) -> str:
        if value is None or value <= 0.0:
            raw = "-/s"
        else:
            raw = f"{value:.2f}/s"
        return f"{MAGENTA}{raw:>8}{RESET}"

    def _format_summary_row(
        self,
        label: str,
        total: int,
        ready: Optional[int],
        success: Optional[int],
        failure: Optional[int],
        rate: Optional[float],
    ) -> str:
        return (
            f"{BOLD}{label:<10}{RESET} "
            f"{self._fmt_column(total, CYAN)} "
            f"{self._fmt_column(ready, CYAN)} "
            f"{self._fmt_column(success, GREEN)} "
            f"{self._fmt_column(failure, RED)} "
            f"{self._fmt_rate(rate)}"
        )

    def _build_summary_table(self, snapshot: dict[str, Any]) -> list[str]:
        counts = snapshot.get("counts", {})
        total = int(counts.get("files_total", 0) or 0)
        prepared = int(counts.get("prepared_total", 0) or 0)
        processed = int(counts.get("processed_total", 0) or 0)
        prepared_failures = int(
            counts.get("prepared_failures", max(total - prepared, 0))
        )
        commit_success = int(counts.get("commit_success", 0) or 0)
        commit_failure = int(counts.get("commit_failure", 0) or 0)
        deletions_total = int(counts.get("deletions_total", 0) or 0)
        deletion_success = int(counts.get("deletions_success", 0) or 0)
        deletion_failure = int(counts.get("deletions_failure", 0) or 0)
        rate = float(snapshot.get("rate_commits_per_sec", 0.0) or 0.0)

        lines = [
            f"{BOLD}{CYAN}{'Phase':<10} {'Total':>6} {'Ready':>6} {'✓':>6} {'✗':>6} {'Rate':>8}{RESET}",
            self._format_summary_row(
                "Prepare", total, prepared, prepared, prepared_failures, None
            ),
            self._format_summary_row(
                "Commit", processed, None, commit_success, commit_failure, rate
            ),
        ]
        if deletions_total or deletion_success or deletion_failure:
            lines.append(
                self._format_summary_row(
                    "Deletions",
                    deletions_total,
                    None,
                    deletion_success,
                    deletion_failure,
                    None,
                )
            )
        return lines

    def _format_overall_status(self, snapshot: dict[str, Any]) -> str:
        counts = snapshot.get("counts", {})
        success = int(counts.get("overall_success", 0) or 0)
        failure = int(counts.get("overall_failure", 0) or 0)
        return f"{GREEN}{success}✓{RESET} / {RED}{failure}✗{RESET}"

    def _render_snapshot_summary(
        self,
        snapshot: dict[str, Any],
        heading: str = "Run Summary",
        *,
        verbose: bool = False,
    ) -> None:
        if heading:
            print(f"{BOLD}{CYAN}{heading}{RESET}")
        for line in self._build_summary_table(snapshot):
            print(line)

        duration = float(snapshot.get("duration_seconds", 0.0) or 0.0)
        rate = float(snapshot.get("rate_commits_per_sec", 0.0) or 0.0)
        print(
            f"{CYAN}Duration{RESET} {duration:.2f}s  " f"{CYAN}Rate{RESET} {rate:.2f}/s"
        )

        checklist: list[tuple[str, str]] = [
            ("Provider", snapshot.get("provider", "-")),
            ("Model", snapshot.get("model", "-")),
            ("Retries", str(snapshot.get("max_retries", "-"))),
            ("Commit status", self._format_overall_status(snapshot)),
        ]
        auto_push_state = snapshot.get("auto_push_state")
        if auto_push_state:
            checklist.append(("Auto-push", auto_push_state))
        width = max(len(label) for label, _ in checklist)
        for label, value in checklist:
            print(f"{CYAN}{label:<{width}}{RESET} {value}")

        summary_line = snapshot.get("summary")
        if summary_line:
            print()
            self._print_info(summary_line)

        errors = snapshot.get("errors") or []
        if errors:
            print()
            self._print_warning("Errors:")
            for err in errors:
                self._print_error(f"  - {err}")

        if verbose and snapshot.get("commits"):
            print()
            self._print_heading("Commits")
            for entry in snapshot["commits"]:
                if not isinstance(entry, dict):
                    continue
                label = entry.get("file_path") or entry.get("message") or "(commit)"
                if entry.get("success"):
                    self._print_success(f"✓ {label}")
                else:
                    message = entry.get("error") or "failed"
                    self._print_warning(f"✗ {label}: {message}")

        subjects = snapshot.get("subjects") or []
        if subjects:
            print()
            self._print_success(f"Latest commit: {subjects[-1]}")

    def _display_compact_results(self, snapshot: dict[str, Any], verbose: bool) -> None:
        print()
        self._render_snapshot_summary(snapshot, verbose=verbose)

    def _execute_status(self, args: argparse.Namespace, repo_root: Path) -> int:
        snapshot = self._load_run_snapshot(repo_root)
        if not snapshot:
            self._print_warning("No kcmt run history found for this repository.")
            return 1
        if getattr(args, "raw", False):
            print(
                json.dumps(
                    snapshot,
                    indent=2,
                    ensure_ascii=False,
                    cls=DecimalFriendlyJSONEncoder,
                )
            )
            return 0

        verbose_flag = bool(getattr(args, "verbose", False))
        self._display_status_summary(snapshot, verbose_flag)
        return 0

    def _display_status_summary(self, snapshot: dict[str, Any], verbose: bool) -> None:
        repo_display = snapshot.get("repo_path") or (
            str(self._repo_root) if self._repo_root else "<unknown>"
        )
        header = f"{BOLD}{CYAN}kcmt status{RESET} :: {CYAN}{repo_display}{RESET}"
        print(header)
        timestamp = snapshot.get("timestamp")
        duration = float(snapshot.get("duration_seconds", 0.0) or 0.0)
        if timestamp:
            print(
                f"{CYAN}Run time{RESET} {timestamp}  "
                f"{CYAN}Duration{RESET} {duration:.2f}s"
            )
        else:
            print(f"{CYAN}Duration{RESET} {duration:.2f}s")
        print()
        self._render_snapshot_summary(snapshot, heading="Summary", verbose=verbose)

    # ------------------------------------------------------------------
    # Output helpers
    # ------------------------------------------------------------------
    def _print_banner(
        self, config: Config, args: Optional[argparse.Namespace] = None
    ) -> None:
        repo = Path(config.git_repo_path).resolve()
        banner = f"{BOLD}{CYAN}kcmt :: provider {config.provider} :: repo {repo}{RESET}"
        print(banner)

    def _print_heading(self, title: str) -> None:
        print(f"\n{BOLD}{CYAN}{title}{RESET}")

    def _print_info(self, message: str) -> None:
        print(f"{CYAN}{message}{RESET}")

    def _print_success(self, message: str) -> None:
        print(f"{GREEN}{message}{RESET}")

    def _print_warning(self, message: str) -> None:
        print(f"{YELLOW}{message}{RESET}")

    def _print_error(self, message: str) -> None:
        print(f"{RED}{message}{RESET}", file=sys.stderr)

    def _display_results(
        self,
        results: dict[str, Any],
        verbose: bool,
        snapshot: Optional[dict[str, Any]] = None,
    ) -> None:
        if self._compact_mode and snapshot is not None:
            self._display_compact_results(snapshot, verbose)
            return

        deletions = results.get("deletions_committed", [])
        file_commits = results.get("file_commits", [])
        errors = results.get("errors", [])
        pushed = results.get("pushed")

        self._print_heading("Workflow Summary")

        successful_deletions = [r for r in deletions if r.success]
        failed_deletions = [r for r in deletions if not r.success]
        successful_commits = [r for r in file_commits if r.success]
        failed_commits = [r for r in file_commits if not r.success]

        summary_rows: list[tuple[str, str, str]] = []
        if deletions:
            plain = (
                f"{len(successful_deletions)} success / {len(failed_deletions)} fail"
            )
            styled = (
                f"{GREEN}{len(successful_deletions):>3}{RESET} ✓  "
                f"{RED}{len(failed_deletions):>3}{RESET} ✗"
            )
            summary_rows.append(("Deletions", plain, styled))

        if file_commits:
            successful_commits = [r for r in file_commits if r.success]
            failed_commits = [r for r in file_commits if not r.success]
            if successful_commits:
                self._print_success(f"✓ Committed {len(successful_commits)} file(s)")
            if failed_commits:
                self._print_warning(f"✗ Failed to commit {len(failed_commits)} file(s)")
                for result in failed_commits:
                    if hasattr(result, "file_path") and result.file_path:
                        self._print_error(f"  {result.file_path}: {result.error}")
                    else:
                        self._print_error(f"  {result.error}")

        if errors:
            self._print_warning("Encountered errors:")
            for error in errors:
                self._print_error(f"  - {error}")

        if pushed is True:
            self._print_success("Auto-push: pushed")
        elif pushed is False:
            self._print_info("Auto-push: not triggered")

        summary_text = results.get("summary")
        if summary_text:
            self._print_info(summary_text)


def main(argv: Optional[list[str]] = None) -> int:
    """Entry point used by console scripts."""
    return CLI().run(argv)


if __name__ == "__main__":
    sys.exit(main())
