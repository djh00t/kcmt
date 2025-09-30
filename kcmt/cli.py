"""Command-line interface for kcmt."""

from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
import time
from dataclasses import asdict, is_dataclass
from datetime import datetime
from collections.abc import Callable, Iterator
from contextlib import contextmanager
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
            "--auto-push",
            action="store_true",
            help=(
                "Automatically git push after successful workflow "
                "(or set KLINGON_CMT_AUTO_PUSH=1)"
            ),
        )
        parser.add_argument(
            "--compact",
            "--summary",
            dest="compact",
            action="store_true",
            help=(
                "Use condensed output with a summary table and checklist"
            ),
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

            if parsed_args.list_models:
                return self._execute_list_models(parsed_args)
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
        return overrides

    def _run_configuration(self, args: argparse.Namespace, repo_root: Path) -> int:
        detected = detect_available_providers()
        provider = args.provider or self._prompt_provider(detected)

        provider_meta = DEFAULT_MODELS[provider]
        model = args.model or self._prompt_model(provider, provider_meta["model"])
        endpoint = args.endpoint or self._prompt_endpoint(
            provider, provider_meta["endpoint"]
        )
        api_key_env = args.api_key_env or self._prompt_api_key_env(provider, detected)

        config = Config(
            provider=provider,
            model=model,
            llm_endpoint=endpoint,
            api_key_env=api_key_env,
            git_repo_path=str(repo_root.expanduser().resolve(strict=False)),
        )
        save_config(config, repo_root)

        self._print_success(
            "Configuration saved to {}".format(
                (repo_root / ".kcmt" / "config.json").relative_to(repo_root)
            )
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
        for prov in ("openai", "anthropic", "xai"):
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
                if prov == "openai":
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
                    if prov == "openai":
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

        print(
            json.dumps(
                out,
                indent=2,
                ensure_ascii=False,
                cls=DecimalFriendlyJSONEncoder,
            )
        )
        return 0

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
    def _print_compact_header(
        self, config: Config, args: argparse.Namespace
    ) -> None:
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
        if is_dataclass(result):
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

        commit_success = sum(1 for item in file_commits if getattr(item, "success", False))
        commit_failure = len(file_commits) - commit_success
        deletion_success = sum(1 for item in deletions if getattr(item, "success", False))
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
            "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
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
                return json.load(handle)
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
        prepared_failures = int(counts.get("prepared_failures", max(total - prepared, 0)))
        commit_success = int(counts.get("commit_success", 0) or 0)
        commit_failure = int(counts.get("commit_failure", 0) or 0)
        deletions_total = int(counts.get("deletions_total", 0) or 0)
        deletion_success = int(counts.get("deletions_success", 0) or 0)
        deletion_failure = int(counts.get("deletions_failure", 0) or 0)
        rate = float(snapshot.get("rate_commits_per_sec", 0.0) or 0.0)

        lines = [
            f"{BOLD}{CYAN}{'Phase':<10} {'Total':>6} {'Ready':>6} {'✓':>6} {'✗':>6} {'Rate':>8}{RESET}",
            self._format_summary_row("Prepare", total, prepared, prepared, prepared_failures, None),
            self._format_summary_row("Commit", processed, None, commit_success, commit_failure, rate),
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
            f"{CYAN}Duration{RESET} {duration:.2f}s  "
            f"{CYAN}Rate{RESET} {rate:.2f}/s"
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
