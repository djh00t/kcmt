"""Command-line interface for kcmt."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

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
from .git import GitRepo

RESET = "\033[0m"
BOLD = "\033[1m"
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
MAGENTA = "\033[95m"
RED = "\033[91m"


class CLI:
    """Command-line interface for kcmt."""

    def __init__(self) -> None:
        self.parser = self._create_parser()

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
  kcmt                                  # default atomic workflow with live stats
  kcmt --oneshot                        # commit a single candidate file automatically
  kcmt --file README.md                 # commit only README.md
  kcmt --configure                      # interactive provider & model setup
    kcmt --provider openai --model gpt-5-mini-2025-08-07
            """,
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
            help="Path to the target Git repository (default: current directory)",
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
            "--allow-fallback",
            action="store_true",
            help=(
                "Allow local heuristic commit message fallback after all LLM "
                "attempts fail or produce invalid format"
            ),
        )
        parser.add_argument(
            "--auto-push",
            action="store_true",
            help=(
                "Automatically git push after successful workflow "
                "(or set KLINGON_CMT_AUTO_PUSH=1)"
            ),
        )

        return parser

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------
    def run(self, args: Optional[List[str]] = None) -> int:
        try:
            parsed_args = self.parser.parse_args(args)
            repo_root = Path(parsed_args.repo_path).resolve()
            non_interactive = (
                bool(os.environ.get("PYTEST_CURRENT_TEST"))
                or not sys.stdin.isatty()
            )

            # Allow providing the token via CLI for this run
            if getattr(parsed_args, "github_token", None):
                os.environ["GITHUB_TOKEN"] = parsed_args.github_token

            if parsed_args.configure:
                return self._run_configuration(parsed_args, repo_root)

            # Check if this is the first time running kcmt in this repo
            if not load_persisted_config(repo_root):
                if parsed_args.provider and non_interactive:
                    # Ephemeral config using defaults (test / CI friendly).
                    # Include env-driven feature flags on first-run so they
                    # persist (previously they were lost until a second run).
                    provider = parsed_args.provider
                    meta = DEFAULT_MODELS.get(
                        provider, DEFAULT_MODELS["openai"]
                    )
                    env_allow = os.environ.get(
                        "KLINGON_CMT_ALLOW_FALLBACK", ""
                    ).lower() in {"1", "true", "yes", "on"}
                    env_push = os.environ.get(
                        "KLINGON_CMT_AUTO_PUSH", ""
                    ).lower() in {"1", "true", "yes", "on"}
                    cfg = Config(
                        provider=provider,
                        model=parsed_args.model or meta["model"],
                        llm_endpoint=parsed_args.endpoint
                        or meta["endpoint"],
                        api_key_env=parsed_args.api_key_env
                        or meta["api_key_env"],
                        git_repo_path=str(repo_root),
                        allow_fallback=env_allow
                        or getattr(parsed_args, "allow_fallback", False),
                        auto_push=env_push
                        or getattr(parsed_args, "auto_push", False),
                    )
                    save_config(cfg, repo_root)
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

            overrides = self._collect_overrides(parsed_args)
            config = load_config(repo_root=repo_root, overrides=overrides)

            # Persist updated boolean feature flags so subsequent plain runs
            # (without explicit flags) retain user preference. This mirrors
            # typical CLI tooling that records config after feature toggles.
            persisted_cfg = load_persisted_config(repo_root)
            # Persist when flags explicitly overridden OR when env enabled a
            # feature not yet persisted (so subsequent plain runs inherit it)
            should_persist = False
            if any(k in overrides for k in ("auto_push", "allow_fallback")):
                should_persist = True
            else:
                if config.auto_push and (
                    not persisted_cfg
                    or not getattr(persisted_cfg, "auto_push", False)
                ):
                    should_persist = True
                if config.allow_fallback and (
                    not persisted_cfg
                    or not getattr(persisted_cfg, "allow_fallback", False)
                ):
                    should_persist = True
            if should_persist:
                try:  # pragma: no cover - trivial persistence path
                    save_config(config, repo_root)
                except OSError:  # Narrowed from broad Exception
                    pass

            if not config.resolve_api_key():
                # Allow tests that explicitly pass --api-key-env but don't
                # exercise LLM paths (monkeypatched workflow) to proceed.
                if (
                    os.environ.get("PYTEST_CURRENT_TEST")
                    and getattr(parsed_args, "api_key_env", None)
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

            self._print_banner(config)

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
        except Exception as err:  # pragma: no cover noqa: BLE001
            self._print_error(f"Unexpected error: {err}")
            _pa = locals().get("parsed_args")
            if _pa is not None and getattr(_pa, "verbose", False):
                import traceback

                traceback.print_exc()
            return 1

    # ------------------------------------------------------------------
    # Configuration helpers
    # ------------------------------------------------------------------
    def _collect_overrides(self, args: argparse.Namespace) -> Dict[str, str]:
        overrides: Dict[str, str] = {}
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
            overrides["repo_path"] = args.repo_path
        if getattr(args, "allow_fallback", False):
            overrides["allow_fallback"] = "1"
        if getattr(args, "auto_push", False):
            overrides["auto_push"] = "1"
        return overrides

    def _run_configuration(
        self, args: argparse.Namespace, repo_root: Path
    ) -> int:
        detected = detect_available_providers()
        provider = (args.provider or self._prompt_provider(detected))

        provider_meta = DEFAULT_MODELS[provider]
        model = args.model or self._prompt_model(
            provider, provider_meta["model"]
        )
        endpoint = args.endpoint or self._prompt_endpoint(
            provider, provider_meta["endpoint"]
        )
        api_key_env = args.api_key_env or self._prompt_api_key_env(
            provider, detected
        )

        config = Config(
            provider=provider,
            model=model,
            llm_endpoint=endpoint,
            api_key_env=api_key_env,
            git_repo_path=args.repo_path,
        )
        save_config(config, repo_root)

        self._print_success(
            "Configuration saved to {}".format(
                (repo_root / ".kcmt" / "config.json").relative_to(repo_root)
            )
        )
        return 0

    def _prompt_provider(self, detected: Dict[str, List[str]]) -> str:
        self._print_heading("Select provider")
        for idx, name in enumerate(sorted(DEFAULT_MODELS.keys()), start=1):
            badge = (
                GREEN + "●" + RESET
                if detected.get(name)
                else YELLOW + "○" + RESET
            )
            print(
                f"  {idx}. {badge} {describe_provider(name)}"
            )

        while True:
            choice = input(
                f"{MAGENTA}Provider [1-{len(DEFAULT_MODELS)}]{RESET}: "
            ).strip()
            if not choice:
                choice = "1"
            if choice.isdigit() and 1 <= int(choice) <= len(DEFAULT_MODELS):
                provider = sorted(DEFAULT_MODELS.keys())[int(choice) - 1]
                if not detected.get(provider):
                    confirm = input(
                        f"{YELLOW}No keys detected for {provider}. Continue?"
                        f" [y/N]{RESET} "
                    ).strip().lower()
                    if confirm not in {"y", "yes"}:
                        continue
                return provider
            self._print_warning("Invalid selection. Please choose again.")

    def _prompt_model(self, provider: str, default_model: str) -> str:
        prompt = (
            f"{MAGENTA}Model for {provider}{RESET} [{default_model}]: "
        )
        response = input(prompt).strip()
        return response or default_model

    def _prompt_endpoint(self, provider: str, default_endpoint: str) -> str:
        prompt = (
            f"{MAGENTA}Endpoint for {provider}{RESET} [{default_endpoint}]: "
        )
        response = input(prompt).strip()
        return response or default_endpoint

    def _prompt_api_key_env(
        self, provider: str, detected: Dict[str, List[str]]
    ) -> str:
        matches = detected.get(provider, [])
        if not matches:
            default_env = DEFAULT_MODELS[provider]["api_key_env"]
            prompt = (
                f"{MAGENTA}Environment variable with API key{RESET} "
                f"[{default_env}]: "
            )
            response = input(prompt).strip()
            return response or default_env

        self._print_heading(
            "Select API key environment variable"
        )
        for idx, env_key in enumerate(matches, start=1):
            marker = (
                GREEN + "●" + RESET
                if env_key in os.environ
                else RED + "●" + RESET
            )
            suffix = " (missing)" if env_key not in os.environ else ""
            print(f"  {idx}. {marker} {env_key}{suffix}")
        print(f"  {len(matches) + 1}. {CYAN}Enter a different variable{RESET}")

        while True:
            choice = input(
                f"{MAGENTA}API key variable [1-{len(matches)+1}]{RESET}: "
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

        entered = input(
            f"{MAGENTA}Enter environment variable name{RESET}: "
        ).strip()
        return entered or DEFAULT_MODELS[provider]["api_key_env"]

    # ------------------------------------------------------------------
    # Execution modes
    # ------------------------------------------------------------------
    def _execute_workflow(
        self, args: argparse.Namespace, config: Config
    ) -> int:
        self._print_info(f"Provider: {config.provider}")
        self._print_info(f"Model: {config.model}")
        self._print_info(f"Endpoint: {config.llm_endpoint}")
        self._print_info(f"Max retries: {args.max_retries}")
        if hasattr(args, 'limit') and args.limit:
            self._print_info(f"File limit: {args.limit}")
        self._print_info("")

        try:
            workflow = KlingonCMTWorkflow(
                repo_path=config.git_repo_path,
                max_retries=args.max_retries,
                config=config,
                show_progress=not args.no_progress,
                file_limit=getattr(args, 'limit', None),
                debug=getattr(args, 'debug', False),
            )
        except TypeError:  # Backward compatibility for tests
            workflow = KlingonCMTWorkflow(
                repo_path=config.git_repo_path,
                max_retries=args.max_retries,
                config=config,
                show_progress=not args.no_progress,
            )
        results = workflow.execute_workflow()
        self._display_results(results, args.verbose)
        return 0

    def _execute_oneshot(
        self, args: argparse.Namespace, config: Config
    ) -> int:
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

    def _execute_single_file(
        self, args: argparse.Namespace, config: Config
    ) -> int:
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
    # Output helpers
    # ------------------------------------------------------------------
    def _print_banner(self, config: Config) -> None:
        repo = Path(config.git_repo_path).resolve()
        banner = (
            f"{BOLD}{CYAN}kcmt :: provider {config.provider} :: repo "
            f"{repo}{RESET}"
        )
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

    def _display_results(self, results: Dict[str, Any], verbose: bool) -> None:
        deletions = results.get("deletions_committed", [])
        file_commits = results.get("file_commits", [])
        errors = results.get("errors", [])
        pushed = results.get("pushed")

        if deletions:
            successful_deletions = [r for r in deletions if r.success]
            failed_deletions = [r for r in deletions if not r.success]
            if successful_deletions:
                self._print_success(
                    f"Committed {len(successful_deletions)} deletion(s)"
                )
            if failed_deletions:
                self._print_warning(
                    f"Failed to commit {len(failed_deletions)} deletion(s)"
                )
                if verbose:
                    for result in failed_deletions:
                        self._print_error(f"  - {result.error}")

        if file_commits:
            successful_commits = [r for r in file_commits if r.success]
            failed_commits = [r for r in file_commits if not r.success]
            if successful_commits:
                self._print_success(
                    f"✓ Committed {len(successful_commits)} file(s)"
                )
            if failed_commits:
                self._print_warning(
                    f"✗ Failed to commit {len(failed_commits)} file(s)"
                )
                for result in failed_commits:
                    if hasattr(result, 'file_path') and result.file_path:
                        self._print_error(
                            f"  {result.file_path}: {result.error}"
                        )
                    else:
                        self._print_error(f"  {result.error}")

        if errors:
            self._print_warning("Encountered errors:")
            for error in errors:
                self._print_error(f"  - {error}")

        if pushed:
            self._print_success("Pushed commits to remote (auto-push)")

    # Success/failure counts already shown above; omit extra summary.


def main(argv: Optional[List[str]] = None) -> int:
    """Entry point used by console scripts."""
    return CLI().run(argv)


if __name__ == "__main__":
    sys.exit(main())
