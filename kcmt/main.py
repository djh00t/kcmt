"""Main entry point for kcmt."""

import sys

from .cli import main as cli_main


def main() -> int:
    """Main entry point that delegates to CLI and returns its exit code."""
    return cli_main()


if __name__ == "__main__":
    sys.exit(main())
