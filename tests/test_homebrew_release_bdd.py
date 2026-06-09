from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Any

from pytest_bdd import given, scenarios, then, when

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "sync_homebrew_formula.py"
)
SCRIPT_SPEC = spec_from_file_location("sync_homebrew_formula", SCRIPT_PATH)
assert SCRIPT_SPEC is not None and SCRIPT_SPEC.loader is not None
SCRIPT_MODULE = module_from_spec(SCRIPT_SPEC)
SCRIPT_SPEC.loader.exec_module(SCRIPT_MODULE)
sync_formula = SCRIPT_MODULE.sync_formula

scenarios("features/homebrew_release.feature")

RELEASE_VERSION = "0.3.2"
RELEASE_CHECKSUM = "1fa8338b016e839e3c1f1a02805e186986bba75c0dcb535b06137be0c71d205a"


@given("a placeholder kcmt-homebrew formula", target_fixture="homebrew_context")
def placeholder_homebrew_formula(tmp_path: Path) -> dict[str, Path]:
    tap_repo = tmp_path / "kcmt-homebrew"
    formula_dir = tap_repo / "Formula"
    formula_dir.mkdir(parents=True)
    (formula_dir / "kcmt.rb").write_text(
        """class Kcmt < Formula
  desc "Rust CLI for generating conventional commits"
  homepage "https://github.com/djh00t/kcmt"
  version "0.3.1"
  url "https://github.com/djh00t/kcmt/archive/refs/tags/v#{version}.tar.gz"
  sha256 "placeholder"
  license "MIT"
end
""",
        encoding="utf-8",
    )
    return {"tap_repo": tap_repo, "sums_file": tmp_path / "dist" / "SHA256SUMS"}


@given("a release checksum file for version 0.3.2")
def release_checksum_file(homebrew_context: dict[str, Path]) -> dict[str, Path]:
    sums_file = homebrew_context["sums_file"]
    sums_file.parent.mkdir(parents=True, exist_ok=True)
    sums_file.write_text(
        f"{RELEASE_CHECKSUM}  dist/kcmt-{RELEASE_VERSION}-source.tar.gz\n",
        encoding="utf-8",
    )
    return homebrew_context


@when("the Homebrew sync helper updates the formula for version 0.3.2")
def sync_homebrew_formula(homebrew_context: dict[str, Path]) -> dict[str, Any]:
    formula_path = sync_formula(
        homebrew_context["tap_repo"],
        RELEASE_VERSION,
        homebrew_context["sums_file"],
    )
    homebrew_context["formula_path"] = formula_path
    return homebrew_context


@then("the formula version is updated to 0.3.2")
def formula_version_is_updated(homebrew_context: dict[str, Any]) -> None:
    formula_text = Path(homebrew_context["formula_path"]).read_text(encoding="utf-8")
    assert f'version "{RELEASE_VERSION}"' in formula_text


@then("the formula checksum is updated from the release checksums")
def formula_checksum_is_updated(homebrew_context: dict[str, Any]) -> None:
    formula_text = Path(homebrew_context["formula_path"]).read_text(encoding="utf-8")
    assert f'sha256 "{RELEASE_CHECKSUM}"' in formula_text
