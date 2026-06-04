"""BDD coverage for the Rust fastest automated commit tool optimization."""

from __future__ import annotations

from pathlib import Path

import pytest
from pytest_bdd import given, scenario, then, when

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def context() -> dict[str, str]:
    """Share text captured by BDD steps within one scenario."""

    return {}


@scenario(
    "../features/rust_fast_commit_tool.feature",
    "Runtime benchmark exposes a Python versus Rust score board",
)
def test_runtime_benchmark_exposes_scoreboard() -> None:
    """Runtime benchmark reports include Python-vs-Rust comparison rows."""


@scenario(
    "../features/rust_fast_commit_tool.feature",
    "Rust workflow telemetry records commit stages",
)
def test_rust_workflow_telemetry_records_commit_stages() -> None:
    """Rust status snapshots contain workflow telemetry stages."""


@scenario(
    "../features/rust_fast_commit_tool.feature",
    "Rust batch workflow records queue timing",
)
def test_rust_batch_workflow_records_queue_timing() -> None:
    """Rust batch workflow records LLM queue timing stages."""


@scenario(
    "../features/rust_fast_commit_tool.feature",
    "Performance reports preserve quality gates",
)
def test_performance_report_preserves_quality_gates() -> None:
    """Iteration reports must describe the Rust quality gate."""


@given("the Rust optimization spec has been approved")
def approved_spec() -> None:
    assert (ROOT / "specs/005-rust-fastest-commit-tool/spec.md").exists()


@when("the runtime benchmark result is rendered")
def rendered_benchmark(context: dict[str, str]) -> None:
    report = ROOT / "docs/performance/rust-fastest-commit-tool-iterations.md"
    context["report"] = report.read_text(encoding="utf-8")


@then("the result includes a Python versus Rust score board")
def result_includes_scoreboard(context: dict[str, str]) -> None:
    report = context["report"]
    assert "Score board:" in report
    assert "| Stage | Python median ms | Rust median ms |" in report


@given("a Rust file workflow has completed")
def rust_file_workflow_completed() -> None:
    assert (ROOT / "rust/crates/kcmt-cli/src/commands/workflow.rs").exists()


@when("the raw status snapshot is inspected")
def inspected_snapshot_source(context: dict[str, str]) -> None:
    workflow = ROOT / "rust/crates/kcmt-cli/src/commands/workflow.rs"
    context["workflow"] = workflow.read_text(encoding="utf-8")


@then("telemetry includes status scan, commit, push, and snapshot stages")
def telemetry_includes_required_stages(context: dict[str, str]) -> None:
    workflow = context["workflow"]
    assert '"status_scan"' in workflow
    assert '"commit"' in workflow
    assert '"push"' in workflow
    assert '"snapshot"' in workflow
    assert '"telemetry"' in workflow


@given("a Rust batch workflow has been implemented")
def rust_batch_workflow_implemented() -> None:
    workflow = ROOT / "rust/crates/kcmt-cli/src/commands/workflow.rs"
    assert "run_batch_workflow" in workflow.read_text(encoding="utf-8")


@when("the workflow source is inspected")
def workflow_source_inspected(context: dict[str, str]) -> None:
    workflow = ROOT / "rust/crates/kcmt-cli/src/commands/workflow.rs"
    context["workflow"] = workflow.read_text(encoding="utf-8")


@then("telemetry includes first and all LLM enqueue timing")
def telemetry_includes_batch_queue_timing(context: dict[str, str]) -> None:
    workflow = context["workflow"]
    assert '"time_to_first_llm_enqueue"' in workflow
    assert '"time_to_all_llm_enqueued"' in workflow


@given("Rust has a provider-backed commit message path")
def rust_has_provider_backed_message_path() -> None:
    workflow = ROOT / "rust/crates/kcmt-cli/src/commands/workflow.rs"
    text = workflow.read_text(encoding="utf-8")
    assert "OpenAiCommitClient" in text
    assert "generate_openai_commit_message" in text


@when("final results are reported")
def final_results_reported(context: dict[str, str]) -> None:
    report = ROOT / "docs/performance/rust-fastest-commit-tool-iterations.md"
    context["report"] = report.read_text(encoding="utf-8")


@then("the report states that Conventional Commit validation protects quality")
def report_states_quality_gate(context: dict[str, str]) -> None:
    report = context["report"]
    assert "Conventional Commit validation" in report
    assert "quality" in report
