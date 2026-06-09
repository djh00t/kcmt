from __future__ import annotations

from pathlib import Path

from pytest_bdd import given, scenarios, then, when

WORKFLOW_PATH = (
    Path(__file__).resolve().parents[1] / ".github/workflows/dependency-advisor.yml"
)

scenarios("features/dependency_advisor_workflow.feature")


@given("the dependency advisor workflow file", target_fixture="workflow_text")
def dependency_advisor_workflow_file() -> str:
    return WORKFLOW_PATH.read_text(encoding="utf-8")


@when("I inspect the workflow configuration", target_fixture="workflow_text")
def inspect_workflow(workflow_text: str) -> str:
    return workflow_text


@then("it uses the published dependency advisor action")
def uses_published_dependency_advisor_action(workflow_text: str) -> None:
    assert "uses: djh00t/dependency-advisor-action@v0.0.2-action-test" in workflow_text


@then("it scans the repository root")
def scans_repository_root(workflow_text: str) -> None:
    assert "path: ${{ github.workspace }}" in workflow_text
    assert 'fail-on-findings: "false"' in workflow_text


@then("it uploads dependency advisor reports")
def uploads_dependency_advisor_reports(workflow_text: str) -> None:
    assert "Upload SARIF to code scanning" in workflow_text
    assert "Upload dependency advisor reports" in workflow_text
