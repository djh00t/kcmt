Feature: Dependency Advisor workflow
  The repo installs the Dependency Advisor scan as a CI signal.

  Scenario: The workflow uses the published dependency advisor action
    Given the dependency advisor workflow file
    When I inspect the workflow configuration
    Then it uses the published dependency advisor action
    And it scans the repository root
    And it uploads dependency advisor reports
