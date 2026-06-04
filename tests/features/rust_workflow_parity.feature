Feature: Rust workflow parity
  The Rust CLI preserves provider-free workflow behavior while it replaces Python.

  Scenario: Oneshot commits multiple changed files separately
    Given a git repository with two changed tracked files
    When the Rust commit command runs in oneshot mode
    Then both changed files are committed separately
    And the repository worktree is clean

  Scenario: Oneshot commits a deleted tracked file
    Given a git repository with one deleted tracked file
    When the Rust kcmt command runs in oneshot mode
    Then the deleted file is committed with the deletion message
    And the deletion is recorded in the raw status snapshot

  Scenario: File workflow records CLI config overrides
    Given a git repository with one changed tracked file
    When the Rust kcmt command commits the file with config overrides
    Then the raw status snapshot records the config overrides

  Scenario: File workflow uses sanitized provider output
    Given a git repository with one changed tracked file
    When the Rust kcmt command commits the file with wrapped provider output
    Then the latest commit uses the sanitized provider message

  Scenario: File workflow rejects invalid provider output
    Given a git repository with one changed tracked file
    When the Rust kcmt command receives invalid provider output
    Then the workflow fails before committing the file

  Scenario: Oneshot auto-push records remote push success
    Given a git repository with one changed tracked file and a pushable origin
    When the Rust kcmt command runs in oneshot mode with auto-push
    Then the latest commit is pushed to origin
    And the raw status snapshot records auto-push success

  Scenario: Oneshot auto-push records push failure without reverting commits
    Given a git repository with one changed tracked file and a broken origin
    When the Rust kcmt command runs in oneshot mode with auto-push
    Then the latest commit remains in the local repository
    And the raw status snapshot records auto-push failure

  Scenario: Oneshot auto-push skips repositories without origin
    Given a git repository with one changed tracked file
    When the Rust kcmt command runs in oneshot mode with auto-push
    Then the latest commit remains in the local repository without auto-push failure
    And the raw status snapshot records auto-push skip

  Scenario: Oneshot OpenAI batch queues all file prompts before committing
    Given a git repository with two changed tracked files and a mocked OpenAI batch provider
    When the Rust kcmt command runs in oneshot batch mode
    Then the batch provider receives both file prompts before commits are written
    And both files are committed with the batch provider messages

  Scenario: File workflow falls back to the next configured provider
    Given a git repository with one changed tracked file and a fallback provider chain
    When the Rust kcmt command commits the file using configured provider fallback
    Then the fallback provider is used after the primary provider fails
    And the latest commit uses the fallback provider message

  Scenario: Python entrypoint dispatches covered workflows to Rust by default
    Given a git repository with one changed tracked file
    When the Python kcmt entrypoint commits the file in auto runtime mode
    Then the latest commit is written by the Rust runtime

  Scenario: Rust configure writes compatible config
    Given an empty runtime configuration home
    When the Rust kcmt command configures Anthropic non-interactively
    Then the Rust configuration file contains the Anthropic provider settings

  Scenario: Rust list-models shows supported default models
    Given an empty runtime configuration home
    When the Rust kcmt command lists models
    Then the model list includes all supported providers

  Scenario: Rust verify-keys reports provider environment status
    Given an empty runtime configuration home
    When the Rust kcmt command verifies API keys
    Then the key verification output shows present and missing providers

  Scenario: Rust provider benchmark reports JSON and CSV output
    Given an empty runtime configuration home
    When the Rust kcmt command benchmarks a provider with structured outputs
    Then the benchmark output includes leaderboard JSON and CSV sections
    And the provider benchmark snapshot is persisted
