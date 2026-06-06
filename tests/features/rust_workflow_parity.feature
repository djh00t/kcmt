Feature: Rust workflow parity
  The Rust CLI preserves provider-free workflow behavior while it replaces Python.

  Scenario: Oneshot commits one automatically selected file
    Given a git repository with two changed tracked files
    When the Rust commit command runs in oneshot mode
    Then exactly one changed file is committed
    And one changed file remains uncommitted

  Scenario: Default non-interactive workflow commits multiple files
    Given a git repository with two changed tracked files
    When the Rust kcmt command runs in default workflow mode
    Then both changed files are committed separately
    And the repository worktree is clean

  Scenario: Default workflow skips ignored files and expands untracked directories
    Given a git repository with ignored files and an untracked directory
    When the Rust kcmt command runs in default workflow mode
    Then the untracked directory files are committed separately
    And ignored files are not committed
    And the repository worktree is clean

  Scenario: Worker override is recorded in runtime telemetry
    Given a git repository with two changed tracked files
    When the Rust kcmt command runs in default workflow mode with two workers
    Then both changed files are committed separately
    And the raw status snapshot records two prepare workers

  Scenario: Compact verbose profile flags control workflow output
    Given a git repository with one changed tracked file
    When the Rust kcmt command runs in compact verbose profile mode
    Then the compact workflow output includes summary and commit details
    And the compact workflow output includes profile timings

  Scenario: Oneshot limit commits only the requested number of files
    Given a git repository with two changed tracked files
    When the Rust commit command runs in oneshot mode with a file limit
    Then exactly one changed file is committed
    And one changed file remains uncommitted

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

  Scenario: File workflow summarizes binary diffs for provider prompts
    Given a git repository with one changed binary file and a mocked OpenAI provider
    When the Rust kcmt command commits the binary file with the mocked provider
    Then the mocked provider prompt includes the binary diff summary
    And the latest commit uses the binary provider message

  Scenario: File workflow retries malformed provider output with a simplified prompt
    Given a git repository with one changed tracked file and a malformed-then-valid OpenAI provider
    When the Rust kcmt command commits the file with malformed then valid provider output
    Then the mocked provider receives a simplified retry prompt
    And the latest commit uses the simplified retry provider message

  Scenario: File workflow rejects invalid provider output
    Given a git repository with one changed tracked file
    When the Rust kcmt command receives invalid provider output
    Then the workflow fails before committing the file

  Scenario: Default workflow records per-file failure without blocking other commits
    Given a git repository with one deleted tracked file and one changed tracked file
    When the Rust kcmt command runs default mode with invalid provider output
    Then the deleted file is committed with the deletion message
    And the changed file remains uncommitted with a recorded prepare failure

  Scenario: Provider response fixture requires explicit opt in
    Given a git repository with one changed tracked file
    When the Rust kcmt command receives fixture provider output without opt in
    Then the fixture provider output is ignored before committing the file

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

  Scenario: Default OpenAI batch queues all file prompts before committing
    Given a git repository with two changed tracked files and a mocked OpenAI batch provider
    When the Rust kcmt command runs in default batch mode
    Then the batch provider receives both file prompts before commits are written
    And both files are committed with the batch provider messages

  Scenario: Default OpenAI batch reports partial invalid provider output
    Given a git repository with two changed tracked files and a partially invalid OpenAI batch provider
    When the Rust kcmt command runs in default batch mode
    Then the batch provider receives both file prompts before commits are written
    And only the valid batch file is committed
    And provider output and status remain secret-free

  Scenario: Default xAI batch queues all file prompts before committing
    Given a git repository with two changed tracked files and a mocked xAI batch provider
    When the Rust kcmt command runs in default xAI batch mode
    Then the xAI batch provider receives both file prompts before commits are written
    And both files are committed with the batch provider messages

  Scenario: File workflow falls back to the next configured provider
    Given a git repository with one changed tracked file and a fallback provider chain
    When the Rust kcmt command commits the file using configured provider fallback
    Then the fallback provider is used after the primary provider fails
    And the latest commit uses the fallback provider message

  Scenario: GitHub token flag is used for GitHub Models
    Given a git repository with one changed tracked file and a mocked GitHub Models provider
    When the Rust kcmt command commits the file using a GitHub token flag
    Then the GitHub provider receives the CLI token
    And the latest commit uses the GitHub provider message

  Scenario: Max retries zero disables provider retry attempts
    Given a git repository with one changed tracked file and a transiently failing provider
    When the Rust kcmt command commits the file with max retries set to zero
    Then the provider receives exactly one retry-limited request
    And no commit is written after the retry-limited provider failure

  Scenario: Python entrypoint dispatches covered workflows to Rust by default
    Given a git repository with one changed tracked file
    When the Python kcmt entrypoint commits the file in auto runtime mode
    Then the latest commit is written by the Rust runtime

  Scenario: Python entrypoint dispatches default non-interactive workflow to Rust
    Given a git repository with one changed tracked file
    When the Python kcmt entrypoint runs the default workflow in auto runtime mode
    Then the latest commit is written by the Rust runtime

  Scenario: Rust configure writes compatible config
    Given an empty runtime configuration home
    When the Rust kcmt command configures Anthropic non-interactively
    Then the Rust configuration file contains the Anthropic provider settings
    And the Rust preferences file contains default selector preferences

  Scenario: Anthropic latest Haiku rule selects the Haiku model
    Given a git repository with one changed tracked file and Anthropic latest Haiku preferences
    When the Rust kcmt command commits the file with Anthropic preferences
    Then the Anthropic provider receives the latest Haiku model
    And the latest commit uses the Anthropic provider message

  Scenario: Rust stats reports aggregate usage telemetry
    Given a git repository with one changed tracked file
    When the Rust kcmt command runs in oneshot mode
    Then the Rust stats command reports usage telemetry

  Scenario: Rust list-models shows supported default models
    Given an empty runtime configuration home
    When the Rust kcmt command lists models
    Then the model list includes all supported providers

  Scenario: Rust list-models debug emits structured provider data
    Given an empty runtime configuration home
    When the Rust kcmt command lists models in debug mode
    Then the debug model list is structured JSON

  Scenario: Rust verify-keys reports provider environment status
    Given an empty runtime configuration home
    When the Rust kcmt command verifies API keys
    Then the key verification output shows present and missing providers

  Scenario: Rust provider benchmark reports JSON and CSV output
    Given an empty runtime configuration home
    When the Rust kcmt command benchmarks a provider with structured outputs
    Then the benchmark output includes leaderboard JSON and CSV sections
    And the provider benchmark snapshot is persisted

  Scenario: Runtime benchmark reports Rust snapshot stage telemetry
    Given a checked-in runtime benchmark corpus
    When the Rust kcmt runtime benchmark runs against the corpus
    Then the benchmark report includes Rust workflow stage timings
