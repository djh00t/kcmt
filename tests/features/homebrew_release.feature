Feature: Homebrew release sync
  The release helper refreshes the tap formula from the latest release checksum.

  Scenario: Syncing the tap formula updates the version and checksum
    Given a placeholder kcmt-homebrew formula
    And a release checksum file for version 0.3.2
    When the Homebrew sync helper updates the formula for version 0.3.2
    Then the formula version is updated to 0.3.2
    And the formula checksum is updated from the release checksums

  Scenario: README install section documents Rust-only Homebrew install
    Given the README install section
    Then the install section includes the kcmt Homebrew tap
    And the install section installs kcmt from Homebrew
    And the install section states Homebrew installs the Rust CLI without the legacy Python package
