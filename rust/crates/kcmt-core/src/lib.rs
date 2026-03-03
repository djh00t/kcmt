//! Shared core library for workflow orchestration and domain models.

pub mod config;
pub mod error;
pub mod git;
pub mod metrics;
pub mod model;
pub mod workflow;

/// Returns the workspace package name for smoke checks.
pub fn package_name() -> &'static str {
    "kcmt-core"
}
