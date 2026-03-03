//! Recovery helpers for single-file commit flows.

use std::path::Path;
use std::process::Command;

use crate::error::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AtomicCommitOutcome {
    Completed,
    Recovered,
}

pub fn rollback_staged_file(repo_path: &Path, file_path: &str) -> Result<AtomicCommitOutcome> {
    let status = Command::new("git")
        .current_dir(repo_path)
        .args(["reset", "--", file_path])
        .status()?;

    if status.success() {
        Ok(AtomicCommitOutcome::Recovered)
    } else {
        Ok(AtomicCommitOutcome::Completed)
    }
}
