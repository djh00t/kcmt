//! File-scoped commit execution helpers.

use std::path::Path;
use std::process::Command;

use crate::error::{KcmtError, Result};

pub fn commit_file(repo_path: &Path, file_path: &str, message: &str, dry_run: bool) -> Result<()> {
    if dry_run {
        return Ok(());
    }

    let add_status = Command::new("git")
        .current_dir(repo_path)
        .args(["add", "--", file_path])
        .status()?;
    if !add_status.success() {
        return Err(KcmtError::Message(format!(
            "failed to stage path for commit: {file_path}"
        )));
    }

    let commit_status = Command::new("git")
        .current_dir(repo_path)
        .args(["commit", "-m", message, "--", file_path])
        .status()?;

    if commit_status.success() {
        Ok(())
    } else {
        Err(KcmtError::Message(format!(
            "git commit failed for pathspec {file_path}"
        )))
    }
}
