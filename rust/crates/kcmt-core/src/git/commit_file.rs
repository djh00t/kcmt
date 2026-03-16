//! File-scoped commit execution helpers.

use std::collections::HashMap;
use std::env;
use std::path::Path;
use std::process::Command;

use crate::error::{KcmtError, Result};

fn commit_env() -> HashMap<String, String> {
    let mut values: HashMap<String, String> = env::vars().collect();

    let author_name = values
        .get("GIT_AUTHOR_NAME")
        .cloned()
        .or_else(|| values.get("KCMT_GIT_AUTHOR_NAME").cloned())
        .unwrap_or_else(|| "kcmt-bot".to_string());
    let author_email = values
        .get("GIT_AUTHOR_EMAIL")
        .cloned()
        .or_else(|| values.get("KCMT_GIT_AUTHOR_EMAIL").cloned())
        .unwrap_or_else(|| "kcmt@example.com".to_string());

    values.entry("GIT_AUTHOR_NAME".to_string()).or_insert(author_name.clone());
    values
        .entry("GIT_COMMITTER_NAME".to_string())
        .or_insert(author_name);
    values
        .entry("GIT_AUTHOR_EMAIL".to_string())
        .or_insert(author_email.clone());
    values
        .entry("GIT_COMMITTER_EMAIL".to_string())
        .or_insert(author_email);
    values
}

pub fn commit_file(
    repo_path: &Path,
    file_path: &str,
    message: &str,
    dry_run: bool,
) -> Result<()> {
    if dry_run {
        return Ok(());
    }

    let add_status = Command::new("git")
        .current_dir(repo_path)
        .args(["add", "-A", "--", file_path])
        .status()?;
    if !add_status.success() {
        return Err(KcmtError::Message(format!(
            "failed to stage path for commit: {file_path}"
        )));
    }

    let commit_status = Command::new("git")
        .current_dir(repo_path)
        .envs(commit_env())
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
