use std::path::PathBuf;

use kcmt_core::git::repo::{CliGitRepository, GitRepository};

pub fn render_status(repo_path: Option<PathBuf>) -> Result<String, String> {
    let repo = CliGitRepository::from_path(repo_path.unwrap_or_else(|| PathBuf::from(".")));
    repo.status_porcelain().map_err(|err| err.to_string())
}
