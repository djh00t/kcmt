//! Backward-compatible persisted config loader.

use std::fs;
use std::path::Path;

use crate::error::{KcmtError, Result};
use crate::model::WorkflowConfig;

pub fn load_persisted_config(repo_root: &Path, path: &Path) -> Result<WorkflowConfig> {
    let content = fs::read_to_string(path)?;
    let config: WorkflowConfig =
        serde_json::from_str(&content).map_err(|err| KcmtError::Message(err.to_string()))?;

    let git_repo_path = if config.git_repo_path.trim().is_empty() {
        repo_root.to_path_buf()
    } else {
        let candidate = Path::new(&config.git_repo_path).to_path_buf();
        if candidate.is_absolute() {
            candidate
        } else {
            repo_root.join(candidate)
        }
    };

    Ok(WorkflowConfig {
        git_repo_path: git_repo_path.to_string_lossy().to_string(),
        ..config
    })
}
