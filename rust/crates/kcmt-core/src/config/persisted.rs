//! Backward-compatible persisted config loader.

use std::fs;
use std::path::Path;

use crate::error::{KcmtError, Result};
use crate::model::WorkflowConfig;

pub fn load_persisted_config(path: &Path) -> Result<WorkflowConfig> {
    let content = fs::read_to_string(path)?;
    let config: WorkflowConfig =
        serde_json::from_str(&content).map_err(|err| KcmtError::Message(err.to_string()))?;
    Ok(config)
}
