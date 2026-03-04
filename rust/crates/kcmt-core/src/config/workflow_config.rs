//! Workflow config parsing and validation.

use crate::error::{KcmtError, Result};
use crate::model::WorkflowConfig;

pub fn validate_config(config: &WorkflowConfig) -> Result<()> {
    if config.provider.trim().is_empty() {
        return Err(KcmtError::Message("provider is required".to_string()));
    }
    if config.model.trim().is_empty() {
        return Err(KcmtError::Message("model is required".to_string()));
    }
    if config.max_commit_length == 0 {
        return Err(KcmtError::Message(
            "max_commit_length must be greater than 0".to_string(),
        ));
    }
    Ok(())
}
