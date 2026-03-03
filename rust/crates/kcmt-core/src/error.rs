//! Shared error and result types used across core workflows.

use std::process::ExitStatus;

#[derive(Debug, thiserror::Error)]
pub enum KcmtError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    #[error("command failed: {command} (status: {status:?}) stderr: {stderr}")]
    CommandFailure {
        command: String,
        status: Option<i32>,
        stderr: String,
    },

    #[error("{0}")]
    Message(String),
}

pub type Result<T> = std::result::Result<T, KcmtError>;

pub fn status_code(status: ExitStatus) -> Option<i32> {
    status.code()
}
