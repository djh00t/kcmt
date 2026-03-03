//! Git command runner abstraction with a default subprocess implementation.

use std::path::Path;
use std::process::Command;

use crate::error::{status_code, KcmtError, Result};

#[derive(Debug, Clone)]
pub struct CommandOutput {
    pub stdout: String,
    pub stderr: String,
    pub status: Option<i32>,
}

pub trait CommandRunner {
    fn run(&self, repo_path: &Path, args: &[&str]) -> Result<CommandOutput>;
}

#[derive(Debug, Default, Clone, Copy)]
pub struct GitCliRunner;

impl CommandRunner for GitCliRunner {
    fn run(&self, repo_path: &Path, args: &[&str]) -> Result<CommandOutput> {
        let output = Command::new("git")
            .current_dir(repo_path)
            .args(args)
            .output()?;

        let status = status_code(output.status);
        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();

        if output.status.success() {
            Ok(CommandOutput {
                stdout,
                stderr,
                status,
            })
        } else {
            Err(KcmtError::CommandFailure {
                command: format!("git {}", args.join(" ")),
                status,
                stderr,
            })
        }
    }
}
