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

#[cfg(test)]
mod tests {
    use super::{CommandRunner, GitCliRunner};
    use crate::error::KcmtError;
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::process::Command;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};

    fn unique_temp_dir(label: &str) -> PathBuf {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock should be after unix epoch")
            .as_nanos();
        let suffix = COUNTER.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!("kcmt-core-{label}-{nanos}-{suffix}"));
        fs::create_dir_all(&path).expect("temp dir should be created");
        path
    }

    fn git(repo: &Path, args: &[&str]) {
        let output = Command::new("git")
            .current_dir(repo)
            .args(args)
            .output()
            .expect("git command should run");
        assert!(
            output.status.success(),
            "git {:?} failed: {}",
            args,
            String::from_utf8_lossy(&output.stderr)
        );
    }

    #[test]
    fn git_cli_runner_returns_output_on_success() {
        let repo = unique_temp_dir("runner-success");
        git(&repo, &["init", "-q"]);

        let output = GitCliRunner
            .run(&repo, &["rev-parse", "--show-toplevel"])
            .expect("git rev-parse should succeed");

        assert!(output.stderr.is_empty());
        assert!(output.status.is_some());
        assert_eq!(
            fs::canonicalize(output.stdout.trim()).expect("stdout path canonical"),
            fs::canonicalize(&repo).expect("repo path canonical")
        );
    }

    #[test]
    fn git_cli_runner_wraps_nonzero_exit_as_command_failure() {
        let repo = unique_temp_dir("runner-failure");
        git(&repo, &["init", "-q"]);

        let err = GitCliRunner
            .run(&repo, &["definitely-not-a-command"])
            .expect_err("invalid git command should fail");

        match err {
            KcmtError::CommandFailure {
                command,
                status,
                stderr,
            } => {
                assert_eq!(command, "git definitely-not-a-command");
                assert_ne!(status, Some(0));
                assert!(!stderr.trim().is_empty());
            }
            other => panic!("expected command failure, got {other:?}"),
        }
    }
}
