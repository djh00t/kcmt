//! Git repository adapter built on top of the command runner abstraction.

use std::path::{Path, PathBuf};

use crate::error::Result;
use crate::git::runner::{CommandRunner, GitCliRunner};

pub trait GitRepository {
    fn staged_files(&self) -> Result<Vec<String>>;
    fn status_porcelain(&self) -> Result<String>;
}

pub struct CliGitRepository<R: CommandRunner> {
    repo_path: PathBuf,
    runner: R,
}

impl<R: CommandRunner> CliGitRepository<R> {
    pub fn new(repo_path: impl AsRef<Path>, runner: R) -> Self {
        Self {
            repo_path: repo_path.as_ref().to_path_buf(),
            runner,
        }
    }
}

impl CliGitRepository<GitCliRunner> {
    pub fn from_path(repo_path: impl AsRef<Path>) -> Self {
        Self::new(repo_path, GitCliRunner)
    }
}

impl<R: CommandRunner> GitRepository for CliGitRepository<R> {
    fn staged_files(&self) -> Result<Vec<String>> {
        let output = self
            .runner
            .run(&self.repo_path, &["diff", "--cached", "--name-only"])?;
        Ok(output
            .stdout
            .lines()
            .map(str::trim)
            .filter(|line| !line.is_empty())
            .map(ToOwned::to_owned)
            .collect())
    }

    fn status_porcelain(&self) -> Result<String> {
        let output = self.runner.run(&self.repo_path, &["status", "--porcelain"])?;
        Ok(output.stdout)
    }
}
