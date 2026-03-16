//! Git repository adapter built on top of the command runner abstraction.

use std::path::{Path, PathBuf};
use std::process::Command;

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

pub fn find_git_repo_root(start_path: impl AsRef<Path>) -> Option<PathBuf> {
    let mut path = start_path.as_ref().to_path_buf();
    if path.is_file() {
        path = path.parent()?.to_path_buf();
    }

    let output = Command::new("git")
        .current_dir(&path)
        .args(["rev-parse", "--show-toplevel"])
        .output()
        .ok()?;
    if output.status.success() {
        let top = String::from_utf8_lossy(&output.stdout).trim().to_string();
        if !top.is_empty() {
            return Some(PathBuf::from(top));
        }
    }

    for candidate in std::iter::once(path.as_path()).chain(path.ancestors()) {
        if candidate.join(".git").exists() {
            return Some(candidate.to_path_buf());
        }
    }

    None
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
        let output = self.runner.run(
            &self.repo_path,
            &["status", "--porcelain=v1", "-z", "--untracked-files=all"],
        )?;
        Ok(render_porcelain_lines(&output.stdout))
    }
}

fn render_porcelain_lines(raw: &str) -> String {
    let mut rendered = String::new();
    let mut entries = raw.split('\0').filter(|entry| !entry.is_empty()).peekable();

    while let Some(entry) = entries.next() {
        if entry.len() < 3 {
            continue;
        }
        let status = &entry[0..2];
        let mut path = if entry.as_bytes().get(2) == Some(&b' ') {
            entry[3..].to_string()
        } else {
            entry[2..].to_string()
        };

        let is_rename_or_copy = status.contains('R') || status.contains('C');
        if is_rename_or_copy {
            if let Some(renamed_path) = entries.next() {
                path = renamed_path.to_string();
            }
        }

        rendered.push_str(status);
        rendered.push(' ');
        rendered.push_str(&path);
        rendered.push('\n');
    }

    rendered
}

#[cfg(test)]
mod tests {
    use super::{find_git_repo_root, render_porcelain_lines, CliGitRepository, GitRepository};
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
    fn finds_top_level_repo_from_nested_directory() {
        let repo = unique_temp_dir("repo");
        git(&repo, &["init", "-q"]);
        let nested = repo.join("pkg").join("sub");
        fs::create_dir_all(&nested).expect("nested dir");

        let found = find_git_repo_root(&nested).expect("repo root should be found");
        let found_canonical = fs::canonicalize(found).expect("found path canonical");
        let repo_canonical = fs::canonicalize(repo).expect("repo path canonical");
        assert_eq!(found_canonical, repo_canonical);
    }

    #[test]
    fn render_porcelain_lines_expands_nested_untracked_files() {
        let raw = "?? src/app.py\0?? src/lib/util.py\0";

        let rendered = render_porcelain_lines(raw);

        assert_eq!(rendered, "?? src/app.py\n?? src/lib/util.py\n");
    }

    #[test]
    fn status_porcelain_lists_nested_untracked_files() {
        let repo = unique_temp_dir("status-porcelain");
        git(&repo, &["init", "-q"]);
        let nested = repo.join("src").join("module_000");
        fs::create_dir_all(&nested).expect("nested dir");
        fs::write(nested.join("file_0000.py"), "print('alpha')\n").expect("alpha file");
        fs::write(nested.join("file_0001.py"), "print('beta')\n").expect("beta file");

        let status = CliGitRepository::from_path(&repo)
            .status_porcelain()
            .expect("status should render");

        assert!(status.contains("?? src/module_000/file_0000.py"));
        assert!(status.contains("?? src/module_000/file_0001.py"));
    }
}
