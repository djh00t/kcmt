//! Recovery helpers for single-file commit flows.

use std::path::Path;
use std::process::Command;

use crate::error::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AtomicCommitOutcome {
    Completed,
    Recovered,
}

pub fn rollback_staged_file(repo_path: &Path, file_path: &str) -> Result<AtomicCommitOutcome> {
    let status = Command::new("git")
        .current_dir(repo_path)
        .args(["reset", "--", file_path])
        .status()?;

    if status.success() {
        Ok(AtomicCommitOutcome::Recovered)
    } else {
        Ok(AtomicCommitOutcome::Completed)
    }
}

#[cfg(test)]
mod tests {
    use super::{rollback_staged_file, AtomicCommitOutcome};
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

    fn git(repo: &Path, args: &[&str]) -> String {
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
        String::from_utf8_lossy(&output.stdout).trim().to_string()
    }

    fn init_repo(repo: &Path) {
        git(repo, &["init", "-q"]);
        git(repo, &["config", "user.name", "kcmt test"]);
        git(repo, &["config", "user.email", "kcmt-test@example.com"]);
    }

    #[test]
    fn rollback_staged_file_returns_recovered_on_successful_git_reset() {
        let repo = unique_temp_dir("rollback-recovered");
        init_repo(&repo);
        fs::write(repo.join("tracked.py"), "print('seed')\n").expect("tracked file");
        git(&repo, &["add", "tracked.py"]);

        let outcome = rollback_staged_file(&repo, "tracked.py").expect("rollback should succeed");

        assert_eq!(outcome, AtomicCommitOutcome::Recovered);
        assert_eq!(git(&repo, &["status", "--short"]), "?? tracked.py");
    }

    #[test]
    fn rollback_staged_file_returns_completed_when_git_reset_fails() {
        let repo = unique_temp_dir("rollback-completed");
        git(&repo, &["init", "--bare", "-q"]);

        let outcome = rollback_staged_file(&repo, "missing.py").expect("rollback should return");

        assert_eq!(outcome, AtomicCommitOutcome::Completed);
    }
}
