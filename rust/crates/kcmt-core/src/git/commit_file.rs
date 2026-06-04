//! File-scoped commit execution helpers.

use std::collections::HashMap;
use std::env;
use std::fs;
use std::path::Path;
use std::path::PathBuf;
use std::process::Command;
use std::time::Instant;

use crate::error::{KcmtError, Result};

#[derive(Debug, Clone, Copy, Default)]
pub struct CommitFileOutcome {
    pub stage_path_ms: f64,
    pub create_commit_ms: f64,
}

fn commit_env() -> HashMap<String, String> {
    let mut values: HashMap<String, String> = env::vars().collect();

    let author_name = values
        .get("GIT_AUTHOR_NAME")
        .cloned()
        .or_else(|| values.get("KCMT_GIT_AUTHOR_NAME").cloned())
        .unwrap_or_else(|| "kcmt-bot".to_string());
    let author_email = values
        .get("GIT_AUTHOR_EMAIL")
        .cloned()
        .or_else(|| values.get("KCMT_GIT_AUTHOR_EMAIL").cloned())
        .unwrap_or_else(|| "kcmt@example.com".to_string());

    values
        .entry("GIT_AUTHOR_NAME".to_string())
        .or_insert(author_name.clone());
    values
        .entry("GIT_COMMITTER_NAME".to_string())
        .or_insert(author_name);
    values
        .entry("GIT_AUTHOR_EMAIL".to_string())
        .or_insert(author_email.clone());
    values
        .entry("GIT_COMMITTER_EMAIL".to_string())
        .or_insert(author_email);
    values
}

pub fn commit_file(
    repo_path: &Path,
    file_path: &str,
    message: &str,
    dry_run: bool,
) -> Result<CommitFileOutcome> {
    if dry_run {
        return Ok(CommitFileOutcome::default());
    }

    let stage_start = Instant::now();
    let add_status = Command::new("git")
        .current_dir(repo_path)
        .args(["add", "-A", "--", file_path])
        .status()?;
    let stage_path_ms = stage_start.elapsed().as_secs_f64() * 1000.0;
    if !add_status.success() {
        return Err(KcmtError::Message(format!(
            "failed to stage path for commit: {file_path}"
        )));
    }

    let commit_start = Instant::now();
    let commit_status = Command::new("git")
        .current_dir(repo_path)
        .envs(commit_env())
        .args(["commit", "-m", message, "--", file_path])
        .status()?;
    let create_commit_ms = commit_start.elapsed().as_secs_f64() * 1000.0;

    if commit_status.success() {
        Ok(CommitFileOutcome {
            stage_path_ms,
            create_commit_ms,
        })
    } else {
        Err(KcmtError::Message(format!(
            "git commit failed for pathspec {file_path}"
        )))
    }
}

pub fn recent_commit_hash(repo_path: &Path) -> Result<Option<String>> {
    let Some(git_dir) = git_dir_path(repo_path)? else {
        return Ok(None);
    };
    let head_path = git_dir.join("HEAD");
    let head = match fs::read_to_string(&head_path) {
        Ok(head) => head,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(err) => return Err(err.into()),
    };
    let head = head.trim();
    if let Some(ref_name) = head.strip_prefix("ref:").map(str::trim) {
        return read_ref_hash(&git_dir, ref_name);
    }
    Ok(valid_hash(head).map(ToOwned::to_owned))
}

fn git_dir_path(repo_path: &Path) -> Result<Option<PathBuf>> {
    let dot_git = repo_path.join(".git");
    if dot_git.is_dir() {
        return Ok(Some(dot_git));
    }
    if dot_git.is_file() {
        let content = fs::read_to_string(&dot_git)?;
        let Some(path) = content.trim().strip_prefix("gitdir:").map(str::trim) else {
            return Ok(None);
        };
        let git_dir = PathBuf::from(path);
        return Ok(Some(if git_dir.is_absolute() {
            git_dir
        } else {
            repo_path.join(git_dir)
        }));
    }
    Ok(None)
}

fn read_ref_hash(git_dir: &Path, ref_name: &str) -> Result<Option<String>> {
    let loose_ref = git_dir.join(ref_name);
    match fs::read_to_string(&loose_ref) {
        Ok(hash) => return Ok(valid_hash(hash.trim()).map(ToOwned::to_owned)),
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => {}
        Err(err) => return Err(err.into()),
    }

    let packed_refs = git_dir.join("packed-refs");
    let packed = match fs::read_to_string(&packed_refs) {
        Ok(packed) => packed,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(err) => return Err(err.into()),
    };
    for line in packed.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') || line.starts_with('^') {
            continue;
        }
        let Some((hash, name)) = line.split_once(' ') else {
            continue;
        };
        if name == ref_name {
            return Ok(valid_hash(hash).map(ToOwned::to_owned));
        }
    }
    Ok(None)
}

fn valid_hash(value: &str) -> Option<&str> {
    let len = value.len();
    if (len == 40 || len == 64) && value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        Some(value)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::recent_commit_hash;
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
            .env("GIT_AUTHOR_NAME", "kcmt-bot")
            .env("GIT_AUTHOR_EMAIL", "kcmt@example.com")
            .env("GIT_COMMITTER_NAME", "kcmt-bot")
            .env("GIT_COMMITTER_EMAIL", "kcmt@example.com")
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
    fn reads_recent_commit_hash_without_git_log() {
        let repo = unique_temp_dir("recent-commit");
        git(&repo, &["init", "-q"]);
        fs::write(repo.join("tracked.py"), "print('seed')\n").expect("tracked file");
        git(&repo, &["add", "tracked.py"]);
        git(&repo, &["commit", "-m", "chore(repo): seed"]);

        let expected = Command::new("git")
            .current_dir(&repo)
            .args(["rev-parse", "HEAD"])
            .output()
            .expect("git rev-parse should run");
        assert!(expected.status.success());
        let expected = String::from_utf8_lossy(&expected.stdout).trim().to_string();

        let actual = recent_commit_hash(&repo).expect("hash should be read");

        assert_eq!(actual.as_deref(), Some(expected.as_str()));
    }
}
