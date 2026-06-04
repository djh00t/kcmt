//! File-scoped commit execution helpers.

use std::collections::BTreeMap;
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
    pub stage_path_invoked: bool,
    pub create_commit_ms: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommitStaging {
    StagePath,
    DirectPath,
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
    commit_file_with_staging(
        repo_path,
        file_path,
        message,
        dry_run,
        CommitStaging::StagePath,
    )
}

pub fn commit_file_with_staging(
    repo_path: &Path,
    file_path: &str,
    message: &str,
    dry_run: bool,
    staging: CommitStaging,
) -> Result<CommitFileOutcome> {
    if dry_run {
        return Ok(CommitFileOutcome::default());
    }

    if env::var("KCMT_GIT_COMMIT_BACKEND").as_deref() == Ok("gix") {
        return commit_file_with_gix(repo_path, file_path, message);
    }

    let (stage_path_ms, stage_path_invoked) = match staging {
        CommitStaging::StagePath => {
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
            (stage_path_ms, true)
        }
        CommitStaging::DirectPath => (0.0, false),
    };

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
            stage_path_invoked,
            create_commit_ms,
        })
    } else {
        Err(KcmtError::Message(format!(
            "git commit failed for pathspec {file_path}"
        )))
    }
}

fn commit_file_with_gix(
    repo_path: &Path,
    file_path: &str,
    message: &str,
) -> Result<CommitFileOutcome> {
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
    let repo = gix::open(repo_path)
        .map_err(|err| KcmtError::Message(format!("gix open failed: {err}")))?;
    let index = repo
        .open_index()
        .map_err(|err| KcmtError::Message(format!("gix open index failed: {err}")))?;
    let tree_id = write_index_tree(&repo, &index)?;
    let parents = recent_commit_hash(repo_path)?
        .and_then(|hash| gix::hash::ObjectId::from_hex(hash.as_bytes()).ok())
        .into_iter()
        .collect::<Vec<_>>();
    let signature = gix_signature();
    let commit_id = {
        let mut time_buf = gix::actor::date::parse::TimeBuf::default();
        let signature_ref = signature.to_ref(&mut time_buf);
        repo.commit_as(
            signature_ref,
            signature_ref,
            "HEAD",
            message,
            tree_id,
            parents,
        )
        .map_err(|err| KcmtError::Message(format!("gix commit failed: {err}")))?
    };
    let create_commit_ms = commit_start.elapsed().as_secs_f64() * 1000.0;
    if commit_id.is_null() {
        return Err(KcmtError::Message(
            "gix commit produced a null commit id".to_string(),
        ));
    }
    Ok(CommitFileOutcome {
        stage_path_ms,
        stage_path_invoked: true,
        create_commit_ms,
    })
}

#[derive(Default)]
struct TreeNode {
    files: BTreeMap<Vec<u8>, (gix::objs::tree::EntryMode, gix::hash::ObjectId)>,
    dirs: BTreeMap<Vec<u8>, TreeNode>,
}

fn write_index_tree(
    repo: &gix::Repository,
    index: &gix::index::File,
) -> Result<gix::hash::ObjectId> {
    let mut root = TreeNode::default();
    for (path, (mode, oid)) in index.entries_with_paths_by_filter_map(|_path, entry| {
        if entry.stage() != gix::index::entry::Stage::Unconflicted {
            return None;
        }
        entry.mode.to_tree_entry_mode().map(|mode| (mode, entry.id))
    }) {
        insert_tree_entry(&mut root, path.as_ref(), mode, oid)?;
    }
    write_tree_node(repo, root)
}

fn insert_tree_entry(
    node: &mut TreeNode,
    path: &[u8],
    mode: gix::objs::tree::EntryMode,
    oid: gix::hash::ObjectId,
) -> Result<()> {
    let mut parts = path
        .split(|byte| *byte == b'/')
        .filter(|part| !part.is_empty());
    let Some(first) = parts.next() else {
        return Ok(());
    };
    let mut current = node;
    let mut name = first;
    for next in parts {
        current = current.dirs.entry(name.to_vec()).or_default();
        name = next;
    }
    current.files.insert(name.to_vec(), (mode, oid));
    Ok(())
}

fn write_tree_node(repo: &gix::Repository, node: TreeNode) -> Result<gix::hash::ObjectId> {
    let mut entries = Vec::with_capacity(node.files.len() + node.dirs.len());
    for (name, child) in node.dirs {
        let oid = write_tree_node(repo, child)?;
        entries.push(gix::objs::tree::Entry {
            mode: gix::objs::tree::EntryKind::Tree.into(),
            filename: name.into(),
            oid,
        });
    }
    for (name, (mode, oid)) in node.files {
        entries.push(gix::objs::tree::Entry {
            mode,
            filename: name.into(),
            oid,
        });
    }
    entries.sort();
    let tree = gix::objs::Tree { entries };
    Ok(repo
        .write_object(tree)
        .map_err(|err| KcmtError::Message(format!("gix write tree failed: {err}")))?
        .detach())
}

fn gix_signature() -> gix::actor::Signature {
    let env = commit_env();
    let name = env
        .get("GIT_AUTHOR_NAME")
        .cloned()
        .unwrap_or_else(|| "kcmt-bot".to_string());
    let email = env
        .get("GIT_AUTHOR_EMAIL")
        .cloned()
        .unwrap_or_else(|| "kcmt@example.com".to_string());
    gix::actor::Signature {
        name: name.into(),
        email: email.into(),
        time: gix::actor::date::Time::now_local_or_utc(),
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
    use super::{commit_file_with_gix, recent_commit_hash};
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

    fn git_output(repo: &Path, args: &[&str]) -> String {
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
        String::from_utf8_lossy(&output.stdout).trim().to_string()
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

    #[test]
    fn gix_commit_backend_commits_tracked_file_and_updates_index() {
        let repo = unique_temp_dir("gix-commit-tracked");
        git(&repo, &["init", "-q"]);
        fs::write(repo.join("tracked.py"), "print('seed')\n").expect("tracked file");
        git(&repo, &["add", "tracked.py"]);
        git(&repo, &["commit", "-m", "chore(repo): seed"]);
        fs::write(repo.join("tracked.py"), "print('changed')\n").expect("tracked change");

        let outcome = commit_file_with_gix(&repo, "tracked.py", "chore(repo): update tracked")
            .expect("gix commit");

        assert!(outcome.stage_path_invoked);
        assert!(outcome.create_commit_ms >= 0.0);
        assert_eq!(git_output(&repo, &["status", "--short"]), "");
        assert_eq!(
            git_output(&repo, &["log", "-1", "--pretty=%s"]),
            "chore(repo): update tracked"
        );
    }

    #[test]
    fn gix_commit_backend_commits_untracked_file() {
        let repo = unique_temp_dir("gix-commit-untracked");
        git(&repo, &["init", "-q"]);
        fs::write(repo.join("tracked.py"), "print('seed')\n").expect("tracked file");
        git(&repo, &["add", "tracked.py"]);
        git(&repo, &["commit", "-m", "chore(repo): seed"]);
        fs::write(repo.join("new_file.py"), "print('new')\n").expect("new file");

        commit_file_with_gix(&repo, "new_file.py", "chore(repo): add new file")
            .expect("gix commit");

        assert_eq!(git_output(&repo, &["status", "--short"]), "");
        assert_eq!(
            git_output(&repo, &["log", "-1", "--pretty=%s"]),
            "chore(repo): add new file"
        );
    }
}
