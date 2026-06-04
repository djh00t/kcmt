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
use gix::bstr::ByteSlice;
use gix::objs::Write;

#[derive(Debug, Clone, Default)]
pub struct CommitFileOutcome {
    pub stage_path_ms: f64,
    pub stage_path_invoked: bool,
    pub create_commit_ms: f64,
    pub commit_hash: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommitStaging {
    StagePath,
    DirectPath,
}

pub fn gix_commit_backend_enabled() -> bool {
    env::var("KCMT_GIT_COMMIT_BACKEND").as_deref() == Ok("gix")
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

    if gix_commit_backend_enabled() {
        return commit_file_with_gix(repo_path, file_path, message, staging);
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
            commit_hash: None,
        })
    } else {
        Err(KcmtError::Message(format!(
            "git commit failed for pathspec {file_path}"
        )))
    }
}

pub struct GixCommitSession {
    repo_path: PathBuf,
    repo: gix::Repository,
    index: gix::index::File,
    parent: Option<gix::hash::ObjectId>,
    signature: gix::actor::Signature,
    index_dirty: bool,
    defer_index_writes: bool,
}

impl GixCommitSession {
    pub fn open(repo_path: &Path) -> Result<Self> {
        Self::open_with_deferred_index_writes(repo_path, false)
    }

    pub fn open_with_deferred_index_writes(
        repo_path: &Path,
        defer_index_writes: bool,
    ) -> Result<Self> {
        let repo = gix::open(repo_path)
            .map_err(|err| KcmtError::Message(format!("gix open failed: {err}")))?;
        let index = repo
            .open_index()
            .map_err(|err| KcmtError::Message(format!("gix open index failed: {err}")))?;
        let parent = recent_commit_hash(repo_path)?
            .and_then(|hash| gix::hash::ObjectId::from_hex(hash.as_bytes()).ok());
        let signature = gix_signature();
        Ok(Self {
            repo_path: repo_path.to_path_buf(),
            repo,
            index,
            parent,
            signature,
            index_dirty: false,
            defer_index_writes,
        })
    }

    pub fn commit_path(
        &mut self,
        file_path: &str,
        message: &str,
        staging: CommitStaging,
    ) -> Result<CommitFileOutcome> {
        let commit_start = Instant::now();
        let (stage_path_ms, stage_path_invoked, index_dirty) = prepare_index_for_gix_commit(
            &self.repo,
            &mut self.index,
            &self.repo_path,
            file_path,
            staging,
            !self.defer_index_writes,
        )?;
        self.index_dirty |= index_dirty;
        let tree_id = write_commit_tree(&self.repo, &self.index, file_path)?;
        let parents = self.parent.into_iter().collect::<Vec<_>>();
        let commit_id = {
            let mut time_buf = gix::actor::date::parse::TimeBuf::default();
            let signature_ref = self.signature.to_ref(&mut time_buf);
            self.repo
                .commit_as(
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
        let commit_id = commit_id.detach();
        self.parent = Some(commit_id);
        Ok(CommitFileOutcome {
            stage_path_ms,
            stage_path_invoked,
            create_commit_ms,
            commit_hash: Some(commit_id.to_string()),
        })
    }

    pub fn flush_index(&mut self) -> Result<f64> {
        if !self.index_dirty {
            return Ok(0.0);
        }
        let flush_start = Instant::now();
        self.index
            .write(gix::index::write::Options::default())
            .map_err(|err| KcmtError::Message(format!("gix index write failed: {err}")))?;
        self.index_dirty = false;
        Ok(flush_start.elapsed().as_secs_f64() * 1000.0)
    }
}

fn commit_file_with_gix(
    repo_path: &Path,
    file_path: &str,
    message: &str,
    staging: CommitStaging,
) -> Result<CommitFileOutcome> {
    let commit_start = Instant::now();
    let repo = gix::open(repo_path)
        .map_err(|err| KcmtError::Message(format!("gix open failed: {err}")))?;
    let mut stage_path_ms = 0.0;
    let mut stage_path_invoked = false;
    let mut index = match repo.open_index() {
        Ok(index) => index,
        Err(_) if staging == CommitStaging::StagePath => {
            let stage_result = stage_path_with_git(repo_path, file_path)?;
            stage_path_ms += stage_result;
            stage_path_invoked = true;
            repo.open_index()
                .map_err(|err| KcmtError::Message(format!("gix open index failed: {err}")))?
        }
        Err(err) => return Err(KcmtError::Message(format!("gix open index failed: {err}"))),
    };
    if !stage_path_invoked {
        let (prepare_stage_ms, prepare_stage_invoked, _) =
            prepare_index_for_gix_commit(&repo, &mut index, repo_path, file_path, staging, true)?;
        stage_path_ms += prepare_stage_ms;
        stage_path_invoked = prepare_stage_invoked;
    }
    let tree_id = write_commit_tree(&repo, &index, file_path)?;
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
        stage_path_invoked,
        create_commit_ms,
        commit_hash: Some(commit_id.to_string()),
    })
}

fn prepare_index_for_gix_commit(
    repo: &gix::Repository,
    index: &mut gix::index::File,
    repo_path: &Path,
    file_path: &str,
    staging: CommitStaging,
    write_index: bool,
) -> Result<(f64, bool, bool)> {
    if update_path_in_index(repo, index, repo_path, file_path, staging).is_ok() {
        if write_index {
            index
                .write(gix::index::write::Options::default())
                .map_err(|err| KcmtError::Message(format!("gix index write failed: {err}")))?;
        }
        return Ok((0.0, false, !write_index));
    }

    let stage_path_ms = stage_path_with_git(repo_path, file_path)?;
    *index = repo
        .open_index()
        .map_err(|err| KcmtError::Message(format!("gix reopen index failed: {err}")))?;
    Ok((stage_path_ms, true, false))
}

fn stage_path_with_git(repo_path: &Path, file_path: &str) -> Result<f64> {
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
    Ok(stage_path_ms)
}

fn update_path_in_index(
    repo: &gix::Repository,
    index: &mut gix::index::File,
    repo_path: &Path,
    file_path: &str,
    staging: CommitStaging,
) -> Result<()> {
    let worktree_path = repo_path.join(file_path);
    let path = file_path.as_bytes().as_bstr();
    if !worktree_path.exists() {
        if staging != CommitStaging::DirectPath {
            return Err(KcmtError::Message(format!(
                "path requires git staging fallback: {file_path}"
            )));
        }
        let mut removed = false;
        index.remove_entries(|_, entry_path, entry| {
            let remove =
                entry_path == path && entry.stage() == gix::index::entry::Stage::Unconflicted;
            removed |= remove;
            remove
        });
        if !removed {
            return Err(KcmtError::Message(format!(
                "tracked path not found in index: {file_path}"
            )));
        }
        index.remove_tree();
        return Ok(());
    }

    let entry_exists = index
        .entry_mut_by_path_and_stage(path, gix::index::entry::Stage::Unconflicted)
        .is_some();
    if staging == CommitStaging::DirectPath && !entry_exists {
        return Err(KcmtError::Message(format!(
            "tracked path not found in index: {file_path}"
        )));
    }

    let metadata = gix::index::fs::Metadata::from_path_no_follow(&worktree_path)?;
    if !metadata.is_file() {
        return Err(KcmtError::Message(format!(
            "tracked path is not a regular file: {file_path}"
        )));
    }
    let mut file = fs::File::open(&worktree_path)?;
    let blob_id = repo
        .objects
        .write_stream(gix::objs::Kind::Blob, metadata.len(), &mut file)
        .map_err(|err| KcmtError::Message(format!("gix write blob failed: {err}")))?;
    let stat = gix::index::entry::Stat::from_fs(&metadata)
        .map_err(|err| KcmtError::Message(format!("gix stat conversion failed: {err}")))?;
    if entry_exists {
        let entry = index
            .entry_mut_by_path_and_stage(path, gix::index::entry::Stage::Unconflicted)
            .ok_or_else(|| {
                KcmtError::Message(format!("tracked path not found in index: {file_path}"))
            })?;
        entry.id = blob_id;
        entry.stat = stat;
        if let Some(change) = entry.mode.change_to_match_fs(&metadata, true, true) {
            entry.mode = change.apply(entry.mode);
        }
    } else {
        let mode = if metadata.is_executable() {
            gix::index::entry::Mode::FILE_EXECUTABLE
        } else {
            gix::index::entry::Mode::FILE
        };
        index.dangerously_push_entry(
            stat,
            blob_id,
            gix::index::entry::Flags::from_stage(gix::index::entry::Stage::Unconflicted),
            mode,
            path,
        );
        index.sort_entries();
    }
    index.remove_tree();
    Ok(())
}

#[derive(Default)]
struct TreeNode {
    files: BTreeMap<Vec<u8>, (gix::objs::tree::EntryMode, gix::hash::ObjectId)>,
    dirs: BTreeMap<Vec<u8>, TreeNode>,
}

fn write_commit_tree(
    repo: &gix::Repository,
    index: &gix::index::File,
    file_path: &str,
) -> Result<gix::hash::ObjectId> {
    match env::var("KCMT_GIT_TREE_BACKEND").as_deref() {
        Ok("full") => write_index_tree(repo, index),
        Ok("path") => write_changed_path_tree(repo, index, file_path)
            .or_else(|_| write_index_tree(repo, index)),
        _ if index.entries().len() >= 512 => write_changed_path_tree(repo, index, file_path)
            .or_else(|_| write_index_tree(repo, index)),
        _ => write_index_tree(repo, index),
    }
}

fn write_changed_path_tree(
    repo: &gix::Repository,
    index: &gix::index::File,
    file_path: &str,
) -> Result<gix::hash::ObjectId> {
    let components = file_path
        .as_bytes()
        .split(|byte| *byte == b'/')
        .filter(|component| !component.is_empty())
        .map(|component| component.to_vec())
        .collect::<Vec<_>>();
    if components.is_empty() {
        return Err(KcmtError::Message("empty commit path".to_string()));
    }
    let entry = index_tree_entry_for_path(index, file_path)?;
    let root_id = repo
        .head_tree_id_or_empty()
        .map_err(|err| KcmtError::Message(format!("gix head tree lookup failed: {err}")))?
        .detach();
    let (tree_id, _) = write_changed_path_tree_at(repo, root_id, &components, entry)?;
    Ok(tree_id)
}

fn index_tree_entry_for_path(
    index: &gix::index::File,
    file_path: &str,
) -> Result<Option<gix::objs::tree::Entry>> {
    let path = file_path.as_bytes().as_bstr();
    let Some(entry) = index.entry_by_path_and_stage(path, gix::index::entry::Stage::Unconflicted)
    else {
        return Ok(None);
    };
    let mode = entry
        .mode
        .to_tree_entry_mode()
        .ok_or_else(|| KcmtError::Message(format!("invalid index mode for {file_path}")))?;
    Ok(Some(gix::objs::tree::Entry {
        mode,
        filename: components_leaf(file_path)?.to_vec().into(),
        oid: entry.id,
    }))
}

fn components_leaf(file_path: &str) -> Result<&[u8]> {
    file_path
        .as_bytes()
        .rsplit(|byte| *byte == b'/')
        .find(|component| !component.is_empty())
        .ok_or_else(|| KcmtError::Message("empty commit path".to_string()))
}

fn write_changed_path_tree_at(
    repo: &gix::Repository,
    tree_id: gix::hash::ObjectId,
    components: &[Vec<u8>],
    replacement: Option<gix::objs::tree::Entry>,
) -> Result<(gix::hash::ObjectId, bool)> {
    let mut entries = repo
        .find_tree(tree_id)
        .map_err(|err| KcmtError::Message(format!("gix find tree failed: {err}")))?
        .decode()
        .map_err(|err| KcmtError::Message(format!("gix decode tree failed: {err}")))?
        .entries
        .into_iter()
        .map(Into::into)
        .collect::<Vec<gix::objs::tree::Entry>>();

    let name = components[0].as_slice();
    if components.len() == 1 {
        entries.retain(|entry| entry.filename.as_slice() != name);
        if let Some(replacement) = replacement {
            entries.push(replacement);
        }
    } else {
        let existing_child = entries
            .iter()
            .find(|entry| entry.filename.as_slice() == name && entry.mode.is_tree())
            .map(|entry| entry.oid);
        let child_id =
            existing_child.unwrap_or_else(|| gix::hash::ObjectId::empty_tree(repo.object_hash()));
        let (new_child_id, child_is_empty) =
            write_changed_path_tree_at(repo, child_id, &components[1..], replacement)?;
        entries.retain(|entry| entry.filename.as_slice() != name);
        if !child_is_empty {
            entries.push(gix::objs::tree::Entry {
                mode: gix::objs::tree::EntryKind::Tree.into(),
                filename: name.to_vec().into(),
                oid: new_child_id,
            });
        }
    }

    entries.sort();
    let is_empty = entries.is_empty();
    let tree = gix::objs::Tree { entries };
    let new_id = repo
        .write_object(tree)
        .map_err(|err| KcmtError::Message(format!("gix write tree failed: {err}")))?
        .detach();
    Ok((new_id, is_empty))
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
    let Some(git_dirs) = git_dirs(repo_path)? else {
        return Ok(None);
    };
    let head_path = git_dirs.git_dir.join("HEAD");
    let head = match fs::read_to_string(&head_path) {
        Ok(head) => head,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(err) => return Err(err.into()),
    };
    let head = head.trim();
    if let Some(ref_name) = head.strip_prefix("ref:").map(str::trim) {
        return read_ref_hash(&git_dirs, ref_name);
    }
    Ok(valid_hash(head).map(ToOwned::to_owned))
}

#[derive(Debug)]
struct GitDirs {
    git_dir: PathBuf,
    common_dir: PathBuf,
}

fn git_dirs(repo_path: &Path) -> Result<Option<GitDirs>> {
    let dot_git = repo_path.join(".git");
    let git_dir = if dot_git.is_dir() {
        dot_git
    } else if dot_git.is_file() {
        let content = fs::read_to_string(&dot_git)?;
        let Some(path) = content.trim().strip_prefix("gitdir:").map(str::trim) else {
            return Ok(None);
        };
        let git_dir = PathBuf::from(path);
        if git_dir.is_absolute() {
            git_dir
        } else {
            repo_path.join(git_dir)
        }
    } else {
        return Ok(None);
    };
    let common_dir = common_git_dir(&git_dir)?;
    Ok(Some(GitDirs {
        git_dir,
        common_dir,
    }))
}

fn common_git_dir(git_dir: &Path) -> Result<PathBuf> {
    let common_dir_path = git_dir.join("commondir");
    let common_dir = match fs::read_to_string(&common_dir_path) {
        Ok(value) => {
            let path = PathBuf::from(value.trim());
            if path.is_absolute() {
                path
            } else {
                git_dir.join(path)
            }
        }
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => git_dir.to_path_buf(),
        Err(err) => return Err(err.into()),
    };
    Ok(common_dir)
}

fn read_ref_hash(git_dirs: &GitDirs, ref_name: &str) -> Result<Option<String>> {
    for dir in [&git_dirs.git_dir, &git_dirs.common_dir] {
        let loose_ref = dir.join(ref_name);
        match fs::read_to_string(&loose_ref) {
            Ok(hash) => return Ok(valid_hash(hash.trim()).map(ToOwned::to_owned)),
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => {}
            Err(err) => return Err(err.into()),
        }
    }

    for dir in [&git_dirs.git_dir, &git_dirs.common_dir] {
        let packed_refs = dir.join("packed-refs");
        let packed = match fs::read_to_string(&packed_refs) {
            Ok(packed) => packed,
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => continue,
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
    use super::{commit_file_with_gix, recent_commit_hash, CommitStaging, GixCommitSession};
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
    fn reads_recent_commit_hash_from_linked_worktree_common_dir() {
        let repo = unique_temp_dir("recent-worktree-source");
        git(&repo, &["init", "-q"]);
        fs::write(repo.join("tracked.py"), "print('seed')\n").expect("tracked file");
        git(&repo, &["add", "tracked.py"]);
        git(&repo, &["commit", "-m", "chore(repo): seed"]);
        let worktree = unique_temp_dir("recent-worktree-linked");
        fs::remove_dir_all(&worktree).expect("empty worktree path should be removable");
        git(
            &repo,
            &[
                "worktree",
                "add",
                "-q",
                "-b",
                "kcmt-test-worktree",
                worktree.to_str().expect("utf8 worktree path"),
            ],
        );

        let expected = Command::new("git")
            .current_dir(&worktree)
            .args(["rev-parse", "HEAD"])
            .output()
            .expect("git rev-parse should run");
        assert!(expected.status.success());
        let expected = String::from_utf8_lossy(&expected.stdout).trim().to_string();

        let actual = recent_commit_hash(&worktree).expect("hash should be read");

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

        let outcome = commit_file_with_gix(
            &repo,
            "tracked.py",
            "chore(repo): update tracked",
            CommitStaging::DirectPath,
        )
        .expect("gix commit");

        assert!(!outcome.stage_path_invoked);
        assert_eq!(outcome.stage_path_ms, 0.0);
        assert!(outcome.commit_hash.is_some());
        assert!(outcome.create_commit_ms >= 0.0);
        assert_eq!(git_output(&repo, &["status", "--short"]), "");
        assert_eq!(
            git_output(&repo, &["log", "-1", "--pretty=%s"]),
            "chore(repo): update tracked"
        );
    }

    #[test]
    fn gix_commit_backend_commits_tracked_delete_without_staging() {
        let repo = unique_temp_dir("gix-commit-delete");
        git(&repo, &["init", "-q"]);
        fs::write(repo.join("tracked.py"), "print('seed')\n").expect("tracked file");
        git(&repo, &["add", "tracked.py"]);
        git(&repo, &["commit", "-m", "chore(repo): seed"]);
        fs::remove_file(repo.join("tracked.py")).expect("delete tracked file");

        let outcome = commit_file_with_gix(
            &repo,
            "tracked.py",
            "chore(repo): remove tracked",
            CommitStaging::DirectPath,
        )
        .expect("gix commit");

        assert!(!outcome.stage_path_invoked);
        assert_eq!(outcome.stage_path_ms, 0.0);
        assert!(outcome.commit_hash.is_some());
        assert_eq!(git_output(&repo, &["status", "--short"]), "");
        assert_eq!(
            git_output(&repo, &["log", "-1", "--pretty=%s"]),
            "chore(repo): remove tracked"
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

        let outcome = commit_file_with_gix(
            &repo,
            "new_file.py",
            "chore(repo): add new file",
            CommitStaging::StagePath,
        )
        .expect("gix commit");

        assert!(!outcome.stage_path_invoked);
        assert_eq!(outcome.stage_path_ms, 0.0);
        assert!(outcome.commit_hash.is_some());
        assert_eq!(git_output(&repo, &["status", "--short"]), "");
        assert_eq!(
            git_output(&repo, &["log", "-1", "--pretty=%s"]),
            "chore(repo): add new file"
        );
    }

    #[test]
    fn gix_commit_session_chains_multiple_file_commits() {
        let repo = unique_temp_dir("gix-session-multiple");
        git(&repo, &["init", "-q"]);
        fs::write(repo.join("one.py"), "print('one')\n").expect("one seed");
        fs::write(repo.join("two.py"), "print('two')\n").expect("two seed");
        git(&repo, &["add", "."]);
        git(&repo, &["commit", "-m", "chore(repo): seed"]);
        fs::write(repo.join("one.py"), "print('one changed')\n").expect("one change");
        fs::write(repo.join("two.py"), "print('two changed')\n").expect("two change");

        let mut session = GixCommitSession::open_with_deferred_index_writes(&repo, true)
            .expect("session should open");
        let first = session
            .commit_path(
                "one.py",
                "chore(repo): update one",
                CommitStaging::DirectPath,
            )
            .expect("first commit");
        let second = session
            .commit_path(
                "two.py",
                "chore(repo): update two",
                CommitStaging::DirectPath,
            )
            .expect("second commit");

        assert!(first.commit_hash.is_some());
        assert!(second.commit_hash.is_some());
        assert_ne!(first.commit_hash, second.commit_hash);
        assert_ne!(git_output(&repo, &["status", "--short"]), "");
        let flush_ms = session.flush_index().expect("index flush");
        assert!(flush_ms >= 0.0);
        assert_eq!(git_output(&repo, &["status", "--short"]), "");
        assert_eq!(
            git_output(&repo, &["log", "--pretty=%s", "-2"]),
            "chore(repo): update two\nchore(repo): update one"
        );
        assert_eq!(
            git_output(&repo, &["rev-parse", "HEAD^"]),
            first.commit_hash.expect("first hash")
        );
    }

    #[test]
    fn gix_commit_session_open_writes_single_file_index_immediately() {
        let repo = unique_temp_dir("gix-session-single-flush");
        git(&repo, &["init", "-q"]);
        fs::write(repo.join("one.py"), "print('one')\n").expect("one seed");
        git(&repo, &["add", "."]);
        git(&repo, &["commit", "-m", "chore(repo): seed"]);
        fs::write(repo.join("one.py"), "print('one changed')\n").expect("one change");

        let mut session = GixCommitSession::open(&repo).expect("session should open");
        session
            .commit_path(
                "one.py",
                "chore(repo): update one",
                CommitStaging::DirectPath,
            )
            .expect("commit");

        assert_eq!(session.flush_index().expect("flush"), 0.0);
        assert_eq!(git_output(&repo, &["status", "--short"]), "");
    }

    #[test]
    fn gix_commit_backend_commits_nested_tracked_file() {
        let repo = unique_temp_dir("gix-commit-nested");
        git(&repo, &["init", "-q"]);
        fs::create_dir_all(repo.join("src").join("module")).expect("nested dir");
        fs::write(
            repo.join("src").join("module").join("tracked.py"),
            "print('seed')\n",
        )
        .expect("tracked file");
        fs::write(repo.join("src").join("other.py"), "print('other')\n").expect("other file");
        git(&repo, &["add", "."]);
        git(&repo, &["commit", "-m", "chore(repo): seed"]);
        fs::write(
            repo.join("src").join("module").join("tracked.py"),
            "print('changed')\n",
        )
        .expect("tracked change");

        let outcome = commit_file_with_gix(
            &repo,
            "src/module/tracked.py",
            "chore(src): update tracked",
            CommitStaging::DirectPath,
        )
        .expect("gix commit");

        assert!(!outcome.stage_path_invoked);
        assert_eq!(git_output(&repo, &["status", "--short"]), "");
        assert_eq!(
            git_output(&repo, &["show", "HEAD:src/module/tracked.py"]),
            "print('changed')"
        );
        assert_eq!(
            git_output(&repo, &["show", "HEAD:src/other.py"]),
            "print('other')"
        );
    }

    #[test]
    fn gix_commit_backend_removes_empty_nested_tree_after_delete() {
        let repo = unique_temp_dir("gix-commit-nested-delete");
        git(&repo, &["init", "-q"]);
        fs::create_dir_all(repo.join("src").join("module")).expect("nested dir");
        fs::write(
            repo.join("src").join("module").join("tracked.py"),
            "print('seed')\n",
        )
        .expect("tracked file");
        fs::write(repo.join("src").join("keep.py"), "print('keep')\n").expect("keep file");
        git(&repo, &["add", "."]);
        git(&repo, &["commit", "-m", "chore(repo): seed"]);
        fs::remove_file(repo.join("src").join("module").join("tracked.py"))
            .expect("delete tracked file");

        let outcome = commit_file_with_gix(
            &repo,
            "src/module/tracked.py",
            "chore(src): remove tracked",
            CommitStaging::DirectPath,
        )
        .expect("gix commit");

        assert!(!outcome.stage_path_invoked);
        assert_eq!(git_output(&repo, &["status", "--short"]), "");
        assert_eq!(
            git_output(&repo, &["ls-tree", "--name-only", "HEAD:src"]),
            "keep.py"
        );
        assert_eq!(
            git_output(&repo, &["show", "HEAD:src/keep.py"]),
            "print('keep')"
        );
    }

    #[test]
    fn gix_commit_backend_commits_first_untracked_file() {
        let repo = unique_temp_dir("gix-commit-first-untracked");
        git(&repo, &["init", "-q"]);
        fs::write(repo.join("first.py"), "print('first')\n").expect("new file");

        let outcome = commit_file_with_gix(
            &repo,
            "first.py",
            "chore(repo): add first file",
            CommitStaging::StagePath,
        )
        .expect("gix commit");

        assert!(outcome.stage_path_invoked);
        assert!(outcome.commit_hash.is_some());
        assert_eq!(git_output(&repo, &["status", "--short"]), "");
        assert_eq!(
            git_output(&repo, &["log", "-1", "--pretty=%s"]),
            "chore(repo): add first file"
        );
    }
}
