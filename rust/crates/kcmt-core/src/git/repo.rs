//! Git repository adapter built on top of the command runner abstraction.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;

use crate::error::{KcmtError, Result};
use crate::git::runner::{CommandRunner, GitCliRunner};
use gix::bstr::ByteSlice;

pub trait GitRepository {
    fn staged_files(&self) -> Result<Vec<String>>;
    fn status_porcelain(&self) -> Result<String>;
    fn status_porcelain_for_path(&self, file_path: &str) -> Result<String>;
}

pub struct CliGitRepository<R: CommandRunner> {
    repo_path: PathBuf,
    runner: R,
}

pub fn find_git_repo_root(start_path: impl AsRef<Path>) -> Option<PathBuf> {
    let start_path = start_path.as_ref();
    let mut path = fs::canonicalize(start_path).unwrap_or_else(|_| {
        if start_path.is_absolute() {
            start_path.to_path_buf()
        } else {
            std::env::current_dir()
                .map(|cwd| cwd.join(start_path))
                .unwrap_or_else(|_| start_path.to_path_buf())
        }
    });
    if path.is_file() {
        path = path.parent()?.to_path_buf();
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
        if std::env::var("KCMT_GIT_STATUS_BACKEND").as_deref() != Ok("cli") {
            return status_porcelain_gix(&self.repo_path, Vec::new());
        }

        self.status_porcelain_cli()
    }

    fn status_porcelain_for_path(&self, file_path: &str) -> Result<String> {
        if std::env::var("KCMT_GIT_STATUS_BACKEND").as_deref() != Ok("cli") {
            if let Ok(status) = status_porcelain_for_path_fast(&self.repo_path, file_path) {
                return Ok(status);
            }
            return status_porcelain_gix(
                &self.repo_path,
                vec![gix::bstr::BString::from(file_path.as_bytes().to_vec())],
            );
        }

        self.status_porcelain_cli_for_path(file_path)
    }
}

impl<R: CommandRunner> CliGitRepository<R> {
    fn status_porcelain_cli(&self) -> Result<String> {
        let output = self.runner.run(
            &self.repo_path,
            &["status", "--porcelain=v1", "-z", "--untracked-files=all"],
        )?;
        Ok(render_porcelain_lines(&output.stdout))
    }

    fn status_porcelain_cli_for_path(&self, file_path: &str) -> Result<String> {
        let output = self.runner.run(
            &self.repo_path,
            &[
                "status",
                "--porcelain=v1",
                "-z",
                "--untracked-files=all",
                "--",
                file_path,
            ],
        )?;
        Ok(render_porcelain_lines(&output.stdout))
    }
}

fn status_porcelain_for_path_fast(repo_path: &Path, file_path: &str) -> Result<String> {
    let repo = gix::open(repo_path)
        .map_err(|err| KcmtError::Message(format!("gix open failed: {err}")))?;
    let index = repo
        .open_index()
        .map_err(|err| KcmtError::Message(format!("gix open index failed: {err}")))?;
    let path = file_path.as_bytes().as_bstr();
    let index_entry = index.entry_by_path_and_stage(path, gix::index::entry::Stage::Unconflicted);
    let worktree_path = repo_path.join(file_path);

    let index_oid = index_entry.map(|entry| entry.id);
    let worktree_code = if worktree_path.exists() {
        let metadata = fs::metadata(&worktree_path)?;
        if !metadata.is_file() {
            return Err(KcmtError::Message(format!(
                "path status requires fallback for non-file: {file_path}"
            )));
        }
        let Some(index_entry) = index_entry else {
            return Ok(format!("?? {file_path}\n"));
        };
        let index_oid = index_entry.id;
        if metadata.len() != u64::from(index_entry.stat.size) {
            "M"
        } else {
            let blob_oid = hash_worktree_blob(&repo, &worktree_path, metadata.len())?;
            if blob_oid != index_oid {
                "M"
            } else {
                " "
            }
        }
    } else if index_oid.is_some() {
        "D"
    } else {
        " "
    };

    let head_oid = head_tree_entry_oid(&repo, file_path)?;
    let staged_code = match (head_oid, index_oid) {
        (None, Some(_)) => "A",
        (Some(_), None) => "D",
        (Some(head), Some(index)) if head != index => "M",
        _ => " ",
    };

    if staged_code == " " && worktree_code == " " {
        return Ok(String::new());
    }
    Ok(format!("{staged_code}{worktree_code} {file_path}\n"))
}

fn head_tree_entry_oid(
    repo: &gix::Repository,
    file_path: &str,
) -> Result<Option<gix::hash::ObjectId>> {
    let tree_id = repo
        .head_tree_id_or_empty()
        .map_err(|err| KcmtError::Message(format!("gix head tree lookup failed: {err}")))?;
    let tree = repo
        .find_tree(tree_id)
        .map_err(|err| KcmtError::Message(format!("gix find tree failed: {err}")))?;
    let entry = tree
        .lookup_entry_by_path(Path::new(file_path))
        .map_err(|err| KcmtError::Message(format!("gix tree path lookup failed: {err}")))?;
    Ok(entry.map(|entry| entry.object_id()))
}

fn hash_worktree_blob(
    repo: &gix::Repository,
    path: &Path,
    size: u64,
) -> Result<gix::hash::ObjectId> {
    let mut file = fs::File::open(path)?;
    let mut progress = gix::progress::Discard;
    let interrupt = AtomicBool::new(false);
    gix::objs::compute_stream_hash(
        repo.object_hash(),
        gix::objs::Kind::Blob,
        &mut file,
        size,
        &mut progress,
        &interrupt,
    )
    .map_err(|err| KcmtError::Message(format!("gix hash blob failed: {err}")))
}

fn status_porcelain_gix(repo_path: &Path, patterns: Vec<gix::bstr::BString>) -> Result<String> {
    use gix::bstr::ByteSlice;

    let repo = gix::open(repo_path)
        .map_err(|err| KcmtError::Message(format!("gix open failed: {err}")))?;
    let status = repo
        .status(gix::progress::Discard)
        .map_err(|err| KcmtError::Message(format!("gix status failed: {err}")))?
        .untracked_files(gix::status::UntrackedFiles::Files)
        .index_worktree_submodules(None)
        .tree_index_track_renames(gix::status::tree_index::TrackRenames::Disabled)
        .index_worktree_options_mut(|opts| {
            opts.sorting = Some(
                gix::status::plumbing::index_as_worktree_with_renames::Sorting::ByPathCaseSensitive,
            );
            opts.thread_limit = Some(1);
        });
    let mut rows = Vec::new();

    for item in status
        .into_iter(patterns)
        .map_err(|err| KcmtError::Message(format!("gix status iterator failed: {err}")))?
    {
        let item =
            item.map_err(|err| KcmtError::Message(format!("gix status item failed: {err}")))?;
        match item {
            gix::status::Item::IndexWorktree(change) => {
                use gix::status::index_worktree::Item;
                use gix::status::plumbing::index_as_worktree::{Change, EntryStatus};

                match change {
                    Item::Modification {
                        rela_path, status, ..
                    } => match status {
                        EntryStatus::Change(Change::Removed) => {
                            rows.push(format!(" D {}", rela_path.as_bstr().to_str_lossy()));
                        }
                        EntryStatus::Change(Change::Modification { .. })
                        | EntryStatus::Change(Change::SubmoduleModification(_))
                        | EntryStatus::Change(Change::Type { .. }) => {
                            rows.push(format!(" M {}", rela_path.as_bstr().to_str_lossy()));
                        }
                        EntryStatus::Conflict(_) => {
                            rows.push(format!("UU {}", rela_path.as_bstr().to_str_lossy()));
                        }
                        EntryStatus::IntentToAdd | EntryStatus::NeedsUpdate(_) => {}
                    },
                    Item::DirectoryContents { entry, .. } => {
                        if matches!(entry.status, gix::dir::entry::Status::Untracked) {
                            rows.push(format!("?? {}", entry.rela_path.as_bstr().to_str_lossy()));
                        }
                    }
                    Item::Rewrite {
                        dirwalk_entry,
                        copy,
                        ..
                    } => {
                        let status = if copy { " C" } else { " R" };
                        let path = dirwalk_entry.rela_path.as_bstr().to_str_lossy();
                        rows.push(format!("{status} {path}"));
                    }
                }
            }
            gix::status::Item::TreeIndex(change) => {
                let path = change.location().to_str_lossy();
                let status = match change {
                    gix::diff::index::Change::Addition { .. } => "A ",
                    gix::diff::index::Change::Deletion { .. } => "D ",
                    gix::diff::index::Change::Modification { .. } => "M ",
                    gix::diff::index::Change::Rewrite { copy, .. } => {
                        if copy {
                            "C "
                        } else {
                            "R "
                        }
                    }
                };
                rows.push(format!("{status} {path}"));
            }
        }
    }

    rows.sort();
    let mut rendered = rows.join("\n");
    if !rendered.is_empty() {
        rendered.push('\n');
    }
    Ok(rendered)
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
    use super::{
        find_git_repo_root, render_porcelain_lines, status_porcelain_for_path_fast,
        status_porcelain_gix, CliGitRepository, GitRepository,
    };
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

    fn git_output(repo: &Path, args: &[&str]) -> String {
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
        String::from_utf8_lossy(&output.stdout).to_string()
    }

    fn init_repo(repo: &Path) {
        git(repo, &["init", "-q"]);
        git(repo, &["config", "user.name", "kcmt test"]);
        git(repo, &["config", "user.email", "kcmt-test@example.com"]);
    }

    fn sorted_lines(value: &str) -> Vec<String> {
        let mut lines = value
            .lines()
            .filter(|line| !line.is_empty())
            .map(ToOwned::to_owned)
            .collect::<Vec<_>>();
        lines.sort();
        lines
    }

    #[test]
    fn finds_top_level_repo_from_nested_directory() {
        let repo = unique_temp_dir("repo");
        init_repo(&repo);
        let nested = repo.join("pkg").join("sub");
        fs::create_dir_all(&nested).expect("nested dir");

        let found = find_git_repo_root(&nested).expect("repo root should be found");
        let found_canonical = fs::canonicalize(found).expect("found path canonical");
        let repo_canonical = fs::canonicalize(repo).expect("repo path canonical");
        assert_eq!(found_canonical, repo_canonical);
    }

    #[test]
    fn finds_top_level_repo_from_relative_current_directory() {
        let found = find_git_repo_root(".").expect("current repo root should be found");
        let found_canonical = fs::canonicalize(found).expect("found path canonical");
        let expected = git_output(Path::new("."), &["rev-parse", "--show-toplevel"]);
        let expected_canonical =
            fs::canonicalize(expected.trim()).expect("git top level canonical");

        assert_eq!(found_canonical, expected_canonical);
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
        init_repo(&repo);
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

    #[test]
    fn gix_status_matches_git_cli_for_common_file_states() {
        let repo = unique_temp_dir("gix-status-parity");
        init_repo(&repo);
        fs::write(repo.join("tracked.py"), "print('seed')\n").expect("tracked seed");
        fs::write(repo.join("delete_me.txt"), "bye\n").expect("delete seed");
        git(&repo, &["add", "tracked.py", "delete_me.txt"]);
        git(&repo, &["commit", "-m", "chore(repo): seed"]);

        fs::write(repo.join("tracked.py"), "print('changed')\n").expect("tracked change");
        fs::remove_file(repo.join("delete_me.txt")).expect("delete file");
        fs::write(repo.join("untracked.py"), "print('new')\n").expect("untracked file");
        fs::write(repo.join("staged_new.py"), "print('staged')\n").expect("staged file");
        git(&repo, &["add", "staged_new.py"]);

        let git_status = render_porcelain_lines(&git_output(
            &repo,
            &["status", "--porcelain=v1", "-z", "--untracked-files=all"],
        ));
        let gix_status = status_porcelain_gix(&repo, Vec::new()).expect("gix status should render");

        assert_eq!(sorted_lines(&gix_status), sorted_lines(&git_status));
    }

    #[test]
    fn path_status_limits_gix_status_to_requested_file() {
        let repo = unique_temp_dir("gix-path-status");
        init_repo(&repo);
        fs::write(repo.join("tracked.py"), "print('seed')\n").expect("tracked seed");
        fs::write(repo.join("other.py"), "print('seed')\n").expect("other seed");
        git(&repo, &["add", "tracked.py", "other.py"]);
        git(&repo, &["commit", "-m", "chore(repo): seed"]);

        fs::write(repo.join("tracked.py"), "print('changed')\n").expect("tracked change");
        fs::write(repo.join("other.py"), "print('changed')\n").expect("other change");

        let status = CliGitRepository::from_path(&repo)
            .status_porcelain_for_path("tracked.py")
            .expect("path status should render");

        assert_eq!(sorted_lines(&status), vec![" M tracked.py".to_string()]);
    }

    #[test]
    fn fast_path_status_hashes_same_size_tracked_modification() {
        let repo = unique_temp_dir("gix-path-status-same-size");
        init_repo(&repo);
        fs::write(repo.join("tracked.py"), "abc\n").expect("tracked seed");
        git(&repo, &["add", "tracked.py"]);
        git(&repo, &["commit", "-m", "chore(repo): seed"]);
        fs::write(repo.join("tracked.py"), "xyz\n").expect("same size change");

        let status = status_porcelain_for_path_fast(&repo, "tracked.py")
            .expect("fast path status should render");

        assert_eq!(status, " M tracked.py\n");
    }

    #[test]
    fn fast_path_status_returns_empty_for_clean_file() {
        let repo = unique_temp_dir("gix-path-status-clean");
        init_repo(&repo);
        fs::write(repo.join("tracked.py"), "print('seed')\n").expect("tracked seed");
        git(&repo, &["add", "tracked.py"]);
        git(&repo, &["commit", "-m", "chore(repo): seed"]);

        let status = status_porcelain_for_path_fast(&repo, "tracked.py")
            .expect("fast path status should render");

        assert_eq!(status, "");
    }

    #[test]
    fn fast_path_status_detects_tracked_delete() {
        let repo = unique_temp_dir("gix-path-status-delete");
        init_repo(&repo);
        fs::write(repo.join("tracked.py"), "print('seed')\n").expect("tracked seed");
        git(&repo, &["add", "tracked.py"]);
        git(&repo, &["commit", "-m", "chore(repo): seed"]);
        fs::remove_file(repo.join("tracked.py")).expect("delete tracked");

        let status = status_porcelain_for_path_fast(&repo, "tracked.py")
            .expect("fast path status should render");

        assert_eq!(status, " D tracked.py\n");
    }

    #[test]
    fn fast_path_status_detects_untracked_file() {
        let repo = unique_temp_dir("gix-path-status-untracked");
        init_repo(&repo);
        git(
            &repo,
            &["commit", "--allow-empty", "-m", "chore(repo): seed"],
        );
        fs::write(repo.join("untracked.py"), "print('new')\n").expect("untracked file");

        let status = status_porcelain_for_path_fast(&repo, "untracked.py")
            .expect("fast path status should render");

        assert_eq!(status, "?? untracked.py\n");
    }

    #[test]
    fn fast_path_status_detects_staged_file() {
        let repo = unique_temp_dir("gix-path-status-staged");
        init_repo(&repo);
        fs::write(repo.join("tracked.py"), "print('seed')\n").expect("tracked seed");
        git(&repo, &["add", "tracked.py"]);
        git(&repo, &["commit", "-m", "chore(repo): seed"]);
        fs::write(repo.join("tracked.py"), "print('changed')\n").expect("tracked change");
        git(&repo, &["add", "tracked.py"]);

        let status = status_porcelain_for_path_fast(&repo, "tracked.py")
            .expect("fast path status should render");

        assert_eq!(status, "M  tracked.py\n");
    }

    #[test]
    fn fast_path_status_detects_staged_and_unstaged_file() {
        let repo = unique_temp_dir("gix-path-status-mixed");
        init_repo(&repo);
        fs::write(repo.join("tracked.py"), "print('seed')\n").expect("tracked seed");
        git(&repo, &["add", "tracked.py"]);
        git(&repo, &["commit", "-m", "chore(repo): seed"]);
        fs::write(repo.join("tracked.py"), "print('staged')\n").expect("staged change");
        git(&repo, &["add", "tracked.py"]);
        fs::write(repo.join("tracked.py"), "print('unstaged')\n").expect("unstaged change");

        let status = status_porcelain_for_path_fast(&repo, "tracked.py")
            .expect("fast path status should render");

        assert_eq!(status, "MM tracked.py\n");
    }
}
