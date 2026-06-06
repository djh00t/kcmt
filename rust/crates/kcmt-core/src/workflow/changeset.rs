//! Change-set collection and parsing from git porcelain output.

use crate::error::Result;
use crate::git::repo::GitRepository;
use crate::model::ChangeSet;

pub fn collect_changes(repo: &dyn GitRepository) -> Result<Vec<ChangeSet>> {
    let status = repo.status_porcelain()?;
    let mut changes = Vec::new();

    for line in status.lines() {
        if line.len() < 4 {
            continue;
        }
        let code = &line[0..2];
        if code == "!!" {
            continue;
        }
        let path = parse_porcelain_path(&line[3..]);
        if path.is_empty() {
            continue;
        }
        let change_type = code
            .chars()
            .find(|status| *status != ' ')
            .unwrap_or('M')
            .to_string();
        changes.push(ChangeSet {
            file_path: path,
            change_type,
            diff_content: String::new(),
            staged: code.chars().next().unwrap_or(' ') != ' ',
            ignored: false,
        });
    }

    Ok(changes)
}

fn parse_porcelain_path(raw_path: &str) -> String {
    let path = raw_path.trim();
    path.rsplit_once(" -> ")
        .map(|(_, destination)| destination.trim())
        .unwrap_or(path)
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::collect_changes;
    use crate::error::Result;
    use crate::git::repo::GitRepository;

    struct StaticRepo {
        status: &'static str,
    }

    impl GitRepository for StaticRepo {
        fn staged_files(&self) -> Result<Vec<String>> {
            Ok(Vec::new())
        }

        fn status_porcelain(&self) -> Result<String> {
            Ok(self.status.to_string())
        }

        fn status_porcelain_for_path(&self, _file_path: &str) -> Result<String> {
            Ok(self.status.to_string())
        }
    }

    #[test]
    fn collect_changes_skips_ignored_rows_and_uses_worktree_status() {
        let repo = StaticRepo {
            status: " M tracked.py\n!! ignored.log\n?? new.py\n",
        };

        let changes = collect_changes(&repo).expect("changes");

        assert_eq!(changes.len(), 2);
        assert_eq!(changes[0].file_path, "tracked.py");
        assert_eq!(changes[0].change_type, "M");
        assert!(!changes[0].staged);
        assert_eq!(changes[1].file_path, "new.py");
        assert_eq!(changes[1].change_type, "?");
    }

    #[test]
    fn collect_changes_uses_rename_and_copy_destination_paths() {
        let repo = StaticRepo {
            status: "R  old.py -> new.py\n C template.py -> copied.py\n",
        };

        let changes = collect_changes(&repo).expect("changes");

        assert_eq!(changes.len(), 2);
        assert_eq!(changes[0].file_path, "new.py");
        assert_eq!(changes[0].change_type, "R");
        assert!(changes[0].staged);
        assert_eq!(changes[1].file_path, "copied.py");
        assert_eq!(changes[1].change_type, "C");
        assert!(!changes[1].staged);
    }
}
