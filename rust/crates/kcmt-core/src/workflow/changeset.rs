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
        let path = line[3..].trim();
        if path.is_empty() {
            continue;
        }
        let change_type = code.chars().next().unwrap_or('M').to_string();
        changes.push(ChangeSet {
            file_path: path.to_string(),
            change_type,
            diff_content: String::new(),
            staged: code.chars().next().unwrap_or(' ') != ' ',
            ignored: false,
        });
    }

    Ok(changes)
}
