use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use kcmt_core::config::loader::{load_config, ConfigOverrides};
use kcmt_core::error::KcmtError;
use kcmt_core::error::Result;
use kcmt_core::git::commit_file::commit_file;
use kcmt_core::git::repo::{CliGitRepository, GitRepository};
use serde_json::json;

use super::history::snapshot_path;

pub fn run_file_workflow(repo_path: PathBuf, file_path: &str) -> Result<String> {
    let config = load_config(&repo_path, &ConfigOverrides::default())?;
    let repo = CliGitRepository::from_path(&repo_path);
    let entries = parse_status_entries(&repo.status_porcelain()?);
    if !entries.iter().any(|(_, path)| path == file_path) {
        return Ok("No changes detected in the specified file.\n".to_string());
    }

    let message = heuristic_commit_message(file_path, config.max_commit_length);
    commit_file(&repo_path, file_path, &message, false)?;

    let mut lines = vec![format!("✓ {file_path}"), format!("  {message}")];
    let commit_hash = recent_commit_hash(&repo_path)?;
    if let Some(hash) = &commit_hash {
        lines.push(format!("  {}", truncate_hash(&hash)));
    }
    persist_run_snapshot(&repo_path, &config, file_path, &message, commit_hash.as_deref())?;

    let mut output = lines.join("\n");
    output.push('\n');
    Ok(output)
}

pub fn run_oneshot_workflow(repo_path: PathBuf) -> Result<String> {
    let repo = CliGitRepository::from_path(&repo_path);
    let entries = parse_status_entries(&repo.status_porcelain()?);
    if entries.is_empty() {
        return Ok("No changes to commit.\n".to_string());
    }

    let target = entries
        .iter()
        .find(|(status, _)| !status.contains('D'))
        .map(|(_, path)| path.clone())
        .unwrap_or_else(|| entries[0].1.clone());

    run_file_workflow(repo_path, &target)
}

fn parse_status_entries(status: &str) -> Vec<(String, String)> {
    status
        .lines()
        .filter_map(|line| {
            if line.len() < 4 {
                return None;
            }
            let code = line[0..2].to_string();
            let path = line[3..].trim().to_string();
            if path.is_empty() {
                None
            } else {
                Some((code, path))
            }
        })
        .collect()
}

fn heuristic_commit_message(file_path: &str, max_commit_length: usize) -> String {
    let path = Path::new(file_path);
    let stem = path
        .file_stem()
        .and_then(|stem| stem.to_str())
        .filter(|stem| !stem.is_empty())
        .unwrap_or("file");
    let scope = path
        .parent()
        .and_then(|parent| parent.file_name())
        .and_then(|name| name.to_str())
        .filter(|name| !name.is_empty())
        .unwrap_or("repo");
    let mut message = format!("chore({scope}): update {stem}");
    if message.len() > max_commit_length {
        message.truncate(max_commit_length);
    }
    message
}

fn recent_commit_hash(repo_path: &Path) -> Result<Option<String>> {
    let output = Command::new("git")
        .current_dir(repo_path)
        .args(["log", "-1", "--pretty=%H"])
        .output()?;
    if !output.status.success() {
        return Ok(None);
    }

    let hash = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if hash.is_empty() {
        Ok(None)
    } else {
        Ok(Some(hash))
    }
}

fn truncate_hash(hash: &str) -> &str {
    let max = 8.min(hash.len());
    &hash[..max]
}

fn persist_run_snapshot(
    repo_path: &Path,
    config: &kcmt_core::model::WorkflowConfig,
    file_path: &str,
    message: &str,
    commit_hash: Option<&str>,
) -> Result<()> {
    let snapshot = json!({
        "schema_version": 1,
        "timestamp": "",
        "repo_path": repo_path.display().to_string(),
        "provider": config.provider,
        "model": config.model,
        "endpoint": config.llm_endpoint,
        "duration_seconds": 0.0,
        "rate_commits_per_sec": 0.0,
        "counts": {
            "files_total": 1,
            "prepared_total": 1,
            "processed_total": 1,
            "prepared_failures": 0,
            "commit_success": 1,
            "commit_failure": 0,
            "deletions_total": 0,
            "deletions_success": 0,
            "deletions_failure": 0,
            "overall_success": 1,
            "overall_failure": 0,
            "errors": 0
        },
        "pushed": false,
        "summary": "Successfully completed 1 commits. Committed 1 file change(s)",
        "errors": [],
        "commits": [{
            "success": true,
            "commit_hash": commit_hash,
            "message": message,
            "error": null,
            "file_path": file_path
        }],
        "deletions": [],
        "subjects": [message],
        "stats": {}
    });

    let path = snapshot_path(repo_path);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let rendered =
        serde_json::to_string_pretty(&snapshot).map_err(|err| KcmtError::Message(err.to_string()))?;
    fs::write(path, rendered)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{heuristic_commit_message, parse_status_entries};

    #[test]
    fn parses_porcelain_entries() {
        let entries = parse_status_entries(" M alpha.py\n?? beta.py\n");

        assert_eq!(entries, vec![
            (" M".to_string(), "alpha.py".to_string()),
            ("??".to_string(), "beta.py".to_string()),
        ]);
    }

    #[test]
    fn builds_heuristic_file_message() {
        assert_eq!(
            heuristic_commit_message("src/example.py", 72),
            "chore(src): update example"
        );
    }

    #[test]
    fn truncates_heuristic_message_to_config_limit() {
        assert_eq!(heuristic_commit_message("src/very_long_filename_here.py", 18).len(), 18);
    }
}
