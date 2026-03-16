use std::fs;
use std::path::{Path, PathBuf};

use serde_json::Value;

use super::history::{normalize_repo_path, snapshot_path};

pub fn render_status(repo_path: Option<PathBuf>, raw: bool) -> Result<String, String> {
    let repo_path = normalize_repo_path(repo_path.unwrap_or_else(|| PathBuf::from(".")));
    let snapshot = load_run_snapshot(&repo_path)
        .ok_or_else(|| "No kcmt run history found for this repository.".to_string())?;

    if raw {
        return serde_json::to_string_pretty(&snapshot)
            .map(|mut rendered| {
                rendered.push('\n');
                rendered
            })
            .map_err(|err| err.to_string());
    }

    Ok(render_status_summary(&repo_path, &snapshot))
}

fn render_status_summary(repo_path: &Path, snapshot: &Value) -> String {
    let repo_display = snapshot
        .get("repo_path")
        .and_then(Value::as_str)
        .map(ToOwned::to_owned)
        .unwrap_or_else(|| repo_path.display().to_string());
    let timestamp = snapshot.get("timestamp").and_then(Value::as_str).unwrap_or("");
    let duration = snapshot
        .get("duration_seconds")
        .and_then(Value::as_f64)
        .unwrap_or(0.0);
    let summary = snapshot.get("summary").and_then(Value::as_str).unwrap_or("");
    let counts = snapshot.get("counts").and_then(Value::as_object);
    let success_count = counts
        .and_then(|counts| counts.get("commit_success"))
        .and_then(Value::as_i64)
        .unwrap_or(0);
    let failure_count = counts
        .and_then(|counts| counts.get("commit_failure"))
        .and_then(Value::as_i64)
        .unwrap_or(0);

    let mut lines = Vec::new();
    lines.push(format!("kcmt status :: {repo_display}"));
    if timestamp.is_empty() {
        lines.push(format!("Duration {duration:.2}s"));
    } else {
        lines.push(format!("Run time {timestamp}  Duration {duration:.2}s"));
    }
    lines.push(String::new());
    lines.push("Summary".to_string());
    if summary.is_empty() {
        lines.push("No summary captured.".to_string());
    } else {
        lines.push(summary.to_string());
    }
    lines.push(String::new());
    lines.push("Commit status".to_string());
    lines.push(format!("Success: {success_count}"));
    lines.push(format!("Failure: {failure_count}"));

    let mut output = lines.join("\n");
    output.push('\n');
    output
}

fn load_run_snapshot(repo_path: &Path) -> Option<Value> {
    let path = snapshot_path(repo_path);
    let content = fs::read_to_string(path).ok()?;
    let parsed = serde_json::from_str::<Value>(&content).ok()?;
    parsed.as_object()?;
    Some(parsed)
}

#[cfg(test)]
mod tests {
    use super::{render_status, snapshot_path};
    use crate::commands::history::state_dir;
    use serde_json::json;
    use std::env;
    use std::fs;
    use std::path::PathBuf;
    use std::sync::{Mutex, OnceLock};
    use std::time::{SystemTime, UNIX_EPOCH};

    fn env_lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

    fn unique_temp_dir(label: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock should be after unix epoch")
            .as_nanos();
        let path = env::temp_dir().join(format!("kcmt-{label}-{nanos}"));
        fs::create_dir_all(&path).expect("temp dir should be created");
        path
    }

    #[test]
    fn render_status_returns_error_without_snapshot() {
        let _guard = env_lock().lock().expect("env lock");
        let config_home_dir = unique_temp_dir("status-missing");
        env::set_var("KCMT_CONFIG_HOME", &config_home_dir);

        let err = render_status(Some(PathBuf::from("/tmp/repo")), false)
            .expect_err("missing snapshot should error");
        assert_eq!(err, "No kcmt run history found for this repository.");
    }

    #[test]
    fn render_status_supports_raw_and_summary_output() {
        let _guard = env_lock().lock().expect("env lock");
        let config_home_dir = unique_temp_dir("status-present");
        env::set_var("KCMT_CONFIG_HOME", &config_home_dir);

        let repo = unique_temp_dir("repo");
        let snapshot = json!({
            "repo_path": repo.display().to_string(),
            "timestamp": "2026-03-16T00:00:00Z",
            "duration_seconds": 1.25,
            "summary": "Successfully completed 1 commits.",
            "counts": {
                "commit_success": 1,
                "commit_failure": 0
            }
        });
        let path = snapshot_path(&repo);
        fs::create_dir_all(path.parent().expect("snapshot parent"))
            .expect("snapshot dir should be created");
        fs::write(&path, serde_json::to_string_pretty(&snapshot).expect("snapshot json"))
            .expect("snapshot should be written");

        let raw = render_status(Some(repo.clone()), true).expect("raw status");
        assert!(raw.contains("\"summary\": \"Successfully completed 1 commits.\""));

        let summary = render_status(Some(repo.clone()), false).expect("summary status");
        assert!(summary.contains("kcmt status"));
        assert!(summary.contains("Summary"));
        assert!(summary.contains("Commit status"));
        assert!(summary.contains("Success: 1"));
        assert!(state_dir(&repo).starts_with(&config_home_dir));
    }
}
