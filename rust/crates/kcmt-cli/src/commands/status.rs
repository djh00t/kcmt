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
    let schema_version = snapshot
        .get("schema_version")
        .and_then(Value::as_i64)
        .unwrap_or(1);
    let repo_display = snapshot
        .get("repo_path")
        .and_then(Value::as_str)
        .map(ToOwned::to_owned)
        .unwrap_or_else(|| repo_path.display().to_string());
    let provider = snapshot
        .get("provider")
        .and_then(Value::as_str)
        .unwrap_or("-");
    let model = snapshot.get("model").and_then(Value::as_str).unwrap_or("-");
    let timestamp = snapshot
        .get("timestamp")
        .and_then(Value::as_str)
        .unwrap_or("");
    let duration = snapshot
        .get("duration_seconds")
        .and_then(Value::as_f64)
        .unwrap_or(0.0);
    let summary = snapshot
        .get("summary")
        .and_then(Value::as_str)
        .unwrap_or("");
    let counts = snapshot.get("counts").and_then(Value::as_object);
    let files_total = counts
        .and_then(|counts| counts.get("files_total"))
        .and_then(Value::as_i64)
        .unwrap_or(0);
    let prepared_count = counts
        .and_then(|counts| counts.get("prepared_total"))
        .and_then(Value::as_i64)
        .unwrap_or(0);
    let processed_count = counts
        .and_then(|counts| counts.get("processed_total"))
        .and_then(Value::as_i64)
        .unwrap_or(0);
    let prepared_failures = counts
        .and_then(|counts| counts.get("prepared_failures"))
        .and_then(Value::as_i64)
        .unwrap_or(0);
    let success_count = counts
        .and_then(|counts| counts.get("commit_success"))
        .and_then(Value::as_i64)
        .unwrap_or(0);
    let failure_count = counts
        .and_then(|counts| counts.get("commit_failure"))
        .and_then(Value::as_i64)
        .unwrap_or(0);
    let deletions_success = counts
        .and_then(|counts| counts.get("deletions_success"))
        .and_then(Value::as_i64)
        .unwrap_or(0);
    let deletions_failure = counts
        .and_then(|counts| counts.get("deletions_failure"))
        .and_then(Value::as_i64)
        .unwrap_or(0);
    let overall_success = counts
        .and_then(|counts| counts.get("overall_success"))
        .and_then(Value::as_i64)
        .unwrap_or(success_count + deletions_success);
    let overall_failure = counts
        .and_then(|counts| counts.get("overall_failure"))
        .and_then(Value::as_i64)
        .unwrap_or(failure_count + deletions_failure);
    let rate = snapshot
        .get("rate_commits_per_sec")
        .and_then(Value::as_f64)
        .unwrap_or(0.0);
    let auto_push_state = snapshot
        .get("auto_push_state")
        .and_then(Value::as_str)
        .unwrap_or("not triggered");
    let pushed = snapshot
        .get("pushed")
        .and_then(Value::as_bool)
        .unwrap_or(false);

    let mut lines = Vec::new();
    lines.push(format!("kcmt status :: {repo_display}"));
    lines.push(format!("Schema version: {schema_version}"));
    lines.push(format!("Provider: {provider}  Model: {model}"));
    if timestamp.is_empty() {
        lines.push(format!("Duration {duration:.2}s  Rate {rate:.2}/s"));
    } else {
        lines.push(format!(
            "Run time {timestamp}  Duration {duration:.2}s  Rate {rate:.2}/s"
        ));
    }
    lines.push(format!(
        "Auto-push: {auto_push_state}{}",
        if pushed { " (pushed)" } else { "" }
    ));
    lines.push(String::new());
    lines.push("Summary".to_string());
    if summary.is_empty() {
        lines.push("No summary captured.".to_string());
    } else {
        lines.push(summary.to_string());
    }
    lines.push(String::new());
    lines.push("Preparation status".to_string());
    lines.push(format!("Files: {files_total}"));
    lines.push(format!("Prepared: {prepared_count}"));
    lines.push(format!("Processed: {processed_count}"));
    lines.push(format!("Prepare failures: {prepared_failures}"));
    lines.push(String::new());
    lines.push("Commit status".to_string());
    lines.push(format!("Success: {success_count}"));
    lines.push(format!("Failure: {failure_count}"));
    lines.push(format!("Deletions success: {deletions_success}"));
    lines.push(format!("Deletions failure: {deletions_failure}"));
    lines.push(format!("Overall success: {overall_success}"));
    lines.push(format!("Overall failure: {overall_failure}"));

    append_latest_commit(&mut lines, snapshot);
    append_errors(&mut lines, snapshot);
    append_stage_timings(&mut lines, snapshot);

    let mut output = lines.join("\n");
    output.push('\n');
    output
}

fn append_latest_commit(lines: &mut Vec<String>, snapshot: &Value) {
    let latest_subject = snapshot
        .get("subjects")
        .and_then(Value::as_array)
        .and_then(|subjects| subjects.last())
        .and_then(Value::as_str)
        .or_else(|| {
            snapshot
                .get("commits")
                .and_then(Value::as_array)
                .and_then(|commits| commits.last())
                .and_then(|commit| commit.get("message"))
                .and_then(Value::as_str)
        });
    if let Some(subject) = latest_subject {
        lines.push(String::new());
        lines.push("Latest commit".to_string());
        lines.push(subject.to_string());
    }
}

fn append_errors(lines: &mut Vec<String>, snapshot: &Value) {
    let errors = snapshot
        .get("errors")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .collect::<Vec<_>>();
    if errors.is_empty() {
        return;
    }
    lines.push(String::new());
    lines.push("Errors".to_string());
    for error in errors {
        lines.push(format!("- {error}"));
    }
}

fn append_stage_timings(lines: &mut Vec<String>, snapshot: &Value) {
    let Some(stages) = snapshot
        .get("telemetry")
        .and_then(|telemetry| telemetry.get("stages"))
        .and_then(Value::as_array)
    else {
        return;
    };
    if stages.is_empty() {
        return;
    }

    lines.push(String::new());
    lines.push("Telemetry stages".to_string());
    lines.push("Stage\tDuration ms\tItems".to_string());
    for stage in stages {
        let name = stage
            .get("stage")
            .and_then(Value::as_str)
            .unwrap_or("unknown");
        let duration_ms = stage
            .get("duration_ms")
            .and_then(Value::as_f64)
            .unwrap_or(0.0);
        let items = stage.get("items").and_then(Value::as_i64).unwrap_or(0);
        lines.push(format!("{name}\t{duration_ms:.1}\t{items}"));
    }
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
            "schema_version": 1,
            "repo_path": repo.display().to_string(),
            "timestamp": "2026-03-16T00:00:00Z",
            "duration_seconds": 1.25,
            "rate_commits_per_sec": 0.8,
            "provider": "openai",
            "model": "gpt-test",
            "summary": "Successfully completed 1 commits.",
            "counts": {
                "files_total": 1,
                "prepared_total": 1,
                "processed_total": 1,
                "prepared_failures": 0,
                "commit_success": 1,
                "commit_failure": 0,
                "deletions_success": 0,
                "deletions_failure": 0,
                "overall_success": 1,
                "overall_failure": 0
            },
            "pushed": false,
            "auto_push_state": "not triggered",
            "subjects": ["chore(repo): update tracked"],
            "errors": [],
            "telemetry": {
                "stages": [
                    {"stage": "status_scan", "duration_ms": 1.0, "items": 1},
                    {"stage": "workflow_total", "duration_ms": 2.0, "items": 1}
                ]
            },
        });
        let path = snapshot_path(&repo);
        fs::create_dir_all(path.parent().expect("snapshot parent"))
            .expect("snapshot dir should be created");
        fs::write(
            &path,
            serde_json::to_string_pretty(&snapshot).expect("snapshot json"),
        )
        .expect("snapshot should be written");

        let raw = render_status(Some(repo.clone()), true).expect("raw status");
        assert!(raw.contains("\"summary\": \"Successfully completed 1 commits.\""));

        let summary = render_status(Some(repo.clone()), false).expect("summary status");
        assert!(summary.contains("kcmt status"));
        assert!(summary.contains("Summary"));
        assert!(summary.contains("Provider: openai  Model: gpt-test"));
        assert!(summary.contains("Rate 0.80/s"));
        assert!(summary.contains("Auto-push: not triggered"));
        assert!(summary.contains("Preparation status"));
        assert!(summary.contains("Commit status"));
        assert!(summary.contains("Success: 1"));
        assert!(summary.contains("Overall success: 1"));
        assert!(summary.contains("Latest commit"));
        assert!(summary.contains("Telemetry stages"));
        assert!(state_dir(&repo).starts_with(&config_home_dir));
    }
}
