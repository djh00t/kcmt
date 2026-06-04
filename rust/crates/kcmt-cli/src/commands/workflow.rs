use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::thread;
use std::time::Instant;

use kcmt_core::config::loader::{load_config, ConfigOverrides};
use kcmt_core::error::KcmtError;
use kcmt_core::error::Result;
use kcmt_core::git::commit_file::commit_file;
use kcmt_core::git::repo::{CliGitRepository, GitRepository};
use kcmt_core::workflow::quality::is_conventional_commit;
use kcmt_core::workflow::telemetry::{StageTiming, WorkflowTelemetry};
use kcmt_provider::clients::openai::OpenAiCommitClient;
use serde_json::json;

use super::history::snapshot_path;

const MAX_LLM_EVIDENCE_BYTES: usize = 24_000;

struct PreparedChange {
    file_path: String,
    evidence: String,
}

pub fn run_file_workflow(repo_path: PathBuf, file_path: &str) -> Result<String> {
    let workflow_start = Instant::now();
    let mut telemetry = WorkflowTelemetry::new("rust", "workflow");

    let config = load_config(&repo_path, &ConfigOverrides::default())?;
    let repo = CliGitRepository::from_path(&repo_path);
    let status_start = elapsed_ms(&workflow_start);
    let entries = parse_status_entries(&repo.status_porcelain()?);
    telemetry.record(StageTiming::completed(
        "status_scan",
        status_start,
        elapsed_ms(&workflow_start),
        None,
    ));
    run_file_workflow_from_entries(
        repo_path,
        file_path,
        &config,
        &entries,
        &workflow_start,
        &mut telemetry,
    )
}

fn run_file_workflow_from_entries(
    repo_path: PathBuf,
    file_path: &str,
    config: &kcmt_core::model::WorkflowConfig,
    entries: &[(String, String)],
    workflow_start: &Instant,
    telemetry: &mut WorkflowTelemetry,
) -> Result<String> {
    if !entries.iter().any(|(_, path)| path == file_path) {
        return Ok("No changes detected in the specified file.\n".to_string());
    }

    let file_status = entries
        .iter()
        .find(|(_, path)| path == file_path)
        .map(|(status, _)| status.as_str());
    let message = generate_commit_message(
        &repo_path,
        file_path,
        file_status,
        config,
        config.max_commit_length,
        workflow_start,
        telemetry,
    )?;
    let validation_start = elapsed_ms(&workflow_start);
    if !is_conventional_commit(&message) {
        return Err(KcmtError::Message(format!(
            "generated commit message is not conventional: {message}"
        )));
    }
    telemetry.record(StageTiming::completed(
        "response_validation",
        validation_start,
        elapsed_ms(&workflow_start),
        Some(file_path.to_string()),
    ));

    let commit_start = elapsed_ms(&workflow_start);
    commit_file(&repo_path, file_path, &message, false)?;
    telemetry.record(StageTiming::completed(
        "commit",
        commit_start,
        elapsed_ms(&workflow_start),
        Some(file_path.to_string()),
    ));

    let mut lines = vec![format!("✓ {file_path}"), format!("  {message}")];
    let commit_hash = recent_commit_hash(&repo_path)?;
    if let Some(hash) = &commit_hash {
        lines.push(format!("  {}", truncate_hash(&hash)));
    }
    push_if_configured(&repo_path, config, &workflow_start, telemetry)?;
    let snapshot_start = elapsed_ms(&workflow_start);
    telemetry.record(StageTiming::completed(
        "snapshot",
        snapshot_start,
        snapshot_start,
        None,
    ));
    persist_run_snapshot(
        &repo_path,
        &config,
        file_path,
        &message,
        commit_hash.as_deref(),
        &telemetry,
    )?;
    persist_telemetry_export(&telemetry)?;

    let mut output = lines.join("\n");
    output.push('\n');
    Ok(output)
}

fn generate_commit_message(
    repo_path: &Path,
    file_path: &str,
    file_status: Option<&str>,
    config: &kcmt_core::model::WorkflowConfig,
    max_commit_length: usize,
    workflow_start: &Instant,
    telemetry: &mut WorkflowTelemetry,
) -> Result<String> {
    let llm_like = fake_llm_response().is_some() || rust_llm_enabled();
    let evidence = if llm_like {
        let diff_start = elapsed_ms(workflow_start);
        let evidence = prepare_file_evidence(repo_path, file_path, file_status)?;
        telemetry.record(StageTiming::completed(
            "diff_preparation",
            diff_start,
            elapsed_ms(workflow_start),
            Some(file_path.to_string()),
        ));
        Some(evidence)
    } else {
        None
    };
    let enqueue_start = elapsed_ms(workflow_start);
    telemetry.record(StageTiming::completed(
        "llm_enqueue",
        enqueue_start,
        elapsed_ms(workflow_start),
        Some(file_path.to_string()),
    ));
    let wait_start = elapsed_ms(workflow_start);
    let message = if let Some(fake_response) = fake_llm_response() {
        fake_response
    } else if rust_llm_enabled() {
        let evidence = evidence.unwrap_or_else(|| format!("File: {file_path}"));
        generate_openai_commit_message(file_path, &evidence, config)?
    } else {
        heuristic_commit_message(file_path, max_commit_length)
    };
    telemetry.record(StageTiming::completed(
        "llm_wait",
        wait_start,
        elapsed_ms(workflow_start),
        Some(file_path.to_string()),
    ));
    Ok(message)
}

fn fake_llm_response() -> Option<String> {
    std::env::var("KCMT_FAKE_LLM_RESPONSE")
        .ok()
        .filter(|value| !value.trim().is_empty())
}

fn rust_llm_enabled() -> bool {
    std::env::var("KCMT_RUST_LLM")
        .map(|value| {
            matches!(
                value.to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(false)
}

fn generate_openai_commit_message(
    file_path: &str,
    evidence: &str,
    config: &kcmt_core::model::WorkflowConfig,
) -> Result<String> {
    let api_key = std::env::var(&config.api_key_env).map_err(|_| {
        KcmtError::Message(format!(
            "Rust LLM provider requires environment variable {}",
            config.api_key_env
        ))
    })?;
    let client = OpenAiCommitClient::new(&config.llm_endpoint, &config.model, api_key)
        .map_err(|err| KcmtError::Message(err.to_string()))?;
    let runtime = tokio::runtime::Runtime::new()
        .map_err(|err| KcmtError::Message(format!("failed to start Rust LLM runtime: {err}")))?;
    runtime
        .block_on(client.generate_commit_message(evidence, &format!("File: {file_path}")))
        .map_err(|err| KcmtError::Message(err.to_string()))
}

fn prepare_file_evidence(
    repo_path: &Path,
    file_path: &str,
    file_status: Option<&str>,
) -> Result<String> {
    let mut sections = Vec::new();
    sections.push(format!("file: {file_path}"));
    if let Some(status) = file_status {
        sections.push(format!("git status: {}", status.trim()));
    }
    if file_status.is_some_and(|status| status.contains('?')) {
        let path = repo_path.join(file_path);
        let content = fs::read_to_string(&path).map_err(|err| {
            KcmtError::Message(format!("failed to read untracked file {file_path}: {err}"))
        })?;
        sections.push(format!("new file content:\n{}", cap_evidence(&content)));
        return Ok(sections.join("\n\n"));
    }

    let working_diff = git_output(
        repo_path,
        &["diff", "--no-ext-diff", "--text", "--", file_path],
    )?;
    if !working_diff.trim().is_empty() {
        sections.push(format!(
            "working tree diff:\n{}",
            cap_evidence(&working_diff)
        ));
    }

    let staged_diff = git_output(
        repo_path,
        &[
            "diff",
            "--cached",
            "--no-ext-diff",
            "--text",
            "--",
            file_path,
        ],
    )?;
    if !staged_diff.trim().is_empty() {
        sections.push(format!("staged diff:\n{}", cap_evidence(&staged_diff)));
    }

    Ok(sections.join("\n\n"))
}

fn git_output(repo_path: &Path, args: &[&str]) -> Result<String> {
    let output = Command::new("git")
        .current_dir(repo_path)
        .args(args)
        .output()?;
    if output.status.success() {
        Ok(String::from_utf8_lossy(&output.stdout).to_string())
    } else {
        Err(KcmtError::Message(format!(
            "git {:?} failed: {}",
            args,
            String::from_utf8_lossy(&output.stderr).trim()
        )))
    }
}

fn cap_evidence(value: &str) -> String {
    if value.len() <= MAX_LLM_EVIDENCE_BYTES {
        return value.to_string();
    }

    let mut end = MAX_LLM_EVIDENCE_BYTES;
    while !value.is_char_boundary(end) {
        end -= 1;
    }
    format!(
        "{}\n[truncated to {} bytes for LLM prompt]",
        &value[..end],
        MAX_LLM_EVIDENCE_BYTES
    )
}

fn persist_telemetry_export(telemetry: &WorkflowTelemetry) -> Result<()> {
    let Ok(path) = std::env::var("KCMT_RUNTIME_TELEMETRY_PATH") else {
        return Ok(());
    };
    if path.trim().is_empty() {
        return Ok(());
    }
    let path = PathBuf::from(path);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let rendered =
        serde_json::to_string(telemetry).map_err(|err| KcmtError::Message(err.to_string()))?;
    fs::write(path, rendered)?;
    Ok(())
}

fn elapsed_ms(start: &Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1000.0
}

pub fn run_oneshot_workflow(repo_path: PathBuf) -> Result<String> {
    let workflow_start = Instant::now();
    let mut telemetry = WorkflowTelemetry::new("rust", "workflow");
    let config = load_config(&repo_path, &ConfigOverrides::default())?;
    let repo = CliGitRepository::from_path(&repo_path);
    let status_start = elapsed_ms(&workflow_start);
    let entries = parse_status_entries(&repo.status_porcelain()?);
    telemetry.record(StageTiming::completed(
        "status_scan",
        status_start,
        elapsed_ms(&workflow_start),
        None,
    ));
    if entries.is_empty() {
        return Ok("No changes to commit.\n".to_string());
    }

    let target = entries
        .iter()
        .find(|(status, _)| !status.contains('D'))
        .map(|(_, path)| path.clone())
        .unwrap_or_else(|| entries[0].1.clone());

    run_file_workflow_from_entries(
        repo_path,
        &target,
        &config,
        &entries,
        &workflow_start,
        &mut telemetry,
    )
}

pub fn run_batch_workflow(repo_path: PathBuf) -> Result<String> {
    let workflow_start = Instant::now();
    let mut telemetry = WorkflowTelemetry::new("rust", "workflow");
    let config = load_config(&repo_path, &ConfigOverrides::default())?;
    let repo = CliGitRepository::from_path(&repo_path);
    let status_start = elapsed_ms(&workflow_start);
    let entries = parse_status_entries(&repo.status_porcelain()?);
    telemetry.record(StageTiming::completed(
        "status_scan",
        status_start,
        elapsed_ms(&workflow_start),
        None,
    ));
    if entries.is_empty() {
        return Ok("No changes to commit.\n".to_string());
    }

    let prepare_start = elapsed_ms(&workflow_start);
    let prepared = prepare_batch_changes(&repo_path, &entries, &workflow_start, &mut telemetry)?;
    telemetry.record(StageTiming::completed(
        "time_to_all_llm_enqueued",
        prepare_start,
        elapsed_ms(&workflow_start),
        None,
    ));

    let messages =
        generate_batch_commit_messages(&prepared, &config, &workflow_start, &mut telemetry)?;

    let mut lines = Vec::new();
    let mut commits = Vec::new();
    for (change, message) in prepared.iter().zip(messages.iter()) {
        let validation_start = elapsed_ms(&workflow_start);
        if !is_conventional_commit(message) {
            return Err(KcmtError::Message(format!(
                "generated commit message is not conventional: {message}"
            )));
        }
        telemetry.record(StageTiming::completed(
            "response_validation",
            validation_start,
            elapsed_ms(&workflow_start),
            Some(change.file_path.clone()),
        ));

        let commit_start = elapsed_ms(&workflow_start);
        commit_file(&repo_path, &change.file_path, message, false)?;
        telemetry.record(StageTiming::completed(
            "commit",
            commit_start,
            elapsed_ms(&workflow_start),
            Some(change.file_path.clone()),
        ));
        let commit_hash = recent_commit_hash(&repo_path)?;
        lines.push(format!("✓ {}", change.file_path));
        lines.push(format!("  {message}"));
        if let Some(hash) = &commit_hash {
            lines.push(format!("  {}", truncate_hash(hash)));
        }
        commits.push((change.file_path.clone(), message.clone(), commit_hash));
    }

    push_if_configured(&repo_path, &config, &workflow_start, &mut telemetry)?;
    let snapshot_start = elapsed_ms(&workflow_start);
    telemetry.record(StageTiming::completed(
        "snapshot",
        snapshot_start,
        snapshot_start,
        None,
    ));
    persist_batch_snapshot(&repo_path, &config, &commits, &telemetry)?;
    persist_telemetry_export(&telemetry)?;

    let mut output = lines.join("\n");
    output.push('\n');
    Ok(output)
}

fn prepare_batch_changes(
    repo_path: &Path,
    entries: &[(String, String)],
    workflow_start: &Instant,
    telemetry: &mut WorkflowTelemetry,
) -> Result<Vec<PreparedChange>> {
    let batch_prepare_start = elapsed_ms(workflow_start);
    let mut handles = Vec::new();
    for (status, file_path) in entries.iter().filter(|(status, _)| !status.contains('D')) {
        let repo_path = repo_path.to_path_buf();
        let status = status.clone();
        let file_path = file_path.clone();
        let workflow_start = *workflow_start;
        handles.push(thread::spawn(move || {
            let start = elapsed_ms(&workflow_start);
            let evidence = prepare_file_evidence(&repo_path, &file_path, Some(&status))?;
            let end = elapsed_ms(&workflow_start);
            Ok::<_, KcmtError>((
                PreparedChange {
                    file_path,
                    evidence,
                },
                start,
                end,
            ))
        }));
    }

    let mut prepared = Vec::new();
    let mut first_enqueue_recorded = false;
    for handle in handles {
        let (change, start, end) = handle
            .join()
            .map_err(|_| KcmtError::Message("batch preparation worker panicked".to_string()))??;
        telemetry.record(StageTiming::completed(
            "diff_preparation",
            start,
            end,
            Some(change.file_path.clone()),
        ));
        if !first_enqueue_recorded {
            telemetry.record(StageTiming::completed(
                "time_to_first_llm_enqueue",
                batch_prepare_start,
                end,
                Some(change.file_path.clone()),
            ));
            first_enqueue_recorded = true;
        }
        prepared.push(change);
    }

    Ok(prepared)
}

fn generate_batch_commit_messages(
    prepared: &[PreparedChange],
    config: &kcmt_core::model::WorkflowConfig,
    workflow_start: &Instant,
    telemetry: &mut WorkflowTelemetry,
) -> Result<Vec<String>> {
    for change in prepared {
        let enqueue_start = elapsed_ms(workflow_start);
        telemetry.record(StageTiming::completed(
            "llm_enqueue",
            enqueue_start,
            elapsed_ms(workflow_start),
            Some(change.file_path.clone()),
        ));
    }

    if let Some(fake_response) = fake_llm_response() {
        let wait_start = elapsed_ms(workflow_start);
        let messages = prepared.iter().map(|_| fake_response.clone()).collect();
        telemetry.record(StageTiming::completed(
            "llm_wait",
            wait_start,
            elapsed_ms(workflow_start),
            None,
        ));
        return Ok(messages);
    }

    if rust_llm_enabled() {
        return generate_openai_batch_commit_messages(prepared, config, workflow_start, telemetry);
    }

    Ok(prepared
        .iter()
        .map(|change| heuristic_commit_message(&change.file_path, config.max_commit_length))
        .collect())
}

fn generate_openai_batch_commit_messages(
    prepared: &[PreparedChange],
    config: &kcmt_core::model::WorkflowConfig,
    workflow_start: &Instant,
    telemetry: &mut WorkflowTelemetry,
) -> Result<Vec<String>> {
    let api_key = std::env::var(&config.api_key_env).map_err(|_| {
        KcmtError::Message(format!(
            "Rust LLM provider requires environment variable {}",
            config.api_key_env
        ))
    })?;
    let client = OpenAiCommitClient::new(&config.llm_endpoint, &config.model, api_key)
        .map_err(|err| KcmtError::Message(err.to_string()))?;
    let runtime = tokio::runtime::Runtime::new()
        .map_err(|err| KcmtError::Message(format!("failed to start Rust LLM runtime: {err}")))?;
    let wait_start = elapsed_ms(workflow_start);
    let mut messages = vec![String::new(); prepared.len()];
    let results = runtime.block_on(async {
        let mut tasks = tokio::task::JoinSet::new();
        for (index, change) in prepared.iter().enumerate() {
            let client = client.clone();
            let evidence = change.evidence.clone();
            let context = format!("File: {}", change.file_path);
            tasks.spawn(async move {
                let message = client.generate_commit_message(&evidence, &context).await;
                (index, message)
            });
        }

        let mut results = Vec::new();
        while let Some(joined) = tasks.join_next().await {
            results.push(joined.map_err(|err| KcmtError::Message(err.to_string()))?);
        }
        Ok::<_, KcmtError>(results)
    })?;
    telemetry.record(StageTiming::completed(
        "llm_wait",
        wait_start,
        elapsed_ms(workflow_start),
        None,
    ));

    for (index, result) in results {
        messages[index] = result.map_err(|err| KcmtError::Message(err.to_string()))?;
    }
    Ok(messages)
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

fn push_if_configured(
    repo_path: &Path,
    config: &kcmt_core::model::WorkflowConfig,
    workflow_start: &Instant,
    telemetry: &mut WorkflowTelemetry,
) -> Result<()> {
    let push_start = elapsed_ms(workflow_start);
    if !config.auto_push || git_remote_names(repo_path)?.is_empty() {
        telemetry.record(StageTiming::completed(
            "push",
            push_start,
            elapsed_ms(workflow_start),
            None,
        ));
        return Ok(());
    }

    let output = Command::new("git")
        .current_dir(repo_path)
        .arg("push")
        .output()?;
    if output.status.success() {
        telemetry.record(StageTiming::completed(
            "push",
            push_start,
            elapsed_ms(workflow_start),
            None,
        ));
        Ok(())
    } else {
        let error = String::from_utf8_lossy(&output.stderr).trim().to_string();
        telemetry.record(StageTiming::failed(
            "push",
            push_start,
            elapsed_ms(workflow_start),
            &error,
        ));
        Err(KcmtError::Message(format!("git push failed: {error}")))
    }
}

fn git_remote_names(repo_path: &Path) -> Result<Vec<String>> {
    let output = Command::new("git")
        .current_dir(repo_path)
        .args(["remote"])
        .output()?;
    if !output.status.success() {
        return Ok(Vec::new());
    }
    Ok(String::from_utf8_lossy(&output.stdout)
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .map(ToOwned::to_owned)
        .collect())
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
    telemetry: &WorkflowTelemetry,
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
        "stats": {
            "telemetry": telemetry
        }
    });

    let path = snapshot_path(repo_path);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let rendered =
        serde_json::to_string(&snapshot).map_err(|err| KcmtError::Message(err.to_string()))?;
    fs::write(path, rendered)?;
    Ok(())
}

fn persist_batch_snapshot(
    repo_path: &Path,
    config: &kcmt_core::model::WorkflowConfig,
    commits: &[(String, String, Option<String>)],
    telemetry: &WorkflowTelemetry,
) -> Result<()> {
    let commit_rows: Vec<_> = commits
        .iter()
        .map(|(file_path, message, commit_hash)| {
            json!({
                "success": true,
                "commit_hash": commit_hash,
                "message": message,
                "error": null,
                "file_path": file_path
            })
        })
        .collect();
    let subjects: Vec<_> = commits
        .iter()
        .map(|(_, message, _)| message.clone())
        .collect();

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
            "files_total": commits.len(),
            "prepared_total": commits.len(),
            "processed_total": commits.len(),
            "prepared_failures": 0,
            "commit_success": commits.len(),
            "commit_failure": 0,
            "deletions_total": 0,
            "deletions_success": 0,
            "deletions_failure": 0,
            "overall_success": commits.len(),
            "overall_failure": 0,
            "errors": 0
        },
        "pushed": false,
        "summary": format!("Successfully completed {} commits. Committed {} file change(s)", commits.len(), commits.len()),
        "errors": [],
        "commits": commit_rows,
        "deletions": [],
        "subjects": subjects,
        "stats": {
            "telemetry": telemetry
        }
    });

    let path = snapshot_path(repo_path);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let rendered =
        serde_json::to_string(&snapshot).map_err(|err| KcmtError::Message(err.to_string()))?;
    fs::write(path, rendered)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        heuristic_commit_message, parse_status_entries, prepare_file_evidence,
        MAX_LLM_EVIDENCE_BYTES,
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
        let path = std::env::temp_dir().join(format!("kcmt-workflow-{label}-{nanos}-{suffix}"));
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

    fn init_repo(label: &str) -> PathBuf {
        let repo = unique_temp_dir(label);
        git(&repo, &["init", "-q"]);
        git(&repo, &["config", "user.name", "Tester"]);
        git(&repo, &["config", "user.email", "tester@example.com"]);
        repo
    }

    #[test]
    fn parses_porcelain_entries() {
        let entries = parse_status_entries(" M alpha.py\n?? beta.py\n");

        assert_eq!(
            entries,
            vec![
                (" M".to_string(), "alpha.py".to_string()),
                ("??".to_string(), "beta.py".to_string()),
            ]
        );
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
        assert_eq!(
            heuristic_commit_message("src/very_long_filename_here.py", 18).len(),
            18
        );
    }

    #[test]
    fn prepares_modified_file_diff_for_llm() {
        let repo = init_repo("modified-evidence");
        fs::write(repo.join("app.py"), "print('seed')\n").expect("seed write");
        git(&repo, &["add", "app.py"]);
        git(&repo, &["commit", "-m", "chore(repo): seed"]);
        fs::write(repo.join("app.py"), "print('changed')\n").expect("change write");

        let evidence = prepare_file_evidence(&repo, "app.py", Some(" M")).expect("file evidence");

        assert!(evidence.contains("file: app.py"));
        assert!(evidence.contains("git status: M"));
        assert!(evidence.contains("working tree diff:"));
        assert!(evidence.contains("-print('seed')"));
        assert!(evidence.contains("+print('changed')"));
    }

    #[test]
    fn prepares_untracked_file_content_for_llm() {
        let repo = init_repo("untracked-evidence");
        fs::write(repo.join("new.py"), "print('new')\n").expect("new write");

        let evidence = prepare_file_evidence(&repo, "new.py", Some("??")).expect("file evidence");

        assert!(evidence.contains("file: new.py"));
        assert!(evidence.contains("git status: ??"));
        assert!(evidence.contains("new file content:"));
        assert!(evidence.contains("print('new')"));
    }

    #[test]
    fn caps_large_llm_evidence() {
        let repo = init_repo("large-evidence");
        let content = "x".repeat(MAX_LLM_EVIDENCE_BYTES + 100);
        fs::write(repo.join("large.txt"), content).expect("large write");

        let evidence =
            prepare_file_evidence(&repo, "large.txt", Some("??")).expect("file evidence");

        assert!(evidence.contains("[truncated to 24000 bytes for LLM prompt]"));
        assert!(evidence.len() < MAX_LLM_EVIDENCE_BYTES + 200);
    }
}
