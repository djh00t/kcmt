use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Duration;
use std::time::Instant;

use kcmt_core::config::loader::{load_config, ConfigOverrides};
use kcmt_core::error::KcmtError;
use kcmt_core::error::Result;
use kcmt_core::git::commit_file::commit_file;
use kcmt_core::git::repo::{CliGitRepository, GitRepository};
use kcmt_core::message::{build_prompt, sanitize_commit_output};
use kcmt_provider::clients::{
    AnthropicClient, GitHubModelsClient, OpenAiBatchJob, OpenAiClient, ProviderMessage, XaiClient,
};
use kcmt_provider::transport::{AsyncTransport, RetryPolicy};
use serde_json::json;

use super::history::snapshot_path;

#[derive(Debug, Clone, PartialEq, Eq)]
struct StatusEntry {
    code: String,
    path: String,
}

impl StatusEntry {
    fn is_deletion(&self) -> bool {
        self.code.contains('D')
    }
}

#[derive(Debug, Clone)]
struct WorkflowCommit {
    file_path: String,
    message: String,
    commit_hash: Option<String>,
    is_deletion: bool,
}

#[derive(Debug, Clone)]
struct PreparedEntry {
    entry: StatusEntry,
    message: String,
}

#[derive(Debug, Clone)]
struct ProviderCandidate {
    provider: String,
    model: String,
    endpoint: String,
    api_key_env: String,
}

#[derive(Debug, Clone)]
struct PushOutcome {
    pushed: bool,
    state: &'static str,
    errors: Vec<String>,
}

impl PushOutcome {
    fn not_triggered() -> Self {
        Self {
            pushed: false,
            state: "not triggered",
            errors: Vec::new(),
        }
    }
}

#[derive(Debug, Clone)]
struct WorkflowStageTiming {
    stage: &'static str,
    duration_ms: f64,
    items: usize,
}

#[derive(Debug, Clone, Default)]
struct WorkflowTelemetry {
    stages: Vec<WorkflowStageTiming>,
}

impl WorkflowTelemetry {
    fn record_since(&mut self, stage: &'static str, start: Instant, items: usize) {
        self.stages.push(WorkflowStageTiming {
            stage,
            duration_ms: start.elapsed().as_secs_f64() * 1000.0,
            items,
        });
    }
}

pub fn run_file_workflow(
    repo_path: PathBuf,
    file_path: &str,
    overrides: ConfigOverrides,
) -> Result<String> {
    let config = load_config(&repo_path, &overrides)?;
    let repo = CliGitRepository::from_path(&repo_path);
    let mut telemetry = WorkflowTelemetry::default();
    let status_start = Instant::now();
    let entries = parse_status_entries(&repo.status_porcelain()?);
    telemetry.record_since("status_scan", status_start, entries.len());
    let Some(entry) = entries.into_iter().find(|entry| entry.path == file_path) else {
        return Ok("No changes detected in the specified file.\n".to_string());
    };

    run_entries_workflow(repo_path, &config, vec![entry], telemetry)
}

pub fn run_oneshot_workflow(repo_path: PathBuf, overrides: ConfigOverrides) -> Result<String> {
    let config = load_config(&repo_path, &overrides)?;
    let repo = CliGitRepository::from_path(&repo_path);
    let mut telemetry = WorkflowTelemetry::default();
    let status_start = Instant::now();
    let entries = parse_status_entries(&repo.status_porcelain()?);
    telemetry.record_since("status_scan", status_start, entries.len());
    if entries.is_empty() {
        return Ok("No changes to commit.\n".to_string());
    }

    run_entries_workflow(repo_path, &config, entries, telemetry)
}

fn run_entries_workflow(
    repo_path: PathBuf,
    config: &kcmt_core::model::WorkflowConfig,
    entries: Vec<StatusEntry>,
    mut telemetry: WorkflowTelemetry,
) -> Result<String> {
    let mut lines = Vec::new();
    let mut commits = Vec::new();

    let prepare_start = Instant::now();
    let prepared = prepare_messages_for_entries(&repo_path, entries, config)?;
    telemetry.record_since("llm_wait", prepare_start, prepared.len());

    let commit_start = Instant::now();
    for prepared_entry in prepared {
        let entry = prepared_entry.entry;
        let message = prepared_entry.message;
        commit_file(&repo_path, &entry.path, &message, false)?;
        let is_deletion = entry.is_deletion();

        lines.push(format!("✓ {}", entry.path));
        lines.push(format!("  {message}"));

        let commit_hash = recent_commit_hash(&repo_path)?;
        if let Some(hash) = &commit_hash {
            lines.push(format!("  {}", truncate_hash(hash)));
        }

        commits.push(WorkflowCommit {
            file_path: entry.path,
            message,
            commit_hash,
            is_deletion,
        });
    }
    telemetry.record_since("commit", commit_start, commits.len());

    let push_start = Instant::now();
    let push_outcome = auto_push_if_configured(&repo_path, config, !commits.is_empty());
    telemetry.record_since("push", push_start, usize::from(push_outcome.pushed));
    match push_outcome.state {
        "pushed" => lines.push("Auto-push: pushed".to_string()),
        "failed" => lines.push("Auto-push: failed".to_string()),
        _ => {}
    }

    persist_run_snapshot(&repo_path, config, &commits, &push_outcome, &mut telemetry)?;

    let mut output = lines.join("\n");
    output.push('\n');
    Ok(output)
}

fn parse_status_entries(status: &str) -> Vec<StatusEntry> {
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
                Some(StatusEntry { code, path })
            }
        })
        .collect()
}

fn commit_message_for_entry(
    repo_path: &Path,
    entry: &StatusEntry,
    config: &kcmt_core::model::WorkflowConfig,
) -> Result<String> {
    if entry.is_deletion() {
        Ok(deletion_commit_message(
            &entry.path,
            config.max_commit_length,
        ))
    } else if let Ok(raw_response) = env::var("KCMT_PROVIDER_RESPONSE") {
        sanitize_commit_output(&raw_response)
            .map(|message| limit_subject(message, config.max_commit_length))
            .map_err(KcmtError::Message)
    } else if runtime_benchmark_enabled() {
        Ok(heuristic_commit_message(
            &entry.path,
            config.max_commit_length,
        ))
    } else if configured_api_key(config).is_some() {
        invoke_provider_with_fallback(repo_path, entry, config)
    } else if local_synthesis_enabled() {
        Ok(heuristic_commit_message(
            &entry.path,
            config.max_commit_length,
        ))
    } else {
        Err(KcmtError::Message(format!(
            "No API key available for {}; run `kcmt --configure` or set {}",
            config.provider, config.api_key_env
        )))
    }
}

fn local_synthesis_enabled() -> bool {
    env_truthy("KCMT_ALLOW_LOCAL_SYNTHESIS") || runtime_benchmark_enabled()
}

fn runtime_benchmark_enabled() -> bool {
    env_truthy("KCMT_RUNTIME_BENCHMARK")
}

fn env_truthy(key: &str) -> bool {
    env::var(key)
        .map(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(false)
}

fn prepare_messages_for_entries(
    repo_path: &Path,
    entries: Vec<StatusEntry>,
    config: &kcmt_core::model::WorkflowConfig,
) -> Result<Vec<PreparedEntry>> {
    if should_use_openai_batch(config) {
        if let Some(api_key) = configured_api_key(config) {
            return prepare_openai_batch_messages(repo_path, entries, config, &api_key);
        }
    }

    entries
        .into_iter()
        .map(|entry| {
            let message = commit_message_for_entry(repo_path, &entry, config)?;
            Ok(PreparedEntry { entry, message })
        })
        .collect()
}

fn should_use_openai_batch(config: &kcmt_core::model::WorkflowConfig) -> bool {
    config.provider == "openai" && config.use_batch
}

fn prepare_openai_batch_messages(
    repo_path: &Path,
    entries: Vec<StatusEntry>,
    config: &kcmt_core::model::WorkflowConfig,
    api_key: &str,
) -> Result<Vec<PreparedEntry>> {
    let mut deletions = Vec::new();
    let mut batch_entries = Vec::new();
    for entry in entries {
        if entry.is_deletion() {
            let message = deletion_commit_message(&entry.path, config.max_commit_length);
            deletions.push(PreparedEntry { entry, message });
        } else {
            batch_entries.push(entry);
        }
    }
    if batch_entries.is_empty() {
        return Ok(deletions);
    }

    let system = "You generate strictly valid Conventional Commit messages.";
    let mut jobs = Vec::new();
    for entry in &batch_entries {
        let diff = diff_for_entry(repo_path, entry)?;
        let context = format!("File: {}", entry.path);
        let prompt = build_prompt(&diff, &context, "conventional");
        jobs.push(OpenAiBatchJob {
            custom_id: entry.path.clone(),
            messages: vec![
                ProviderMessage::system(system),
                ProviderMessage::user(prompt),
            ],
        });
    }

    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|err| {
            KcmtError::Message(format!("failed to initialize provider runtime: {err}"))
        })?;
    let transport = AsyncTransport::new(
        Duration::from_secs(60),
        RetryPolicy {
            max_attempts: 3,
            base_backoff: Duration::from_millis(250),
        },
    )
    .map_err(|err| KcmtError::Message(format!("failed to initialize provider transport: {err}")))?;
    let batch_model = config
        .batch_model
        .as_deref()
        .filter(|value| !value.trim().is_empty())
        .unwrap_or(&config.model);
    let results = runtime
        .block_on(OpenAiClient::invoke_batch(
            &transport,
            &config.llm_endpoint,
            api_key,
            batch_model,
            &jobs,
            Duration::from_secs(config.batch_timeout_seconds),
            Duration::from_millis(50),
        ))
        .map_err(|err| KcmtError::Message(format!("provider batch request failed: {err}")))?;
    let mut messages_by_id = results
        .into_iter()
        .map(|result| {
            sanitize_commit_output(&result.content)
                .map(|message| limit_subject(message, config.max_commit_length))
                .map(|message| (result.custom_id, message))
                .map_err(KcmtError::Message)
        })
        .collect::<Result<std::collections::BTreeMap<_, _>>>()?;

    let mut prepared = Vec::new();
    for entry in batch_entries {
        let message = messages_by_id
            .remove(&entry.path)
            .ok_or_else(|| KcmtError::Message(format!("batch response missing {}", entry.path)))?;
        prepared.push(PreparedEntry { entry, message });
    }
    prepared.extend(deletions);
    Ok(prepared)
}

fn configured_api_key(config: &kcmt_core::model::WorkflowConfig) -> Option<String> {
    env::var(&config.api_key_env)
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

fn provider_candidates(config: &kcmt_core::model::WorkflowConfig) -> Vec<ProviderCandidate> {
    let mut candidates = Vec::new();
    candidates.push(ProviderCandidate {
        provider: config.provider.clone(),
        model: config.model.clone(),
        endpoint: config.llm_endpoint.clone(),
        api_key_env: config.api_key_env.clone(),
    });
    for preference in config.model_priority.iter().skip(1) {
        if preference.provider.trim().is_empty() || preference.model.trim().is_empty() {
            continue;
        }
        let entry = config.providers.get(&preference.provider);
        let endpoint = entry
            .and_then(|entry| entry.endpoint.clone())
            .unwrap_or_else(|| default_provider_endpoint(&preference.provider).to_string());
        let api_key_env = entry
            .and_then(|entry| entry.api_key_env.clone())
            .unwrap_or_else(|| default_provider_api_key_env(&preference.provider).to_string());
        candidates.push(ProviderCandidate {
            provider: preference.provider.clone(),
            model: preference.model.clone(),
            endpoint,
            api_key_env,
        });
    }
    let mut seen = std::collections::BTreeSet::new();
    candidates
        .into_iter()
        .filter(|candidate| {
            seen.insert(format!(
                "{}:{}:{}",
                candidate.provider, candidate.model, candidate.api_key_env
            ))
        })
        .collect()
}

fn default_provider_endpoint(provider: &str) -> &'static str {
    match provider {
        "anthropic" => "https://api.anthropic.com/v1",
        "xai" => "https://api.x.ai/v1",
        "github" => "https://models.github.ai/inference",
        _ => "https://api.openai.com/v1",
    }
}

fn default_provider_api_key_env(provider: &str) -> &'static str {
    match provider {
        "anthropic" => "ANTHROPIC_API_KEY",
        "xai" => "XAI_API_KEY",
        "github" => "GITHUB_TOKEN",
        _ => "OPENAI_API_KEY",
    }
}

fn api_key_for_env(api_key_env: &str) -> Option<String> {
    env::var(api_key_env)
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

fn invoke_provider_with_fallback(
    repo_path: &Path,
    entry: &StatusEntry,
    config: &kcmt_core::model::WorkflowConfig,
) -> Result<String> {
    let mut last_error = None;
    for candidate in provider_candidates(config) {
        let Some(api_key) = api_key_for_env(&candidate.api_key_env) else {
            continue;
        };
        match invoke_provider_candidate(repo_path, entry, &candidate, &api_key).and_then(|raw| {
            sanitize_commit_output(&raw)
                .map(|message| limit_subject(message, config.max_commit_length))
                .map_err(KcmtError::Message)
        }) {
            Ok(message) => return Ok(message),
            Err(err) => last_error = Some(err),
        }
    }
    Err(last_error.unwrap_or_else(|| {
        KcmtError::Message("LLM unavailable; no providers succeeded".to_string())
    }))
}

fn invoke_provider_candidate(
    repo_path: &Path,
    entry: &StatusEntry,
    candidate: &ProviderCandidate,
    api_key: &str,
) -> Result<String> {
    let diff = diff_for_entry(repo_path, entry)?;
    let context = format!("File: {}", entry.path);
    let prompt = build_prompt(&diff, &context, "conventional");
    let system = "You generate strictly valid Conventional Commit messages.";
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|err| {
            KcmtError::Message(format!("failed to initialize provider runtime: {err}"))
        })?;
    let transport = AsyncTransport::new(
        Duration::from_secs(60),
        RetryPolicy {
            max_attempts: 3,
            base_backoff: Duration::from_millis(250),
        },
    )
    .map_err(|err| KcmtError::Message(format!("failed to initialize provider transport: {err}")))?;
    runtime
        .block_on(async {
            match candidate.provider.as_str() {
                "openai" => {
                    let messages = vec![
                        ProviderMessage::system(system),
                        ProviderMessage::user(prompt),
                    ];
                    OpenAiClient::invoke_chat(
                        &transport,
                        &candidate.endpoint,
                        api_key,
                        &candidate.model,
                        &messages,
                    )
                    .await
                }
                "xai" => {
                    let messages = vec![
                        ProviderMessage::system(system),
                        ProviderMessage::user(prompt),
                    ];
                    XaiClient::invoke_chat(
                        &transport,
                        &candidate.endpoint,
                        api_key,
                        &candidate.model,
                        &messages,
                    )
                    .await
                }
                "github" => {
                    let messages = vec![
                        ProviderMessage::system(system),
                        ProviderMessage::user(prompt),
                    ];
                    GitHubModelsClient::invoke_chat(
                        &transport,
                        &candidate.endpoint,
                        api_key,
                        &candidate.model,
                        &messages,
                    )
                    .await
                }
                "anthropic" => {
                    AnthropicClient::invoke_messages(
                        &transport,
                        &candidate.endpoint,
                        api_key,
                        &candidate.model,
                        system,
                        &prompt,
                    )
                    .await
                }
                other => Err(anyhow::anyhow!("unsupported provider: {other}")),
            }
        })
        .map_err(|err| KcmtError::Message(format!("provider request failed: {err}")))
}

fn diff_for_entry(repo_path: &Path, entry: &StatusEntry) -> Result<String> {
    let output = Command::new("git")
        .current_dir(repo_path)
        .args(["diff", "--", &entry.path])
        .output()?;
    if output.status.success() {
        let diff = String::from_utf8_lossy(&output.stdout).to_string();
        if !diff.trim().is_empty() {
            return Ok(diff);
        }
    }

    let path = repo_path.join(&entry.path);
    match fs::read_to_string(&path) {
        Ok(content) => Ok(format!(
            "New or changed file: {}\n\n{}",
            entry.path, content
        )),
        Err(_) => Ok(format!("Changed file: {}", entry.path)),
    }
}

fn deletion_commit_message(file_path: &str, max_commit_length: usize) -> String {
    let scope = sanitize_deletion_scope(file_path);
    let mut message = format!("chore({scope}): file deleted");
    if message.len() > max_commit_length {
        message.truncate(max_commit_length);
    }
    message
}

fn limit_subject(message: String, max_commit_length: usize) -> String {
    let mut lines = message.lines();
    let Some(subject) = lines.next() else {
        return message;
    };
    if subject.len() <= max_commit_length {
        return message;
    }

    let mut limited = subject.to_string();
    limited.truncate(max_commit_length);
    let body = lines.collect::<Vec<_>>().join("\n");
    if body.trim().is_empty() {
        limited
    } else {
        format!("{limited}\n{body}")
    }
}

fn sanitize_deletion_scope(file_path: &str) -> String {
    let sanitized: String = file_path
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || matches!(ch, '_' | '-') {
                ch
            } else {
                '-'
            }
        })
        .collect();
    sanitized.trim_matches('-').to_string()
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

fn auto_push_if_configured(
    repo_path: &Path,
    config: &kcmt_core::model::WorkflowConfig,
    any_success: bool,
) -> PushOutcome {
    if !any_success || !config.auto_push {
        return PushOutcome::not_triggered();
    }

    if !has_origin_remote(repo_path) {
        return PushOutcome {
            pushed: false,
            state: "skipped",
            errors: Vec::new(),
        };
    }

    match push_current_branch(repo_path) {
        Ok(()) => PushOutcome {
            pushed: true,
            state: "pushed",
            errors: Vec::new(),
        },
        Err(err) => PushOutcome {
            pushed: false,
            state: "failed",
            errors: vec![format!("Auto-push failed: {err}")],
        },
    }
}

fn has_origin_remote(repo_path: &Path) -> bool {
    Command::new("git")
        .current_dir(repo_path)
        .args(["config", "--get", "remote.origin.url"])
        .output()
        .map(|output| output.status.success() && !output.stdout.is_empty())
        .unwrap_or(false)
}

fn push_current_branch(repo_path: &Path) -> Result<()> {
    let branch_output = Command::new("git")
        .current_dir(repo_path)
        .args(["rev-parse", "--abbrev-ref", "HEAD"])
        .output()?;
    if !branch_output.status.success() {
        return Err(KcmtError::Message(format!(
            "git rev-parse failed: {}",
            String::from_utf8_lossy(&branch_output.stderr).trim()
        )));
    }

    let branch = String::from_utf8_lossy(&branch_output.stdout)
        .trim()
        .to_string();
    if branch.is_empty() || branch == "HEAD" {
        return Err(KcmtError::Message(
            "cannot auto-push from detached HEAD".to_string(),
        ));
    }

    let push_output = Command::new("git")
        .current_dir(repo_path)
        .args(["push", "origin", &branch])
        .output()?;
    if push_output.status.success() {
        Ok(())
    } else {
        Err(KcmtError::Message(
            String::from_utf8_lossy(&push_output.stderr)
                .trim()
                .to_string(),
        ))
    }
}

fn truncate_hash(hash: &str) -> &str {
    let max = 8.min(hash.len());
    &hash[..max]
}

fn persist_run_snapshot(
    repo_path: &Path,
    config: &kcmt_core::model::WorkflowConfig,
    commits: &[WorkflowCommit],
    push_outcome: &PushOutcome,
    telemetry: &mut WorkflowTelemetry,
) -> Result<()> {
    let snapshot_start = Instant::now();
    let file_commits: Vec<_> = commits
        .iter()
        .filter(|commit| !commit.is_deletion)
        .collect();
    let deletions: Vec<_> = commits.iter().filter(|commit| commit.is_deletion).collect();
    let commit_success = file_commits.len();
    let deletions_success = deletions.len();
    let overall_success = commits.len();
    let subjects: Vec<&str> = commits
        .iter()
        .map(|commit| commit.message.as_str())
        .collect();
    let summary = format!(
        "Successfully completed {overall_success} commits. Committed {overall_success} file change(s)"
    );
    telemetry.record_since("snapshot", snapshot_start, 1);

    let snapshot = json!({
        "schema_version": 1,
        "timestamp": "",
        "repo_path": repo_path.display().to_string(),
        "provider": config.provider,
        "model": config.model,
        "endpoint": config.llm_endpoint,
        "config": {
            "provider": config.provider,
            "model": config.model,
            "endpoint": config.llm_endpoint,
            "api_key_env": config.api_key_env,
            "git_repo_path": config.git_repo_path,
            "max_commit_length": config.max_commit_length,
            "auto_push": config.auto_push,
            "use_batch": config.use_batch,
            "batch_model": config.batch_model,
            "batch_timeout_seconds": config.batch_timeout_seconds
        },
        "batch": {
            "use_batch": config.use_batch,
            "batch_model": config.batch_model,
            "batch_timeout_seconds": config.batch_timeout_seconds
        },
        "duration_seconds": 0.0,
        "rate_commits_per_sec": 0.0,
        "counts": {
            "files_total": commits.len(),
            "prepared_total": commits.len(),
            "processed_total": commits.len(),
            "prepared_failures": 0,
            "commit_success": commit_success,
            "commit_failure": 0,
            "deletions_total": deletions.len(),
            "deletions_success": deletions_success,
            "deletions_failure": 0,
            "overall_success": overall_success,
            "overall_failure": 0,
            "errors": push_outcome.errors.len()
        },
        "pushed": push_outcome.pushed,
        "auto_push_state": push_outcome.state,
        "summary": summary,
        "errors": &push_outcome.errors,
        "commits": file_commits.iter().map(|commit| json!({
            "success": true,
            "commit_hash": commit.commit_hash,
            "message": commit.message,
            "error": null,
            "file_path": commit.file_path
        })).collect::<Vec<_>>(),
        "deletions": deletions.iter().map(|commit| json!({
            "success": true,
            "commit_hash": commit.commit_hash,
            "message": commit.message,
            "error": null,
            "file_path": commit.file_path
        })).collect::<Vec<_>>(),
        "subjects": subjects,
        "stats": {},
        "telemetry": {
            "schema_version": 1,
            "stages": telemetry.stages.iter().map(|stage| json!({
                "stage": stage.stage,
                "duration_ms": stage.duration_ms,
                "items": stage.items
            })).collect::<Vec<_>>()
        }
    });

    let path = snapshot_path(repo_path);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let rendered = serde_json::to_string_pretty(&snapshot)
        .map_err(|err| KcmtError::Message(err.to_string()))?;
    fs::write(path, rendered)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        deletion_commit_message, heuristic_commit_message, parse_status_entries, StatusEntry,
    };

    #[test]
    fn parses_porcelain_entries() {
        let entries = parse_status_entries(" M alpha.py\n?? beta.py\n");

        assert_eq!(
            entries,
            vec![
                StatusEntry {
                    code: " M".to_string(),
                    path: "alpha.py".to_string()
                },
                StatusEntry {
                    code: "??".to_string(),
                    path: "beta.py".to_string()
                },
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
    fn builds_python_compatible_deletion_message() {
        assert_eq!(
            deletion_commit_message("delete_me.txt", 72),
            "chore(delete_me-txt): file deleted"
        );
    }
}
