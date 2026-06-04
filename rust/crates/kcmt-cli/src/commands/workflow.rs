use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::thread;
use std::time::Duration;
use std::time::Instant;

use kcmt_core::config::loader::{load_config, ConfigOverrides};
use kcmt_core::error::KcmtError;
use kcmt_core::error::Result;
use kcmt_core::git::commit_file::{
    commit_file_with_staging, gix_commit_backend_enabled, recent_commit_hash, CommitStaging,
    GixCommitSession,
};
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

    fn is_new_file(&self) -> bool {
        self.code == "??" || self.code.contains('A')
    }

    fn commit_staging(&self) -> CommitStaging {
        if self.requires_staging_before_commit() {
            CommitStaging::StagePath
        } else {
            CommitStaging::DirectPath
        }
    }

    fn requires_staging_before_commit(&self) -> bool {
        self.code == "??" || self.code.contains('A') || self.code.contains('U')
    }
}

#[derive(Debug, Clone)]
struct WorkflowCommit {
    file_path: String,
    message: String,
    commit_hash: Option<String>,
    is_deletion: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WorkflowFailureStage {
    Prepare,
    Commit,
}

#[derive(Debug, Clone)]
struct WorkflowFailure {
    file_path: String,
    error: String,
    is_deletion: bool,
    stage: WorkflowFailureStage,
}

impl WorkflowFailure {
    fn prepare(entry: &StatusEntry, error: impl Into<String>) -> Self {
        Self {
            file_path: entry.path.clone(),
            error: error.into(),
            is_deletion: entry.is_deletion(),
            stage: WorkflowFailureStage::Prepare,
        }
    }

    fn commit(entry: &StatusEntry, error: impl Into<String>) -> Self {
        Self {
            file_path: entry.path.clone(),
            error: error.into(),
            is_deletion: entry.is_deletion(),
            stage: WorkflowFailureStage::Commit,
        }
    }
}

#[derive(Debug, Clone)]
struct PreparedEntry {
    entry: StatusEntry,
    message: String,
}

type PreparedOutcome = std::result::Result<PreparedEntry, WorkflowFailure>;

#[derive(Debug, Clone, Copy, Default)]
struct PreparationTelemetry {
    diff_preparation_ms: f64,
    diff_preparation_items: usize,
    llm_enqueue_ms: f64,
    llm_enqueue_items: usize,
}

#[derive(Debug)]
struct PreparationResult {
    outcomes: Vec<PreparedOutcome>,
    telemetry: PreparationTelemetry,
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

#[derive(Debug, Clone, Default)]
pub struct WorkflowOutputOptions {
    pub compact: bool,
    pub verbose: bool,
    pub profile_startup: bool,
    pub startup_stages: Vec<WorkflowStageTiming>,
}

impl WorkflowOutputOptions {
    pub fn record_startup_stage(&mut self, stage: &'static str, duration_ms: f64, items: usize) {
        self.startup_stages.push(WorkflowStageTiming {
            stage,
            duration_ms,
            items,
        });
    }
}

#[derive(Debug, Clone, Copy)]
pub struct WorkflowStageTiming {
    pub stage: &'static str,
    pub duration_ms: f64,
    pub items: usize,
}

#[derive(Debug, Clone, Default)]
struct WorkflowTelemetry {
    stages: Vec<WorkflowStageTiming>,
    prepare_workers: usize,
}

impl WorkflowTelemetry {
    fn from_startup_stages(stages: Vec<WorkflowStageTiming>) -> Self {
        Self {
            stages,
            prepare_workers: 0,
        }
    }

    fn record_since(&mut self, stage: &'static str, start: Instant, items: usize) {
        self.stages.push(WorkflowStageTiming {
            stage,
            duration_ms: start.elapsed().as_secs_f64() * 1000.0,
            items,
        });
    }

    fn record_duration(&mut self, stage: &'static str, duration_ms: f64, items: usize) {
        self.stages.push(WorkflowStageTiming {
            stage,
            duration_ms,
            items,
        });
    }

    fn record_empty(&mut self, stage: &'static str) {
        self.stages.push(WorkflowStageTiming {
            stage,
            duration_ms: 0.0,
            items: 0,
        });
    }
}

pub fn run_file_workflow(
    repo_path: PathBuf,
    file_path: &str,
    overrides: ConfigOverrides,
    output_options: WorkflowOutputOptions,
) -> Result<String> {
    let workflow_start = Instant::now();
    let config_start = Instant::now();
    let config = load_config(&repo_path, &overrides)?;
    let mut telemetry =
        WorkflowTelemetry::from_startup_stages(output_options.startup_stages.clone());
    telemetry.record_since("config_load", config_start, 1);
    let repo = CliGitRepository::from_path(&repo_path);
    let status_start = Instant::now();
    let entries = parse_status_entries(&repo.status_porcelain_for_path(file_path)?);
    telemetry.record_since("status_scan", status_start, entries.len());
    let Some(entry) = entries.into_iter().find(|entry| entry.path == file_path) else {
        return Ok("No changes detected in the specified file.\n".to_string());
    };

    run_entries_workflow(
        repo_path,
        &config,
        vec![entry],
        telemetry,
        max_attempts_from_retries(overrides.max_retries),
        overrides.prepare_workers,
        output_options,
        workflow_start,
    )
}

pub fn run_oneshot_workflow(
    repo_path: PathBuf,
    overrides: ConfigOverrides,
    output_options: WorkflowOutputOptions,
) -> Result<String> {
    let workflow_start = Instant::now();
    let config_start = Instant::now();
    let config = load_config(&repo_path, &overrides)?;
    let mut telemetry =
        WorkflowTelemetry::from_startup_stages(output_options.startup_stages.clone());
    telemetry.record_since("config_load", config_start, 1);
    let repo = CliGitRepository::from_path(&repo_path);
    let status_start = Instant::now();
    let mut entries = parse_status_entries(&repo.status_porcelain()?);
    if let Some(limit) = overrides.file_limit.filter(|limit| *limit > 0) {
        entries.truncate(limit);
    }
    telemetry.record_since("status_scan", status_start, entries.len());
    if entries.is_empty() {
        return Ok("No changes to commit.\n".to_string());
    }

    let entry = select_oneshot_entry(entries).expect("entries were checked above");
    run_entries_workflow(
        repo_path,
        &config,
        vec![entry],
        telemetry,
        max_attempts_from_retries(overrides.max_retries),
        overrides.prepare_workers,
        output_options,
        workflow_start,
    )
}

pub fn run_default_workflow(
    repo_path: PathBuf,
    overrides: ConfigOverrides,
    output_options: WorkflowOutputOptions,
) -> Result<String> {
    let workflow_start = Instant::now();
    let config_start = Instant::now();
    let config = load_config(&repo_path, &overrides)?;
    let mut telemetry =
        WorkflowTelemetry::from_startup_stages(output_options.startup_stages.clone());
    telemetry.record_since("config_load", config_start, 1);
    let repo = CliGitRepository::from_path(&repo_path);
    let status_start = Instant::now();
    let mut entries = parse_status_entries(&repo.status_porcelain()?);
    if let Some(limit) = overrides.file_limit.filter(|limit| *limit > 0) {
        entries.truncate(limit);
    }
    telemetry.record_since("status_scan", status_start, entries.len());
    if entries.is_empty() {
        return Ok("No changes to commit.\n".to_string());
    }

    run_entries_workflow(
        repo_path,
        &config,
        entries,
        telemetry,
        max_attempts_from_retries(overrides.max_retries),
        overrides.prepare_workers,
        output_options,
        workflow_start,
    )
}

fn select_oneshot_entry(entries: Vec<StatusEntry>) -> Option<StatusEntry> {
    entries
        .iter()
        .position(|entry| !entry.is_deletion())
        .map(|index| entries[index].clone())
        .or_else(|| entries.into_iter().next())
}

fn run_entries_workflow(
    repo_path: PathBuf,
    config: &kcmt_core::model::WorkflowConfig,
    entries: Vec<StatusEntry>,
    mut telemetry: WorkflowTelemetry,
    max_attempts: usize,
    prepare_workers_override: Option<usize>,
    output_options: WorkflowOutputOptions,
    workflow_start: Instant,
) -> Result<String> {
    let total_entries = entries.len();
    let mut commits = Vec::new();
    let mut failures = Vec::new();

    let prepare_workers = select_prepare_workers(entries.len(), prepare_workers_override);
    telemetry.prepare_workers = prepare_workers;
    let wait_start = Instant::now();
    let preparation =
        prepare_messages_for_entries(&repo_path, entries, config, max_attempts, prepare_workers)?;
    telemetry.record_duration(
        "diff_preparation",
        preparation.telemetry.diff_preparation_ms,
        preparation.telemetry.diff_preparation_items,
    );
    telemetry.record_duration(
        "llm_enqueue",
        preparation.telemetry.llm_enqueue_ms,
        preparation.telemetry.llm_enqueue_items,
    );
    let prepared_outcomes = preparation.outcomes;
    let prepared_count = prepared_outcomes
        .iter()
        .filter(|outcome| outcome.is_ok())
        .count();
    telemetry.record_since("llm_wait", wait_start, prepared_outcomes.len());
    telemetry.record_empty("response_validation");

    let commit_start = Instant::now();
    let mut commit_stage_path_ms = 0.0;
    let mut commit_create_ms = 0.0;
    let mut commit_read_hash_ms = 0.0;
    let mut commit_hash_reads = 0;
    let mut commit_stage_path_invocations = 0;
    let mut gix_session = if gix_commit_backend_enabled() {
        Some(GixCommitSession::open_with_deferred_index_writes(
            &repo_path,
            total_entries > 1,
        )?)
    } else {
        None
    };
    for outcome in prepared_outcomes {
        let prepared_entry = match outcome {
            Ok(prepared_entry) => prepared_entry,
            Err(failure) => {
                failures.push(failure);
                continue;
            }
        };
        let entry = prepared_entry.entry;
        let message = prepared_entry.message;
        let staging = entry.commit_staging();
        let commit_outcome = match gix_session.as_mut() {
            Some(session) => session.commit_path(&entry.path, &message, staging),
            None => commit_file_with_staging(&repo_path, &entry.path, &message, false, staging),
        };
        let commit_outcome = match commit_outcome {
            Ok(outcome) => outcome,
            Err(err) => {
                unstage_path(&repo_path, &entry.path);
                failures.push(WorkflowFailure::commit(&entry, err.to_string()));
                continue;
            }
        };
        commit_stage_path_ms += commit_outcome.stage_path_ms;
        commit_stage_path_invocations += usize::from(commit_outcome.stage_path_invoked);
        commit_create_ms += commit_outcome.create_commit_ms;
        let is_deletion = entry.is_deletion();

        let commit_hash = if let Some(commit_hash) = commit_outcome.commit_hash {
            Some(commit_hash)
        } else {
            let hash_start = Instant::now();
            let commit_hash = recent_commit_hash(&repo_path)?;
            commit_read_hash_ms += hash_start.elapsed().as_secs_f64() * 1000.0;
            commit_hash_reads += usize::from(commit_hash.is_some());
            commit_hash
        };

        commits.push(WorkflowCommit {
            file_path: entry.path,
            message,
            commit_hash,
            is_deletion,
        });
    }
    let commit_index_flush_ms = if let Some(session) = gix_session.as_mut() {
        session.flush_index()?
    } else {
        0.0
    };
    telemetry.record_duration(
        "commit_stage_path",
        commit_stage_path_ms,
        commit_stage_path_invocations,
    );
    telemetry.record_duration("commit_create", commit_create_ms, commits.len());
    telemetry.record_duration(
        "commit_index_flush",
        commit_index_flush_ms,
        usize::from(commit_index_flush_ms > 0.0),
    );
    telemetry.record_duration("commit_read_hash", commit_read_hash_ms, commit_hash_reads);
    telemetry.record_since("commit", commit_start, commits.len());

    let push_start = Instant::now();
    let push_outcome = auto_push_if_configured(&repo_path, config, !commits.is_empty());
    telemetry.record_since("push", push_start, usize::from(push_outcome.pushed));
    telemetry.record_empty("output_render");

    persist_run_snapshot(
        &repo_path,
        config,
        &commits,
        &failures,
        prepared_count,
        total_entries,
        &push_outcome,
        &mut telemetry,
        workflow_start,
    )?;

    if commits.is_empty() && total_entries <= 1 {
        if let Some(failure) = failures.first() {
            return Err(KcmtError::Message(failure.error.clone()));
        }
    }

    if commits.is_empty() && total_entries <= 1 {
        Ok("No changes to commit.\n".to_string())
    } else {
        Ok(render_workflow_output(
            config,
            &commits,
            &failures,
            &push_outcome,
            &telemetry,
            output_options,
        ))
    }
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

fn render_workflow_output(
    config: &kcmt_core::model::WorkflowConfig,
    commits: &[WorkflowCommit],
    failures: &[WorkflowFailure],
    push_outcome: &PushOutcome,
    telemetry: &WorkflowTelemetry,
    options: WorkflowOutputOptions,
) -> String {
    let mut lines = Vec::new();
    if options.compact {
        lines.push(format!(
            "provider {}  model {}  retries -",
            config.provider, config.model
        ));
        lines.push(String::new());
        lines.push("Run Summary".to_string());
        lines.push(format!(
            "Commits {}  Failures {}",
            commits.len(),
            failures.len()
        ));
        lines.push(format!("Auto-push {}", push_outcome.state));
        if let Some(subject) = commits.last().map(|commit| commit.message.as_str()) {
            lines.push(format!("Latest commit: {subject}"));
        }
        if options.verbose {
            lines.push(String::new());
            lines.push("Commits".to_string());
            append_commit_lines(&mut lines, commits);
            append_failure_lines(&mut lines, failures);
        }
    } else {
        append_commit_lines(&mut lines, commits);
        append_failure_lines(&mut lines, failures);
        match push_outcome.state {
            "pushed" => lines.push("Auto-push: pushed".to_string()),
            "failed" => lines.push("Auto-push: failed".to_string()),
            _ => {}
        }
    }

    if options.profile_startup {
        lines.push(String::new());
        for stage in &telemetry.stages {
            lines.push(format!(
                "[kcmt-profile] {}: {:.1} ms items={}",
                stage.stage, stage.duration_ms, stage.items
            ));
        }
    }

    let mut output = lines.join("\n");
    output.push('\n');
    output
}

fn append_commit_lines(lines: &mut Vec<String>, commits: &[WorkflowCommit]) {
    for commit in commits {
        lines.push(format!("✓ {}", commit.file_path));
        lines.push(format!("  {}", commit.message));
        if let Some(hash) = &commit.commit_hash {
            lines.push(format!("  {}", truncate_hash(hash)));
        }
    }
}

fn append_failure_lines(lines: &mut Vec<String>, failures: &[WorkflowFailure]) {
    for failure in failures {
        lines.push(format!("✗ {}", failure.file_path));
        lines.push(format!("  {}", failure.error));
    }
}

fn commit_message_for_entry(
    repo_path: &Path,
    entry: &StatusEntry,
    config: &kcmt_core::model::WorkflowConfig,
    max_attempts: usize,
) -> Result<String> {
    if entry.is_deletion() {
        Ok(deletion_commit_message(
            &entry.path,
            config.max_commit_length,
        ))
    } else if let Some(raw_response) = fixture_provider_response() {
        sanitize_commit_output(&raw_response)
            .map(|message| limit_subject(message, config.max_commit_length))
            .map_err(KcmtError::Message)
    } else if runtime_benchmark_enabled() {
        Ok(heuristic_commit_message(
            &entry.path,
            config.max_commit_length,
        ))
    } else if configured_api_key(config).is_some() {
        invoke_provider_with_fallback(repo_path, entry, config, max_attempts)
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

fn max_attempts_from_retries(max_retries: Option<usize>) -> usize {
    max_retries.unwrap_or(3).saturating_add(1).max(1)
}

fn select_prepare_workers(file_count: usize, override_workers: Option<usize>) -> usize {
    if file_count <= 1 {
        return 1;
    }
    let desired = override_workers.or_else(|| {
        env::var("KCMT_PREPARE_WORKERS")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
    });
    if let Some(workers) = desired.filter(|workers| *workers > 0) {
        return workers.min(file_count).max(1);
    }
    let cpu_hint = thread::available_parallelism()
        .map(usize::from)
        .unwrap_or(4);
    file_count.min(8).min(cpu_hint).max(1)
}

fn local_synthesis_enabled() -> bool {
    env_truthy("KCMT_ALLOW_LOCAL_SYNTHESIS") || runtime_benchmark_enabled()
}

fn fixture_provider_response() -> Option<String> {
    if !env_truthy("KCMT_ALLOW_PROVIDER_RESPONSE_FIXTURE") && !runtime_benchmark_enabled() {
        return None;
    }
    env::var("KCMT_PROVIDER_RESPONSE")
        .ok()
        .filter(|value| !value.trim().is_empty())
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
    max_attempts: usize,
    prepare_workers: usize,
) -> Result<PreparationResult> {
    if should_use_openai_batch(config) {
        if let Some(api_key) = configured_api_key(config) {
            return prepare_openai_batch_messages(
                repo_path,
                entries,
                config,
                &api_key,
                max_attempts,
            );
        }
    }

    if prepare_workers <= 1 || entries.len() <= 1 {
        let outcomes = entries
            .into_iter()
            .map(
                |entry| match commit_message_for_entry(repo_path, &entry, config, max_attempts) {
                    Ok(message) => Ok(PreparedEntry { entry, message }),
                    Err(err) => Err(WorkflowFailure::prepare(&entry, err.to_string())),
                },
            )
            .collect::<Vec<_>>();
        return Ok(PreparationResult {
            outcomes,
            telemetry: PreparationTelemetry::default(),
        });
    }

    let outcomes =
        prepare_messages_in_workers(repo_path, entries, config, max_attempts, prepare_workers)?;
    Ok(PreparationResult {
        outcomes,
        telemetry: PreparationTelemetry::default(),
    })
}

fn prepare_messages_in_workers(
    repo_path: &Path,
    entries: Vec<StatusEntry>,
    config: &kcmt_core::model::WorkflowConfig,
    max_attempts: usize,
    prepare_workers: usize,
) -> Result<Vec<PreparedOutcome>> {
    let worker_count = prepare_workers.min(entries.len()).max(1);
    let mut buckets = vec![Vec::<(usize, StatusEntry)>::new(); worker_count];
    for (index, entry) in entries.into_iter().enumerate() {
        buckets[index % worker_count].push((index, entry));
    }

    let handles = buckets
        .into_iter()
        .filter(|bucket| !bucket.is_empty())
        .map(|bucket| {
            let repo_path = repo_path.to_path_buf();
            let config = config.clone();
            thread::spawn(move || {
                bucket
                    .into_iter()
                    .map(|(index, entry)| {
                        let outcome = match commit_message_for_entry(
                            &repo_path,
                            &entry,
                            &config,
                            max_attempts,
                        ) {
                            Ok(message) => Ok(PreparedEntry { entry, message }),
                            Err(err) => Err(WorkflowFailure::prepare(&entry, err.to_string())),
                        };
                        (index, outcome)
                    })
                    .collect::<Vec<_>>()
            })
        })
        .collect::<Vec<_>>();

    let mut indexed = Vec::new();
    for handle in handles {
        let prepared = handle
            .join()
            .map_err(|_| KcmtError::Message("prepare worker panicked".to_string()))?;
        indexed.extend(prepared);
    }
    indexed.sort_by_key(|(index, _)| *index);
    Ok(indexed.into_iter().map(|(_, entry)| entry).collect())
}

fn should_use_openai_batch(config: &kcmt_core::model::WorkflowConfig) -> bool {
    config.provider == "openai" && config.use_batch
}

fn prepare_openai_batch_messages(
    repo_path: &Path,
    entries: Vec<StatusEntry>,
    config: &kcmt_core::model::WorkflowConfig,
    api_key: &str,
    max_attempts: usize,
) -> Result<PreparationResult> {
    let mut outcomes = Vec::new();
    let mut batch_entries = Vec::new();
    let mut jobs = Vec::new();
    let mut telemetry = PreparationTelemetry::default();
    let system = "You generate strictly valid Conventional Commit messages.";
    for entry in entries {
        if entry.is_deletion() {
            let message = deletion_commit_message(&entry.path, config.max_commit_length);
            outcomes.push(Ok(PreparedEntry { entry, message }));
            continue;
        }

        let diff_start = Instant::now();
        let diff = match diff_for_entry(repo_path, &entry) {
            Ok(diff) => diff,
            Err(err) => {
                telemetry.diff_preparation_ms += diff_start.elapsed().as_secs_f64() * 1000.0;
                telemetry.diff_preparation_items += 1;
                outcomes.push(Err(WorkflowFailure::prepare(&entry, err.to_string())));
                continue;
            }
        };
        telemetry.diff_preparation_ms += diff_start.elapsed().as_secs_f64() * 1000.0;
        telemetry.diff_preparation_items += 1;
        let enqueue_start = Instant::now();
        let context = format!("File: {}", entry.path);
        let prompt = build_prompt(&diff, &context, "conventional");
        jobs.push(OpenAiBatchJob {
            custom_id: entry.path.clone(),
            messages: vec![
                ProviderMessage::system(system),
                ProviderMessage::user(prompt),
            ],
        });
        telemetry.llm_enqueue_ms += enqueue_start.elapsed().as_secs_f64() * 1000.0;
        telemetry.llm_enqueue_items += 1;
        batch_entries.push(entry);
    }
    if batch_entries.is_empty() {
        return Ok(PreparationResult {
            outcomes,
            telemetry,
        });
    }

    let runtime = match tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
    {
        Ok(runtime) => runtime,
        Err(err) => {
            let error = format!("failed to initialize provider runtime: {err}");
            outcomes.extend(
                batch_entries
                    .into_iter()
                    .map(|entry| Err(WorkflowFailure::prepare(&entry, error.clone()))),
            );
            return Ok(PreparationResult {
                outcomes,
                telemetry,
            });
        }
    };
    let transport = match AsyncTransport::new(
        Duration::from_secs(60),
        RetryPolicy {
            max_attempts,
            base_backoff: Duration::from_millis(250),
        },
    ) {
        Ok(transport) => transport,
        Err(err) => {
            let error = format!("failed to initialize provider transport: {err}");
            outcomes.extend(
                batch_entries
                    .into_iter()
                    .map(|entry| Err(WorkflowFailure::prepare(&entry, error.clone()))),
            );
            return Ok(PreparationResult {
                outcomes,
                telemetry,
            });
        }
    };
    let batch_model = config
        .batch_model
        .as_deref()
        .filter(|value| !value.trim().is_empty())
        .unwrap_or(&config.model);
    let results = match runtime.block_on(OpenAiClient::invoke_batch(
        &transport,
        &config.llm_endpoint,
        api_key,
        batch_model,
        &jobs,
        Duration::from_secs(config.batch_timeout_seconds),
        Duration::from_millis(50),
    )) {
        Ok(results) => results,
        Err(err) => {
            let error = format!("provider batch request failed: {err}");
            outcomes.extend(
                batch_entries
                    .into_iter()
                    .map(|entry| Err(WorkflowFailure::prepare(&entry, error.clone()))),
            );
            return Ok(PreparationResult {
                outcomes,
                telemetry,
            });
        }
    };
    let mut messages_by_id = std::collections::BTreeMap::new();
    let mut errors_by_id = std::collections::BTreeMap::new();
    for result in results {
        match sanitize_commit_output(&result.content) {
            Ok(message) => {
                messages_by_id.insert(
                    result.custom_id,
                    limit_subject(message, config.max_commit_length),
                );
            }
            Err(err) => {
                errors_by_id.insert(result.custom_id, err);
            }
        }
    }

    for entry in batch_entries {
        if let Some(message) = messages_by_id.remove(&entry.path) {
            outcomes.push(Ok(PreparedEntry { entry, message }));
        } else if let Some(error) = errors_by_id.remove(&entry.path) {
            outcomes.push(Err(WorkflowFailure::prepare(&entry, error)));
        } else {
            outcomes.push(Err(WorkflowFailure::prepare(
                &entry,
                format!("batch response missing {}", entry.path),
            )));
        }
    }
    Ok(PreparationResult {
        outcomes,
        telemetry,
    })
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
    max_attempts: usize,
) -> Result<String> {
    let mut last_error = None;
    for candidate in provider_candidates(config) {
        let Some(api_key) = api_key_for_env(&candidate.api_key_env) else {
            continue;
        };
        match invoke_provider_candidate(repo_path, entry, &candidate, &api_key, max_attempts)
            .and_then(|raw| {
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
    max_attempts: usize,
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
            max_attempts,
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
    if entry.is_new_file() {
        return file_content_diff(repo_path, entry);
    }

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

    file_content_diff(repo_path, entry)
}

fn file_content_diff(repo_path: &Path, entry: &StatusEntry) -> Result<String> {
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
    match local_origin_remote_probe(repo_path) {
        OriginRemoteProbe::Present => return true,
        OriginRemoteProbe::Absent => return false,
        OriginRemoteProbe::Unknown => {}
    }

    Command::new("git")
        .current_dir(repo_path)
        .args(["config", "--get", "remote.origin.url"])
        .output()
        .map(|output| output.status.success() && !output.stdout.is_empty())
        .unwrap_or(false)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OriginRemoteProbe {
    Present,
    Absent,
    Unknown,
}

fn local_origin_remote_probe(repo_path: &Path) -> OriginRemoteProbe {
    let Some(config_path) = git_common_config_path(repo_path) else {
        return OriginRemoteProbe::Unknown;
    };
    let Ok(config) = fs::read_to_string(config_path) else {
        return OriginRemoteProbe::Unknown;
    };
    if git_config_value(&config, "remote \"origin\"", "url")
        .map(|value| !value.trim().is_empty())
        .unwrap_or(false)
    {
        return OriginRemoteProbe::Present;
    }
    if git_config_has_include(&config) {
        return OriginRemoteProbe::Unknown;
    }
    OriginRemoteProbe::Absent
}

fn git_common_config_path(repo_path: &Path) -> Option<PathBuf> {
    let git_path = repo_path.join(".git");
    if git_path.is_dir() {
        return Some(git_path.join("config"));
    }

    let git_file = fs::read_to_string(&git_path).ok()?;
    let git_dir = git_file.lines().find_map(|line| {
        let trimmed = line.trim();
        trimmed
            .strip_prefix("gitdir:")
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(|value| resolve_git_path(repo_path, value))
    })?;
    let common_dir_path = git_dir.join("commondir");
    let common_dir = fs::read_to_string(&common_dir_path)
        .ok()
        .map(|value| resolve_git_path(&git_dir, value.trim()))
        .unwrap_or(git_dir);
    Some(common_dir.join("config"))
}

fn resolve_git_path(base: &Path, value: &str) -> PathBuf {
    let path = Path::new(value);
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        base.join(path)
    }
}

fn git_config_value(config: &str, section: &str, key: &str) -> Option<String> {
    let mut in_target_section = false;
    for raw_line in config.lines() {
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') || line.starts_with(';') {
            continue;
        }
        if let Some(section_name) = git_config_section_name(line) {
            in_target_section = section_name.eq_ignore_ascii_case(section);
            continue;
        }
        if !in_target_section {
            continue;
        }
        let Some((candidate_key, value)) = line.split_once('=') else {
            continue;
        };
        if candidate_key.trim().eq_ignore_ascii_case(key) {
            return Some(unquote_git_config_value(value.trim()));
        }
    }
    None
}

fn git_config_has_include(config: &str) -> bool {
    config.lines().any(|raw_line| {
        git_config_section_name(raw_line.trim())
            .map(|section| {
                section.eq_ignore_ascii_case("include")
                    || section.to_ascii_lowercase().starts_with("includeif ")
            })
            .unwrap_or(false)
    })
}

fn git_config_section_name(line: &str) -> Option<&str> {
    line.strip_prefix('[')
        .and_then(|value| value.strip_suffix(']'))
        .map(str::trim)
        .filter(|value| !value.is_empty())
}

fn unquote_git_config_value(value: &str) -> String {
    value
        .strip_prefix('"')
        .and_then(|inner| inner.strip_suffix('"'))
        .unwrap_or(value)
        .to_string()
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

fn workflow_summary(
    commit_success: usize,
    commit_failure: usize,
    deletions_success: usize,
    deletions_failure: usize,
) -> String {
    let overall_success = commit_success + deletions_success;
    let overall_failure = commit_failure + deletions_failure;
    let mut parts = Vec::new();
    if overall_success > 0 {
        parts.push(format!("Successfully completed {overall_success} commits"));
    } else {
        parts.push("No commits were made".to_string());
    }
    if deletions_success > 0 || deletions_failure > 0 {
        parts.push(format!("Committed {deletions_success} deletion(s)"));
    }
    if commit_success > 0 || commit_failure > 0 {
        parts.push(format!("Committed {commit_success} file change(s)"));
    }
    if overall_failure > 0 {
        parts.push(format!("Encountered {overall_failure} file failure(s)"));
    }
    parts.join(". ")
}

fn unstage_path(repo_path: &Path, file_path: &str) {
    let _ = Command::new("git")
        .current_dir(repo_path)
        .args(["reset", "--", file_path])
        .status();
}

fn persist_run_snapshot(
    repo_path: &Path,
    config: &kcmt_core::model::WorkflowConfig,
    commits: &[WorkflowCommit],
    failures: &[WorkflowFailure],
    prepared_count: usize,
    total_entries: usize,
    push_outcome: &PushOutcome,
    telemetry: &mut WorkflowTelemetry,
    workflow_start: Instant,
) -> Result<()> {
    let snapshot_start = Instant::now();
    let file_commits: Vec<_> = commits
        .iter()
        .filter(|commit| !commit.is_deletion)
        .collect();
    let deletions: Vec<_> = commits.iter().filter(|commit| commit.is_deletion).collect();
    let file_failures: Vec<_> = failures
        .iter()
        .filter(|failure| !failure.is_deletion)
        .collect();
    let deletion_failure_records: Vec<_> = failures
        .iter()
        .filter(|failure| failure.is_deletion)
        .collect();
    let commit_success = file_commits.len();
    let commit_failure = file_failures.len();
    let deletions_success = deletions.len();
    let deletions_failure = deletion_failure_records.len();
    let overall_success = commits.len();
    let prepared_failures = failures
        .iter()
        .filter(|failure| failure.stage == WorkflowFailureStage::Prepare)
        .count();
    let overall_failure = failures.len();
    let subjects: Vec<&str> = commits
        .iter()
        .map(|commit| commit.message.as_str())
        .collect();
    let summary = workflow_summary(
        commit_success,
        commit_failure,
        deletions_success,
        deletions_failure,
    );
    if runtime_benchmark_enabled() {
        telemetry.record_since("snapshot", snapshot_start, 1);
        telemetry.record_since("workflow_total", workflow_start, total_entries);
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
            "summary": summary,
            "counts": {
                "files_total": total_entries,
                "prepared_total": prepared_count,
                "processed_total": commits.len() + failures.len(),
                "prepared_failures": prepared_failures,
                "commit_success": commit_success,
                "commit_failure": commit_failure,
                "deletions_total": deletions.len() + deletion_failure_records.len(),
                "deletions_success": deletions_success,
                "deletions_failure": deletions_failure,
                "overall_success": overall_success,
                "overall_failure": overall_failure,
                "errors": push_outcome.errors.len()
            },
            "pushed": push_outcome.pushed,
            "auto_push_state": push_outcome.state,
            "errors": &push_outcome.errors,
            "commits": [],
            "deletions": [],
            "subjects": subjects,
            "stats": {},
            "telemetry": {
                "schema_version": 1,
                "prepare_workers": telemetry.prepare_workers,
                "stages": telemetry.stages.iter().map(|stage| json!({
                    "stage": stage.stage,
                    "duration_ms": stage.duration_ms,
                    "items": stage.items
                })).collect::<Vec<_>>()
            }
        });
        return write_run_snapshot(repo_path, &snapshot);
    }

    let mut commit_records = Vec::new();
    for commit in &file_commits {
        commit_records.push(json!({
            "success": true,
            "commit_hash": commit.commit_hash,
            "message": commit.message,
            "error": null,
            "file_path": commit.file_path
        }));
    }
    for failure in &file_failures {
        commit_records.push(json!({
            "success": false,
            "commit_hash": null,
            "message": null,
            "error": failure.error,
            "file_path": failure.file_path
        }));
    }

    let mut deletion_records = Vec::new();
    for commit in &deletions {
        deletion_records.push(json!({
            "success": true,
            "commit_hash": commit.commit_hash,
            "message": commit.message,
            "error": null,
            "file_path": commit.file_path
        }));
    }
    for failure in &deletion_failure_records {
        deletion_records.push(json!({
            "success": false,
            "commit_hash": null,
            "message": null,
            "error": failure.error,
            "file_path": failure.file_path
        }));
    }
    telemetry.record_since("snapshot", snapshot_start, 1);
    telemetry.record_since("workflow_total", workflow_start, total_entries);

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
            "files_total": total_entries,
            "prepared_total": prepared_count,
            "processed_total": commits.len() + failures.len(),
            "prepared_failures": prepared_failures,
            "commit_success": commit_success,
            "commit_failure": commit_failure,
            "deletions_total": deletions.len() + deletion_failure_records.len(),
            "deletions_success": deletions_success,
            "deletions_failure": deletions_failure,
            "overall_success": overall_success,
            "overall_failure": overall_failure,
            "errors": push_outcome.errors.len()
        },
        "pushed": push_outcome.pushed,
        "auto_push_state": push_outcome.state,
        "summary": summary,
        "errors": &push_outcome.errors,
        "commits": commit_records,
        "deletions": deletion_records,
        "subjects": subjects,
        "stats": {},
        "telemetry": {
            "schema_version": 1,
            "prepare_workers": telemetry.prepare_workers,
            "stages": telemetry.stages.iter().map(|stage| json!({
                "stage": stage.stage,
                "duration_ms": stage.duration_ms,
                "items": stage.items
            })).collect::<Vec<_>>()
        }
    });

    write_run_snapshot(repo_path, &snapshot)
}

fn write_run_snapshot(repo_path: &Path, snapshot: &serde_json::Value) -> Result<()> {
    let path = snapshot_path(repo_path);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let rendered =
        serde_json::to_string(snapshot).map_err(|err| KcmtError::Message(err.to_string()))?;
    fs::write(path, rendered)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        deletion_commit_message, diff_for_entry, git_common_config_path, git_config_has_include,
        git_config_value, heuristic_commit_message, local_origin_remote_probe,
        parse_status_entries, select_prepare_workers, OriginRemoteProbe, StatusEntry,
    };
    use kcmt_core::git::commit_file::CommitStaging;
    use std::fs;
    use std::path::{Path, PathBuf};
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

    fn write_git_config(repo: &Path, config: &str) {
        let git_dir = repo.join(".git");
        fs::create_dir_all(&git_dir).expect("git dir should be created");
        fs::write(git_dir.join("config"), config).expect("git config should be written");
    }

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
    fn diff_for_new_file_entry_reads_file_without_git_diff() {
        let repo = unique_temp_dir("new-file-diff");
        fs::write(repo.join("new.md"), "# New file\n").expect("untracked file");

        for code in ["??", "A ", "AM"] {
            let entry = StatusEntry {
                code: code.to_string(),
                path: "new.md".to_string(),
            };
            let diff = diff_for_entry(&repo, &entry).expect("new file diff");

            assert_eq!(diff, "New or changed file: new.md\n\n# New file\n");
        }
    }

    #[test]
    fn tracked_statuses_can_commit_directly_without_staging() {
        for code in [" M", " D", "M ", "D ", "MM", "MD", "DM"] {
            let entry = StatusEntry {
                code: code.to_string(),
                path: "tracked.py".to_string(),
            };
            assert_eq!(entry.commit_staging(), CommitStaging::DirectPath, "{code}");
        }

        for code in ["??", "A ", "AM", "UU"] {
            let entry = StatusEntry {
                code: code.to_string(),
                path: "new.py".to_string(),
            };
            assert_eq!(entry.commit_staging(), CommitStaging::StagePath, "{code}");
        }
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
    fn single_file_prepare_uses_one_worker_without_env_or_cpu_probe() {
        assert_eq!(select_prepare_workers(0, None), 1);
        assert_eq!(select_prepare_workers(1, None), 1);
        assert_eq!(select_prepare_workers(1, Some(8)), 1);
    }

    #[test]
    fn builds_python_compatible_deletion_message() {
        assert_eq!(
            deletion_commit_message("delete_me.txt", 72),
            "chore(delete_me-txt): file deleted"
        );
    }

    #[test]
    fn reads_origin_url_from_local_git_config() {
        let config = r#"
[core]
    repositoryformatversion = 0
[remote "origin"]
    url = git@example.com:owner/repo.git
"#;

        assert_eq!(
            git_config_value(config, "remote \"origin\"", "url").as_deref(),
            Some("git@example.com:owner/repo.git")
        );
    }

    #[test]
    fn detects_git_config_includes_for_command_fallback() {
        let config = r#"
[includeIf "gitdir:~/work/"]
    path = ~/.gitconfig-work
"#;

        assert!(git_config_has_include(config));
    }

    #[test]
    fn local_origin_probe_detects_missing_origin_without_fallback() {
        let repo = unique_temp_dir("no-origin");
        write_git_config(
            &repo,
            r#"
[core]
    repositoryformatversion = 0
"#,
        );

        assert_eq!(local_origin_remote_probe(&repo), OriginRemoteProbe::Absent);
    }

    #[test]
    fn local_origin_probe_detects_configured_origin() {
        let repo = unique_temp_dir("origin");
        write_git_config(
            &repo,
            r#"
[remote "origin"]
    url = /tmp/origin.git
"#,
        );

        assert_eq!(local_origin_remote_probe(&repo), OriginRemoteProbe::Present);
    }

    #[test]
    fn worktree_git_file_resolves_common_git_config() {
        let repo = unique_temp_dir("worktree");
        let common_git_dir = unique_temp_dir("common-git");
        let worktree_git_dir = unique_temp_dir("linked-worktree-git");
        fs::write(
            worktree_git_dir.join("commondir"),
            common_git_dir.to_string_lossy().as_bytes(),
        )
        .expect("commondir should be written");
        fs::write(
            repo.join(".git"),
            format!("gitdir: {}\n", worktree_git_dir.display()),
        )
        .expect("git file should be written");

        let expected_config_path = common_git_dir.join("config");
        assert_eq!(
            git_common_config_path(&repo).as_deref(),
            Some(expected_config_path.as_path())
        );
    }
}
