use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc;
use std::sync::Arc;
use std::sync::Mutex;
use std::thread;
use std::time::Duration;
use std::time::Instant;

use kcmt_core::config::loader::{load_config, ConfigOverrides};
use kcmt_core::error::KcmtError;
use kcmt_core::error::Result;
use kcmt_core::git::commit_file::{
    commit_file_with_staging, commit_paths_with_staging, gix_commit_backend_enabled,
    recent_commit_hash, CommitStaging, GixCommitSession,
};
use kcmt_core::git::repo::{CliGitRepository, GitRepository};
use kcmt_core::message::{
    build_prompt_with_profile, build_prompt_with_profile_style, sanitize_commit_output,
    selected_prompt_profile,
};
use kcmt_core::preferences::{
    default_keychain_account, load_preferences, resolve_credential, CredentialRequest,
    CredentialSource, OsKeychainStore, Preferences, PromptProfile,
};
use kcmt_core::selector::{select_model, ModelSelection};
use kcmt_core::telemetry::{load_usage_summary, record_usage, TelemetryRunRecord};
use kcmt_provider::clients::{
    AnthropicClient, GitHubModelsClient, OpenAiBatchJob, OpenAiClient, ProviderMessage, XaiClient,
};
use kcmt_provider::error_map::normalize_error;
use kcmt_provider::transport::{AsyncTransport, RetryPolicy};
use kcmt_tui::{
    spawn_workflow_tui, WorkflowTuiContext, WorkflowTuiEvent, WorkflowTuiSession, WorkflowTuiState,
};
use serde_json::json;
use time::format_description::well_known::Rfc3339;
use time::OffsetDateTime;

use super::history::snapshot_path;
use super::model_discovery::{cached_or_static_catalog_for_config, catalog_to_selector_candidates};

#[derive(Debug, Clone, PartialEq, Eq)]
struct StatusEntry {
    code: String,
    path: String,
    source_path: Option<String>,
}

impl StatusEntry {
    fn index_status(&self) -> char {
        self.code.chars().next().unwrap_or(' ')
    }

    fn worktree_status(&self) -> char {
        self.code.chars().nth(1).unwrap_or(' ')
    }

    fn is_deletion(&self) -> bool {
        self.code.contains('D')
    }

    fn is_untracked(&self) -> bool {
        self.code == "??"
    }

    fn is_rename_or_copy(&self) -> bool {
        self.code.contains('R') || self.code.contains('C')
    }

    fn has_staged_changes(&self) -> bool {
        self.index_status() != ' ' && self.index_status() != '?'
    }

    fn has_worktree_changes(&self) -> bool {
        self.worktree_status() != ' ' && self.worktree_status() != '?'
    }

    fn commit_staging(&self) -> CommitStaging {
        if self.requires_staging_before_commit() {
            CommitStaging::StagePath
        } else {
            CommitStaging::DirectPath
        }
    }

    fn commit_pathspecs(&self) -> Vec<&str> {
        let mut pathspecs = Vec::new();
        if let Some(source_path) = self.source_path.as_deref() {
            pathspecs.push(source_path);
        }
        pathspecs.push(self.path.as_str());
        pathspecs
    }

    fn requires_staging_before_commit(&self) -> bool {
        self.code == "??"
            || self.code.contains('A')
            || self.code.contains('U')
            || self.is_rename_or_copy()
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
struct WorkflowProgress {
    enabled: bool,
    live_tui: bool,
    tui_echo: bool,
    mode: &'static str,
    total: usize,
    queued: Arc<AtomicUsize>,
    completed: Arc<AtomicUsize>,
    failed: Arc<AtomicUsize>,
    tui_state: Option<Arc<Mutex<WorkflowTuiState>>>,
}

impl WorkflowProgress {
    fn new(
        mode: &'static str,
        total: usize,
        options: &WorkflowOutputOptions,
        context: WorkflowTuiContext,
    ) -> Self {
        let tui_state = if options.tui || options.tui_model_export {
            let mut state = WorkflowTuiState::new(context);
            state.apply(WorkflowTuiEvent::Discovered { total_files: total });
            Some(Arc::new(Mutex::new(state)))
        } else {
            None
        };
        let progress = Self {
            enabled: progress_enabled(options) && total > 0,
            live_tui: options.tui,
            tui_echo: !options.tui && options.tui_model_export && progress_enabled(options),
            mode,
            total,
            queued: Arc::new(AtomicUsize::new(0)),
            completed: Arc::new(AtomicUsize::new(0)),
            failed: Arc::new(AtomicUsize::new(0)),
            tui_state,
        };
        progress.summary("start");
        progress
    }

    fn start_tui_session(&self) -> Option<WorkflowTuiSession> {
        if !self.enabled || !kcmt_tui::should_enable_tui(false) {
            return None;
        }
        self.tui_state
            .as_ref()
            .and_then(|state| spawn_workflow_tui(Arc::clone(state)).ok())
    }

    fn queued(&self, stage: &'static str, file_path: &str) {
        self.queued.fetch_add(1, Ordering::Relaxed);
        self.apply_tui(WorkflowTuiEvent::Queued {
            file_path: file_path.to_string(),
            stage: stage.to_string(),
        });
        self.render(stage, Some(file_path), None);
    }

    fn event(&self, stage: &'static str, file_path: &str) {
        if stage == "llm" {
            self.apply_tui(WorkflowTuiEvent::RequestSent {
                file_path: file_path.to_string(),
            });
        }
        self.render(stage, Some(file_path), None);
    }

    fn prepared(&self, file_path: &str, subject: &str) {
        self.completed.fetch_add(1, Ordering::Relaxed);
        self.apply_tui(WorkflowTuiEvent::Prepared {
            file_path: file_path.to_string(),
            subject: subject.to_string(),
        });
        self.render("done", Some(file_path), Some("ok"));
    }

    fn prepare_failed(&self, file_path: &str, error: &str) {
        self.failed.fetch_add(1, Ordering::Relaxed);
        self.apply_tui(WorkflowTuiEvent::PrepareFailed {
            file_path: file_path.to_string(),
            error: error.to_string(),
        });
        self.render("done", Some(file_path), Some("failed"));
    }

    fn commit_started(&self, file_path: &str) {
        self.apply_tui(WorkflowTuiEvent::CommitStarted {
            file_path: file_path.to_string(),
        });
        self.render("commit", Some(file_path), None);
    }

    fn commit_succeeded(&self, file_path: &str, subject: &str, commit_hash: Option<&str>) {
        self.apply_tui(WorkflowTuiEvent::CommitSucceeded {
            file_path: file_path.to_string(),
            subject: subject.to_string(),
            commit_hash: commit_hash.map(str::to_string),
        });
        self.render("committed", Some(file_path), Some("ok"));
    }

    fn commit_failed(&self, file_path: &str, error: &str) {
        self.apply_tui(WorkflowTuiEvent::CommitFailed {
            file_path: file_path.to_string(),
            error: error.to_string(),
        });
        self.render("committed", Some(file_path), Some("failed"));
    }

    fn push_started(&self) {
        self.apply_tui(WorkflowTuiEvent::PushStarted);
        self.summary("push");
    }

    fn push_finished(&self, state: &str) {
        self.apply_tui(WorkflowTuiEvent::PushFinished {
            state: state.to_string(),
        });
        self.summary("push");
    }

    fn finished(&self) {
        self.apply_tui(WorkflowTuiEvent::Finished);
        self.summary("finished");
    }

    fn snapshot_json(&self) -> Option<String> {
        self.tui_state.as_ref().and_then(|state| {
            state
                .lock()
                .ok()
                .and_then(|state| state.to_json_line().ok())
        })
    }

    fn summary(&self, stage: &'static str) {
        self.render(stage, None, None);
    }

    fn render(&self, stage: &'static str, file_path: Option<&str>, status: Option<&str>) {
        if !self.enabled || self.live_tui {
            return;
        }
        let queued = self.queued.load(Ordering::Relaxed);
        let completed = self.completed.load(Ordering::Relaxed);
        let failed = self.failed.load(Ordering::Relaxed);
        let pending = self.total.saturating_sub(completed + failed);
        let mut line = format!(
            "kcmt progress: mode={} stage={} total={} queued={} pending={} completed={} failed={}",
            self.mode, stage, self.total, queued, pending, completed, failed
        );
        if let Some(status) = status {
            line.push_str(&format!(" status={status}"));
        }
        if let Some(file_path) = file_path {
            line.push_str(&format!(" file={file_path}"));
        }
        eprintln!("{line}");
    }

    fn apply_tui(&self, event: WorkflowTuiEvent) {
        let Some(state) = &self.tui_state else {
            return;
        };
        let Ok(mut state) = state.lock() else {
            return;
        };
        state.apply(event);
        if self.tui_echo {
            let lines = state.render_lines();
            if let Some(summary) = lines.get(1) {
                eprintln!("kcmt tui: {summary}");
            }
        }
    }
}

struct BatchProgress {
    stop: mpsc::Sender<()>,
    handle: thread::JoinHandle<()>,
}

impl BatchProgress {
    fn stop(self) {
        let _ = self.stop.send(());
        let _ = self.handle.join();
    }
}

#[derive(Debug, Clone)]
struct ProviderCandidate {
    provider: String,
    model: String,
    endpoint: String,
    api_key_env: String,
    keychain_account: String,
}

#[derive(Debug, Clone)]
struct WorkflowRuntime {
    prompt_profile: PromptProfile,
    model_selection: ModelSelection,
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
    pub no_progress: bool,
    pub tui: bool,
    pub tui_model_export: bool,
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
    let preferences = load_preferences().unwrap_or_else(|_| Preferences::default());
    let usage_summary = load_usage_summary(&repo_path).unwrap_or_default();
    let model_catalog =
        catalog_to_selector_candidates(&cached_or_static_catalog_for_config(config, &preferences));
    let available = available_providers(config);
    let mut model_selection = select_model(
        config,
        &preferences,
        &model_catalog,
        &usage_summary,
        &available,
    );
    if !preferences.provider_rules.contains_key(&config.provider)
        && config.model != provider_default_model(&config.provider)
    {
        model_selection = ModelSelection {
            provider: config.provider.clone(),
            model: config.model.clone(),
            rule_applied: None,
            fallback_reason: None,
        };
    }
    let selected_config = selected_workflow_config(config, &model_selection);
    let runtime = WorkflowRuntime {
        prompt_profile: selected_prompt_profile(&preferences),
        model_selection,
    };
    let config = &selected_config;

    let prepare_workers = select_prepare_workers(entries.len(), prepare_workers_override);
    telemetry.prepare_workers = prepare_workers;
    let progress_mode = if should_use_provider_batch(config) {
        "batch"
    } else {
        "direct"
    };
    let progress = WorkflowProgress::new(
        progress_mode,
        entries.len(),
        &output_options,
        WorkflowTuiContext {
            repo_path: repo_path.display().to_string(),
            provider: config.provider.clone(),
            model: config.model.clone(),
            mode: progress_mode.to_string(),
            total_files: entries.len(),
            last_screen: preferences.tui.last_screen.clone(),
        },
    );
    let _workflow_tui_session = progress.start_tui_session();
    let wait_start = Instant::now();
    let preparation = prepare_messages_for_entries(
        &repo_path,
        entries,
        config,
        &runtime,
        max_attempts,
        prepare_workers,
        progress.clone(),
    )?;
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
    let can_use_gix_session = prepared_outcomes.iter().all(|outcome| {
        outcome
            .as_ref()
            .map(|prepared| prepared.entry.source_path.is_none())
            .unwrap_or(true)
    });
    let mut gix_session = if gix_commit_backend_enabled() && can_use_gix_session {
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
        progress.commit_started(&entry.path);
        let commit_outcome = match gix_session.as_mut() {
            Some(session) => session.commit_path(&entry.path, &message, staging),
            None if entry.source_path.is_some() => {
                let pathspecs = entry.commit_pathspecs();
                commit_paths_with_staging(&repo_path, &pathspecs, &message, false, staging)
            }
            None => commit_file_with_staging(&repo_path, &entry.path, &message, false, staging),
        };
        let commit_outcome = match commit_outcome {
            Ok(outcome) => outcome,
            Err(err) => {
                if let Some(source_path) = &entry.source_path {
                    unstage_path(&repo_path, source_path);
                }
                unstage_path(&repo_path, &entry.path);
                progress.commit_failed(&entry.path, &err.to_string());
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

        progress.commit_succeeded(&entry.path, &message, commit_hash.as_deref());
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
    progress.push_started();
    let push_outcome = auto_push_if_configured(&repo_path, config, !commits.is_empty());
    progress.push_finished(push_outcome.state);
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
    let _ = record_usage(
        &repo_path,
        &TelemetryRunRecord {
            provider: config.provider.clone(),
            model: config.model.clone(),
            selected_rule: runtime
                .model_selection
                .rule_applied
                .as_ref()
                .map(|rule| rule.as_str().to_string()),
            success: failures.is_empty() && !commits.is_empty(),
            latency_ms: workflow_start.elapsed().as_secs_f64() * 1000.0,
            fallback_count: u64::from(runtime.model_selection.fallback_reason.is_some()),
            request_count: prepared_count as u64,
        },
    );

    if commits.is_empty() && total_entries <= 1 {
        if let Some(failure) = failures.first() {
            progress.finished();
            emit_tui_model_to_stderr(&progress, &output_options);
            return Err(KcmtError::Message(failure.error.clone()));
        }
    }

    if commits.is_empty() && total_entries <= 1 {
        Ok("No changes to commit.\n".to_string())
    } else {
        progress.finished();
        let tui_model_json = output_options
            .tui_model_export
            .then(|| progress.snapshot_json())
            .flatten();
        Ok(render_workflow_output(
            config,
            &commits,
            &failures,
            &push_outcome,
            &telemetry,
            output_options,
            tui_model_json.as_deref(),
        ))
    }
}

fn emit_tui_model_to_stderr(progress: &WorkflowProgress, output_options: &WorkflowOutputOptions) {
    if output_options.tui_model_export {
        if let Some(model_json) = progress.snapshot_json() {
            eprintln!("[kcmt-tui-model] {model_json}");
        }
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
            if code == "!!" {
                return None;
            }
            let (path, source_path) = parse_porcelain_path(&line[3..]);
            if path.is_empty() {
                None
            } else {
                Some(StatusEntry {
                    code,
                    path,
                    source_path,
                })
            }
        })
        .collect()
}

fn parse_porcelain_path(raw_path: &str) -> (String, Option<String>) {
    let path = raw_path.trim();
    if let Some((source, destination)) = path.rsplit_once(" -> ") {
        (
            destination.trim().to_string(),
            Some(source.trim().to_string()).filter(|path| !path.is_empty()),
        )
    } else {
        (path.to_string(), None)
    }
}

fn render_workflow_output(
    config: &kcmt_core::model::WorkflowConfig,
    commits: &[WorkflowCommit],
    failures: &[WorkflowFailure],
    push_outcome: &PushOutcome,
    telemetry: &WorkflowTelemetry,
    options: WorkflowOutputOptions,
    tui_model_json: Option<&str>,
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

    if let Some(model_json) = tui_model_json {
        lines.push(String::new());
        lines.push(format!("[kcmt-tui-model] {model_json}"));
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
    runtime: &WorkflowRuntime,
    max_attempts: usize,
) -> Result<String> {
    if entry.is_deletion() {
        Ok(deletion_commit_message(
            &entry.path,
            config.max_commit_length,
        ))
    } else if let Some(raw_response) = fixture_provider_response() {
        match sanitize_commit_output(&raw_response) {
            Ok(message) => Ok(limit_subject(message, config.max_commit_length)),
            Err(_) => Ok(heuristic_commit_message(
                &entry.path,
                config.max_commit_length,
            )),
        }
    } else if runtime_benchmark_enabled() {
        Ok(heuristic_commit_message(
            &entry.path,
            config.max_commit_length,
        ))
    } else if configured_api_key(config).is_some() {
        invoke_provider_with_fallback(repo_path, entry, config, runtime, max_attempts)
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

fn progress_enabled(options: &WorkflowOutputOptions) -> bool {
    !runtime_benchmark_enabled() && !options.no_progress && !env_truthy("KCMT_NO_PROGRESS")
}

fn start_batch_progress(
    progress: WorkflowProgress,
    model: &str,
    timeout: Duration,
) -> Option<BatchProgress> {
    if !progress.enabled {
        return None;
    }
    progress.summary("submitted");
    let (stop, receiver) = mpsc::channel();
    let model = model.to_string();
    let handle = thread::spawn(move || {
        let started = Instant::now();
        loop {
            match receiver.recv_timeout(Duration::from_secs(15)) {
                Ok(()) | Err(mpsc::RecvTimeoutError::Disconnected) => break,
                Err(mpsc::RecvTimeoutError::Timeout) => {
                    progress.summary("waiting");
                    eprintln!(
                        "kcmt progress detail: mode=batch stage=waiting model={model} timeout={}s elapsed={:.0}s",
                        timeout.as_secs(),
                        started.elapsed().as_secs_f64()
                    );
                }
            }
        }
    });
    Some(BatchProgress { stop, handle })
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
    runtime: &WorkflowRuntime,
    max_attempts: usize,
    prepare_workers: usize,
    progress: WorkflowProgress,
) -> Result<PreparationResult> {
    if should_use_provider_batch(config) {
        if let Some(api_key) = configured_api_key(config) {
            return prepare_provider_batch_messages(
                repo_path,
                entries,
                config,
                runtime,
                &api_key,
                max_attempts,
                progress,
            );
        }
    }

    if prepare_workers <= 1 || entries.len() <= 1 {
        let outcomes = entries
            .into_iter()
            .map(|entry| {
                progress.queued("diff", &entry.path);
                progress.event("llm", &entry.path);
                match commit_message_for_entry(repo_path, &entry, config, runtime, max_attempts) {
                    Ok(message) => {
                        progress.prepared(&entry.path, &message);
                        Ok(PreparedEntry { entry, message })
                    }
                    Err(err) => {
                        let error = err.to_string();
                        progress.prepare_failed(&entry.path, &error);
                        Err(WorkflowFailure::prepare(&entry, error))
                    }
                }
            })
            .collect::<Vec<_>>();
        progress.summary("results");
        return Ok(PreparationResult {
            outcomes,
            telemetry: PreparationTelemetry::default(),
        });
    }

    let outcomes = prepare_messages_in_workers(
        repo_path,
        entries,
        config,
        runtime,
        max_attempts,
        prepare_workers,
        progress.clone(),
    )?;
    progress.summary("results");
    Ok(PreparationResult {
        outcomes,
        telemetry: PreparationTelemetry::default(),
    })
}

fn prepare_messages_in_workers(
    repo_path: &Path,
    entries: Vec<StatusEntry>,
    config: &kcmt_core::model::WorkflowConfig,
    runtime: &WorkflowRuntime,
    max_attempts: usize,
    prepare_workers: usize,
    progress: WorkflowProgress,
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
            let runtime = runtime.clone();
            let progress = progress.clone();
            thread::spawn(move || {
                bucket
                    .into_iter()
                    .map(|(index, entry)| {
                        progress.queued("diff", &entry.path);
                        progress.event("llm", &entry.path);
                        let outcome = match commit_message_for_entry(
                            &repo_path,
                            &entry,
                            &config,
                            &runtime,
                            max_attempts,
                        ) {
                            Ok(message) => {
                                progress.prepared(&entry.path, &message);
                                Ok(PreparedEntry { entry, message })
                            }
                            Err(err) => {
                                let error = err.to_string();
                                progress.prepare_failed(&entry.path, &error);
                                Err(WorkflowFailure::prepare(&entry, error))
                            }
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

fn should_use_provider_batch(config: &kcmt_core::model::WorkflowConfig) -> bool {
    matches!(config.provider.as_str(), "openai" | "xai") && config.use_batch
}

fn commit_message_system_prompt(max_commit_length: usize) -> String {
    format!(
        "You generate strictly valid Conventional Commit messages. Return only the commit message. Keep the subject <={max_commit_length} characters. Prefer a single subject line for simple diffs. Use a body only when it materially improves quality; limit body to three concise factual bullets."
    )
}

fn provider_default_model(provider: &str) -> &'static str {
    match provider {
        "anthropic" => "claude-3-5-haiku-latest",
        "xai" => "grok-code-fast",
        "github" => "openai/gpt-4.1-mini",
        _ => "gpt-5-mini-2025-08-07",
    }
}

fn system_prompt_for_runtime(
    runtime_context: &WorkflowRuntime,
    max_commit_length: usize,
) -> String {
    if runtime_context.prompt_profile.id == "conventional"
        || runtime_context
            .prompt_profile
            .system_instruction
            .trim()
            .is_empty()
    {
        commit_message_system_prompt(max_commit_length)
    } else {
        format!(
            "{}\n{}",
            commit_message_system_prompt(max_commit_length),
            runtime_context.prompt_profile.system_instruction
        )
    }
}

fn prepare_provider_batch_messages(
    repo_path: &Path,
    entries: Vec<StatusEntry>,
    config: &kcmt_core::model::WorkflowConfig,
    runtime_context: &WorkflowRuntime,
    api_key: &str,
    max_attempts: usize,
    progress: WorkflowProgress,
) -> Result<PreparationResult> {
    let mut outcomes = Vec::new();
    let mut batch_entries = Vec::new();
    let mut jobs = Vec::new();
    let mut telemetry = PreparationTelemetry::default();
    let system = system_prompt_for_runtime(runtime_context, config.max_commit_length);
    for entry in entries {
        if entry.is_deletion() {
            let message = deletion_commit_message(&entry.path, config.max_commit_length);
            progress.queued("local", &entry.path);
            progress.prepared(&entry.path, &message);
            outcomes.push(Ok(PreparedEntry { entry, message }));
            continue;
        }

        progress.queued("diff", &entry.path);
        let diff_start = Instant::now();
        let diff = match diff_for_entry(repo_path, &entry) {
            Ok(diff) => diff,
            Err(err) => {
                telemetry.diff_preparation_ms += diff_start.elapsed().as_secs_f64() * 1000.0;
                telemetry.diff_preparation_items += 1;
                let error = err.to_string();
                progress.prepare_failed(&entry.path, &error);
                outcomes.push(Err(WorkflowFailure::prepare(&entry, error)));
                continue;
            }
        };
        telemetry.diff_preparation_ms += diff_start.elapsed().as_secs_f64() * 1000.0;
        telemetry.diff_preparation_items += 1;
        let enqueue_start = Instant::now();
        let context = format!("File: {}", entry.path);
        let prompt = build_prompt_with_profile(&diff, &context, &runtime_context.prompt_profile);
        jobs.push(OpenAiBatchJob {
            custom_id: entry.path.clone(),
            messages: vec![
                ProviderMessage::system(system.as_str()),
                ProviderMessage::user(prompt),
            ],
        });
        telemetry.llm_enqueue_ms += enqueue_start.elapsed().as_secs_f64() * 1000.0;
        telemetry.llm_enqueue_items += 1;
        progress.event("queued", &entry.path);
        batch_entries.push(entry);
    }
    if batch_entries.is_empty() {
        return Ok(PreparationResult {
            outcomes,
            telemetry,
        });
    }

    if runtime_benchmark_enabled() {
        let fixture_response = fixture_provider_response();
        for entry in batch_entries {
            let entry_path = entry.path.clone();
            let raw = fixture_response
                .clone()
                .unwrap_or_else(|| heuristic_commit_message(&entry_path, config.max_commit_length));
            match sanitize_commit_output(&raw) {
                Ok(message) => outcomes.push(Ok(PreparedEntry {
                    entry,
                    message: limit_subject(message, config.max_commit_length),
                })),
                Err(_) => outcomes.push(Ok(PreparedEntry {
                    entry,
                    message: heuristic_commit_message(&entry_path, config.max_commit_length),
                })),
            }
        }
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
    let batch_timeout = Duration::from_secs(config.batch_timeout_seconds);
    let batch_wait = start_batch_progress(progress.clone(), batch_model, batch_timeout);
    let batch_output = match runtime.block_on(async {
        match config.provider.as_str() {
            "openai" => {
                OpenAiClient::invoke_batch(
                    &transport,
                    &config.llm_endpoint,
                    api_key,
                    batch_model,
                    &jobs,
                    batch_timeout,
                    Duration::from_millis(50),
                )
                .await
            }
            "xai" => {
                XaiClient::invoke_batch(
                    &transport,
                    &config.llm_endpoint,
                    api_key,
                    batch_model,
                    &jobs,
                    batch_timeout,
                    Duration::from_millis(5000),
                )
                .await
            }
            other => Err(anyhow::anyhow!("unsupported batch provider: {other}")),
        }
    }) {
        Ok(output) => {
            if let Some(batch_wait) = batch_wait {
                batch_wait.stop();
            }
            output
        }
        Err(err) => {
            if let Some(batch_wait) = batch_wait {
                batch_wait.stop();
            }
            let error = provider_error_message("provider batch request failed", &err.to_string());
            outcomes.extend(batch_entries.into_iter().map(|entry| {
                progress.prepare_failed(&entry.path, &error);
                Err(WorkflowFailure::prepare(&entry, error.clone()))
            }));
            progress.summary("results");
            return Ok(PreparationResult {
                outcomes,
                telemetry,
            });
        }
    };
    let mut messages_by_id = std::collections::BTreeMap::new();
    let mut errors_by_id = std::collections::BTreeMap::new();
    for result in batch_output.results {
        let result_path = result.custom_id.clone();
        match sanitize_commit_output(&result.content) {
            Ok(message) => {
                messages_by_id.insert(
                    result.custom_id,
                    limit_subject(message, config.max_commit_length),
                );
            }
            Err(_) => {
                messages_by_id.insert(
                    result.custom_id,
                    heuristic_commit_message(&result_path, config.max_commit_length),
                );
            }
        }
    }
    for failure in batch_output.failures {
        errors_by_id.insert(
            failure.custom_id,
            provider_error_message("provider batch response failed", &failure.error),
        );
    }

    for entry in batch_entries {
        if let Some(message) = messages_by_id.remove(&entry.path) {
            progress.prepared(&entry.path, &message);
            outcomes.push(Ok(PreparedEntry { entry, message }));
        } else if let Some(error) = errors_by_id.remove(&entry.path) {
            progress.prepare_failed(&entry.path, &error);
            outcomes.push(Err(WorkflowFailure::prepare(&entry, error)));
        } else {
            let error = format!("batch response missing {}", entry.path);
            progress.prepare_failed(&entry.path, &error);
            outcomes.push(Err(WorkflowFailure::prepare(&entry, error)));
        }
    }
    progress.summary("results");
    Ok(PreparationResult {
        outcomes,
        telemetry,
    })
}

fn configured_api_key(config: &kcmt_core::model::WorkflowConfig) -> Option<String> {
    let candidate = ProviderCandidate {
        provider: config.provider.clone(),
        model: config.model.clone(),
        endpoint: config.llm_endpoint.clone(),
        api_key_env: config.api_key_env.clone(),
        keychain_account: config
            .providers
            .get(&config.provider)
            .and_then(|entry| entry.keychain_account.clone())
            .unwrap_or_else(|| default_keychain_account(&config.provider)),
    };
    credential_for_candidate(&candidate).map(|credential| credential.0)
}

fn provider_candidates(config: &kcmt_core::model::WorkflowConfig) -> Vec<ProviderCandidate> {
    let mut candidates = Vec::new();
    candidates.push(ProviderCandidate {
        provider: config.provider.clone(),
        model: config.model.clone(),
        endpoint: config.llm_endpoint.clone(),
        api_key_env: config.api_key_env.clone(),
        keychain_account: config
            .providers
            .get(&config.provider)
            .and_then(|entry| entry.keychain_account.clone())
            .unwrap_or_else(|| default_keychain_account(&config.provider)),
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
        let keychain_account = entry
            .and_then(|entry| entry.keychain_account.clone())
            .unwrap_or_else(|| default_keychain_account(&preference.provider));
        candidates.push(ProviderCandidate {
            provider: preference.provider.clone(),
            model: preference.model.clone(),
            endpoint,
            api_key_env,
            keychain_account,
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
        "anthropic" => "https://api.anthropic.com",
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

fn credential_for_candidate(candidate: &ProviderCandidate) -> Option<(String, CredentialSource)> {
    let explicit_provider = env::var("KCMT_EXPLICIT_API_KEY_PROVIDER").ok();
    let explicit_secret = if explicit_provider.as_deref() == Some(candidate.provider.as_str()) {
        env::var("KCMT_EXPLICIT_API_KEY").ok()
    } else {
        None
    };
    let request = CredentialRequest {
        provider: candidate.provider.clone(),
        explicit_secret,
        keychain_account: Some(candidate.keychain_account.clone()),
        env_var: candidate.api_key_env.clone(),
    };
    resolve_credential(&request, &OsKeychainStore)
        .ok()
        .flatten()
        .map(|credential| (credential.secret, credential.source))
}

fn available_providers(config: &kcmt_core::model::WorkflowConfig) -> Vec<String> {
    let candidates = provider_candidates(config);
    if fixture_provider_response().is_some() {
        return candidates
            .into_iter()
            .map(|candidate| candidate.provider)
            .collect();
    }
    if runtime_benchmark_enabled() {
        return vec![config.provider.clone()];
    }
    candidates
        .into_iter()
        .filter(|candidate| credential_for_candidate(candidate).is_some())
        .map(|candidate| candidate.provider)
        .collect()
}

fn selected_workflow_config(
    config: &kcmt_core::model::WorkflowConfig,
    selection: &ModelSelection,
) -> kcmt_core::model::WorkflowConfig {
    let mut selected = config.clone();
    let provider_changed = selected.provider != selection.provider;
    selected.provider = selection.provider.clone();
    selected.model = selection.model.clone();
    if let Some(candidate) = provider_candidates(config)
        .into_iter()
        .find(|candidate| candidate.provider == selection.provider)
    {
        selected.llm_endpoint = candidate.endpoint;
        selected.api_key_env = candidate.api_key_env;
    }
    if !matches!(selected.provider.as_str(), "openai" | "xai") {
        selected.use_batch = false;
        selected.batch_model = None;
    } else if provider_changed {
        selected.batch_model = default_provider_batch_model(&selected.provider).map(str::to_string);
    }
    selected
}

fn default_provider_batch_model(provider: &str) -> Option<&'static str> {
    match provider {
        "openai" => Some("gpt-5-mini-2025-08-07"),
        "xai" => Some("grok-4.3"),
        _ => None,
    }
}

fn invoke_provider_with_fallback(
    repo_path: &Path,
    entry: &StatusEntry,
    config: &kcmt_core::model::WorkflowConfig,
    runtime_context: &WorkflowRuntime,
    max_attempts: usize,
) -> Result<String> {
    let mut last_error = None;
    for candidate in provider_candidates(config) {
        let Some((api_key, _source)) = credential_for_candidate(&candidate) else {
            continue;
        };
        match invoke_provider_candidate(
            repo_path,
            entry,
            &candidate,
            &api_key,
            runtime_context,
            max_attempts,
            config.max_commit_length,
        ) {
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
    runtime_context: &WorkflowRuntime,
    max_attempts: usize,
    max_commit_length: usize,
) -> Result<String> {
    let diff = diff_for_entry(repo_path, entry)?;
    let context = format!("File: {}", entry.path);
    let system = system_prompt_for_runtime(runtime_context, max_commit_length);
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
    let raw = invoke_provider_candidate_with_style(
        &runtime,
        &transport,
        candidate,
        api_key,
        runtime_context,
        &diff,
        &context,
        &system,
        "conventional",
    )?;
    match sanitize_commit_output(&raw) {
        Ok(message) => Ok(limit_subject(message, max_commit_length)),
        Err(first_error) => {
            let retry_raw = invoke_provider_candidate_with_style(
                &runtime,
                &transport,
                candidate,
                api_key,
                runtime_context,
                &diff,
                &context,
                &system,
                "simple",
            )?;
            match sanitize_commit_output(&retry_raw) {
                Ok(message) => Ok(limit_subject(message, max_commit_length)),
                Err(retry_error) => {
                    let fallback = heuristic_commit_message(&entry.path, max_commit_length);
                    if fallback.trim().is_empty() {
                        Err(KcmtError::Message(format!(
                            "{first_error}; simplified prompt retry failed: {retry_error}"
                        )))
                    } else {
                        Ok(fallback)
                    }
                }
            }
        }
    }
}

fn invoke_provider_candidate_with_style(
    runtime: &tokio::runtime::Runtime,
    transport: &AsyncTransport,
    candidate: &ProviderCandidate,
    api_key: &str,
    runtime_context: &WorkflowRuntime,
    diff: &str,
    context: &str,
    system: &str,
    style: &str,
) -> Result<String> {
    let prompt =
        build_prompt_with_profile_style(diff, context, style, &runtime_context.prompt_profile);
    runtime
        .block_on(async {
            match candidate.provider.as_str() {
                "openai" => {
                    let messages = vec![
                        ProviderMessage::system(system),
                        ProviderMessage::user(prompt),
                    ];
                    OpenAiClient::invoke_model(
                        transport,
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
                        transport,
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
                        transport,
                        &candidate.endpoint,
                        api_key,
                        &candidate.model,
                        &messages,
                    )
                    .await
                }
                "anthropic" => {
                    AnthropicClient::invoke_messages(
                        transport,
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
        .map_err(|err| {
            KcmtError::Message(provider_error_message(
                "provider request failed",
                &err.to_string(),
            ))
        })
}

fn provider_error_message(prefix: &str, raw_error: &str) -> String {
    let normalized = normalize_error(raw_error, provider_status_from_error(raw_error));
    format!("{prefix}: {}", normalized.message)
}

fn provider_status_from_error(raw_error: &str) -> Option<u16> {
    raw_error.split_whitespace().find_map(|part| {
        part.trim_matches(|ch: char| !ch.is_ascii_digit())
            .parse::<u16>()
            .ok()
            .filter(|status| (400..=599).contains(status))
    })
}

fn diff_for_entry(repo_path: &Path, entry: &StatusEntry) -> Result<String> {
    if entry.is_untracked() {
        return file_content_diff(repo_path, entry);
    }

    if entry.has_staged_changes() && !entry.has_worktree_changes() {
        if let Some(diff) = git_diff_for_entry(repo_path, entry, &["--cached"])? {
            return Ok(diff);
        }
    }

    if entry.has_staged_changes() && entry.has_worktree_changes() {
        if let Some(diff) = git_diff_for_entry(repo_path, entry, &["HEAD"])? {
            return Ok(diff);
        }
    }

    if entry.has_worktree_changes() {
        if let Some(diff) = git_diff_for_entry(repo_path, entry, &[])? {
            return Ok(diff);
        }
    }

    if let Some(diff) = git_diff_for_entry(repo_path, entry, &["HEAD"])? {
        return Ok(diff);
    }

    file_content_diff(repo_path, entry)
}

fn git_diff_for_entry(
    repo_path: &Path,
    entry: &StatusEntry,
    extra_args: &[&str],
) -> Result<Option<String>> {
    let mut args = vec!["diff", "--no-color", "--no-ext-diff"];
    args.extend(extra_args);
    args.extend(["--", entry.path.as_str()]);
    let output = Command::new("git")
        .current_dir(repo_path)
        .args(args)
        .output()?;
    if output.status.success() {
        let diff = String::from_utf8_lossy(&output.stdout).to_string();
        if !diff.trim().is_empty() {
            return Ok(Some(diff));
        }
    }

    Ok(None)
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
    truncate_to_char_boundary(&mut message, max_commit_length);
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
    truncate_to_char_boundary(&mut limited, max_commit_length);
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
    truncate_to_char_boundary(&mut message, max_commit_length);
    message
}

fn truncate_to_char_boundary(message: &mut String, max_len: usize) {
    if message.len() <= max_len {
        return;
    }

    let mut boundary = max_len;
    while boundary > 0 && !message.is_char_boundary(boundary) {
        boundary -= 1;
    }
    message.truncate(boundary);
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

fn timestamp_utc() -> Result<String> {
    OffsetDateTime::now_utc()
        .format(&Rfc3339)
        .map_err(|err| KcmtError::Message(format!("failed to format UTC timestamp: {err}")))
}

fn commits_per_second(commits: usize, duration_seconds: f64) -> f64 {
    if commits == 0 || duration_seconds <= 0.0 {
        0.0
    } else {
        commits as f64 / duration_seconds
    }
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
    let timestamp = timestamp_utc()?;
    if runtime_benchmark_enabled() {
        telemetry.record_since("snapshot", snapshot_start, 1);
        telemetry.record_since("workflow_total", workflow_start, total_entries);
        let duration_seconds = workflow_start.elapsed().as_secs_f64();
        let rate_commits_per_sec = commits_per_second(overall_success, duration_seconds);
        let stats = json!({
            "total_files": total_entries,
            "diffs_built": prepared_count,
            "requests": prepared_count,
            "responses": prepared_count,
            "prepared": prepared_count,
            "processed": commits.len() + failures.len(),
            "successes": overall_success,
            "failures": overall_failure,
            "elapsed": duration_seconds,
            "rate": rate_commits_per_sec
        });
        let snapshot = json!({
            "schema_version": 1,
            "timestamp": timestamp,
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
            "duration_seconds": duration_seconds,
            "rate_commits_per_sec": rate_commits_per_sec,
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
            "stats": stats,
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
    let duration_seconds = workflow_start.elapsed().as_secs_f64();
    let rate_commits_per_sec = commits_per_second(overall_success, duration_seconds);
    let stats = json!({
        "total_files": total_entries,
        "diffs_built": prepared_count,
        "requests": prepared_count,
        "responses": prepared_count,
        "prepared": prepared_count,
        "processed": commits.len() + failures.len(),
        "successes": overall_success,
        "failures": overall_failure,
        "elapsed": duration_seconds,
        "rate": rate_commits_per_sec
    });

    let snapshot = json!({
        "schema_version": 1,
        "timestamp": timestamp,
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
        "duration_seconds": duration_seconds,
        "rate_commits_per_sec": rate_commits_per_sec,
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
        "stats": stats,
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
        default_provider_batch_model, deletion_commit_message, diff_for_entry,
        git_common_config_path, git_config_has_include, git_config_value, heuristic_commit_message,
        local_origin_remote_probe, parse_status_entries, select_prepare_workers,
        selected_workflow_config, OriginRemoteProbe, StatusEntry, WorkflowOutputOptions,
        WorkflowProgress,
    };
    use kcmt_core::git::commit_file::CommitStaging;
    use kcmt_core::model::{ModelPreference, ProviderConfigEntry, WorkflowConfig};
    use kcmt_core::selector::ModelSelection;
    use kcmt_tui::WorkflowTuiContext;
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

    fn tui_context(total_files: usize) -> WorkflowTuiContext {
        WorkflowTuiContext {
            repo_path: "/repo".to_string(),
            provider: "openai".to_string(),
            model: "gpt-test".to_string(),
            mode: "direct".to_string(),
            total_files,
            last_screen: None,
        }
    }

    fn git(repo: &Path, args: &[&str]) {
        let output = std::process::Command::new("git")
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

    #[test]
    fn parses_porcelain_entries() {
        let entries = parse_status_entries(" M alpha.py\n?? beta.py\n!! ignored.log\n");

        assert_eq!(
            entries,
            vec![
                StatusEntry {
                    code: " M".to_string(),
                    path: "alpha.py".to_string(),
                    source_path: None
                },
                StatusEntry {
                    code: "??".to_string(),
                    path: "beta.py".to_string(),
                    source_path: None
                },
            ]
        );
    }

    #[test]
    fn workflow_tui_echo_respects_no_progress() {
        let options = WorkflowOutputOptions {
            tui: true,
            no_progress: true,
            ..WorkflowOutputOptions::default()
        };

        let progress = WorkflowProgress::new("direct", 1, &options, tui_context(1));

        assert!(!progress.enabled);
        assert!(!progress.tui_echo);
    }

    #[test]
    fn parses_porcelain_rename_and_copy_destinations() {
        let entries =
            parse_status_entries("R  old_name.py -> new_name.py\n C template.py -> copied.py\n");

        assert_eq!(
            entries,
            vec![
                StatusEntry {
                    code: "R ".to_string(),
                    path: "new_name.py".to_string(),
                    source_path: Some("old_name.py".to_string())
                },
                StatusEntry {
                    code: " C".to_string(),
                    path: "copied.py".to_string(),
                    source_path: Some("template.py".to_string())
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
                source_path: None,
            };
            let diff = diff_for_entry(&repo, &entry).expect("new file diff");

            assert_eq!(diff, "New or changed file: new.md\n\n# New file\n");
        }
    }

    #[test]
    fn diff_for_tracked_entry_uses_unified_git_diff() {
        let repo = unique_temp_dir("tracked-diff");
        git(&repo, &["init", "-q"]);
        fs::write(repo.join("tracked.py"), "print('seed')\n").expect("tracked seed");
        git(&repo, &["add", "tracked.py"]);
        git(&repo, &["commit", "-m", "chore(repo): seed"]);
        fs::write(repo.join("tracked.py"), "print('changed')\n").expect("tracked change");
        let entry = StatusEntry {
            code: " M".to_string(),
            path: "tracked.py".to_string(),
            source_path: None,
        };

        let diff = diff_for_entry(&repo, &entry).expect("tracked diff");

        assert!(diff.starts_with("diff --git a/tracked.py b/tracked.py"));
        assert!(diff.contains("-print('seed')"));
        assert!(diff.contains("+print('changed')"));
    }

    #[test]
    fn diff_for_staged_only_entry_uses_cached_diff() {
        let repo = unique_temp_dir("staged-diff");
        git(&repo, &["init", "-q"]);
        fs::write(repo.join("tracked.py"), "print('seed')\n").expect("tracked seed");
        git(&repo, &["add", "tracked.py"]);
        git(&repo, &["commit", "-m", "chore(repo): seed"]);
        fs::write(repo.join("tracked.py"), "print('staged')\n").expect("staged change");
        git(&repo, &["add", "tracked.py"]);
        let entry = StatusEntry {
            code: "M ".to_string(),
            path: "tracked.py".to_string(),
            source_path: None,
        };

        let diff = diff_for_entry(&repo, &entry).expect("staged diff");

        assert!(diff.starts_with("diff --git a/tracked.py b/tracked.py"));
        assert!(diff.contains("-print('seed')"));
        assert!(diff.contains("+print('staged')"));
    }

    #[test]
    fn diff_for_mixed_entry_uses_head_diff() {
        let repo = unique_temp_dir("mixed-diff");
        git(&repo, &["init", "-q"]);
        fs::write(repo.join("tracked.py"), "print('seed')\n").expect("tracked seed");
        git(&repo, &["add", "tracked.py"]);
        git(&repo, &["commit", "-m", "chore(repo): seed"]);
        fs::write(repo.join("tracked.py"), "print('staged')\n").expect("staged change");
        git(&repo, &["add", "tracked.py"]);
        fs::write(repo.join("tracked.py"), "print('working')\n").expect("working change");
        let entry = StatusEntry {
            code: "MM".to_string(),
            path: "tracked.py".to_string(),
            source_path: None,
        };

        let diff = diff_for_entry(&repo, &entry).expect("mixed diff");

        assert!(diff.starts_with("diff --git a/tracked.py b/tracked.py"));
        assert!(diff.contains("-print('seed')"));
        assert!(diff.contains("+print('working')"));
    }

    #[test]
    fn diff_for_binary_tracked_entry_uses_git_binary_summary() {
        let repo = unique_temp_dir("binary-diff");
        git(&repo, &["init", "-q"]);
        fs::write(repo.join("image.bin"), [0_u8, 159, 146, 150]).expect("binary seed");
        git(&repo, &["add", "image.bin"]);
        git(&repo, &["commit", "-m", "chore(repo): seed"]);
        fs::write(repo.join("image.bin"), [0_u8, 159, 146, 151]).expect("binary change");
        let entry = StatusEntry {
            code: " M".to_string(),
            path: "image.bin".to_string(),
            source_path: None,
        };

        let diff = diff_for_entry(&repo, &entry).expect("binary diff");

        assert!(diff.contains("Binary files"));
        assert!(diff.contains("image.bin"));
    }

    #[test]
    fn tracked_statuses_can_commit_directly_without_staging() {
        for code in [" M", " D", "M ", "D ", "MM", "MD", "DM"] {
            let entry = StatusEntry {
                code: code.to_string(),
                path: "tracked.py".to_string(),
                source_path: None,
            };
            assert_eq!(entry.commit_staging(), CommitStaging::DirectPath, "{code}");
        }

        for code in ["??", "A ", "AM", "UU", "R ", " C"] {
            let entry = StatusEntry {
                code: code.to_string(),
                path: "new.py".to_string(),
                source_path: None,
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
    fn truncates_heuristic_message_on_char_boundary() {
        let message = heuristic_commit_message("src/café.py", "chore(src): update café".len() - 1);

        assert_eq!(message, "chore(src): update caf");
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
    fn selected_workflow_config_uses_provider_specific_batch_model_after_selection() {
        let mut providers = std::collections::HashMap::new();
        providers.insert(
            "openai".to_string(),
            ProviderConfigEntry {
                endpoint: Some("https://api.openai.com/v1".to_string()),
                api_key_env: Some("OPENAI_TEST_KEY".to_string()),
                preferred_model: Some("gpt-5-mini-2025-08-07".to_string()),
                ..ProviderConfigEntry::default()
            },
        );
        providers.insert(
            "xai".to_string(),
            ProviderConfigEntry {
                endpoint: Some("https://api.x.ai/v1".to_string()),
                api_key_env: Some("XAI_TEST_KEY".to_string()),
                preferred_model: Some("grok-code-fast".to_string()),
                ..ProviderConfigEntry::default()
            },
        );
        let config = WorkflowConfig {
            provider: "openai".to_string(),
            model: "gpt-5-mini-2025-08-07".to_string(),
            llm_endpoint: "https://api.openai.com/v1".to_string(),
            api_key_env: "OPENAI_TEST_KEY".to_string(),
            use_batch: true,
            batch_model: Some("gpt-5-mini-2025-08-07".to_string()),
            providers,
            model_priority: vec![
                ModelPreference {
                    provider: "openai".to_string(),
                    model: "gpt-5-mini-2025-08-07".to_string(),
                },
                ModelPreference {
                    provider: "xai".to_string(),
                    model: "grok-code-fast".to_string(),
                },
            ],
            ..WorkflowConfig::default()
        };
        let selection = ModelSelection {
            provider: "xai".to_string(),
            model: "grok-code-fast".to_string(),
            rule_applied: None,
            fallback_reason: None,
        };

        let selected = selected_workflow_config(&config, &selection);

        assert_eq!(selected.provider, "xai");
        assert_eq!(selected.model, "grok-code-fast");
        assert_eq!(selected.llm_endpoint, "https://api.x.ai/v1");
        assert_eq!(selected.api_key_env, "XAI_TEST_KEY");
        assert_eq!(selected.batch_model.as_deref(), Some("grok-4.3"));
        assert_eq!(default_provider_batch_model("xai"), Some("grok-4.3"));
    }

    #[test]
    fn selected_workflow_config_disables_batch_for_non_batch_provider_selection() {
        let config = WorkflowConfig {
            provider: "openai".to_string(),
            model: "gpt-5-mini-2025-08-07".to_string(),
            llm_endpoint: "https://api.openai.com/v1".to_string(),
            api_key_env: "OPENAI_TEST_KEY".to_string(),
            use_batch: true,
            batch_model: Some("gpt-5-mini-2025-08-07".to_string()),
            ..WorkflowConfig::default()
        };
        let selection = ModelSelection {
            provider: "anthropic".to_string(),
            model: "claude-3-5-haiku-latest".to_string(),
            rule_applied: None,
            fallback_reason: None,
        };

        let selected = selected_workflow_config(&config, &selection);

        assert_eq!(selected.provider, "anthropic");
        assert!(!selected.use_batch);
        assert!(selected.batch_model.is_none());
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
