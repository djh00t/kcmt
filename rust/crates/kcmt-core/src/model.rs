//! Shared domain entities aligned with migration data-model documentation.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct WorkflowConfig {
    pub provider: String,
    pub model: String,
    pub llm_endpoint: String,
    pub api_key_env: String,
    pub git_repo_path: String,
    pub max_commit_length: usize,
    pub auto_push: bool,
    pub use_batch: bool,
    pub batch_model: Option<String>,
    pub batch_timeout_seconds: u64,
    pub providers: HashMap<String, ProviderConfigEntry>,
    pub model_priority: Vec<ModelPreference>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct ProviderConfigEntry {
    pub name: Option<String>,
    pub endpoint: Option<String>,
    pub api_key_env: Option<String>,
    pub keychain_account: Option<String>,
    pub preferred_model: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct ModelPreference {
    pub provider: String,
    pub model: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ProviderProfile {
    pub provider_id: String,
    pub display_name: String,
    pub endpoint: String,
    pub api_key_env: String,
    pub preferred_model: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ChangeSet {
    pub file_path: String,
    pub change_type: String,
    pub diff_content: String,
    pub staged: bool,
    pub ignored: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CommitRecommendation {
    pub subject: String,
    pub body: Option<String>,
    pub raw_message: String,
    pub provider_id: String,
    pub model: String,
    pub success: bool,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct WorkflowStageTiming {
    pub stage: String,
    pub duration_ms: f64,
    pub items: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct WorkflowRun {
    pub schema_version: u32,
    pub repo_path: String,
    pub provider: String,
    pub model: String,
    pub prepared: Vec<PreparedCommit>,
    pub commits: Vec<CommitResult>,
    pub telemetry: Vec<WorkflowStageTiming>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct PreparedCommit {
    pub file_path: String,
    pub change_type: String,
    pub diff_content: String,
    pub prompt: String,
    pub provider: String,
    pub model: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct CommitResult {
    pub file_path: String,
    pub message: String,
    pub commit_hash: Option<String>,
    pub success: bool,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct ProviderRequest {
    pub provider: String,
    pub model: String,
    pub endpoint: String,
    pub prompt: String,
    pub batch: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct ProviderResponse {
    pub provider: String,
    pub model: String,
    pub raw_message: String,
    pub sanitized_message: Option<String>,
    pub success: bool,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct QualityEvaluation {
    pub message: String,
    pub valid_conventional_commit: bool,
    pub score: f64,
    pub failures: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct BenchmarkResult {
    pub provider: String,
    pub model: String,
    pub avg_latency_ms: f64,
    pub avg_cost_usd: f64,
    pub quality: f64,
    pub success_rate: f64,
    pub runs: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct BenchmarkRun {
    pub timestamp: String,
    pub results: Vec<BenchmarkResult>,
    pub exclusions: Vec<BenchmarkExclusion>,
    pub schema_version: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct BenchmarkExclusion {
    pub provider: String,
    pub model: String,
    pub reason: String,
    pub detail: Option<String>,
}
