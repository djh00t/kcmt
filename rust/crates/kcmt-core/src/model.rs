//! Shared domain entities aligned with migration data-model documentation.

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
