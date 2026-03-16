use serde::{Deserialize, Serialize};

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
    pub schema_version: u32,
    pub timestamp: String,
    pub repo_path: String,
    pub results: Vec<BenchmarkResult>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum RuntimeKind {
    Python,
    Rust,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum RuntimeScenarioStatus {
    Passed,
    Failed,
    Excluded,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeBenchmarkResult {
    pub scenario_id: String,
    pub workflow_contract_id: String,
    pub corpus_id: String,
    pub runtime: RuntimeKind,
    pub command_label: String,
    pub iterations: u32,
    pub status: RuntimeScenarioStatus,
    pub wall_time_ms: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub median_time_ms: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub peak_rss_bytes: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub exit_code: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub failure_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RuntimeSummary {
    pub scenario_count: u32,
    pub passed: u32,
    pub failed: u32,
    pub excluded: u32,
    pub median_wall_time_ms: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeBenchmarkSummary {
    pub python: RuntimeSummary,
    pub rust: RuntimeSummary,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeBenchmarkRun {
    pub schema_version: String,
    pub timestamp: String,
    pub command_set: String,
    pub corpora: Vec<String>,
    pub results: Vec<RuntimeBenchmarkResult>,
    pub summary: RuntimeBenchmarkSummary,
}
