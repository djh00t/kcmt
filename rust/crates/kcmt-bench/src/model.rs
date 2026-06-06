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
    pub stage_timings: Vec<RuntimeStageTiming>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RuntimeStageTiming {
    pub stage: String,
    pub duration_ms: f64,
    pub items: u32,
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
    pub snapshot: RuntimeBenchmarkSnapshot,
    pub results: Vec<RuntimeBenchmarkResult>,
    pub scenario_matrix: Vec<RuntimeScenarioMatrixRow>,
    pub summary: RuntimeBenchmarkSummary,
    pub scorecard: RuntimeBenchmarkScorecard,
    pub optimization_iterations: Vec<OptimizationIteration>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeBenchmarkSnapshot {
    pub snapshot_id: String,
    pub benchmark_kind: String,
    pub schema_version: String,
    pub timestamp: String,
    pub command_set: String,
    pub corpora: Vec<String>,
    pub result_count: u32,
    pub secret_free: bool,
    pub provider_benchmark_kind: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeScenarioMatrixRow {
    pub scenario_id: String,
    pub workflow_contract_id: String,
    pub corpus_id: String,
    pub command_label: String,
    pub python: RuntimeScenarioMatrixCell,
    pub rust: RuntimeScenarioMatrixCell,
    pub comparison: RuntimeScenarioComparison,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeScenarioMatrixCell {
    pub status: RuntimeScenarioStatus,
    pub iterations: u32,
    pub wall_time_ms: Option<f64>,
    pub median_time_ms: Option<f64>,
    pub throughput_commits_per_sec: Option<f64>,
    pub quality_score: Option<f64>,
    pub stage_timings: Vec<RuntimeStageTiming>,
    pub failure_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeScenarioComparison {
    pub comparable: bool,
    pub fastest_runtime: Option<RuntimeKind>,
    pub median_delta_ms: Option<f64>,
    pub speedup_ratio: Option<f64>,
    pub stage_deltas: Vec<RuntimeStageDelta>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeStageDelta {
    pub stage: String,
    pub python_ms: Option<f64>,
    pub rust_ms: Option<f64>,
    pub delta_ms: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeBenchmarkScorecard {
    pub measurement_basis: String,
    pub quality_score_definition: String,
    pub throughput_definition: String,
    pub python_quality_score: Option<f64>,
    pub rust_quality_score: Option<f64>,
    pub provider_throughput_included: bool,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum OptimizationMeasurementStatus {
    Measured,
    Planned,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationIteration {
    pub iteration: u32,
    pub label: String,
    pub baseline: bool,
    pub measurement_status: OptimizationMeasurementStatus,
    pub median_wall_time_ms: Option<f64>,
    pub throughput_commits_per_sec: Option<f64>,
    pub quality_score: Option<f64>,
    pub failures: Option<u32>,
    pub next_bottleneck: String,
}
