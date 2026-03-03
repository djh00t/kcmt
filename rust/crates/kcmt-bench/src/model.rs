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
