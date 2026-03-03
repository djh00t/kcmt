use std::time::{SystemTime, UNIX_EPOCH};

use crate::model::{BenchmarkResult, BenchmarkRun};

pub fn run_benchmark(repo_path: &str) -> BenchmarkRun {
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|dur| dur.as_secs().to_string())
        .unwrap_or_else(|_| "0".to_string());

    BenchmarkRun {
        schema_version: crate::BENCHMARK_SCHEMA_VERSION,
        timestamp,
        repo_path: repo_path.to_string(),
        results: vec![BenchmarkResult {
            provider: "openai".to_string(),
            model: "gpt-4o-mini".to_string(),
            avg_latency_ms: 0.0,
            avg_cost_usd: 0.0,
            quality: 0.0,
            success_rate: 1.0,
            runs: 1,
        }],
    }
}
