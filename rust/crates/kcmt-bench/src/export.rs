use crate::model::{BenchmarkRun, RuntimeBenchmarkRun};

pub fn export_json(run: &BenchmarkRun) -> Result<String, serde_json::Error> {
    serde_json::to_string_pretty(run)
}

pub fn export_csv(run: &BenchmarkRun) -> String {
    let mut out = String::from(
        "provider,model,avg_latency_ms,avg_cost_usd,quality,success_rate,runs\n",
    );
    for result in &run.results {
        out.push_str(&format!(
            "{},{},{:.3},{:.6},{:.2},{:.4},{}\n",
            result.provider,
            result.model,
            result.avg_latency_ms,
            result.avg_cost_usd,
            result.quality,
            result.success_rate,
            result.runs
        ));
    }
    out
}

pub fn export_runtime_json(
    run: &RuntimeBenchmarkRun,
) -> Result<String, serde_json::Error> {
    serde_json::to_string_pretty(run)
}
