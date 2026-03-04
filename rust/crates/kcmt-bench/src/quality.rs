use crate::model::BenchmarkRun;

pub fn average_quality(run: &BenchmarkRun) -> f64 {
    if run.results.is_empty() {
        return 0.0;
    }
    let sum: f64 = run.results.iter().map(|result| result.quality).sum();
    sum / run.results.len() as f64
}
