use std::path::PathBuf;

use kcmt_bench::export::export_runtime_json;
use kcmt_bench::model::{RuntimeBenchmarkRun, RuntimeScenarioStatus};
use kcmt_bench::runner::run_runtime_benchmark;

use crate::args::{BenchmarkArgs, BenchmarkCommand, BenchmarkRuntime};

pub fn run_benchmark_command(repo_path: PathBuf, benchmark: BenchmarkArgs) -> i32 {
    match benchmark.command {
        Some(BenchmarkCommand::Runtime(runtime)) => {
            run_runtime_benchmark_command(repo_path, runtime.runtime, runtime.iterations, runtime.json, runtime.rust_bin)
        }
        None => {
            eprintln!("Benchmark mode requires a subcommand. Try `kcmt benchmark runtime --help`.");
            1
        }
    }
}

fn run_runtime_benchmark_command(
    repo_path: PathBuf,
    runtime: BenchmarkRuntime,
    iterations: usize,
    json: bool,
    rust_bin: Option<String>,
) -> i32 {
    let runtime_label = match runtime {
        BenchmarkRuntime::Python => "python",
        BenchmarkRuntime::Rust => "rust",
        BenchmarkRuntime::Both => "both",
    };
    let rust_bin_path = rust_bin.as_ref().map(PathBuf::from);

    match run_runtime_benchmark(
        &repo_path,
        runtime_label,
        iterations as u32,
        rust_bin_path.as_deref(),
    ) {
        Ok(report) => {
            if json {
                match export_runtime_json(&report) {
                    Ok(rendered) => println!("{rendered}"),
                    Err(err) => {
                        eprintln!("{err}");
                        return 1;
                    }
                }
            } else {
                print!("{}", render_runtime_report(&report));
            }

            if report
                .results
                .iter()
                .any(|result| result.status == RuntimeScenarioStatus::Failed)
            {
                1
            } else {
                0
            }
        }
        Err(err) => {
            eprintln!("{err}");
            1
        }
    }
}

fn render_runtime_report(report: &RuntimeBenchmarkRun) -> String {
    let mut lines = vec![
        "# kcmt Runtime Benchmark".to_string(),
        String::new(),
        format!("- Timestamp: {}", report.timestamp),
        format!("- Command set: {}", report.command_set),
        format!("- Corpora: {}", report.corpora.join(", ")),
        String::new(),
        "| Runtime | Scenarios | Passed | Failed | Excluded | Median wall time (ms) |"
            .to_string(),
        "| --- | --- | --- | --- | --- | --- |".to_string(),
    ];

    append_summary_row(&mut lines, "python", &report.summary.python);
    append_summary_row(&mut lines, "rust", &report.summary.rust);

    lines.push(String::new());
    lines.push(
        "| Scenario | Runtime | Status | Iterations | Total wall time (ms) | Median (ms) | Failure |"
            .to_string(),
    );
    lines.push("| --- | --- | --- | --- | --- | --- | --- |".to_string());
    for result in &report.results {
        let median = result
            .median_time_ms
            .map(|value| format!("{value:.2}"))
            .unwrap_or_else(|| "-".to_string());
        let failure = result
            .failure_reason
            .as_deref()
            .unwrap_or("-");
        lines.push(format!(
            "| {} | {} | {:?} | {} | {:.2} | {} | {} |",
            result.scenario_id,
            runtime_label(result.runtime),
            result.status,
            result.iterations,
            result.wall_time_ms,
            median,
            failure.replace('|', "\\|"),
        ));
    }

    let mut output = lines.join("\n");
    output.push('\n');
    output
}

fn append_summary_row(
    lines: &mut Vec<String>,
    runtime: &str,
    summary: &kcmt_bench::model::RuntimeSummary,
) {
    let median = summary
        .median_wall_time_ms
        .map(|value| format!("{value:.2}"))
        .unwrap_or_else(|| "-".to_string());
    lines.push(format!(
        "| {runtime} | {} | {} | {} | {} | {median} |",
        summary.scenario_count, summary.passed, summary.failed, summary.excluded
    ));
}

fn runtime_label(runtime: kcmt_bench::model::RuntimeKind) -> &'static str {
    match runtime {
        kcmt_bench::model::RuntimeKind::Python => "python",
        kcmt_bench::model::RuntimeKind::Rust => "rust",
    }
}
