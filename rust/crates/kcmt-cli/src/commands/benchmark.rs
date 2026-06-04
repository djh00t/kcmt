use std::cmp::Ordering;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use kcmt_bench::export::export_runtime_json;
use kcmt_bench::model::{RuntimeBenchmarkRun, RuntimeScenarioStatus};
use kcmt_bench::runner::run_runtime_benchmark;
use kcmt_core::config::loader::{load_config, ConfigOverrides};
use kcmt_core::message::{build_prompt, sanitize_commit_output, validate_conventional_commit};
use kcmt_core::model::{BenchmarkExclusion, BenchmarkResult};
use kcmt_provider::clients::{
    AnthropicClient, GitHubModelsClient, OpenAiClient, ProviderMessage, XaiClient,
};
use kcmt_provider::transport::{AsyncTransport, RetryPolicy};
use serde_json::json;

use crate::args::{BenchmarkArgs, BenchmarkCommand, BenchmarkRuntime, CliArgs};
use crate::commands::history::state_dir;

const PROVIDER_DEFAULTS: &[(&str, &str, &str, &str)] = &[
    (
        "openai",
        "gpt-5-mini-2025-08-07",
        "https://api.openai.com/v1",
        "OPENAI_API_KEY",
    ),
    (
        "anthropic",
        "claude-3-5-haiku-latest",
        "https://api.anthropic.com/v1",
        "ANTHROPIC_API_KEY",
    ),
    (
        "xai",
        "grok-code-fast",
        "https://api.x.ai/v1",
        "XAI_API_KEY",
    ),
    (
        "github",
        "openai/gpt-4.1-mini",
        "https://models.github.ai/inference",
        "GITHUB_TOKEN",
    ),
];

const SAMPLE_DIFFS: &[(&str, &str)] = &[
    (
        "modified_function",
        "diff --git a/src/app.py b/src/app.py\n@@\n-def greet():\n-    return 'hi'\n+def greet(name):\n+    return f'hi {name}'\n",
    ),
    (
        "new_config",
        "diff --git a/config/app.toml b/config/app.toml\nnew file mode 100644\n@@\n+[runtime]\n+enabled = true\n",
    ),
    (
        "deleted_file",
        "diff --git a/docs/old.md b/docs/old.md\ndeleted file mode 100644\n@@\n-legacy notes\n",
    ),
    (
        "tests",
        "diff --git a/tests/test_app.py b/tests/test_app.py\n@@\n+def test_greet():\n+    assert greet('Klingon') == 'hi Klingon'\n",
    ),
    (
        "large_diff",
        "diff --git a/src/workflow.py b/src/workflow.py\n@@\n+def prepare():\n+    return ['scan', 'diff', 'llm', 'commit']\n+def validate(message):\n+    return message.startswith(('feat', 'fix', 'chore'))\n",
    ),
];

#[derive(Debug, Clone)]
struct BenchmarkCandidate {
    provider: String,
    model: String,
    endpoint: String,
    api_key_env: String,
}

pub fn run_provider_benchmark(repo_path: PathBuf, args: &CliArgs) -> i32 {
    let overrides = ConfigOverrides {
        provider: args.provider.clone(),
        model: args.model.clone(),
        endpoint: args.endpoint.clone(),
        api_key_env: args.api_key_env.clone(),
        repo_path: Some(repo_path.clone()),
        max_commit_length: args.max_commit_length,
        auto_push: args.auto_push_override(),
        use_batch: args.use_batch_override(),
        batch_model: args.batch_model.clone(),
        batch_timeout_seconds: args.batch_timeout_seconds,
        file_limit: args.limit,
    };

    let config = match load_config(&repo_path, &overrides) {
        Ok(config) => config,
        Err(err) => {
            eprintln!("{err}");
            return 1;
        }
    };

    let candidates = benchmark_candidates(args, &config);
    let mut results = Vec::new();
    let mut exclusions = Vec::new();
    for candidate in candidates {
        let Some(api_key) = env::var(&candidate.api_key_env)
            .ok()
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty())
        else {
            exclusions.push(BenchmarkExclusion {
                provider: candidate.provider,
                model: candidate.model,
                reason: "missing_api_key".to_string(),
                detail: Some(format!("{} is not set", candidate.api_key_env)),
            });
            continue;
        };

        match benchmark_candidate(&candidate, &api_key, args.benchmark_timeout) {
            Ok(result) => results.push(result),
            Err(err) => exclusions.push(BenchmarkExclusion {
                provider: candidate.provider,
                model: candidate.model,
                reason: "provider_error".to_string(),
                detail: Some(err.to_string()),
            }),
        }
    }

    let timestamp = benchmark_timestamp();
    if let Err(err) = persist_provider_benchmark(&repo_path, &timestamp, &results, &exclusions) {
        eprintln!("warning: failed to persist benchmark snapshot: {err}");
    }

    if results.is_empty() {
        print!("{}", render_provider_benchmark(&results, &exclusions));
        return 1;
    }

    print!("{}", render_provider_benchmark(&results, &exclusions));
    if args.benchmark_json {
        println!("{}", render_provider_benchmark_json(&results, &exclusions));
    }
    if args.benchmark_csv {
        print!("{}", render_provider_benchmark_csv(&results, &exclusions));
    }
    0
}

pub fn run_benchmark_command(repo_path: PathBuf, benchmark: BenchmarkArgs) -> i32 {
    match benchmark.command {
        Some(BenchmarkCommand::Runtime(runtime)) => run_runtime_benchmark_command(
            runtime
                .repo_path
                .as_deref()
                .map(PathBuf::from)
                .unwrap_or(repo_path),
            runtime.runtime,
            runtime.iterations,
            runtime.json,
            runtime.rust_bin,
        ),
        None => {
            eprintln!("Benchmark mode requires a subcommand. Try `kcmt benchmark runtime --help`.");
            1
        }
    }
}

fn benchmark_candidates(
    args: &CliArgs,
    config: &kcmt_core::model::WorkflowConfig,
) -> Vec<BenchmarkCandidate> {
    let providers: Vec<String> = args.provider.clone().map_or_else(
        || {
            PROVIDER_DEFAULTS
                .iter()
                .map(|(provider, _, _, _)| provider.to_string())
                .collect()
        },
        |provider| vec![provider],
    );
    let mut candidates = Vec::new();
    for provider in providers {
        let (default_model, default_endpoint, default_api_key_env) = provider_default(&provider)
            .unwrap_or((
                config.model.as_str(),
                config.llm_endpoint.as_str(),
                config.api_key_env.as_str(),
            ));
        let entry = config.providers.get(&provider);
        let endpoint = entry
            .and_then(|entry| entry.endpoint.as_deref())
            .unwrap_or(default_endpoint)
            .to_string();
        let api_key_env = entry
            .and_then(|entry| entry.api_key_env.as_deref())
            .unwrap_or(default_api_key_env)
            .to_string();
        let mut models = if let Some(model) = args.model.as_ref() {
            vec![model.clone()]
        } else {
            vec![entry
                .and_then(|entry| entry.preferred_model.clone())
                .unwrap_or_else(|| default_model.to_string())]
        };
        if args.benchmark_limit > 0 {
            models.truncate(args.benchmark_limit);
        }
        for model in models {
            candidates.push(BenchmarkCandidate {
                provider: provider.clone(),
                model,
                endpoint: endpoint.clone(),
                api_key_env: api_key_env.clone(),
            });
        }
    }
    candidates
}

fn provider_default(provider: &str) -> Option<(&'static str, &'static str, &'static str)> {
    PROVIDER_DEFAULTS
        .iter()
        .find(|(item, _, _, _)| *item == provider)
        .map(|(_, model, endpoint, api_key_env)| (*model, *endpoint, *api_key_env))
}

fn benchmark_candidate(
    candidate: &BenchmarkCandidate,
    api_key: &str,
    timeout: Option<f64>,
) -> anyhow::Result<BenchmarkResult> {
    let mut latencies = Vec::new();
    let mut costs = Vec::new();
    let mut qualities = Vec::new();
    let mut successes = 0u32;
    for (sample_name, diff) in SAMPLE_DIFFS {
        let prompt = build_prompt(diff, &format!("Sample: {sample_name}"), "conventional");
        let started = Instant::now();
        let raw = invoke_benchmark_provider(candidate, api_key, &prompt, timeout)?;
        let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
        latencies.push(elapsed_ms);
        costs.push(estimate_cost_usd(diff, &raw));
        match sanitize_commit_output(&raw) {
            Ok(message) => {
                successes += 1;
                qualities.push(score_message(&message));
            }
            Err(_) => qualities.push(0.0),
        }
    }

    let runs = SAMPLE_DIFFS.len() as u32;
    Ok(BenchmarkResult {
        provider: candidate.provider.clone(),
        model: candidate.model.clone(),
        avg_latency_ms: average(&latencies),
        avg_cost_usd: average(&costs),
        quality: average(&qualities),
        success_rate: successes as f64 / runs as f64,
        runs,
    })
}

fn invoke_benchmark_provider(
    candidate: &BenchmarkCandidate,
    api_key: &str,
    prompt: &str,
    timeout: Option<f64>,
) -> anyhow::Result<String> {
    if let Ok(raw_response) = env::var("KCMT_PROVIDER_RESPONSE") {
        return Ok(raw_response);
    }

    let timeout = timeout
        .filter(|value| *value > 0.0)
        .map(Duration::from_secs_f64)
        .unwrap_or_else(|| Duration::from_secs(60));
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    let transport = AsyncTransport::new(
        timeout,
        RetryPolicy {
            max_attempts: 3,
            base_backoff: Duration::from_millis(250),
        },
    )?;
    let system = "You generate strictly valid Conventional Commit messages.";
    runtime.block_on(async {
        match candidate.provider.as_str() {
            "openai" => {
                let messages = vec![
                    ProviderMessage::system(system),
                    ProviderMessage::user(prompt.to_string()),
                ];
                OpenAiClient::invoke_chat(
                    &transport,
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
                    ProviderMessage::user(prompt.to_string()),
                ];
                XaiClient::invoke_chat(
                    &transport,
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
                    ProviderMessage::user(prompt.to_string()),
                ];
                GitHubModelsClient::invoke_chat(
                    &transport,
                    &candidate.endpoint,
                    api_key,
                    &candidate.model,
                    &messages,
                )
                .await
            }
            "anthropic" => {
                AnthropicClient::invoke_messages(
                    &transport,
                    &candidate.endpoint,
                    api_key,
                    &candidate.model,
                    system,
                    prompt,
                )
                .await
            }
            other => anyhow::bail!("unsupported provider: {other}"),
        }
    })
}

fn score_message(message: &str) -> f64 {
    let header = message.lines().next().unwrap_or_default();
    let mut score: f64 = if validate_conventional_commit(message) {
        60.0
    } else {
        0.0
    };
    if header.contains('(') && header.contains("): ") {
        score += 10.0;
    }
    if header.len() <= 72 {
        score += 10.0;
    }
    if message.lines().count() > 1 {
        score += 10.0;
    }
    if !["update", "change", "misc"]
        .iter()
        .any(|word| header.to_ascii_lowercase().contains(word))
    {
        score += 10.0;
    }
    score.clamp(0.0, 100.0)
}

fn estimate_cost_usd(diff: &str, response: &str) -> f64 {
    let tokens = (diff.len() + response.len()).max(1) as f64 / 4.0;
    tokens / 1_000_000.0
}

fn average(values: &[f64]) -> f64 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f64>() / values.len() as f64
    }
}

fn render_provider_benchmark(
    results: &[BenchmarkResult],
    exclusions: &[BenchmarkExclusion],
) -> String {
    let mut lines = vec![
        "Benchmark Leaderboard".to_string(),
        String::new(),
        "Overall".to_string(),
        "provider\tmodel\tlatency(ms)\tcost\tquality\tsuccess".to_string(),
    ];
    for result in sorted_overall(results) {
        lines.push(format!(
            "{}\t{}\t{:.1}\t${:.6}\t{:.1}\t{:.0}%",
            result.provider,
            result.model,
            result.avg_latency_ms,
            result.avg_cost_usd,
            result.quality,
            result.success_rate * 100.0,
        ));
    }
    if !exclusions.is_empty() {
        lines.extend([
            String::new(),
            "Excluded Models".to_string(),
            "provider\tmodel\treason\tdetail".to_string(),
        ]);
        for exclusion in exclusions {
            lines.push(format!(
                "{}\t{}\t{}\t{}",
                exclusion.provider,
                exclusion.model,
                exclusion.reason,
                exclusion.detail.as_deref().unwrap_or("-"),
            ));
        }
    }
    lines.push(String::new());
    lines.join("\n")
}

fn render_provider_benchmark_json(
    results: &[BenchmarkResult],
    exclusions: &[BenchmarkExclusion],
) -> String {
    let overall = sorted_overall(results);
    let fastest = sorted_by(results, |a, b| {
        a.avg_latency_ms
            .partial_cmp(&b.avg_latency_ms)
            .unwrap_or(Ordering::Equal)
    });
    let cheapest = sorted_by(results, |a, b| {
        a.avg_cost_usd
            .partial_cmp(&b.avg_cost_usd)
            .unwrap_or(Ordering::Equal)
    });
    let best_quality = sorted_by(results, |a, b| {
        b.quality.partial_cmp(&a.quality).unwrap_or(Ordering::Equal)
    });
    let most_stable = sorted_by(results, |a, b| {
        b.success_rate
            .partial_cmp(&a.success_rate)
            .unwrap_or(Ordering::Equal)
    });
    serde_json::to_string_pretty(&json!({
        "overall": overall,
        "fastest": fastest,
        "cheapest": cheapest,
        "best_quality": best_quality,
        "most_stable": most_stable,
        "exclusions": exclusions,
    }))
    .unwrap_or_else(|_| "{}".to_string())
}

fn render_provider_benchmark_csv(
    results: &[BenchmarkResult],
    exclusions: &[BenchmarkExclusion],
) -> String {
    let mut output =
        "provider,model,avg_latency_ms,avg_cost_usd,quality,success_rate,runs\n".to_string();
    for result in results {
        output.push_str(&format!(
            "{},{},{:.3},{:.6},{:.2},{:.2},{}\n",
            csv_escape(&result.provider),
            csv_escape(&result.model),
            result.avg_latency_ms,
            result.avg_cost_usd,
            result.quality,
            result.success_rate,
            result.runs,
        ));
    }
    if !exclusions.is_empty() {
        output.push('\n');
        output.push_str("provider,model,reason,detail\n");
        for exclusion in exclusions {
            output.push_str(&format!(
                "{},{},{},{}\n",
                csv_escape(&exclusion.provider),
                csv_escape(&exclusion.model),
                csv_escape(&exclusion.reason),
                csv_escape(exclusion.detail.as_deref().unwrap_or("")),
            ));
        }
    }
    output
}

fn csv_escape(value: &str) -> String {
    if value.contains([',', '"', '\n']) {
        format!("\"{}\"", value.replace('"', "\"\""))
    } else {
        value.to_string()
    }
}

fn sorted_overall(results: &[BenchmarkResult]) -> Vec<BenchmarkResult> {
    sorted_by(results, |a, b| {
        b.quality
            .partial_cmp(&a.quality)
            .unwrap_or(Ordering::Equal)
            .then_with(|| {
                a.avg_latency_ms
                    .partial_cmp(&b.avg_latency_ms)
                    .unwrap_or(Ordering::Equal)
            })
    })
}

fn sorted_by<F>(results: &[BenchmarkResult], mut compare: F) -> Vec<BenchmarkResult>
where
    F: FnMut(&BenchmarkResult, &BenchmarkResult) -> Ordering,
{
    let mut items = results.to_vec();
    items.sort_by(|a, b| compare(a, b));
    items.truncate(10);
    items
}

fn persist_provider_benchmark(
    repo_path: &Path,
    timestamp: &str,
    results: &[BenchmarkResult],
    exclusions: &[BenchmarkExclusion],
) -> anyhow::Result<()> {
    let report_dir = state_dir(repo_path).join("benchmarks");
    fs::create_dir_all(&report_dir)?;
    let safe_ts = timestamp.replace([':', '.'], "");
    let snapshot = json!({
        "schema_version": 1,
        "timestamp": timestamp,
        "repo_path": repo_path,
        "params": {
            "samples": SAMPLE_DIFFS.len(),
        },
        "results": results,
        "exclusions": exclusions,
    });
    fs::write(
        report_dir.join(format!("benchmark-{safe_ts}.json")),
        serde_json::to_string_pretty(&snapshot)?,
    )?;
    fs::write(
        report_dir.join(format!("benchmark-{safe_ts}.md")),
        render_provider_benchmark(results, exclusions),
    )?;
    Ok(())
}

fn benchmark_timestamp() -> String {
    let seconds = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0);
    format!("{seconds}")
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
        "| Runtime | Scenarios | Passed | Failed | Excluded | Median wall time (ms) |".to_string(),
        "| --- | --- | --- | --- | --- | --- |".to_string(),
    ];

    append_summary_row(&mut lines, "python", &report.summary.python);
    append_summary_row(&mut lines, "rust", &report.summary.rust);

    lines.push(String::new());
    lines.push(
        "| Iteration | Label | Status | Median wall time (ms) | Throughput | Quality | Failures | Next bottleneck |"
            .to_string(),
    );
    lines.push("| --- | --- | --- | --- | --- | --- | --- | --- |".to_string());
    for iteration in &report.optimization_iterations {
        let median = iteration
            .median_wall_time_ms
            .map(|value| format!("{value:.2}"))
            .unwrap_or_else(|| "-".to_string());
        let throughput = iteration
            .throughput_commits_per_sec
            .map(|value| format!("{value:.3}"))
            .unwrap_or_else(|| "-".to_string());
        let quality = iteration
            .quality_score
            .map(|value| format!("{value:.1}"))
            .unwrap_or_else(|| "-".to_string());
        let failures = iteration
            .failures
            .map(|value| value.to_string())
            .unwrap_or_else(|| "-".to_string());
        let measurement_status = match iteration.measurement_status {
            kcmt_bench::model::OptimizationMeasurementStatus::Measured => "measured",
            kcmt_bench::model::OptimizationMeasurementStatus::Planned => "planned",
        };
        lines.push(format!(
            "| {} | {} | {} | {} | {} | {} | {} | {} |",
            iteration.iteration,
            iteration.label.replace('|', "\\|"),
            measurement_status,
            median,
            throughput,
            quality,
            failures,
            iteration.next_bottleneck.replace('|', "\\|"),
        ));
    }

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
        let failure = result.failure_reason.as_deref().unwrap_or("-");
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
