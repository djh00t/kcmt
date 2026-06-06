use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use serde::Deserialize;
use time::format_description::well_known::Rfc3339;
use time::OffsetDateTime;

use crate::model::{
    OptimizationIteration, OptimizationMeasurementStatus, RuntimeBenchmarkResult,
    RuntimeBenchmarkRun, RuntimeBenchmarkScorecard, RuntimeBenchmarkSnapshot,
    RuntimeBenchmarkSummary, RuntimeKind, RuntimeScenarioComparison, RuntimeScenarioMatrixCell,
    RuntimeScenarioMatrixRow, RuntimeScenarioStatus, RuntimeStageDelta, RuntimeStageTiming,
    RuntimeSummary,
};
use crate::RUNTIME_BENCHMARK_SCHEMA_VERSION;

const RUNTIME_BENCHMARK_METADATA_FILENAME: &str = ".kcmt-runtime-corpus.json";
const RUNTIME_BENCHMARK_ENV: &str = "KCMT_RUNTIME_BENCHMARK";

#[derive(Debug, Clone, Deserialize)]
struct CorpusMetadata {
    id: Option<String>,
    kind: Option<String>,
    file_count: Option<usize>,
    git_history_state: Option<String>,
    change_shape: Option<Vec<String>>,
    default_file_target: Option<String>,
    file_targets: Option<Vec<RuntimeFileTarget>>,
    generated_tracked_files: Option<usize>,
}

#[derive(Debug, Clone, Deserialize)]
struct RuntimeFileTarget {
    id: String,
    path: String,
}

#[derive(Debug, Clone)]
struct RuntimeScenario {
    scenario_id: String,
    workflow_contract_id: &'static str,
    command_label: String,
    expected_stdout_fragments: &'static [&'static str],
    file_target: Option<String>,
}

pub fn run_runtime_benchmark(
    repo_path: &Path,
    runtime: &str,
    iterations: u32,
    rust_bin: Option<&Path>,
) -> Result<RuntimeBenchmarkRun> {
    let repo_root = repo_path.to_path_buf();
    if !repo_root.exists() {
        anyhow::bail!(
            "Runtime benchmark corpus does not exist: {}",
            repo_root.display()
        );
    }
    if !matches!(runtime, "python" | "rust" | "both") {
        anyhow::bail!("Unsupported runtime benchmark mode: {runtime}");
    }

    let metadata = load_runtime_benchmark_corpus_metadata(&repo_root)?;
    let scenarios = runtime_benchmark_scenarios(&repo_root, &metadata)?;
    let requested_runtimes = if runtime == "both" {
        vec![RuntimeKind::Python, RuntimeKind::Rust]
    } else if runtime == "python" {
        vec![RuntimeKind::Python]
    } else {
        vec![RuntimeKind::Rust]
    };

    let rust_path = if let Some(path) = rust_bin {
        Some(path.to_path_buf())
    } else if requested_runtimes.contains(&RuntimeKind::Rust) {
        Some(default_runtime_benchmark_rust_bin())
    } else {
        None
    };
    let python_command = resolve_python_command();

    let mut results = Vec::new();
    for selected_runtime in requested_runtimes {
        for scenario in &scenarios {
            results.push(run_runtime_scenario(
                selected_runtime,
                &repo_root,
                &metadata,
                scenario,
                iterations,
                python_command.as_deref(),
                rust_path.as_deref(),
            )?);
        }
    }

    let summary = RuntimeBenchmarkSummary {
        python: build_runtime_summary(&results, RuntimeKind::Python),
        rust: build_runtime_summary(&results, RuntimeKind::Rust),
    };
    let optimization_iterations = build_optimization_iterations(&results, &summary);
    let timestamp = benchmark_timestamp();
    let command_set = "local-workflows-v1".to_string();
    let corpora = vec![metadata.id.clone().unwrap_or_else(|| {
        repo_root
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string()
    })];
    let scenario_matrix = build_scenario_matrix(&results);
    let scorecard = build_scorecard(&results);
    let snapshot = build_runtime_snapshot(&timestamp, &command_set, &corpora, results.len());

    Ok(RuntimeBenchmarkRun {
        schema_version: RUNTIME_BENCHMARK_SCHEMA_VERSION.to_string(),
        timestamp,
        command_set,
        corpora,
        snapshot,
        results: results.clone(),
        scenario_matrix,
        summary,
        scorecard,
        optimization_iterations,
    })
}

fn build_runtime_snapshot(
    timestamp: &str,
    command_set: &str,
    corpora: &[String],
    result_count: usize,
) -> RuntimeBenchmarkSnapshot {
    RuntimeBenchmarkSnapshot {
        snapshot_id: runtime_snapshot_id(timestamp, corpora),
        benchmark_kind: "runtime".to_string(),
        schema_version: RUNTIME_BENCHMARK_SCHEMA_VERSION.to_string(),
        timestamp: timestamp.to_string(),
        command_set: command_set.to_string(),
        corpora: corpora.to_vec(),
        result_count: result_count as u32,
        secret_free: true,
        provider_benchmark_kind: "provider".to_string(),
    }
}

fn runtime_snapshot_id(timestamp: &str, corpora: &[String]) -> String {
    let mut slug = timestamp
        .chars()
        .filter(|ch| ch.is_ascii_alphanumeric())
        .collect::<String>();
    if slug.is_empty() {
        slug = "19700101T000000Z".to_string();
    }
    let corpus = corpora
        .first()
        .map(|value| {
            value
                .chars()
                .map(|ch| {
                    if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' {
                        ch
                    } else {
                        '-'
                    }
                })
                .collect::<String>()
        })
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| "runtime-corpus".to_string());
    format!("runtime-{corpus}-{slug}")
}

fn build_scenario_matrix(results: &[RuntimeBenchmarkResult]) -> Vec<RuntimeScenarioMatrixRow> {
    let mut scenario_ids = Vec::<String>::new();
    for result in results {
        if !scenario_ids.contains(&result.scenario_id) {
            scenario_ids.push(result.scenario_id.clone());
        }
    }

    scenario_ids
        .into_iter()
        .filter_map(|scenario_id| {
            let python = find_runtime_result(results, &scenario_id, RuntimeKind::Python);
            let rust = find_runtime_result(results, &scenario_id, RuntimeKind::Rust);
            let exemplar = python.or(rust)?;
            let python_cell = matrix_cell(python);
            let rust_cell = matrix_cell(rust);
            let comparison = scenario_comparison(python, rust);
            Some(RuntimeScenarioMatrixRow {
                scenario_id,
                workflow_contract_id: exemplar.workflow_contract_id.clone(),
                corpus_id: exemplar.corpus_id.clone(),
                command_label: exemplar.command_label.clone(),
                python: python_cell,
                rust: rust_cell,
                comparison,
            })
        })
        .collect()
}

fn find_runtime_result<'a>(
    results: &'a [RuntimeBenchmarkResult],
    scenario_id: &str,
    runtime: RuntimeKind,
) -> Option<&'a RuntimeBenchmarkResult> {
    results
        .iter()
        .find(|result| result.scenario_id == scenario_id && result.runtime == runtime)
}

fn matrix_cell(result: Option<&RuntimeBenchmarkResult>) -> RuntimeScenarioMatrixCell {
    let Some(result) = result else {
        return RuntimeScenarioMatrixCell {
            status: RuntimeScenarioStatus::Excluded,
            iterations: 0,
            wall_time_ms: None,
            median_time_ms: None,
            throughput_commits_per_sec: None,
            quality_score: None,
            stage_timings: Vec::new(),
            failure_reason: Some("runtime not requested".to_string()),
        };
    };
    RuntimeScenarioMatrixCell {
        status: result.status,
        iterations: result.iterations,
        wall_time_ms: Some(result.wall_time_ms),
        median_time_ms: result.median_time_ms,
        throughput_commits_per_sec: result
            .median_time_ms
            .filter(|value| *value > 0.0)
            .map(|value| 1000.0 / value),
        quality_score: Some(result_quality_score(result)),
        stage_timings: result.stage_timings.clone(),
        failure_reason: result.failure_reason.clone(),
    }
}

fn result_quality_score(result: &RuntimeBenchmarkResult) -> f64 {
    match result.status {
        RuntimeScenarioStatus::Passed => 100.0,
        RuntimeScenarioStatus::Failed | RuntimeScenarioStatus::Excluded => 0.0,
    }
}

fn scenario_comparison(
    python: Option<&RuntimeBenchmarkResult>,
    rust: Option<&RuntimeBenchmarkResult>,
) -> RuntimeScenarioComparison {
    let comparable = matches!(
        (
            python.map(|result| result.status),
            rust.map(|result| result.status)
        ),
        (
            Some(RuntimeScenarioStatus::Passed),
            Some(RuntimeScenarioStatus::Passed)
        )
    );
    let python_median = python.and_then(|result| result.median_time_ms);
    let rust_median = rust.and_then(|result| result.median_time_ms);
    let median_delta_ms = if comparable {
        match (python_median, rust_median) {
            (Some(python_ms), Some(rust_ms)) => Some(rust_ms - python_ms),
            _ => None,
        }
    } else {
        None
    };
    let fastest_runtime = if comparable {
        match (python_median, rust_median) {
            (Some(python_ms), Some(rust_ms)) if rust_ms < python_ms => Some(RuntimeKind::Rust),
            (Some(python_ms), Some(rust_ms)) if python_ms < rust_ms => Some(RuntimeKind::Python),
            (Some(_), Some(_)) => None,
            _ => None,
        }
    } else {
        None
    };
    let speedup_ratio = if comparable {
        match (python_median, rust_median) {
            (Some(python_ms), Some(rust_ms)) if rust_ms > 0.0 => Some(python_ms / rust_ms),
            _ => None,
        }
    } else {
        None
    };
    RuntimeScenarioComparison {
        comparable,
        fastest_runtime,
        median_delta_ms,
        speedup_ratio,
        stage_deltas: stage_deltas(
            python
                .map(|result| result.stage_timings.as_slice())
                .unwrap_or(&[]),
            rust.map(|result| result.stage_timings.as_slice())
                .unwrap_or(&[]),
        ),
    }
}

fn stage_deltas(
    python_stages: &[RuntimeStageTiming],
    rust_stages: &[RuntimeStageTiming],
) -> Vec<RuntimeStageDelta> {
    let mut stage_names = Vec::<String>::new();
    for stage in python_stages.iter().chain(rust_stages.iter()) {
        if !stage_names.contains(&stage.stage) {
            stage_names.push(stage.stage.clone());
        }
    }
    stage_names
        .into_iter()
        .map(|stage| {
            let python_ms = stage_duration(python_stages, &stage);
            let rust_ms = stage_duration(rust_stages, &stage);
            RuntimeStageDelta {
                stage,
                python_ms,
                rust_ms,
                delta_ms: match (python_ms, rust_ms) {
                    (Some(python_ms), Some(rust_ms)) => Some(rust_ms - python_ms),
                    _ => None,
                },
            }
        })
        .collect()
}

fn stage_duration(stages: &[RuntimeStageTiming], stage_name: &str) -> Option<f64> {
    stages
        .iter()
        .find(|stage| stage.stage == stage_name)
        .map(|stage| stage.duration_ms)
}

fn build_scorecard(results: &[RuntimeBenchmarkResult]) -> RuntimeBenchmarkScorecard {
    RuntimeBenchmarkScorecard {
        measurement_basis: "runtime workflow subprocess wall time with local fixture provider responses; pre-LLM Rust heuristic stages are not provider throughput".to_string(),
        quality_score_definition:
            "percentage of requested runtime scenarios that passed without failed or excluded rows"
                .to_string(),
        throughput_definition:
            "commits per second derived from median scenario wall time; provider benchmark throughput remains separate"
                .to_string(),
        python_quality_score: runtime_quality_score(results, RuntimeKind::Python),
        rust_quality_score: runtime_quality_score(results, RuntimeKind::Rust),
        provider_throughput_included: false,
    }
}

fn runtime_quality_score(results: &[RuntimeBenchmarkResult], runtime: RuntimeKind) -> Option<f64> {
    let relevant: Vec<&RuntimeBenchmarkResult> = results
        .iter()
        .filter(|result| result.runtime == runtime)
        .collect();
    if relevant.is_empty() {
        return None;
    }
    let passed = relevant
        .iter()
        .filter(|result| result.status == RuntimeScenarioStatus::Passed)
        .count() as f64;
    Some((passed / relevant.len() as f64) * 100.0)
}

fn build_optimization_iterations(
    results: &[RuntimeBenchmarkResult],
    summary: &RuntimeBenchmarkSummary,
) -> Vec<OptimizationIteration> {
    let baseline_time = summary
        .rust
        .median_wall_time_ms
        .or(summary.python.median_wall_time_ms);
    let passed = results
        .iter()
        .filter(|result| result.status == RuntimeScenarioStatus::Passed)
        .count() as f64;
    let failures = results
        .iter()
        .filter(|result| result.status == RuntimeScenarioStatus::Failed)
        .count() as u32;
    let quality_score = if results.is_empty() {
        0.0
    } else {
        (passed / results.len() as f64) * 100.0
    };
    let labels = [
        ("baseline", true, "git subprocess count"),
        ("reduce git subprocesses", false, "diff preparation fan-out"),
        (
            "parallelize diff preparation",
            false,
            "provider batch enqueue",
        ),
        (
            "optimize batch enqueue",
            false,
            "post-LLM commit serialization",
        ),
        (
            "minimize commit overhead",
            false,
            "snapshot and status rendering",
        ),
        (
            "tighten snapshot overhead",
            false,
            "residual provider latency",
        ),
    ];
    labels
        .iter()
        .enumerate()
        .map(|(index, (label, baseline, bottleneck))| {
            let measurement_status = if *baseline {
                OptimizationMeasurementStatus::Measured
            } else {
                OptimizationMeasurementStatus::Planned
            };
            let median_wall_time_ms = if *baseline { baseline_time } else { None };
            let throughput_commits_per_sec = median_wall_time_ms
                .filter(|value| *value > 0.0)
                .map(|value| 1000.0 / value);
            OptimizationIteration {
                iteration: index as u32,
                label: (*label).to_string(),
                baseline: *baseline,
                measurement_status,
                median_wall_time_ms,
                throughput_commits_per_sec,
                quality_score: if *baseline { Some(quality_score) } else { None },
                failures: if *baseline { Some(failures) } else { None },
                next_bottleneck: (*bottleneck).to_string(),
            }
        })
        .collect()
}

fn build_runtime_summary(
    results: &[RuntimeBenchmarkResult],
    runtime: RuntimeKind,
) -> RuntimeSummary {
    let relevant: Vec<&RuntimeBenchmarkResult> = results
        .iter()
        .filter(|item| item.runtime == runtime)
        .collect();
    let passing: Vec<&RuntimeBenchmarkResult> = relevant
        .iter()
        .copied()
        .filter(|item| item.status == RuntimeScenarioStatus::Passed)
        .collect();
    let median_wall_time_ms = median(
        &passing
            .iter()
            .map(|item| item.median_time_ms.unwrap_or(item.wall_time_ms))
            .collect::<Vec<f64>>(),
    );

    RuntimeSummary {
        scenario_count: relevant.len() as u32,
        passed: relevant
            .iter()
            .filter(|item| item.status == RuntimeScenarioStatus::Passed)
            .count() as u32,
        failed: relevant
            .iter()
            .filter(|item| item.status == RuntimeScenarioStatus::Failed)
            .count() as u32,
        excluded: relevant
            .iter()
            .filter(|item| item.status == RuntimeScenarioStatus::Excluded)
            .count() as u32,
        median_wall_time_ms,
    }
}

fn run_runtime_scenario(
    runtime: RuntimeKind,
    source_repo: &Path,
    metadata: &CorpusMetadata,
    scenario: &RuntimeScenario,
    iterations: u32,
    python_command: Option<&[String]>,
    rust_bin: Option<&Path>,
) -> Result<RuntimeBenchmarkResult> {
    if runtime == RuntimeKind::Rust {
        let unavailable = match rust_bin {
            Some(path) if path.exists() => None,
            Some(path) => Some(format!("Rust binary not available: {}", path.display())),
            None => Some("Rust binary not available".to_string()),
        };
        if let Some(reason) = unavailable {
            return Ok(excluded_result(
                runtime, scenario, metadata, iterations, reason,
            ));
        }
    }
    if runtime == RuntimeKind::Python && python_command.is_none() {
        return Ok(excluded_result(
            runtime,
            scenario,
            metadata,
            iterations,
            "Python runtime executable is unavailable".to_string(),
        ));
    }

    let mut durations = Vec::new();
    let mut stage_samples = Vec::new();
    let mut last_exit_code = 0;
    let mut failure_reason = None;

    for _ in 0..iterations {
        let (repo_path, materialized_metadata) = materialize_runtime_corpus(source_repo, metadata)?;
        let config_home = repo_path
            .parent()
            .unwrap_or(&repo_path)
            .join(".kcmt-config");
        let envs = runtime_benchmark_env(&config_home, runtime);

        let iteration_result = (|| -> Result<()> {
            if scenario.workflow_contract_id == "status-repo-path" {
                if let Some(prep_error) = prepare_status_snapshot(
                    runtime,
                    &repo_path,
                    &materialized_metadata,
                    python_command,
                    rust_bin,
                    &envs,
                )? {
                    failure_reason = Some(prep_error);
                    last_exit_code = 1;
                    return Ok(());
                }
            }

            let command = scenario_command(
                runtime,
                &repo_path,
                scenario,
                &materialized_metadata,
                python_command,
                rust_bin,
            )?;
            let (output, elapsed_ms) = execute_runtime_command(&command, &envs)?;
            last_exit_code = output.status.code().unwrap_or(1);
            durations.push(elapsed_ms);
            if !output.status.success() {
                failure_reason = Some(failure_reason_from_output(&output));
                return Ok(());
            }
            let stdout = String::from_utf8_lossy(&output.stdout);
            if !scenario
                .expected_stdout_fragments
                .iter()
                .any(|fragment| stdout.contains(fragment))
            {
                let excerpt = stdout
                    .lines()
                    .take(8)
                    .collect::<Vec<_>>()
                    .join("\\n")
                    .chars()
                    .take(500)
                    .collect::<String>();
                failure_reason = Some(format!(
                    "missing expected stdout fragment, expected one of {:?}; stdout excerpt: {}",
                    scenario.expected_stdout_fragments, excerpt
                ));
                last_exit_code = 1;
            }
            if failure_reason.is_none() && scenario_records_snapshot_telemetry(scenario) {
                stage_samples.push(load_snapshot_stage_timings(&repo_path, &config_home)?);
            }
            Ok(())
        })();

        let _ = fs::remove_dir_all(repo_path.parent().unwrap_or(&repo_path));
        iteration_result?;

        if failure_reason.is_some() {
            break;
        }
    }

    let median_time_ms = median(&durations);
    let mut stage_timings = aggregate_stage_timings(&stage_samples);
    append_process_overhead_stage(&mut stage_timings, median_time_ms);

    Ok(RuntimeBenchmarkResult {
        scenario_id: scenario.scenario_id.clone(),
        workflow_contract_id: scenario.workflow_contract_id.to_string(),
        corpus_id: metadata
            .id
            .clone()
            .unwrap_or_else(|| "runtime-corpus".to_string()),
        runtime,
        command_label: scenario.command_label.clone(),
        iterations,
        status: if failure_reason.is_some() {
            RuntimeScenarioStatus::Failed
        } else {
            RuntimeScenarioStatus::Passed
        },
        wall_time_ms: durations.iter().sum(),
        median_time_ms,
        peak_rss_bytes: None,
        exit_code: Some(last_exit_code),
        failure_reason,
        stage_timings,
    })
}

fn excluded_result(
    runtime: RuntimeKind,
    scenario: &RuntimeScenario,
    metadata: &CorpusMetadata,
    iterations: u32,
    reason: String,
) -> RuntimeBenchmarkResult {
    RuntimeBenchmarkResult {
        scenario_id: scenario.scenario_id.clone(),
        workflow_contract_id: scenario.workflow_contract_id.to_string(),
        corpus_id: metadata
            .id
            .clone()
            .unwrap_or_else(|| "runtime-corpus".to_string()),
        runtime,
        command_label: scenario.command_label.clone(),
        iterations,
        status: RuntimeScenarioStatus::Excluded,
        wall_time_ms: 0.0,
        median_time_ms: None,
        peak_rss_bytes: None,
        exit_code: None,
        failure_reason: Some(reason),
        stage_timings: Vec::new(),
    }
}

fn scenario_records_snapshot_telemetry(scenario: &RuntimeScenario) -> bool {
    matches!(
        scenario.workflow_contract_id,
        "default-repo-path" | "oneshot-repo-path" | "file-repo-path"
    )
}

fn load_snapshot_stage_timings(
    _repo_path: &Path,
    config_home: &Path,
) -> Result<Vec<RuntimeStageTiming>> {
    let Some(snapshot_path) = latest_snapshot_path(config_home)? else {
        return Ok(Vec::new());
    };
    let raw = fs::read_to_string(&snapshot_path)?;
    let payload: serde_json::Value = serde_json::from_str(&raw)?;
    let Some(stages) = payload
        .get("telemetry")
        .and_then(|telemetry| telemetry.get("stages"))
        .and_then(|stages| stages.as_array())
    else {
        return Ok(Vec::new());
    };

    Ok(stages
        .iter()
        .filter_map(|stage| {
            let stage_name = stage.get("stage")?.as_str()?.to_string();
            let duration_ms = stage.get("duration_ms")?.as_f64()?;
            let items = stage.get("items")?.as_u64().unwrap_or(0) as u32;
            Some(RuntimeStageTiming {
                stage: stage_name,
                duration_ms,
                items,
            })
        })
        .collect())
}

fn latest_snapshot_path(config_home: &Path) -> Result<Option<PathBuf>> {
    let repos_dir = config_home.join("repos");
    if !repos_dir.exists() {
        return Ok(None);
    }
    let mut snapshots = Vec::new();
    for entry in fs::read_dir(repos_dir)? {
        let path = entry?.path().join("last_run.json");
        if path.exists() {
            snapshots.push(path);
        }
    }
    snapshots.sort();
    Ok(snapshots.pop())
}

fn aggregate_stage_timings(samples: &[Vec<RuntimeStageTiming>]) -> Vec<RuntimeStageTiming> {
    let mut order = Vec::<String>::new();
    let mut durations_by_stage = std::collections::BTreeMap::<String, Vec<f64>>::new();
    let mut items_by_stage = std::collections::BTreeMap::<String, Vec<f64>>::new();

    for sample in samples {
        for timing in sample {
            if !durations_by_stage.contains_key(&timing.stage) {
                order.push(timing.stage.clone());
            }
            durations_by_stage
                .entry(timing.stage.clone())
                .or_default()
                .push(timing.duration_ms);
            items_by_stage
                .entry(timing.stage.clone())
                .or_default()
                .push(timing.items as f64);
        }
    }

    order
        .into_iter()
        .filter_map(|stage| {
            let duration_ms = median(durations_by_stage.get(&stage)?)?;
            let items = median(items_by_stage.get(&stage)?).unwrap_or(0.0).round() as u32;
            Some(RuntimeStageTiming {
                stage,
                duration_ms,
                items,
            })
        })
        .collect()
}

fn append_process_overhead_stage(
    stages: &mut Vec<RuntimeStageTiming>,
    median_time_ms: Option<f64>,
) {
    let Some(median_time_ms) = median_time_ms else {
        return;
    };
    if !stages.iter().any(|stage| stage.stage == "workflow_total") {
        return;
    }
    let accounted_ms = stages
        .iter()
        .filter(|stage| {
            matches!(
                stage.stage.as_str(),
                "arg_parse" | "repo_discovery" | "dispatch" | "workflow_total"
            )
        })
        .map(|stage| stage.duration_ms)
        .sum::<f64>();
    stages.push(RuntimeStageTiming {
        stage: "process_overhead".to_string(),
        duration_ms: (median_time_ms - accounted_ms).max(0.0),
        items: 1,
    });
}

fn load_runtime_benchmark_corpus_metadata(repo_path: &Path) -> Result<CorpusMetadata> {
    let metadata_path = repo_path.join(RUNTIME_BENCHMARK_METADATA_FILENAME);
    if metadata_path.exists() {
        let raw = fs::read_to_string(&metadata_path)?;
        let parsed = serde_json::from_str::<CorpusMetadata>(&raw).with_context(|| {
            format!(
                "invalid runtime corpus metadata: {}",
                metadata_path.display()
            )
        })?;
        return Ok(parsed);
    }

    let default_file_target = runtime_benchmark_target_file(
        repo_path,
        &CorpusMetadata {
            id: None,
            kind: None,
            file_count: None,
            git_history_state: None,
            change_shape: None,
            default_file_target: None,
            file_targets: None,
            generated_tracked_files: None,
        },
    )?;
    Ok(CorpusMetadata {
        id: Some(
            repo_path
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .to_string(),
        ),
        kind: None,
        file_count: None,
        git_history_state: None,
        change_shape: None,
        default_file_target: Some(default_file_target),
        file_targets: None,
        generated_tracked_files: None,
    })
}

fn runtime_benchmark_scenarios(
    repo_path: &Path,
    metadata: &CorpusMetadata,
) -> Result<Vec<RuntimeScenario>> {
    let target = runtime_benchmark_target_file(repo_path, metadata)?;
    let corpus_id = metadata
        .id
        .clone()
        .unwrap_or_else(|| "runtime-corpus".to_string());
    let mut scenarios = vec![
        RuntimeScenario {
            scenario_id: format!("{corpus_id}:status-repo-path"),
            workflow_contract_id: "status-repo-path",
            command_label: "kcmt status --repo-path <repo>".to_string(),
            expected_stdout_fragments: &["Commit status"],
            file_target: None,
        },
        RuntimeScenario {
            scenario_id: format!("{corpus_id}:oneshot-repo-path"),
            workflow_contract_id: "oneshot-repo-path",
            command_label: "kcmt --oneshot --repo-path <repo>".to_string(),
            expected_stdout_fragments: &["✓ "],
            file_target: None,
        },
        RuntimeScenario {
            scenario_id: format!("{corpus_id}:file-repo-path"),
            workflow_contract_id: "file-repo-path",
            command_label: format!("kcmt --file {target} --repo-path <repo>"),
            expected_stdout_fragments: &["✓ "],
            file_target: Some(target),
        },
    ];
    if !is_large_untracked_runtime_corpus(metadata) {
        scenarios.insert(
            2,
            RuntimeScenario {
                scenario_id: format!("{corpus_id}:default-repo-path"),
                workflow_contract_id: "default-repo-path",
                command_label: "kcmt --repo-path <repo>".to_string(),
                expected_stdout_fragments: &["✓ ", "Committed "],
                file_target: None,
            },
        );
    }
    if let Some(file_targets) = metadata.file_targets.clone() {
        for file_target in file_targets {
            scenarios.push(RuntimeScenario {
                scenario_id: format!("{corpus_id}:file-{}", file_target.id),
                workflow_contract_id: "file-repo-path",
                command_label: format!("kcmt --file {} --repo-path <repo>", file_target.path),
                expected_stdout_fragments: &["✓ "],
                file_target: Some(file_target.path),
            });
        }
    }
    Ok(scenarios)
}

fn is_large_untracked_runtime_corpus(metadata: &CorpusMetadata) -> bool {
    metadata.kind.as_deref() == Some("synthetic")
        && metadata.git_history_state.as_deref() == Some("no-commits")
        && metadata.file_count.unwrap_or_default() >= 1000
        && metadata
            .change_shape
            .as_ref()
            .is_some_and(|shape| shape.iter().any(|item| item == "untracked"))
}

fn runtime_benchmark_target_file(repo_path: &Path, metadata: &CorpusMetadata) -> Result<String> {
    if let Some(target) = metadata.default_file_target.clone() {
        if !target.trim().is_empty() && repo_path.join(&target).exists() {
            return Ok(target);
        }
    }

    let mut files = Vec::new();
    collect_files(repo_path, &mut files)?;
    files.sort();
    for candidate in files {
        if candidate.extension().and_then(|ext| ext.to_str()) == Some("py") {
            return Ok(candidate.to_string_lossy().to_string());
        }
    }
    Err(anyhow::anyhow!(
        "No benchmarkable files found under {}",
        repo_path.display()
    ))
}

fn materialize_runtime_corpus(
    source_repo: &Path,
    metadata: &CorpusMetadata,
) -> Result<(PathBuf, CorpusMetadata)> {
    let temp_root = unique_temp_dir(metadata.id.as_deref().unwrap_or("runtime-corpus"));
    let repo_path = temp_root.join("repo");
    copy_dir_all(source_repo, &repo_path)?;

    let materialized_metadata = if repo_path.join(".git").exists() {
        git(&repo_path, &["config", "user.name", "kcmt-benchmark"])?;
        git(
            &repo_path,
            &["config", "user.email", "kcmt-benchmark@example.com"],
        )?;
        metadata.clone()
    } else {
        prepare_realistic_runtime_corpus(&repo_path, metadata)?
    };

    Ok((repo_path, materialized_metadata))
}

fn prepare_realistic_runtime_corpus(
    repo_path: &Path,
    metadata: &CorpusMetadata,
) -> Result<CorpusMetadata> {
    seed_generated_runtime_targets(repo_path, metadata)?;
    git_init_main(repo_path)?;
    git(&repo_path, &["config", "user.name", "kcmt-benchmark"])?;
    git(
        &repo_path,
        &["config", "user.email", "kcmt-benchmark@example.com"],
    )?;
    git(&repo_path, &["add", "."])?;
    git(&repo_path, &["commit", "-m", "chore(repo): seed"])?;

    let target = runtime_benchmark_target_file(repo_path, metadata)?;
    let target_path = repo_path.join(target);
    if target_path.exists() {
        append_runtime_mutation(&target_path)?;
    }

    if let Some(file_targets) = metadata.file_targets.as_ref() {
        for file_target in file_targets {
            if file_target.id.contains("modified") {
                let target_path = repo_path.join(&file_target.path);
                if target_path.exists() {
                    append_runtime_mutation(&target_path)?;
                }
            }
        }
    }

    let deleted_path = repo_path.join("docs").join("notes.md");
    if deleted_path.exists() {
        fs::remove_file(deleted_path)?;
    }

    let ignore_file = repo_path.join(".gitignore");
    if !ignore_file.exists() {
        fs::write(&ignore_file, "*.local\n")?;
    }

    let ignored = repo_path.join("debug.local");
    fs::write(ignored, "ignored=true\n")?;

    let extra_file = repo_path.join("notes").join("runtime-benchmark.md");
    if let Some(parent) = extra_file.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(
        extra_file,
        "# Runtime Benchmark Fixture\n\nThis file is intentionally untracked.\n",
    )?;

    Ok(metadata.clone())
}

fn seed_generated_runtime_targets(repo_path: &Path, metadata: &CorpusMetadata) -> Result<()> {
    if let Some(count) = metadata.generated_tracked_files.filter(|count| *count > 0) {
        let generated_root = repo_path.join("generated");
        for index in 0..count {
            let dir = generated_root.join(format!("pkg_{:03}", index / 100));
            fs::create_dir_all(&dir)?;
            fs::write(
                dir.join(format!("file_{index:05}.txt")),
                format!("generated runtime benchmark fixture {index}\n"),
            )?;
        }
    }

    let Some(file_targets) = metadata.file_targets.as_ref() else {
        return Ok(());
    };
    for file_target in file_targets {
        if file_target.id.contains("large") || file_target.id.contains("nested") {
            let target_path = repo_path.join(&file_target.path);
            if let Some(parent) = target_path.parent() {
                fs::create_dir_all(parent)?;
            }
            if !target_path.exists() {
                let content = if file_target.id.contains("large") {
                    let mut content = String::with_capacity(256 * 1024);
                    for index in 0..4096 {
                        content.push_str("runtime benchmark large fixture line ");
                        content.push_str(&index.to_string());
                        content.push('\n');
                    }
                    content
                } else {
                    "print('nested runtime benchmark')\n".to_string()
                };
                fs::write(target_path, content)?;
            }
        }
    }
    Ok(())
}

fn append_runtime_mutation(target_path: &Path) -> Result<()> {
    let mut content = fs::read_to_string(target_path)?;
    if !content.ends_with('\n') {
        content.push('\n');
    }
    content.push_str("\n# runtime benchmark mutation\n");
    fs::write(target_path, content)?;
    Ok(())
}

fn resolve_python_command() -> Option<Vec<String>> {
    if let Ok(configured) = env::var("KCMT_PYTHON_BIN") {
        if !configured.trim().is_empty() {
            let command = vec![configured];
            if python_command_has_runtime_deps(&command) {
                return Some(command);
            }
        }
    }
    if let Ok(virtual_env) = env::var("VIRTUAL_ENV") {
        if let Some(candidate) = python_from_virtual_env_root(Path::new(&virtual_env)) {
            let command = vec![candidate.display().to_string()];
            if python_command_has_runtime_deps(&command) {
                return Some(command);
            }
        }
    }
    if let Some(candidate) = python_from_virtual_env_root(&workspace_root().join(".venv")) {
        let command = vec![candidate.display().to_string()];
        if python_command_has_runtime_deps(&command) {
            return Some(command);
        }
    }
    let uv_command = vec!["uv".to_string(), "run".to_string(), "python".to_string()];
    if python_command_has_runtime_deps(&uv_command) {
        return Some(uv_command);
    }
    for candidate in ["python3", "python"] {
        let command = vec![candidate.to_string()];
        if python_command_has_runtime_deps(&command) {
            return Some(command);
        }
    }
    None
}

fn python_from_virtual_env_root(root: &Path) -> Option<PathBuf> {
    [
        root.join("bin").join("python"),
        root.join("Scripts").join("python.exe"),
        root.join("Scripts").join("python"),
    ]
    .into_iter()
    .find(|candidate| candidate.exists())
}

fn python_command_has_runtime_deps(command: &[String]) -> bool {
    Command::new(&command[0])
        .args(&command[1..])
        .args(["-c", "import httpx"])
        .current_dir(workspace_root())
        .output()
        .map(|output| output.status.success())
        .unwrap_or(false)
}

pub fn default_runtime_benchmark_rust_bin() -> PathBuf {
    if let Ok(configured) = env::var("KCMT_RUST_BIN") {
        if !configured.trim().is_empty() {
            return PathBuf::from(configured);
        }
    }
    workspace_root()
        .join("rust")
        .join("target")
        .join("release")
        .join("kcmt")
}

fn runtime_benchmark_env(config_home: &Path, runtime: RuntimeKind) -> Vec<(String, String)> {
    let mut envs = vec![
        ("KCMT_USE_INK".to_string(), "0".to_string()),
        ("KCMT_AUTO_INSTALL_INK_DEPS".to_string(), "0".to_string()),
        (
            "KCMT_CONFIG_HOME".to_string(),
            config_home.display().to_string(),
        ),
        (RUNTIME_BENCHMARK_ENV.to_string(), "1".to_string()),
        ("KCMT_DISABLE_KEYCHAIN".to_string(), "1".to_string()),
        ("KCMT_NO_SPINNER".to_string(), "1".to_string()),
        ("PYTHONIOENCODING".to_string(), "utf-8".to_string()),
        ("PYTHONUTF8".to_string(), "1".to_string()),
        (
            "PYTEST_CURRENT_TEST".to_string(),
            "kcmt-runtime-benchmark".to_string(),
        ),
        (
            "OPENAI_API_KEY".to_string(),
            "kcmt-runtime-benchmark".to_string(),
        ),
        ("KCMT_RUNTIME_BENCHMARK".to_string(), "1".to_string()),
        ("KCMT_ALLOW_LOCAL_SYNTHESIS".to_string(), "1".to_string()),
        (
            "KCMT_PROVIDER_RESPONSE".to_string(),
            "chore(repo): benchmark fixture response".to_string(),
        ),
        ("KCMT_DISABLE_KEYCHAIN".to_string(), "1".to_string()),
        ("GIT_AUTHOR_NAME".to_string(), "kcmt-benchmark".to_string()),
        (
            "GIT_AUTHOR_EMAIL".to_string(),
            "kcmt-benchmark@example.com".to_string(),
        ),
        (
            "GIT_COMMITTER_NAME".to_string(),
            "kcmt-benchmark".to_string(),
        ),
        (
            "GIT_COMMITTER_EMAIL".to_string(),
            "kcmt-benchmark@example.com".to_string(),
        ),
    ];
    for key in ["KCMT_USE_BATCH", "KCMT_BATCH_MODEL", "KCMT_BATCH_TIMEOUT"] {
        if let Ok(value) = std::env::var(key) {
            envs.push((key.to_string(), value));
        }
    }
    if runtime == RuntimeKind::Python {
        envs.push(("KCMT_RUNTIME".to_string(), "python".to_string()));
    } else {
        for key in ["KCMT_GIT_COMMIT_BACKEND", "KCMT_GIT_TREE_BACKEND"] {
            if let Ok(value) = std::env::var(key) {
                envs.push((key.to_string(), value));
            }
        }
    }
    envs
}

fn prepare_status_snapshot(
    runtime: RuntimeKind,
    repo_path: &Path,
    metadata: &CorpusMetadata,
    python_command: Option<&[String]>,
    rust_bin: Option<&Path>,
    envs: &[(String, String)],
) -> Result<Option<String>> {
    let target = runtime_benchmark_target_file(repo_path, metadata)?;
    let scenario = RuntimeScenario {
        scenario_id: "prep:file-repo-path".to_string(),
        workflow_contract_id: "file-repo-path",
        command_label: format!("kcmt --file {target} --repo-path <repo>"),
        expected_stdout_fragments: &["✓ "],
        file_target: Some(target),
    };
    let command = scenario_command(
        runtime,
        repo_path,
        &scenario,
        metadata,
        python_command,
        rust_bin,
    )?;
    let (output, _) = execute_runtime_command(&command, envs)?;
    if !output.status.success() {
        return Ok(Some(failure_reason_from_output(&output)));
    }
    Ok(None)
}

fn scenario_command(
    runtime: RuntimeKind,
    repo_path: &Path,
    scenario: &RuntimeScenario,
    metadata: &CorpusMetadata,
    python_command: Option<&[String]>,
    rust_bin: Option<&Path>,
) -> Result<Vec<String>> {
    let mut command = match runtime {
        RuntimeKind::Python => {
            let mut command = python_command
                .ok_or_else(|| anyhow::anyhow!("Python runtime executable is unavailable"))?
                .to_vec();
            command.extend(["-m".to_string(), "kcmt.main".to_string()]);
            command
        }
        RuntimeKind::Rust => vec![rust_bin
            .ok_or_else(|| anyhow::anyhow!("Rust binary is not configured"))?
            .display()
            .to_string()],
    };

    match scenario.workflow_contract_id {
        "status-repo-path" => {
            command.extend([
                "status".to_string(),
                "--repo-path".to_string(),
                repo_path.display().to_string(),
            ]);
        }
        "oneshot-repo-path" => {
            command.extend([
                "--oneshot".to_string(),
                "--no-auto-push".to_string(),
                "--repo-path".to_string(),
                repo_path.display().to_string(),
            ]);
        }
        "default-repo-path" => {
            command.extend([
                "--no-auto-push".to_string(),
                "--repo-path".to_string(),
                repo_path.display().to_string(),
            ]);
        }
        _ => {
            let target = scenario
                .file_target
                .clone()
                .map(Ok)
                .unwrap_or_else(|| runtime_benchmark_target_file(repo_path, metadata))?;
            command.extend([
                "--file".to_string(),
                target,
                "--no-auto-push".to_string(),
                "--repo-path".to_string(),
                repo_path.display().to_string(),
            ]);
        }
    }
    Ok(command)
}

fn execute_runtime_command(command: &[String], envs: &[(String, String)]) -> Result<(Output, f64)> {
    let started = Instant::now();
    let output = Command::new(&command[0])
        .args(&command[1..])
        .envs(
            envs.iter()
                .map(|(key, value)| (key.as_str(), value.as_str())),
        )
        .current_dir(workspace_root())
        .output()
        .with_context(|| format!("failed to run {:?}", command))?;
    let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
    Ok((output, elapsed_ms))
}

fn failure_reason_from_output(output: &Output) -> String {
    let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
    let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
    let detail = if !stderr.is_empty() {
        stderr
    } else if !stdout.is_empty() {
        stdout
    } else {
        "command failed without output".to_string()
    };
    format!("exit {}: {detail}", output.status.code().unwrap_or(1))
}

fn benchmark_timestamp() -> String {
    OffsetDateTime::now_utc()
        .format(&Rfc3339)
        .unwrap_or_else(|_| "1970-01-01T00:00:00Z".to_string())
}

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(3)
        .expect("workspace root")
        .to_path_buf()
}

fn unique_temp_dir(label: &str) -> PathBuf {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock should be after unix epoch")
        .as_nanos();
    let suffix = COUNTER.fetch_add(1, Ordering::Relaxed);
    let path = env::temp_dir().join(format!("kcmt-runtime-{label}-{nanos}-{suffix}"));
    fs::create_dir_all(&path).expect("temp dir should be created");
    path
}

fn git_init_main(repo_path: &Path) -> Result<()> {
    let repo_arg = repo_path.display().to_string();
    let init_with_branch = Command::new("git")
        .args(["init", "--initial-branch=main", repo_arg.as_str()])
        .output()
        .context("failed to initialize git repository")?;
    if init_with_branch.status.success() && repo_path.join(".git").exists() {
        return Ok(());
    }

    git(repo_path, &["init"])?;
    git(repo_path, &["branch", "-M", "main"])?;
    Ok(())
}

fn git(repo_path: &Path, args: &[&str]) -> Result<()> {
    let output = Command::new("git")
        .current_dir(repo_path)
        .args(args)
        .output()
        .with_context(|| format!("failed to run git {:?}", args))?;
    if !output.status.success() {
        anyhow::bail!(
            "git {:?} failed: {}",
            args,
            String::from_utf8_lossy(&output.stderr)
        );
    }
    Ok(())
}

fn copy_dir_all(src: &Path, dst: &Path) -> Result<()> {
    fs::create_dir_all(dst)?;
    for entry in fs::read_dir(src)? {
        let entry = entry?;
        let ty = entry.file_type()?;
        let dest_path = dst.join(entry.file_name());
        if ty.is_dir() {
            copy_dir_all(&entry.path(), &dest_path)?;
        } else {
            fs::copy(entry.path(), dest_path)?;
        }
    }
    Ok(())
}

fn collect_files(root: &Path, files: &mut Vec<PathBuf>) -> Result<()> {
    for entry in fs::read_dir(root)? {
        let entry = entry?;
        let path = entry.path();
        let ty = entry.file_type()?;
        if ty.is_dir() {
            if path.file_name().and_then(|name| name.to_str()) == Some(".git") {
                continue;
            }
            collect_files(&path, files)?;
        } else if ty.is_file() {
            files.push(path.strip_prefix(root)?.to_path_buf());
        }
    }
    Ok(())
}

fn median(values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|left, right| left.total_cmp(right));
    let middle = sorted.len() / 2;
    if sorted.len() % 2 == 0 {
        Some((sorted[middle - 1] + sorted[middle]) / 2.0)
    } else {
        Some(sorted[middle])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn python_from_virtual_env_root_prefers_unix_bin_python() {
        let root = unique_temp_dir("venv-python");
        let bin = root.join("bin");
        let scripts = root.join("Scripts");
        fs::create_dir_all(&bin).expect("bin dir");
        fs::create_dir_all(&scripts).expect("scripts dir");
        let unix_python = bin.join("python");
        let windows_python = scripts.join("python.exe");
        fs::write(&unix_python, "").expect("unix python");
        fs::write(&windows_python, "").expect("windows python");

        assert_eq!(
            python_from_virtual_env_root(&root).as_deref(),
            Some(unix_python.as_path())
        );
    }

    #[test]
    fn runtime_benchmark_env_forces_noninteractive_python() {
        let config_home = unique_temp_dir("runtime-env");
        let envs = runtime_benchmark_env(&config_home, RuntimeKind::Python);

        assert_eq!(
            envs.iter()
                .find(|(key, _)| key == "PYTEST_CURRENT_TEST")
                .map(|(_, value)| value.as_str()),
            Some("kcmt-runtime-benchmark")
        );
    }

    #[test]
    fn runtime_benchmark_env_preserves_rust_git_backend() {
        std::env::set_var("KCMT_GIT_COMMIT_BACKEND", "gix");
        let config_home = unique_temp_dir("runtime-rust-env");
        let envs = runtime_benchmark_env(&config_home, RuntimeKind::Rust);
        std::env::remove_var("KCMT_GIT_COMMIT_BACKEND");

        assert_eq!(
            envs.iter()
                .find(|(key, _)| key == "KCMT_GIT_COMMIT_BACKEND")
                .map(|(_, value)| value.as_str()),
            Some("gix")
        );
    }

    #[test]
    fn runtime_benchmark_env_preserves_batch_mode_override() {
        std::env::set_var("KCMT_USE_BATCH", "1");
        let config_home = unique_temp_dir("runtime-batch-env");
        let envs = runtime_benchmark_env(&config_home, RuntimeKind::Rust);
        std::env::remove_var("KCMT_USE_BATCH");

        assert_eq!(
            envs.iter()
                .find(|(key, _)| key == "KCMT_USE_BATCH")
                .map(|(_, value)| value.as_str()),
            Some("1")
        );
    }

    #[test]
    fn large_synthetic_runtime_corpus_excludes_default_workflow() {
        let repo = unique_temp_dir("large-synthetic-runtime");
        let target = repo.join("pkg_000").join("file_00000.py");
        fs::create_dir_all(target.parent().expect("target parent")).expect("target parent dir");
        fs::write(&target, "print('runtime')\n").expect("target file");
        let metadata = CorpusMetadata {
            id: Some("synthetic-untracked-1000".to_string()),
            kind: Some("synthetic".to_string()),
            file_count: Some(1000),
            git_history_state: Some("no-commits".to_string()),
            change_shape: Some(vec!["untracked".to_string(), "nested-paths".to_string()]),
            default_file_target: Some("pkg_000/file_00000.py".to_string()),
            file_targets: None,
            generated_tracked_files: None,
        };

        let scenarios = runtime_benchmark_scenarios(&repo, &metadata).expect("scenarios");

        assert_eq!(
            scenarios
                .iter()
                .map(|scenario| scenario.workflow_contract_id)
                .collect::<Vec<_>>(),
            vec!["status-repo-path", "oneshot-repo-path", "file-repo-path"]
        );
    }
}
