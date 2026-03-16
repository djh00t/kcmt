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
    RuntimeBenchmarkResult, RuntimeBenchmarkRun, RuntimeBenchmarkSummary,
    RuntimeKind, RuntimeScenarioStatus, RuntimeSummary,
};
use crate::RUNTIME_BENCHMARK_SCHEMA_VERSION;

const RUNTIME_BENCHMARK_METADATA_FILENAME: &str = ".kcmt-runtime-corpus.json";
const RUNTIME_BENCHMARK_ENV: &str = "KCMT_RUNTIME_BENCHMARK";

#[derive(Debug, Clone, Deserialize)]
struct CorpusMetadata {
    id: Option<String>,
    default_file_target: Option<String>,
}

#[derive(Debug, Clone)]
struct RuntimeScenario {
    scenario_id: String,
    workflow_contract_id: &'static str,
    command_label: String,
    expected_stdout_fragment: &'static str,
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

    Ok(RuntimeBenchmarkRun {
        schema_version: RUNTIME_BENCHMARK_SCHEMA_VERSION.to_string(),
        timestamp: benchmark_timestamp(),
        command_set: "local-workflows-v1".to_string(),
        corpora: vec![metadata
            .id
            .clone()
            .unwrap_or_else(|| repo_root.file_name().unwrap_or_default().to_string_lossy().to_string())],
        results: results.clone(),
        summary: RuntimeBenchmarkSummary {
            python: build_runtime_summary(&results, RuntimeKind::Python),
            rust: build_runtime_summary(&results, RuntimeKind::Rust),
        },
    })
}

fn build_runtime_summary(
    results: &[RuntimeBenchmarkResult],
    runtime: RuntimeKind,
) -> RuntimeSummary {
    let relevant: Vec<&RuntimeBenchmarkResult> =
        results.iter().filter(|item| item.runtime == runtime).collect();
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
    python_command: Option<&str>,
    rust_bin: Option<&Path>,
) -> Result<RuntimeBenchmarkResult> {
    if runtime == RuntimeKind::Rust {
        let unavailable = match rust_bin {
            Some(path) if path.exists() => None,
            Some(path) => Some(format!("Rust binary not available: {}", path.display())),
            None => Some("Rust binary not available".to_string()),
        };
        if let Some(reason) = unavailable {
            return Ok(excluded_result(runtime, scenario, metadata, iterations, reason));
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
    let mut last_exit_code = 0;
    let mut failure_reason = None;

    for _ in 0..iterations {
        let (repo_path, materialized_metadata) =
            materialize_runtime_corpus(source_repo, metadata)?;
        let config_home = repo_path.parent().unwrap_or(&repo_path).join(".kcmt-config");
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
            if !stdout.contains(scenario.expected_stdout_fragment) {
                failure_reason = Some(format!(
                    "missing expected stdout fragment: {}",
                    scenario.expected_stdout_fragment
                ));
                last_exit_code = 1;
            }
            Ok(())
        })();

        let _ = fs::remove_dir_all(repo_path.parent().unwrap_or(&repo_path));
        iteration_result?;

        if failure_reason.is_some() {
            break;
        }
    }

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
        median_time_ms: median(&durations),
        peak_rss_bytes: None,
        exit_code: Some(last_exit_code),
        failure_reason,
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
    }
}

fn load_runtime_benchmark_corpus_metadata(repo_path: &Path) -> Result<CorpusMetadata> {
    let metadata_path = repo_path.join(RUNTIME_BENCHMARK_METADATA_FILENAME);
    if metadata_path.exists() {
        let raw = fs::read_to_string(&metadata_path)?;
        let parsed = serde_json::from_str::<CorpusMetadata>(&raw)
            .with_context(|| format!("invalid runtime corpus metadata: {}", metadata_path.display()))?;
        return Ok(parsed);
    }

    let default_file_target = runtime_benchmark_target_file(repo_path, &CorpusMetadata {
        id: None,
        default_file_target: None,
    })?;
    Ok(CorpusMetadata {
        id: Some(
            repo_path
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .to_string(),
        ),
        default_file_target: Some(default_file_target),
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
    Ok(vec![
        RuntimeScenario {
            scenario_id: format!("{corpus_id}:status-repo-path"),
            workflow_contract_id: "status-repo-path",
            command_label: "kcmt status --repo-path <repo>".to_string(),
            expected_stdout_fragment: "Commit status",
        },
        RuntimeScenario {
            scenario_id: format!("{corpus_id}:oneshot-repo-path"),
            workflow_contract_id: "oneshot-repo-path",
            command_label: "kcmt --oneshot --repo-path <repo>".to_string(),
            expected_stdout_fragment: "✓ ",
        },
        RuntimeScenario {
            scenario_id: format!("{corpus_id}:file-repo-path"),
            workflow_contract_id: "file-repo-path",
            command_label: format!("kcmt --file {target} --repo-path <repo>"),
            expected_stdout_fragment: "✓ ",
        },
    ])
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
    let temp_root = unique_temp_dir(
        metadata
            .id
            .as_deref()
            .unwrap_or("runtime-corpus"),
    );
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
        let mut content = fs::read_to_string(&target_path)?;
        if !content.ends_with('\n') {
            content.push('\n');
        }
        content.push_str("\n# runtime benchmark mutation\n");
        fs::write(&target_path, content)?;
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

fn resolve_python_command() -> Option<String> {
    if let Ok(configured) = env::var("KCMT_PYTHON_BIN") {
        if !configured.trim().is_empty() {
            return Some(configured);
        }
    }
    for candidate in ["python3", "python"] {
        if let Ok(output) = Command::new(candidate).arg("--version").output() {
            if output.status.success() {
                return Some(candidate.to_string());
            }
        }
    }
    None
}

pub fn default_runtime_benchmark_rust_bin() -> PathBuf {
    if let Ok(configured) = env::var("KCMT_RUST_BIN") {
        if !configured.trim().is_empty() {
            return PathBuf::from(configured);
        }
    }
    workspace_root().join("rust").join("target").join("release").join("kcmt")
}

fn runtime_benchmark_env(config_home: &Path, runtime: RuntimeKind) -> Vec<(String, String)> {
    let mut envs = vec![
        ("KCMT_USE_INK".to_string(), "0".to_string()),
        ("KCMT_AUTO_INSTALL_INK_DEPS".to_string(), "0".to_string()),
        ("KCMT_CONFIG_HOME".to_string(), config_home.display().to_string()),
        (RUNTIME_BENCHMARK_ENV.to_string(), "1".to_string()),
        ("KCMT_NO_SPINNER".to_string(), "1".to_string()),
        ("OPENAI_API_KEY".to_string(), "kcmt-runtime-benchmark".to_string()),
        ("GIT_AUTHOR_NAME".to_string(), "kcmt-benchmark".to_string()),
        (
            "GIT_AUTHOR_EMAIL".to_string(),
            "kcmt-benchmark@example.com".to_string(),
        ),
        ("GIT_COMMITTER_NAME".to_string(), "kcmt-benchmark".to_string()),
        (
            "GIT_COMMITTER_EMAIL".to_string(),
            "kcmt-benchmark@example.com".to_string(),
        ),
    ];
    if runtime == RuntimeKind::Python {
        envs.push(("KCMT_RUNTIME".to_string(), "python".to_string()));
    }
    envs
}

fn prepare_status_snapshot(
    runtime: RuntimeKind,
    repo_path: &Path,
    metadata: &CorpusMetadata,
    python_command: Option<&str>,
    rust_bin: Option<&Path>,
    envs: &[(String, String)],
) -> Result<Option<String>> {
    let target = runtime_benchmark_target_file(repo_path, metadata)?;
    let scenario = RuntimeScenario {
        scenario_id: "prep:file-repo-path".to_string(),
        workflow_contract_id: "file-repo-path",
        command_label: format!("kcmt --file {target} --repo-path <repo>"),
        expected_stdout_fragment: "✓ ",
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
    python_command: Option<&str>,
    rust_bin: Option<&Path>,
) -> Result<Vec<String>> {
    let mut command = match runtime {
        RuntimeKind::Python => vec![
            python_command
                .ok_or_else(|| anyhow::anyhow!("Python runtime executable is unavailable"))?
                .to_string(),
            "-m".to_string(),
            "kcmt.main".to_string(),
        ],
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
                "--repo-path".to_string(),
                repo_path.display().to_string(),
            ]);
        }
        _ => {
            let target = runtime_benchmark_target_file(repo_path, metadata)?;
            command.extend([
                "--file".to_string(),
                target,
                "--repo-path".to_string(),
                repo_path.display().to_string(),
            ]);
        }
    }
    Ok(command)
}

fn execute_runtime_command(
    command: &[String],
    envs: &[(String, String)],
) -> Result<(Output, f64)> {
    let started = Instant::now();
    let output = Command::new(&command[0])
        .args(&command[1..])
        .envs(envs.iter().map(|(key, value)| (key.as_str(), value.as_str())))
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
