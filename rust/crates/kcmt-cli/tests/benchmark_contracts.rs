use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

#[test]
fn benchmark_subcommand_requires_mode() {
    let output = Command::new(env!("CARGO_BIN_EXE_kcmt"))
        .args(["benchmark"])
        .output()
        .expect("kcmt binary should run");

    assert_eq!(output.status.code(), Some(1));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("requires a subcommand"));
}

#[test]
fn runtime_benchmark_rust_missing_binary_is_reported_as_excluded_json() {
    let repo = init_repo();
    seed_runtime_corpus(&repo, "pytest-runtime-rust-missing");

    let output = Command::new(env!("CARGO_BIN_EXE_kcmt"))
        .args(["benchmark", "runtime", "--repo-path"])
        .arg(&repo)
        .args([
            "--runtime",
            "rust",
            "--iterations",
            "1",
            "--rust-bin",
            "/tmp/definitely-missing-kcmt-rust-binary",
            "--json",
        ])
        .output()
        .expect("kcmt benchmark runtime should run");

    assert_eq!(
        output.status.code(),
        Some(0),
        "stdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let payload: serde_json::Value =
        serde_json::from_slice(&output.stdout).expect("runtime benchmark json");
    let results = payload["results"].as_array().expect("results array");
    assert_eq!(results.len(), 3);
    assert!(results.iter().all(|item| item["status"] == "excluded"));
    assert!(results.iter().all(|item| item["failure_reason"]
        .as_str()
        .unwrap_or_default()
        .contains("Rust binary not available")));
}

#[test]
fn runtime_benchmark_python_emits_passing_results_json() {
    let repo = init_repo();
    seed_runtime_corpus(&repo, "pytest-runtime-python");

    let output = Command::new(env!("CARGO_BIN_EXE_kcmt"))
        .env("KCMT_PYTHON_BIN", python_bin())
        .args(["benchmark", "runtime", "--repo-path"])
        .arg(&repo)
        .args(["--runtime", "python", "--iterations", "1", "--json"])
        .output()
        .expect("kcmt benchmark runtime should run");

    assert_eq!(
        output.status.code(),
        Some(0),
        "stdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let payload: serde_json::Value =
        serde_json::from_slice(&output.stdout).expect("runtime benchmark json");
    let results = payload["results"].as_array().expect("results array");
    assert_eq!(results.len(), 3);
    let iterations = payload["optimization_iterations"]
        .as_array()
        .expect("optimization iterations");
    assert_eq!(iterations.len(), 6);
    assert_eq!(iterations[0]["label"], "baseline");
    assert_eq!(iterations[0]["baseline"], true);
    assert_eq!(iterations[0]["measurement_status"], "measured");
    assert!(iterations[0]["median_wall_time_ms"].as_f64().is_some());
    assert!(iterations[0]["throughput_commits_per_sec"]
        .as_f64()
        .is_some());
    assert!(iterations[0]["quality_score"].as_f64().is_some());
    assert!(iterations[0]["failures"].as_u64().is_some());
    assert!(iterations.iter().skip(1).all(|item| {
        item["measurement_status"] == "planned"
            && item["median_wall_time_ms"].is_null()
            && item["throughput_commits_per_sec"].is_null()
            && item["quality_score"].is_null()
            && item["failures"].is_null()
    }));
    assert!(iterations
        .iter()
        .all(|item| item["next_bottleneck"].as_str().unwrap_or_default() != ""));
    assert!(results
        .iter()
        .all(|item| item["runtime"] == "python" && item["status"] == "passed"));
}

#[test]
fn runtime_benchmark_rust_ingests_snapshot_stage_timings_json() {
    let repo = init_repo();
    seed_runtime_corpus(&repo, "pytest-runtime-rust-telemetry");

    let output = Command::new(env!("CARGO_BIN_EXE_kcmt"))
        .args(["benchmark", "runtime", "--repo-path"])
        .arg(&repo)
        .args([
            "--runtime",
            "rust",
            "--iterations",
            "1",
            "--rust-bin",
            env!("CARGO_BIN_EXE_kcmt"),
            "--json",
        ])
        .output()
        .expect("kcmt benchmark runtime should run");

    assert_eq!(
        output.status.code(),
        Some(0),
        "stdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let payload: serde_json::Value =
        serde_json::from_slice(&output.stdout).expect("runtime benchmark json");
    let results = payload["results"].as_array().expect("results array");
    let file_result = results
        .iter()
        .find(|item| item["workflow_contract_id"] == "file-repo-path")
        .expect("file workflow result");
    let stages = file_result["stage_timings"]
        .as_array()
        .expect("stage timings array");
    for expected_stage in [
        "repo_discovery",
        "dispatch",
        "config_load",
        "status_scan",
        "diff_preparation",
        "llm_enqueue",
        "llm_wait",
        "response_validation",
        "commit_stage_path",
        "commit_create",
        "commit_read_hash",
        "commit",
        "push",
        "snapshot",
        "workflow_total",
    ] {
        assert!(
            stages.iter().any(|stage| stage["stage"] == expected_stage),
            "missing normalized telemetry stage {expected_stage}: {stages:?}"
        );
    }
    assert!(stages.iter().all(|stage| {
        stage["duration_ms"].as_f64().is_some() && stage["items"].as_u64().is_some()
    }));
}

#[test]
fn provider_benchmark_emits_json_csv_and_persists_snapshot() {
    let repo = init_repo();
    let config_home = unique_temp_dir("config-home");

    let output = Command::new(env!("CARGO_BIN_EXE_kcmt"))
        .env("KCMT_CONFIG_HOME", &config_home)
        .env("OPENAI_API_KEY", "test-openai-key")
        .env("KCMT_PROVIDER_RESPONSE", "fix(core): update benchmark")
        .env("KCMT_ALLOW_PROVIDER_RESPONSE_FIXTURE", "1")
        .args([
            "--benchmark",
            "--provider",
            "openai",
            "--model",
            "gpt-test",
            "--benchmark-limit",
            "1",
            "--benchmark-timeout",
            "0.1",
            "--benchmark-json",
            "--benchmark-csv",
            "--repo-path",
        ])
        .arg(&repo)
        .output()
        .expect("kcmt provider benchmark should run");

    assert!(
        output.status.success(),
        "stdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Benchmark Leaderboard"));
    assert!(stdout.contains("\"overall\""));
    assert!(stdout.contains("provider,model,avg_latency_ms"));
    assert!(stdout.contains("openai,gpt-test"));

    let json_start = stdout.find("{\n").expect("json payload start");
    let json_end = stdout[json_start..]
        .find("\n}\nprovider,model")
        .map(|idx| json_start + idx + 3)
        .expect("json payload end");
    let payload: serde_json::Value =
        serde_json::from_str(&stdout[json_start..json_end]).expect("benchmark json");
    assert_eq!(payload["overall"][0]["provider"], "openai");
    assert_eq!(payload["overall"][0]["model"], "gpt-test");
    assert_eq!(payload["overall"][0]["runs"], 5);

    let snapshots = benchmark_snapshots(&config_home);
    assert_eq!(snapshots.len(), 1);
    let snapshot: serde_json::Value =
        serde_json::from_slice(&fs::read(&snapshots[0]).expect("snapshot file"))
            .expect("snapshot json");
    assert_eq!(snapshot["schema_version"], 1);
    assert_eq!(snapshot["results"][0]["provider"], "openai");
    assert_eq!(snapshot["results"][0]["model"], "gpt-test");
}

fn unique_temp_dir(label: &str) -> PathBuf {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock should be after unix epoch")
        .as_nanos();
    let suffix = COUNTER.fetch_add(1, Ordering::Relaxed);
    let path = std::env::temp_dir().join(format!("kcmt-bench-{label}-{nanos}-{suffix}"));
    fs::create_dir_all(&path).expect("temp dir should be created");
    path
}

fn python_bin() -> PathBuf {
    if let Ok(configured) = std::env::var("KCMT_PYTHON_BIN") {
        if !configured.trim().is_empty() {
            return PathBuf::from(configured);
        }
    }
    if let Ok(virtual_env) = std::env::var("VIRTUAL_ENV") {
        if let Some(executable) = python_from_venv_root(Path::new(&virtual_env)) {
            return executable;
        }
    }
    if let Some(executable) = python_from_venv_root(&workspace_root().join(".venv")) {
        return executable;
    }
    PathBuf::from(if cfg!(windows) {
        "python.exe"
    } else {
        "python3"
    })
}

fn python_from_venv_root(root: &Path) -> Option<PathBuf> {
    [
        root.join("bin").join("python"),
        root.join("Scripts").join("python.exe"),
        root.join("Scripts").join("python"),
    ]
    .into_iter()
    .find(|candidate| candidate.exists())
}

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(3)
        .expect("workspace root")
        .to_path_buf()
}

fn git(repo: &Path, args: &[&str]) {
    let output = Command::new("git")
        .current_dir(repo)
        .args(args)
        .output()
        .expect("git command should run");
    assert!(
        output.status.success(),
        "git {:?} failed: {}",
        args,
        String::from_utf8_lossy(&output.stderr)
    );
}

fn init_repo() -> PathBuf {
    let repo = unique_temp_dir("repo");
    git(&repo, &["init", "-q"]);
    git(&repo, &["config", "user.name", "Tester"]);
    git(&repo, &["config", "user.email", "tester@example.com"]);
    repo
}

fn seed_runtime_corpus(repo: &Path, corpus_id: &str) {
    let source_file = repo.join("src").join("app.py");
    fs::create_dir_all(source_file.parent().expect("source parent"))
        .expect("source dir should be created");
    fs::write(&source_file, "def greet() -> str:\n    return 'hello'\n").expect("source file");
    fs::write(
        repo.join(".kcmt-runtime-corpus.json"),
        format!(
            "{{\"id\":\"{corpus_id}\",\"kind\":\"synthetic\",\"file_count\":1,\"git_history_state\":\"seeded-history\",\"change_shape\":[\"modified\",\"nested-paths\"],\"default_file_target\":\"src/app.py\"}}"
        ),
    )
    .expect("metadata file");
    git(repo, &["add", "."]);
    git(repo, &["commit", "-m", "chore(repo): seed"]);
    fs::write(
        source_file,
        "def greet() -> str:\n    return 'hello runtime'\n",
    )
    .expect("modified source");
}

fn benchmark_snapshots(config_home: &Path) -> Vec<PathBuf> {
    let repos = config_home.join("repos");
    let mut snapshots = Vec::new();
    let Ok(repo_entries) = fs::read_dir(repos) else {
        return snapshots;
    };
    for repo_entry in repo_entries.flatten() {
        let benchmark_dir = repo_entry.path().join("benchmarks");
        let Ok(entries) = fs::read_dir(benchmark_dir) else {
            continue;
        };
        snapshots.extend(entries.flatten().map(|entry| entry.path()).filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .map(|name| name.starts_with("benchmark-") && name.ends_with(".json"))
                .unwrap_or(false)
        }));
    }
    snapshots
}
