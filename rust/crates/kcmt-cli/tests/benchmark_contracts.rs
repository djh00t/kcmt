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
        .args([
            "benchmark",
            "runtime",
            "--repo-path",
        ])
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

    assert_eq!(output.status.code(), Some(0));
    let payload: serde_json::Value =
        serde_json::from_slice(&output.stdout).expect("runtime benchmark json");
    let results = payload["results"].as_array().expect("results array");
    assert_eq!(results.len(), 3);
    assert!(results
        .iter()
        .all(|item| item["status"] == "excluded"));
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
        .args([
            "benchmark",
            "runtime",
            "--repo-path",
        ])
        .arg(&repo)
        .args([
            "--runtime",
            "python",
            "--iterations",
            "1",
            "--json",
        ])
        .output()
        .expect("kcmt benchmark runtime should run");

    assert_eq!(output.status.code(), Some(0));
    let payload: serde_json::Value =
        serde_json::from_slice(&output.stdout).expect("runtime benchmark json");
    let results = payload["results"].as_array().expect("results array");
    assert_eq!(results.len(), 3);
    assert!(results
        .iter()
        .all(|item| item["runtime"] == "python" && item["status"] == "passed"));
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
    fs::write(&source_file, "def greet() -> str:\n    return 'hello'\n")
        .expect("source file");
    fs::write(
        repo.join(".kcmt-runtime-corpus.json"),
        format!(
            "{{\"id\":\"{corpus_id}\",\"kind\":\"synthetic\",\"file_count\":1,\"git_history_state\":\"seeded-history\",\"change_shape\":[\"modified\",\"nested-paths\"],\"default_file_target\":\"src/app.py\"}}"
        ),
    )
    .expect("metadata file");
    git(repo, &["add", "."]);
    git(repo, &["commit", "-m", "chore(repo): seed"]);
    fs::write(source_file, "def greet() -> str:\n    return 'hello runtime'\n")
        .expect("modified source");
}
