use std::fs;
use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

fn unique_temp_dir(label: &str) -> PathBuf {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock should be after unix epoch")
        .as_nanos();
    let suffix = COUNTER.fetch_add(1, Ordering::Relaxed);
    let path = std::env::temp_dir().join(format!("kcmt-cli-{label}-{nanos}-{suffix}"));
    fs::create_dir_all(&path).expect("temp dir should be created");
    path
}

#[test]
fn commit_status_help_uses_commit_branding() {
    let output = Command::new(env!("CARGO_BIN_EXE_commit"))
        .args(["status", "--help"])
        .output()
        .expect("commit binary should run");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Usage: commit status"));
}

#[test]
fn invalid_flag_returns_non_zero_and_parser_message() {
    let output = Command::new(env!("CARGO_BIN_EXE_kc"))
        .args(["--definitely-invalid-flag"])
        .output()
        .expect("kc binary should run");

    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("--definitely-invalid-flag"));
}

#[test]
fn status_repo_path_without_snapshot_returns_contract_message() {
    let repo = unique_temp_dir("status-repo");
    let config_home = unique_temp_dir("status-config");
    let output = Command::new(env!("CARGO_BIN_EXE_kcmt"))
        .env("KCMT_CONFIG_HOME", &config_home)
        .args(["status", "--repo-path"])
        .arg(&repo)
        .output()
        .expect("kcmt binary should run");

    assert_eq!(output.status.code(), Some(1));
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("No kcmt run history found for this repository."));
}

#[test]
fn file_mode_non_git_repo_returns_explicit_repo_error() {
    let repo = unique_temp_dir("non-git");
    let output = Command::new(env!("CARGO_BIN_EXE_kcmt"))
        .args(["--file", "example.py", "--repo-path"])
        .arg(&repo)
        .output()
        .expect("kcmt binary should run");

    assert_eq!(output.status.code(), Some(1));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("Not a Git repository"));
}
