use std::fs;
use std::io::{Read, Write};
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc;
use std::thread;
use std::time::Duration;
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

fn git(repo: &Path, args: &[&str]) -> String {
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
    String::from_utf8_lossy(&output.stdout).trim().to_string()
}

fn kcmt_command(binary_path: &str) -> Command {
    let mut command = Command::new(binary_path);
    command.env_clear();
    for key in ["PATH", "HOME", "USER", "TMPDIR", "LANG", "LC_ALL"] {
        if let Ok(value) = std::env::var(key) {
            command.env(key, value);
        }
    }
    command.env("KCMT_ALLOW_LOCAL_SYNTHESIS", "1");
    command.env("KCMT_RUNTIME_BENCHMARK", "0");
    command
}

fn init_repo() -> PathBuf {
    let repo = unique_temp_dir("repo");
    git(&repo, &["init", "-q"]);
    git(&repo, &["config", "user.name", "Tester"]);
    git(&repo, &["config", "user.email", "tester@example.com"]);
    repo
}

fn raw_status_snapshot(repo: &Path, config_home: &Path) -> serde_json::Value {
    let status_output = kcmt_command(env!("CARGO_BIN_EXE_kcmt"))
        .env("KCMT_CONFIG_HOME", config_home)
        .args(["status", "--raw", "--repo-path"])
        .arg(repo)
        .output()
        .expect("kcmt status should run");
    assert!(
        status_output.status.success(),
        "stdout: {}\nstderr: {}",
        String::from_utf8_lossy(&status_output.stdout),
        String::from_utf8_lossy(&status_output.stderr)
    );
    serde_json::from_slice(&status_output.stdout).expect("raw status is json")
}

fn telemetry_stage_items(snapshot: &serde_json::Value, stage_name: &str) -> u64 {
    snapshot["telemetry"]["stages"]
        .as_array()
        .expect("telemetry stages")
        .iter()
        .find(|stage| stage["stage"] == stage_name)
        .unwrap_or_else(|| panic!("missing telemetry stage {stage_name}"))
        .get("items")
        .and_then(serde_json::Value::as_u64)
        .expect("stage items should be an integer")
}

fn init_bare_remote() -> PathBuf {
    let remote = unique_temp_dir("remote");
    git(&remote, &["init", "--bare", "-q"]);
    remote
}

fn spawn_provider_response(response_body: &'static str) -> (String, mpsc::Receiver<String>) {
    let listener = TcpListener::bind("127.0.0.1:0").expect("mock provider listener");
    let address = listener.local_addr().expect("mock provider address");
    let (sender, receiver) = mpsc::channel();
    thread::spawn(move || {
        let (mut stream, _) = listener.accept().expect("mock provider connection");
        let mut buffer = [0_u8; 8192];
        let bytes = stream.read(&mut buffer).expect("mock provider read");
        sender
            .send(String::from_utf8_lossy(&buffer[..bytes]).to_string())
            .expect("request should be recorded");
        let response = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
            response_body.len(),
            response_body
        );
        stream
            .write_all(response.as_bytes())
            .expect("mock provider write");
    });
    (format!("http://{address}"), receiver)
}

fn spawn_provider_status_response(
    status: u16,
    response_body: &'static str,
) -> (String, mpsc::Receiver<String>) {
    let listener = TcpListener::bind("127.0.0.1:0").expect("mock provider listener");
    let address = listener.local_addr().expect("mock provider address");
    let (sender, receiver) = mpsc::channel();
    thread::spawn(move || {
        let (mut stream, _) = listener.accept().expect("mock provider connection");
        let mut buffer = [0_u8; 8192];
        let bytes = stream.read(&mut buffer).expect("mock provider read");
        sender
            .send(String::from_utf8_lossy(&buffer[..bytes]).to_string())
            .expect("request should be recorded");
        let response = format!(
            "HTTP/1.1 {status} OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nconnection: close\r\n\r\n{}",
            response_body.len(),
            response_body
        );
        stream
            .write_all(response.as_bytes())
            .expect("mock provider write");
    });
    (format!("http://{address}"), receiver)
}

fn spawn_openai_batch_response() -> (String, mpsc::Receiver<String>) {
    let listener = TcpListener::bind("127.0.0.1:0").expect("mock batch listener");
    let address = listener.local_addr().expect("mock batch address");
    let (sender, receiver) = mpsc::channel();
    thread::spawn(move || {
        let responses = [
            r#"{"id":"file_1"}"#,
            r#"{"id":"batch_1","status":"validating"}"#,
            r#"{"id":"batch_1","status":"completed","output_file_id":"output_1"}"#,
            r#"{"custom_id":"alpha.py","response":{"status_code":200,"body":{"choices":[{"message":{"content":"fix(alpha): batch alpha."}}]}}}
{"custom_id":"beta.py","response":{"status_code":200,"body":{"choices":[{"message":{"content":"fix(beta): batch beta."}}]}}}
"#,
        ];
        for response_body in responses {
            let (mut stream, _) = listener.accept().expect("mock batch connection");
            let mut buffer = [0_u8; 16384];
            let bytes = stream.read(&mut buffer).expect("mock batch read");
            sender
                .send(String::from_utf8_lossy(&buffer[..bytes]).to_string())
                .expect("request should be recorded");
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nconnection: close\r\n\r\n{}",
                response_body.len(),
                response_body
            );
            stream
                .write_all(response.as_bytes())
                .expect("mock batch write");
        }
    });
    (format!("http://{address}"), receiver)
}

fn spawn_provider_fallback_response() -> (String, mpsc::Receiver<String>) {
    let listener = TcpListener::bind("127.0.0.1:0").expect("mock fallback listener");
    let address = listener.local_addr().expect("mock fallback address");
    let (sender, receiver) = mpsc::channel();
    thread::spawn(move || {
        let responses = [
            (500, r#"{"error":{"message":"primary unavailable"}}"#),
            (500, r#"{"error":{"message":"primary unavailable"}}"#),
            (500, r#"{"error":{"message":"primary unavailable"}}"#),
            (500, r#"{"error":{"message":"primary unavailable"}}"#),
            (
                200,
                r#"{"content":[{"type":"text","text":"fix(fallback): use secondary provider."}]}"#,
            ),
        ];
        for (status, response_body) in responses {
            let (mut stream, _) = listener.accept().expect("mock fallback connection");
            let mut buffer = [0_u8; 16384];
            let bytes = stream.read(&mut buffer).expect("mock fallback read");
            sender
                .send(String::from_utf8_lossy(&buffer[..bytes]).to_string())
                .expect("request should be recorded");
            let response = format!(
                "HTTP/1.1 {status} OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nconnection: close\r\n\r\n{}",
                response_body.len(),
                response_body
            );
            stream
                .write_all(response.as_bytes())
                .expect("mock fallback write");
        }
    });
    (format!("http://{address}"), receiver)
}

#[test]
fn file_mode_commits_only_requested_path() {
    let repo = init_repo();
    fs::write(repo.join("example.py"), "print('hello')\n").expect("seed file");
    git(&repo, &["add", "example.py"]);
    git(&repo, &["commit", "-m", "chore(repo): seed"]);

    fs::write(repo.join("example.py"), "print('hello world')\n").expect("modified file");
    fs::write(repo.join("other.py"), "print('other')\n").expect("other file");

    let output = kcmt_command(env!("CARGO_BIN_EXE_kcmt"))
        .args(["--file", "example.py", "--no-auto-push", "--repo-path"])
        .arg(&repo)
        .output()
        .expect("kcmt binary should run");

    assert!(
        output.status.success(),
        "stdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("✓ example.py"));
    assert!(stdout.contains("chore(repo): update example"));
    let log = git(&repo, &["log", "--oneline", "-1"]);
    assert!(log.contains("chore(repo): update example"));

    let status = git(&repo, &["status", "--short"]);
    assert!(status.contains("?? other.py"));
    assert!(!status.contains("example.py"));
}

#[test]
fn oneshot_mode_commits_changed_non_deletions() {
    let repo = init_repo();
    fs::write(repo.join("alpha.py"), "print('alpha')\n").expect("alpha seed");
    fs::write(repo.join("beta.py"), "print('beta')\n").expect("beta seed");
    git(&repo, &["add", "alpha.py", "beta.py"]);
    git(&repo, &["commit", "-m", "chore(repo): seed"]);

    fs::write(repo.join("alpha.py"), "print('alpha updated')\n").expect("alpha change");
    fs::write(repo.join("beta.py"), "print('beta updated')\n").expect("beta change");

    let output = kcmt_command(env!("CARGO_BIN_EXE_commit"))
        .args(["--oneshot", "--no-auto-push", "--repo-path"])
        .arg(&repo)
        .output()
        .expect("commit binary should run");

    assert!(
        output.status.success(),
        "stdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("✓ alpha.py"));
    assert!(!stdout.contains("✓ beta.py"));
    let log = git(&repo, &["log", "--pretty=%s"]);
    let subjects: Vec<&str> = log.lines().collect();
    assert!(subjects.contains(&"chore(repo): update alpha"));
    assert!(!subjects.contains(&"chore(repo): update beta"));

    let status = git(&repo, &["status", "--short"]);
    assert_eq!(status, "M beta.py");
}

#[test]
fn oneshot_mode_persists_single_selected_file_snapshot() {
    let repo = init_repo();
    let config_home = unique_temp_dir("config-home");
    fs::write(repo.join("alpha.py"), "print('alpha')\n").expect("alpha seed");
    fs::write(repo.join("beta.py"), "print('beta')\n").expect("beta seed");
    git(&repo, &["add", "alpha.py", "beta.py"]);
    git(&repo, &["commit", "-m", "chore(repo): seed"]);

    fs::write(repo.join("alpha.py"), "print('alpha updated')\n").expect("alpha change");
    fs::write(repo.join("beta.py"), "print('beta updated')\n").expect("beta change");

    let output = kcmt_command(env!("CARGO_BIN_EXE_commit"))
        .env("KCMT_CONFIG_HOME", &config_home)
        .args(["--oneshot", "--no-auto-push", "--repo-path"])
        .arg(&repo)
        .output()
        .expect("commit binary should run");

    assert!(
        output.status.success(),
        "stdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("✓ alpha.py"));
    assert!(!stdout.contains("✓ beta.py"));

    let log = git(&repo, &["log", "--pretty=%s"]);
    let subjects: Vec<&str> = log.lines().collect();
    assert!(subjects.contains(&"chore(repo): update alpha"));
    assert!(!subjects.contains(&"chore(repo): update beta"));

    let status = git(&repo, &["status", "--short"]);
    assert_eq!(status, "M beta.py");

    let status_output = kcmt_command(env!("CARGO_BIN_EXE_kcmt"))
        .env("KCMT_CONFIG_HOME", &config_home)
        .args(["status", "--raw", "--repo-path"])
        .arg(&repo)
        .output()
        .expect("kcmt status should run");

    assert!(status_output.status.success());
    let snapshot: serde_json::Value =
        serde_json::from_slice(&status_output.stdout).expect("raw status is json");
    assert_eq!(snapshot["counts"]["files_total"], 1);
    assert_eq!(snapshot["counts"]["commit_success"], 1);
    assert_eq!(snapshot["counts"]["overall_success"], 1);
    assert_eq!(snapshot["telemetry"]["schema_version"], 1);
    let stages: Vec<&str> = snapshot["telemetry"]["stages"]
        .as_array()
        .expect("telemetry stages")
        .iter()
        .map(|stage| stage["stage"].as_str().expect("stage name"))
        .collect();
    assert!(stages.contains(&"status_scan"));
    assert!(stages.contains(&"diff_preparation"));
    assert!(stages.contains(&"llm_enqueue"));
    assert!(stages.contains(&"llm_wait"));
    assert!(stages.contains(&"response_validation"));
    assert!(stages.contains(&"commit"));
    assert!(stages.contains(&"push"));
    assert!(stages.contains(&"snapshot"));
    assert_eq!(
        snapshot["commits"].as_array().expect("commits array").len(),
        1
    );
}

#[test]
fn default_mode_commits_all_changed_files_separately() {
    let repo = init_repo();
    let config_home = unique_temp_dir("config-home");
    fs::write(repo.join("alpha.py"), "print('alpha')\n").expect("alpha seed");
    fs::write(repo.join("beta.py"), "print('beta')\n").expect("beta seed");
    git(&repo, &["add", "alpha.py", "beta.py"]);
    git(&repo, &["commit", "-m", "chore(repo): seed"]);

    fs::write(repo.join("alpha.py"), "print('alpha updated')\n").expect("alpha change");
    fs::write(repo.join("beta.py"), "print('beta updated')\n").expect("beta change");

    let output = kcmt_command(env!("CARGO_BIN_EXE_kcmt"))
        .env("KCMT_CONFIG_HOME", &config_home)
        .args(["--no-auto-push", "--repo-path"])
        .arg(&repo)
        .output()
        .expect("kcmt binary should run");

    assert!(
        output.status.success(),
        "stdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("✓ alpha.py"));
    assert!(stdout.contains("✓ beta.py"));

    let log = git(&repo, &["log", "--pretty=%s"]);
    let subjects: Vec<&str> = log.lines().collect();
    assert!(subjects.contains(&"chore(repo): update alpha"));
    assert!(subjects.contains(&"chore(repo): update beta"));
    assert!(git(&repo, &["status", "--short"]).is_empty());
}

#[test]
fn default_mode_records_prepare_failure_without_blocking_deletion_commit() {
    let repo = init_repo();
    let config_home = unique_temp_dir("config-home");
    fs::write(repo.join("delete_me.txt"), "delete me\n").expect("delete seed");
    fs::write(repo.join("tracked.py"), "print('seed')\n").expect("tracked seed");
    git(&repo, &["add", "delete_me.txt", "tracked.py"]);
    git(&repo, &["commit", "-m", "chore(repo): seed"]);

    fs::remove_file(repo.join("delete_me.txt")).expect("delete file");
    fs::write(repo.join("tracked.py"), "print('changed')\n").expect("tracked change");

    let output = kcmt_command(env!("CARGO_BIN_EXE_kcmt"))
        .env("KCMT_CONFIG_HOME", &config_home)
        .env("KCMT_PROVIDER_RESPONSE", "This changes a few files")
        .env("KCMT_ALLOW_PROVIDER_RESPONSE_FIXTURE", "1")
        .args(["--no-auto-push", "--repo-path"])
        .arg(&repo)
        .output()
        .expect("kcmt binary should run");

    assert!(
        output.status.success(),
        "stdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("✓ delete_me.txt"));
    assert!(stdout.contains("✗ tracked.py"));
    assert!(stdout.contains("missing conventional commit header"));

    let log = git(&repo, &["log", "--pretty=%s", "-1"]);
    assert_eq!(log, "chore(delete_me-txt): file deleted");
    let status = git(&repo, &["status", "--short"]);
    assert_eq!(status, "M tracked.py");

    let status_output = kcmt_command(env!("CARGO_BIN_EXE_kcmt"))
        .env("KCMT_CONFIG_HOME", &config_home)
        .args(["status", "--raw", "--repo-path"])
        .arg(&repo)
        .output()
        .expect("kcmt status should run");

    assert!(status_output.status.success());
    let snapshot: serde_json::Value =
        serde_json::from_slice(&status_output.stdout).expect("raw status is json");
    assert_eq!(snapshot["counts"]["files_total"], 2);
    assert_eq!(snapshot["counts"]["prepared_total"], 1);
    assert_eq!(snapshot["counts"]["processed_total"], 2);
    assert_eq!(snapshot["counts"]["prepared_failures"], 1);
    assert_eq!(snapshot["counts"]["commit_success"], 0);
    assert_eq!(snapshot["counts"]["commit_failure"], 1);
    assert_eq!(snapshot["counts"]["deletions_total"], 1);
    assert_eq!(snapshot["counts"]["deletions_success"], 1);
    assert_eq!(snapshot["counts"]["overall_success"], 1);
    assert_eq!(snapshot["counts"]["overall_failure"], 1);
    assert_eq!(snapshot["commits"][0]["success"], false);
    assert_eq!(snapshot["commits"][0]["file_path"], "tracked.py");
    assert_eq!(snapshot["deletions"][0]["success"], true);
    assert_eq!(snapshot["deletions"][0]["file_path"], "delete_me.txt");
}

#[test]
fn workers_flag_controls_prepare_worker_count_in_snapshot() {
    let repo = init_repo();
    let config_home = unique_temp_dir("config-home");
    fs::write(repo.join("alpha.py"), "print('alpha')\n").expect("alpha seed");
    fs::write(repo.join("beta.py"), "print('beta')\n").expect("beta seed");
    git(&repo, &["add", "alpha.py", "beta.py"]);
    git(&repo, &["commit", "-m", "chore(repo): seed"]);

    fs::write(repo.join("alpha.py"), "print('alpha updated')\n").expect("alpha change");
    fs::write(repo.join("beta.py"), "print('beta updated')\n").expect("beta change");

    let output = kcmt_command(env!("CARGO_BIN_EXE_kcmt"))
        .env("KCMT_CONFIG_HOME", &config_home)
        .args(["--workers", "2", "--no-auto-push", "--repo-path"])
        .arg(&repo)
        .output()
        .expect("kcmt binary should run");

    assert!(
        output.status.success(),
        "stdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let status_output = kcmt_command(env!("CARGO_BIN_EXE_kcmt"))
        .env("KCMT_CONFIG_HOME", &config_home)
        .args(["status", "--raw", "--repo-path"])
        .arg(&repo)
        .output()
        .expect("kcmt status should run");

    assert!(status_output.status.success());
    let snapshot: serde_json::Value =
        serde_json::from_slice(&status_output.stdout).expect("raw status is json");
    assert_eq!(snapshot["telemetry"]["prepare_workers"], 2);
    assert_eq!(snapshot["counts"]["overall_success"], 2);
}

#[test]
fn compact_verbose_and_profile_flags_control_workflow_output() {
    let repo = init_repo();
    fs::write(repo.join("tracked.py"), "print('seed')\n").expect("tracked seed");
    git(&repo, &["add", "tracked.py"]);
    git(&repo, &["commit", "-m", "chore(repo): seed"]);
    fs::write(repo.join("tracked.py"), "print('changed')\n").expect("tracked change");

    let compact = kcmt_command(env!("CARGO_BIN_EXE_kcmt"))
        .args(["--compact", "--no-auto-push", "--repo-path"])
        .arg(&repo)
        .output()
        .expect("kcmt binary should run");
    assert!(
        compact.status.success(),
        "stdout: {}\nstderr: {}",
        String::from_utf8_lossy(&compact.stdout),
        String::from_utf8_lossy(&compact.stderr)
    );
    let compact_stdout = String::from_utf8_lossy(&compact.stdout);
    assert!(compact_stdout.contains("Run Summary"));
    assert!(compact_stdout.contains("Commits 1  Failures 0"));
    assert!(!compact_stdout.contains("✓ tracked.py"));

    fs::write(repo.join("tracked.py"), "print('changed again')\n").expect("tracked change");
    let verbose_profile = kcmt_command(env!("CARGO_BIN_EXE_kcmt"))
        .args([
            "--compact",
            "--verbose",
            "--profile-startup",
            "--no-auto-push",
            "--repo-path",
        ])
        .arg(&repo)
        .output()
        .expect("kcmt binary should run");
    assert!(
        verbose_profile.status.success(),
        "stdout: {}\nstderr: {}",
        String::from_utf8_lossy(&verbose_profile.stdout),
        String::from_utf8_lossy(&verbose_profile.stderr)
    );
    let verbose_stdout = String::from_utf8_lossy(&verbose_profile.stdout);
    assert!(verbose_stdout.contains("Run Summary"));
    assert!(verbose_stdout.contains("✓ tracked.py"));
    assert!(verbose_stdout.contains("[kcmt-profile] status_scan:"));
    assert!(verbose_stdout.contains("[kcmt-profile] snapshot:"));
}

#[test]
fn oneshot_mode_respects_file_limit() {
    let repo = init_repo();
    let config_home = unique_temp_dir("config-home");
    fs::write(repo.join("alpha.py"), "print('alpha')\n").expect("alpha seed");
    fs::write(repo.join("beta.py"), "print('beta')\n").expect("beta seed");
    git(&repo, &["add", "alpha.py", "beta.py"]);
    git(&repo, &["commit", "-m", "chore(repo): seed"]);

    fs::write(repo.join("alpha.py"), "print('alpha updated')\n").expect("alpha change");
    fs::write(repo.join("beta.py"), "print('beta updated')\n").expect("beta change");

    let output = kcmt_command(env!("CARGO_BIN_EXE_commit"))
        .env("KCMT_CONFIG_HOME", &config_home)
        .args(["--oneshot", "--limit", "1", "--no-auto-push", "--repo-path"])
        .arg(&repo)
        .output()
        .expect("commit binary should run");

    assert!(
        output.status.success(),
        "stdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    let committed_alpha = stdout.contains("✓ alpha.py");
    let committed_beta = stdout.contains("✓ beta.py");
    assert_ne!(committed_alpha, committed_beta);

    let log = git(&repo, &["log", "--pretty=%s"]);
    let subjects: Vec<&str> = log.lines().collect();
    assert_ne!(
        subjects.contains(&"chore(repo): update alpha"),
        subjects.contains(&"chore(repo): update beta")
    );

    let status = git(&repo, &["status", "--short"]);
    if committed_alpha {
        assert_eq!(status, "M beta.py");
    } else {
        assert_eq!(status, "M alpha.py");
    }
}

#[test]
fn oneshot_limit_applies_before_non_deletion_preference() {
    let repo = init_repo();
    let config_home = unique_temp_dir("config-home");
    fs::write(repo.join("aaa_delete.txt"), "bye\n").expect("delete seed");
    fs::write(repo.join("zzz_keep.py"), "print('seed')\n").expect("keep seed");
    git(&repo, &["add", "aaa_delete.txt", "zzz_keep.py"]);
    git(&repo, &["commit", "-m", "chore(repo): seed"]);

    fs::remove_file(repo.join("aaa_delete.txt")).expect("delete tracked file");
    fs::write(repo.join("zzz_keep.py"), "print('changed')\n").expect("modify later file");

    let output = kcmt_command(env!("CARGO_BIN_EXE_kcmt"))
        .env("KCMT_CONFIG_HOME", &config_home)
        .args(["--oneshot", "--limit", "1", "--no-auto-push", "--repo-path"])
        .arg(&repo)
        .output()
        .expect("kcmt binary should run");

    assert!(
        output.status.success(),
        "stdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("✓ aaa_delete.txt"));
    assert!(!stdout.contains("✓ zzz_keep.py"));

    let status = git(&repo, &["status", "--short"]);
    assert_eq!(status, "M zzz_keep.py");
}

#[test]
fn oneshot_mode_commits_deletion_only_change() {
    let repo = init_repo();
    let config_home = unique_temp_dir("config-home");
    fs::write(repo.join("delete_me.txt"), "bye\n").expect("seed file");
    git(&repo, &["add", "delete_me.txt"]);
    git(&repo, &["commit", "-m", "chore(repo): seed"]);
    fs::remove_file(repo.join("delete_me.txt")).expect("delete file");

    let output = kcmt_command(env!("CARGO_BIN_EXE_kcmt"))
        .env("KCMT_CONFIG_HOME", &config_home)
        .args(["--oneshot", "--no-auto-push", "--repo-path"])
        .arg(&repo)
        .output()
        .expect("kcmt binary should run");

    assert!(
        output.status.success(),
        "stdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("✓ delete_me.txt"));
    assert!(stdout.contains("chore(delete_me-txt): file deleted"));

    let log = git(&repo, &["log", "--oneline", "-1"]);
    assert!(log.contains("chore(delete_me-txt): file deleted"));

    let status = git(&repo, &["status", "--short"]);
    assert!(status.is_empty(), "unexpected dirty status: {status}");

    let status_output = kcmt_command(env!("CARGO_BIN_EXE_kcmt"))
        .env("KCMT_CONFIG_HOME", &config_home)
        .args(["status", "--raw", "--repo-path"])
        .arg(&repo)
        .output()
        .expect("kcmt status should run");

    assert!(status_output.status.success());
    let snapshot: serde_json::Value =
        serde_json::from_slice(&status_output.stdout).expect("raw status is json");
    assert_eq!(snapshot["counts"]["files_total"], 1);
    assert_eq!(snapshot["counts"]["commit_success"], 0);
    assert_eq!(snapshot["counts"]["deletions_total"], 1);
    assert_eq!(snapshot["counts"]["deletions_success"], 1);
    assert_eq!(snapshot["counts"]["overall_success"], 1);
    assert_eq!(
        snapshot["deletions"]
            .as_array()
            .expect("deletions array")
            .len(),
        1
    );
}

#[test]
fn file_mode_persists_snapshot_for_status_view() {
    let repo = init_repo();
    let config_home = unique_temp_dir("config-home");
    fs::write(repo.join("tracked.py"), "print('seed')\n").expect("tracked seed");
    git(&repo, &["add", "tracked.py"]);
    git(&repo, &["commit", "-m", "chore(repo): seed"]);
    fs::write(repo.join("tracked.py"), "print('changed')\n").expect("tracked change");

    let output = kcmt_command(env!("CARGO_BIN_EXE_kcmt"))
        .env("KCMT_CONFIG_HOME", &config_home)
        .args(["--file", "tracked.py", "--no-auto-push", "--repo-path"])
        .arg(&repo)
        .output()
        .expect("kcmt binary should run");
    assert!(output.status.success());

    let status_output = kcmt_command(env!("CARGO_BIN_EXE_kcmt"))
        .env("KCMT_CONFIG_HOME", &config_home)
        .args(["status", "--repo-path"])
        .arg(&repo)
        .output()
        .expect("kcmt status should run");

    assert!(status_output.status.success());
    let stdout = String::from_utf8_lossy(&status_output.stdout);
    assert!(stdout.contains("Commit status"));
    assert!(stdout.contains("Success: 1"));
}

#[test]
fn tracked_modified_file_commit_skips_staging_subprocess() {
    let repo = init_repo();
    let config_home = unique_temp_dir("config-home");
    fs::write(repo.join("tracked.py"), "print('seed')\n").expect("tracked seed");
    git(&repo, &["add", "tracked.py"]);
    git(&repo, &["commit", "-m", "chore(repo): seed"]);
    fs::write(repo.join("tracked.py"), "print('changed')\n").expect("tracked change");

    let output = kcmt_command(env!("CARGO_BIN_EXE_kcmt"))
        .env("KCMT_CONFIG_HOME", &config_home)
        .args(["--file", "tracked.py", "--no-auto-push", "--repo-path"])
        .arg(&repo)
        .output()
        .expect("kcmt binary should run");

    assert!(
        output.status.success(),
        "stdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let snapshot = raw_status_snapshot(&repo, &config_home);
    assert_eq!(snapshot["counts"]["overall_success"], 1);
    assert_eq!(telemetry_stage_items(&snapshot, "commit_stage_path"), 0);
}

#[test]
fn tracked_deleted_file_commit_skips_staging_subprocess() {
    let repo = init_repo();
    let config_home = unique_temp_dir("config-home");
    fs::write(repo.join("delete_me.txt"), "bye\n").expect("seed file");
    git(&repo, &["add", "delete_me.txt"]);
    git(&repo, &["commit", "-m", "chore(repo): seed"]);
    fs::remove_file(repo.join("delete_me.txt")).expect("delete file");

    let output = kcmt_command(env!("CARGO_BIN_EXE_kcmt"))
        .env("KCMT_CONFIG_HOME", &config_home)
        .args(["--file", "delete_me.txt", "--no-auto-push", "--repo-path"])
        .arg(&repo)
        .output()
        .expect("kcmt binary should run");

    assert!(
        output.status.success(),
        "stdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let snapshot = raw_status_snapshot(&repo, &config_home);
    assert_eq!(snapshot["counts"]["overall_success"], 1);
    assert_eq!(snapshot["counts"]["deletions_success"], 1);
    assert_eq!(telemetry_stage_items(&snapshot, "commit_stage_path"), 0);
}

#[test]
fn untracked_file_commit_still_stages_path() {
    let repo = init_repo();
    let config_home = unique_temp_dir("config-home");
    fs::write(repo.join("new_file.py"), "print('new')\n").expect("new file");

    let output = kcmt_command(env!("CARGO_BIN_EXE_kcmt"))
        .env("KCMT_CONFIG_HOME", &config_home)
        .args(["--file", "new_file.py", "--no-auto-push", "--repo-path"])
        .arg(&repo)
        .output()
        .expect("kcmt binary should run");

    assert!(
        output.status.success(),
        "stdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let snapshot = raw_status_snapshot(&repo, &config_home);
    assert_eq!(snapshot["counts"]["overall_success"], 1);
    assert_eq!(telemetry_stage_items(&snapshot, "commit_stage_path"), 1);
}

#[test]
fn gix_commit_backend_commits_tracked_file_without_stage_path() {
    let repo = init_repo();
    let config_home = unique_temp_dir("config-home");
    fs::write(repo.join("tracked.py"), "print('seed')\n").expect("tracked seed");
    git(&repo, &["add", "tracked.py"]);
    git(&repo, &["commit", "-m", "chore(repo): seed"]);
    fs::write(repo.join("tracked.py"), "print('changed')\n").expect("tracked change");

    let output = kcmt_command(env!("CARGO_BIN_EXE_kcmt"))
        .env("KCMT_CONFIG_HOME", &config_home)
        .env("KCMT_GIT_COMMIT_BACKEND", "gix")
        .args(["--file", "tracked.py", "--no-auto-push", "--repo-path"])
        .arg(&repo)
        .output()
        .expect("kcmt binary should run");

    assert!(
        output.status.success(),
        "stdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(git(&repo, &["status", "--short"]), "");
    let snapshot = raw_status_snapshot(&repo, &config_home);
    assert_eq!(snapshot["counts"]["overall_success"], 1);
    assert_eq!(telemetry_stage_items(&snapshot, "commit_stage_path"), 0);
}

#[test]
fn gix_commit_backend_stages_untracked_file_workflow() {
    let repo = init_repo();
    let config_home = unique_temp_dir("config-home");
    fs::write(repo.join("tracked.py"), "print('seed')\n").expect("tracked seed");
    git(&repo, &["add", "tracked.py"]);
    git(&repo, &["commit", "-m", "chore(repo): seed"]);
    fs::write(repo.join("new_file.py"), "print('new')\n").expect("new file");

    let output = kcmt_command(env!("CARGO_BIN_EXE_kcmt"))
        .env("KCMT_CONFIG_HOME", &config_home)
        .env("KCMT_GIT_COMMIT_BACKEND", "gix")
        .args(["--file", "new_file.py", "--no-auto-push", "--repo-path"])
        .arg(&repo)
        .output()
        .expect("kcmt binary should run");

    assert!(
        output.status.success(),
        "stdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(git(&repo, &["status", "--short"]), "");
    let snapshot = raw_status_snapshot(&repo, &config_home);
    assert_eq!(snapshot["counts"]["overall_success"], 1);
    assert_eq!(telemetry_stage_items(&snapshot, "commit_stage_path"), 1);
}

#[test]
fn workflow_flags_override_snapshot_config() {
    let repo = init_repo();
    let config_home = unique_temp_dir("config-home");
    fs::write(repo.join("tracked.py"), "print('seed')\n").expect("tracked seed");
    git(&repo, &["add", "tracked.py"]);
    git(&repo, &["commit", "-m", "chore(repo): seed"]);
    fs::write(repo.join("tracked.py"), "print('changed')\n").expect("tracked change");

    let output = kcmt_command(env!("CARGO_BIN_EXE_kcmt"))
        .env("KCMT_CONFIG_HOME", &config_home)
        .args([
            "--file",
            "tracked.py",
            "--provider",
            "openai",
            "--model",
            "gpt-test",
            "--endpoint",
            "https://example.test/v1",
            "--api-key-env",
            "OPENAI_TEST_KEY",
            "--batch",
            "--batch-model",
            "gpt-batch-test",
            "--batch-timeout",
            "1000",
            "--no-auto-push",
            "--max-commit-length",
            "68",
            "--repo-path",
        ])
        .arg(&repo)
        .output()
        .expect("kcmt binary should run");
    assert!(output.status.success());

    let status_output = kcmt_command(env!("CARGO_BIN_EXE_kcmt"))
        .env("KCMT_CONFIG_HOME", &config_home)
        .args(["status", "--raw", "--repo-path"])
        .arg(&repo)
        .output()
        .expect("kcmt status should run");

    assert!(status_output.status.success());
    let snapshot: serde_json::Value =
        serde_json::from_slice(&status_output.stdout).expect("raw status is json");
    assert_eq!(snapshot["provider"], "openai");
    assert_eq!(snapshot["model"], "gpt-test");
    assert_eq!(snapshot["endpoint"], "https://example.test/v1");
    assert_eq!(snapshot["config"]["api_key_env"], "OPENAI_TEST_KEY");
    assert_eq!(snapshot["batch"]["use_batch"], true);
    assert_eq!(snapshot["batch"]["batch_model"], "gpt-batch-test");
    assert_eq!(snapshot["batch"]["batch_timeout_seconds"], 1000);
    assert_eq!(snapshot["pushed"], false);
    assert_eq!(snapshot["config"]["auto_push"], false);
    assert_eq!(snapshot["config"]["max_commit_length"], 68);
}

#[test]
fn file_mode_uses_sanitized_provider_response_when_available() {
    let repo = init_repo();
    fs::write(repo.join("tracked.py"), "print('seed')\n").expect("tracked seed");
    git(&repo, &["add", "tracked.py"]);
    git(&repo, &["commit", "-m", "chore(repo): seed"]);
    fs::write(repo.join("tracked.py"), "print('changed')\n").expect("tracked change");

    let output = kcmt_command(env!("CARGO_BIN_EXE_kcmt"))
        .env(
            "KCMT_PROVIDER_RESPONSE",
            "```text\nHere is the commit:\n- `fix(core): handle provider output.`\n```",
        )
        .env("KCMT_ALLOW_PROVIDER_RESPONSE_FIXTURE", "1")
        .args(["--file", "tracked.py", "--no-auto-push", "--repo-path"])
        .arg(&repo)
        .output()
        .expect("kcmt binary should run");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("fix(core): handle provider output"));

    let log = git(&repo, &["log", "--pretty=%s", "-1"]);
    assert_eq!(log, "fix(core): handle provider output");
}

#[test]
fn file_mode_invokes_openai_compatible_provider_when_api_key_is_available() {
    let repo = init_repo();
    let (endpoint, request_rx) = spawn_provider_response(
        r#"{"choices":[{"message":{"content":"fix(core): use provider message."}}]}"#,
    );
    fs::write(repo.join("tracked.py"), "print('seed')\n").expect("tracked seed");
    git(&repo, &["add", "tracked.py"]);
    git(&repo, &["commit", "-m", "chore(repo): seed"]);
    fs::write(repo.join("tracked.py"), "print('changed')\n").expect("tracked change");

    let output = kcmt_command(env!("CARGO_BIN_EXE_kcmt"))
        .env("OPENAI_TEST_KEY", "test-key")
        .args([
            "--file",
            "tracked.py",
            "--provider",
            "openai",
            "--endpoint",
            &endpoint,
            "--api-key-env",
            "OPENAI_TEST_KEY",
            "--model",
            "gpt-mock",
            "--no-auto-push",
            "--repo-path",
        ])
        .arg(&repo)
        .output()
        .expect("kcmt binary should run");

    assert!(output.status.success());
    let request = request_rx
        .recv_timeout(Duration::from_secs(2))
        .expect("provider request should be sent");
    assert!(request.starts_with("POST /chat/completions HTTP/1.1"));
    assert!(request
        .to_ascii_lowercase()
        .contains("authorization: bearer test-key"));
    assert!(request.contains("gpt-mock"));
    assert!(request.contains("tracked.py"));
    assert!(request.contains("print('changed')"));

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("fix(core): use provider message"));
    let log = git(&repo, &["log", "--pretty=%s", "-1"]);
    assert_eq!(log, "fix(core): use provider message");
}

#[test]
fn github_token_flag_sets_github_models_api_key() {
    let repo = init_repo();
    let (endpoint, request_rx) = spawn_provider_response(
        r#"{"choices":[{"message":{"content":"fix(github): use cli token."}}]}"#,
    );
    fs::write(repo.join("tracked.py"), "print('seed')\n").expect("tracked seed");
    git(&repo, &["add", "tracked.py"]);
    git(&repo, &["commit", "-m", "chore(repo): seed"]);
    fs::write(repo.join("tracked.py"), "print('changed')\n").expect("tracked change");

    let output = kcmt_command(env!("CARGO_BIN_EXE_kcmt"))
        .args([
            "--file",
            "tracked.py",
            "--provider",
            "github",
            "--endpoint",
            &endpoint,
            "--api-key-env",
            "GITHUB_TOKEN",
            "--model",
            "openai/gpt-test",
            "--github-token",
            "cli-gh-token",
            "--no-auto-push",
            "--repo-path",
        ])
        .arg(&repo)
        .output()
        .expect("kcmt binary should run");

    assert!(
        output.status.success(),
        "stdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let request = request_rx
        .recv_timeout(Duration::from_secs(2))
        .expect("provider request should be recorded");
    assert!(request.contains("authorization: Bearer cli-gh-token"));
    let log = git(&repo, &["log", "--pretty=%s", "-1"]);
    assert_eq!(log, "fix(github): use cli token");
}

#[test]
fn max_retries_zero_attempts_provider_once() {
    let repo = init_repo();
    let (endpoint, request_rx) = spawn_provider_status_response(
        500,
        r#"{"error":{"message":"temporary provider failure"}}"#,
    );
    fs::write(repo.join("tracked.py"), "print('seed')\n").expect("tracked seed");
    git(&repo, &["add", "tracked.py"]);
    git(&repo, &["commit", "-m", "chore(repo): seed"]);
    fs::write(repo.join("tracked.py"), "print('changed')\n").expect("tracked change");

    let output = kcmt_command(env!("CARGO_BIN_EXE_kcmt"))
        .env("OPENAI_TEST_KEY", "test-key")
        .args([
            "--file",
            "tracked.py",
            "--provider",
            "openai",
            "--endpoint",
            &endpoint,
            "--api-key-env",
            "OPENAI_TEST_KEY",
            "--model",
            "gpt-mock",
            "--max-retries",
            "0",
            "--no-auto-push",
            "--repo-path",
        ])
        .arg(&repo)
        .output()
        .expect("kcmt binary should run");

    assert!(
        !output.status.success(),
        "stdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let request = request_rx
        .recv_timeout(Duration::from_secs(2))
        .expect("provider request should be recorded");
    assert!(request.starts_with("POST /chat/completions HTTP/1.1"));
    assert!(request_rx.recv_timeout(Duration::from_millis(200)).is_err());
    let log = git(&repo, &["log", "--pretty=%s", "-1"]);
    assert_eq!(log, "chore(repo): seed");
}

#[test]
fn file_mode_falls_back_to_next_configured_provider_after_primary_failure() {
    let repo = init_repo();
    let config_home = unique_temp_dir("config-home");
    let (endpoint, request_rx) = spawn_provider_fallback_response();
    fs::create_dir_all(&config_home).expect("config home");
    fs::write(
        config_home.join("config.json"),
        format!(
            r#"{{
  "provider": "openai",
  "model": "gpt-primary",
  "llm_endpoint": "{endpoint}",
  "api_key_env": "OPENAI_TEST_KEY",
  "git_repo_path": ".",
  "max_commit_length": 72,
  "auto_push": false,
  "providers": {{
    "openai": {{"endpoint": "{endpoint}", "api_key_env": "OPENAI_TEST_KEY", "preferred_model": "gpt-primary"}},
    "anthropic": {{"endpoint": "{endpoint}", "api_key_env": "ANTHROPIC_TEST_KEY", "preferred_model": "claude-fallback"}}
  }},
  "model_priority": [
    {{"provider": "openai", "model": "gpt-primary"}},
    {{"provider": "anthropic", "model": "claude-fallback"}}
  ]
}}"#
        ),
    )
    .expect("persist fallback config");
    fs::write(repo.join("tracked.py"), "print('seed')\n").expect("tracked seed");
    git(&repo, &["add", "tracked.py"]);
    git(&repo, &["commit", "-m", "chore(repo): seed"]);
    fs::write(repo.join("tracked.py"), "print('changed')\n").expect("tracked change");

    let output = kcmt_command(env!("CARGO_BIN_EXE_kcmt"))
        .env("KCMT_CONFIG_HOME", &config_home)
        .env("OPENAI_TEST_KEY", "primary-key")
        .env("ANTHROPIC_TEST_KEY", "fallback-key")
        .args(["--file", "tracked.py", "--repo-path"])
        .arg(&repo)
        .output()
        .expect("kcmt binary should run");

    assert!(
        output.status.success(),
        "stdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let requests: Vec<String> = (0..5)
        .map(|_| request_rx.recv_timeout(Duration::from_secs(2)).unwrap())
        .collect();
    assert!(requests[0].contains("POST /chat/completions HTTP/1.1"));
    assert!(requests[0].contains("authorization: Bearer primary-key"));
    assert!(requests[4].contains("POST /v1/messages HTTP/1.1"));
    assert!(requests[4].contains("x-api-key: fallback-key"));
    let log = git(&repo, &["log", "--pretty=%s", "-1"]);
    assert_eq!(log, "fix(fallback): use secondary provider");
}

#[test]
fn default_openai_batch_queues_all_file_prompts_before_committing() {
    let repo = init_repo();
    let (endpoint, request_rx) = spawn_openai_batch_response();
    fs::write(repo.join("alpha.py"), "print('alpha')\n").expect("alpha seed");
    fs::write(repo.join("beta.py"), "print('beta')\n").expect("beta seed");
    git(&repo, &["add", "alpha.py", "beta.py"]);
    git(&repo, &["commit", "-m", "chore(repo): seed"]);
    fs::write(repo.join("alpha.py"), "print('alpha changed')\n").expect("alpha change");
    fs::write(repo.join("beta.py"), "print('beta changed')\n").expect("beta change");

    let output = kcmt_command(env!("CARGO_BIN_EXE_kcmt"))
        .env("OPENAI_TEST_KEY", "test-key")
        .args([
            "--provider",
            "openai",
            "--endpoint",
            &endpoint,
            "--api-key-env",
            "OPENAI_TEST_KEY",
            "--model",
            "gpt-direct",
            "--batch",
            "--batch-model",
            "gpt-batch",
            "--batch-timeout",
            "900",
            "--no-auto-push",
            "--repo-path",
        ])
        .arg(&repo)
        .output()
        .expect("kcmt binary should run");

    assert!(
        output.status.success(),
        "stdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let requests: Vec<String> = (0..4)
        .map(|_| request_rx.recv_timeout(Duration::from_secs(2)).unwrap())
        .collect();
    assert!(requests[0].contains("POST /files HTTP/1.1"));
    assert!(requests[0].contains("alpha.py"));
    assert!(requests[0].contains("beta.py"));
    assert!(requests[0].contains("print('alpha changed')"));
    assert!(requests[0].contains("print('beta changed')"));
    assert!(!requests.iter().any(|request| {
        request.contains("POST /chat/completions HTTP/1.1")
            && !request.contains("/v1/chat/completions")
    }));
    assert!(requests[1].contains("POST /batches HTTP/1.1"));
    assert!(requests[2].contains("GET /batches/batch_1 HTTP/1.1"));
    assert!(requests[3].contains("GET /files/output_1/content HTTP/1.1"));

    let log = git(&repo, &["log", "--pretty=%s"]);
    let subjects: Vec<&str> = log.lines().collect();
    assert!(subjects.contains(&"fix(alpha): batch alpha"));
    assert!(subjects.contains(&"fix(beta): batch beta"));
    let status = git(&repo, &["status", "--short"]);
    assert!(status.is_empty(), "unexpected dirty status: {status}");
}

#[test]
fn file_mode_aborts_on_invalid_provider_response() {
    let repo = init_repo();
    fs::write(repo.join("tracked.py"), "print('seed')\n").expect("tracked seed");
    git(&repo, &["add", "tracked.py"]);
    git(&repo, &["commit", "-m", "chore(repo): seed"]);
    fs::write(repo.join("tracked.py"), "print('changed')\n").expect("tracked change");

    let output = kcmt_command(env!("CARGO_BIN_EXE_kcmt"))
        .env("KCMT_PROVIDER_RESPONSE", "This changes a few files")
        .env("KCMT_ALLOW_PROVIDER_RESPONSE_FIXTURE", "1")
        .args(["--file", "tracked.py", "--no-auto-push", "--repo-path"])
        .arg(&repo)
        .output()
        .expect("kcmt binary should run");

    assert_eq!(output.status.code(), Some(1));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("missing conventional commit header"));

    let status = git(&repo, &["status", "--short"]);
    assert!(status.contains("tracked.py"));
    let log = git(&repo, &["log", "--pretty=%s", "-1"]);
    assert_eq!(log, "chore(repo): seed");
}

#[test]
fn file_mode_without_provider_key_aborts_unless_local_synthesis_is_enabled() {
    let repo = init_repo();
    fs::write(repo.join("tracked.py"), "print('seed')\n").expect("tracked seed");
    git(&repo, &["add", "tracked.py"]);
    git(&repo, &["commit", "-m", "chore(repo): seed"]);
    fs::write(repo.join("tracked.py"), "print('changed')\n").expect("tracked change");

    let output = kcmt_command(env!("CARGO_BIN_EXE_kcmt"))
        .env_remove("KCMT_ALLOW_LOCAL_SYNTHESIS")
        .args(["--file", "tracked.py", "--no-auto-push", "--repo-path"])
        .arg(&repo)
        .output()
        .expect("kcmt binary should run");

    assert_eq!(output.status.code(), Some(1));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("No API key available"));
    let status = git(&repo, &["status", "--short"]);
    assert!(status.contains("tracked.py"));
    let log = git(&repo, &["log", "--pretty=%s", "-1"]);
    assert_eq!(log, "chore(repo): seed");
}

#[test]
fn provider_response_fixture_is_ignored_without_explicit_opt_in() {
    let repo = init_repo();
    fs::write(repo.join("tracked.py"), "print('seed')\n").expect("tracked seed");
    git(&repo, &["add", "tracked.py"]);
    git(&repo, &["commit", "-m", "chore(repo): seed"]);
    fs::write(repo.join("tracked.py"), "print('changed')\n").expect("tracked change");

    let output = kcmt_command(env!("CARGO_BIN_EXE_kcmt"))
        .env("KCMT_PROVIDER_RESPONSE", "fix(core): should be ignored")
        .env_remove("KCMT_ALLOW_LOCAL_SYNTHESIS")
        .args(["--file", "tracked.py", "--no-auto-push", "--repo-path"])
        .arg(&repo)
        .output()
        .expect("kcmt binary should run");

    assert_eq!(output.status.code(), Some(1));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("No API key available"));
    let log = git(&repo, &["log", "--pretty=%s", "-1"]);
    assert_eq!(log, "chore(repo): seed");
}

#[test]
fn file_mode_commits_nested_untracked_target_only() {
    let repo = init_repo();
    let nested = repo.join("src").join("module_000");
    fs::create_dir_all(&nested).expect("nested dir");
    fs::write(nested.join("file_0000.py"), "print('alpha')\n").expect("alpha file");
    fs::write(nested.join("file_0001.py"), "print('beta')\n").expect("beta file");

    let output = kcmt_command(env!("CARGO_BIN_EXE_kcmt"))
        .args([
            "--file",
            "src/module_000/file_0000.py",
            "--no-auto-push",
            "--repo-path",
        ])
        .arg(&repo)
        .output()
        .expect("kcmt binary should run");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("✓ src/module_000/file_0000.py"));

    let log = git(&repo, &["log", "--oneline", "-1"]);
    assert!(log.contains("chore(module_000): update file_0000"));

    let status = git(&repo, &["status", "--short", "--untracked-files=all"]);
    assert!(status.contains("?? src/module_000/file_0001.py"));
    assert!(!status.contains("file_0000.py"));
}

#[test]
fn oneshot_mode_commits_one_nested_untracked_file() {
    let repo = init_repo();
    let nested = repo.join("src").join("module_000");
    fs::create_dir_all(&nested).expect("nested dir");
    fs::write(nested.join("file_0000.py"), "print('alpha')\n").expect("alpha file");
    fs::write(nested.join("file_0001.py"), "print('beta')\n").expect("beta file");

    let output = kcmt_command(env!("CARGO_BIN_EXE_kcmt"))
        .args(["--oneshot", "--no-auto-push", "--repo-path"])
        .arg(&repo)
        .output()
        .expect("kcmt binary should run");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("✓ src/module_000/file_0000.py"));
    assert!(!stdout.contains("✓ src/module_000/file_0001.py"));

    let status = git(&repo, &["status", "--short", "--untracked-files=all"]);
    assert!(status.contains("?? src/module_000/file_0001.py"));
    assert!(!status.contains("file_0000.py"));
}

#[test]
fn oneshot_auto_push_pushes_successful_commits_to_origin() {
    let repo = init_repo();
    let remote = init_bare_remote();
    let config_home = unique_temp_dir("config-home");
    git(
        &repo,
        &[
            "remote",
            "add",
            "origin",
            remote.to_str().expect("remote path"),
        ],
    );

    fs::write(repo.join("tracked.py"), "print('seed')\n").expect("tracked seed");
    git(&repo, &["add", "tracked.py"]);
    git(&repo, &["commit", "-m", "chore(repo): seed"]);
    git(&repo, &["push", "origin", "HEAD"]);

    fs::write(repo.join("tracked.py"), "print('changed')\n").expect("tracked change");

    let output = kcmt_command(env!("CARGO_BIN_EXE_kcmt"))
        .env("KCMT_CONFIG_HOME", &config_home)
        .args(["--oneshot", "--auto-push", "--repo-path"])
        .arg(&repo)
        .output()
        .expect("kcmt binary should run");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Auto-push: pushed"));

    let local_head = git(&repo, &["rev-parse", "HEAD"]);
    let branch = git(&repo, &["rev-parse", "--abbrev-ref", "HEAD"]);
    let remote_ref = format!("refs/heads/{branch}");
    let remote_head = git(&remote, &["rev-parse", &remote_ref]);
    assert_eq!(remote_head, local_head);

    let status_output = kcmt_command(env!("CARGO_BIN_EXE_kcmt"))
        .env("KCMT_CONFIG_HOME", &config_home)
        .args(["status", "--raw", "--repo-path"])
        .arg(&repo)
        .output()
        .expect("kcmt status should run");
    assert!(status_output.status.success());
    let snapshot: serde_json::Value =
        serde_json::from_slice(&status_output.stdout).expect("raw status is json");
    assert_eq!(snapshot["pushed"], true);
    assert_eq!(
        snapshot["auto_push_state"],
        serde_json::Value::String("pushed".to_string())
    );
    assert_eq!(
        snapshot["errors"].as_array().expect("errors array").len(),
        0
    );
}

#[test]
fn oneshot_auto_push_failure_is_recorded_without_reverting_commit() {
    let repo = init_repo();
    let config_home = unique_temp_dir("config-home");
    git(
        &repo,
        &[
            "remote",
            "add",
            "origin",
            "/tmp/kcmt-definitely-missing-origin.git",
        ],
    );
    fs::write(repo.join("tracked.py"), "print('seed')\n").expect("tracked seed");
    git(&repo, &["add", "tracked.py"]);
    git(&repo, &["commit", "-m", "chore(repo): seed"]);

    fs::write(repo.join("tracked.py"), "print('changed')\n").expect("tracked change");

    let output = kcmt_command(env!("CARGO_BIN_EXE_kcmt"))
        .env("KCMT_CONFIG_HOME", &config_home)
        .args(["--oneshot", "--auto-push", "--repo-path"])
        .arg(&repo)
        .output()
        .expect("kcmt binary should run");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Auto-push: failed"));

    let log = git(&repo, &["log", "--pretty=%s", "-1"]);
    assert_eq!(log, "chore(repo): update tracked");

    let status_output = kcmt_command(env!("CARGO_BIN_EXE_kcmt"))
        .env("KCMT_CONFIG_HOME", &config_home)
        .args(["status", "--raw", "--repo-path"])
        .arg(&repo)
        .output()
        .expect("kcmt status should run");
    assert!(status_output.status.success());
    let snapshot: serde_json::Value =
        serde_json::from_slice(&status_output.stdout).expect("raw status is json");
    assert_eq!(snapshot["pushed"], false);
    assert_eq!(
        snapshot["auto_push_state"],
        serde_json::Value::String("failed".to_string())
    );
    let errors = snapshot["errors"].as_array().expect("errors array");
    assert_eq!(errors.len(), 1);
    assert!(errors[0]
        .as_str()
        .expect("error text")
        .contains("Auto-push failed:"));
}

#[test]
fn oneshot_auto_push_skips_repo_without_origin() {
    let repo = init_repo();
    let config_home = unique_temp_dir("config-home");
    fs::write(repo.join("tracked.py"), "print('seed')\n").expect("tracked seed");
    git(&repo, &["add", "tracked.py"]);
    git(&repo, &["commit", "-m", "chore(repo): seed"]);

    fs::write(repo.join("tracked.py"), "print('changed')\n").expect("tracked change");

    let output = kcmt_command(env!("CARGO_BIN_EXE_kcmt"))
        .env("KCMT_CONFIG_HOME", &config_home)
        .args(["--oneshot", "--auto-push", "--repo-path"])
        .arg(&repo)
        .output()
        .expect("kcmt binary should run");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(!stdout.contains("Auto-push: failed"));

    let status_output = kcmt_command(env!("CARGO_BIN_EXE_kcmt"))
        .env("KCMT_CONFIG_HOME", &config_home)
        .args(["status", "--raw", "--repo-path"])
        .arg(&repo)
        .output()
        .expect("kcmt status should run");
    assert!(status_output.status.success());
    let snapshot: serde_json::Value =
        serde_json::from_slice(&status_output.stdout).expect("raw status is json");
    assert_eq!(snapshot["pushed"], false);
    assert_eq!(
        snapshot["auto_push_state"],
        serde_json::Value::String("skipped".to_string())
    );
    assert_eq!(
        snapshot["errors"].as_array().expect("errors array").len(),
        0
    );
    let push_stage = snapshot["telemetry"]["stages"]
        .as_array()
        .expect("stage timings")
        .iter()
        .find(|stage| stage["stage"] == "push")
        .expect("push stage");
    assert_eq!(push_stage["items"], 0);
}
