use std::fs;
use std::path::{Path, PathBuf};
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

fn init_repo() -> PathBuf {
    let repo = unique_temp_dir("repo");
    git(&repo, &["init", "-q"]);
    git(&repo, &["config", "user.name", "Tester"]);
    git(&repo, &["config", "user.email", "tester@example.com"]);
    repo
}

#[test]
fn file_mode_commits_only_requested_path() {
    let repo = init_repo();
    fs::write(repo.join("example.py"), "print('hello')\n").expect("seed file");
    git(&repo, &["add", "example.py"]);
    git(&repo, &["commit", "-m", "chore(repo): seed"]);

    fs::write(repo.join("example.py"), "print('hello world')\n").expect("modified file");
    fs::write(repo.join("other.py"), "print('other')\n").expect("other file");

    let output = Command::new(env!("CARGO_BIN_EXE_kcmt"))
        .args(["--file", "example.py", "--repo-path"])
        .arg(&repo)
        .output()
        .expect("kcmt binary should run");

    assert!(output.status.success());
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
fn oneshot_mode_commits_first_changed_non_deletion() {
    let repo = init_repo();
    fs::write(repo.join("alpha.py"), "print('alpha')\n").expect("alpha seed");
    fs::write(repo.join("beta.py"), "print('beta')\n").expect("beta seed");
    git(&repo, &["add", "alpha.py", "beta.py"]);
    git(&repo, &["commit", "-m", "chore(repo): seed"]);

    fs::write(repo.join("alpha.py"), "print('alpha updated')\n").expect("alpha change");
    fs::write(repo.join("beta.py"), "print('beta updated')\n").expect("beta change");

    let output = Command::new(env!("CARGO_BIN_EXE_commit"))
        .args(["--oneshot", "--repo-path"])
        .arg(&repo)
        .output()
        .expect("commit binary should run");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("✓ alpha.py"));
    let log = git(&repo, &["log", "--oneline", "-1"]);
    assert!(log.contains("chore(repo): update alpha"));

    let status = git(&repo, &["status", "--short"]);
    assert!(status.lines().any(|line| line.ends_with("beta.py")));
}

#[test]
fn file_mode_persists_snapshot_for_status_view() {
    let repo = init_repo();
    let config_home = unique_temp_dir("config-home");
    fs::write(repo.join("tracked.py"), "print('seed')\n").expect("tracked seed");
    git(&repo, &["add", "tracked.py"]);
    git(&repo, &["commit", "-m", "chore(repo): seed"]);
    fs::write(repo.join("tracked.py"), "print('changed')\n").expect("tracked change");

    let output = Command::new(env!("CARGO_BIN_EXE_kcmt"))
        .env("KCMT_CONFIG_HOME", &config_home)
        .args(["--file", "tracked.py", "--repo-path"])
        .arg(&repo)
        .output()
        .expect("kcmt binary should run");
    assert!(output.status.success());

    let status_output = Command::new(env!("CARGO_BIN_EXE_kcmt"))
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
fn file_mode_commits_nested_untracked_target_only() {
    let repo = init_repo();
    let nested = repo.join("src").join("module_000");
    fs::create_dir_all(&nested).expect("nested dir");
    fs::write(nested.join("file_0000.py"), "print('alpha')\n").expect("alpha file");
    fs::write(nested.join("file_0001.py"), "print('beta')\n").expect("beta file");

    let output = Command::new(env!("CARGO_BIN_EXE_kcmt"))
        .args(["--file", "src/module_000/file_0000.py", "--repo-path"])
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
fn oneshot_mode_selects_first_nested_untracked_file() {
    let repo = init_repo();
    let nested = repo.join("src").join("module_000");
    fs::create_dir_all(&nested).expect("nested dir");
    fs::write(nested.join("file_0000.py"), "print('alpha')\n").expect("alpha file");
    fs::write(nested.join("file_0001.py"), "print('beta')\n").expect("beta file");

    let output = Command::new(env!("CARGO_BIN_EXE_kcmt"))
        .args(["--oneshot", "--repo-path"])
        .arg(&repo)
        .output()
        .expect("kcmt binary should run");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("✓ src/module_000/file_0000.py"));

    let status = git(&repo, &["status", "--short", "--untracked-files=all"]);
    assert!(status.contains("?? src/module_000/file_0001.py"));
    assert!(!status.contains("file_0000.py"));
}
