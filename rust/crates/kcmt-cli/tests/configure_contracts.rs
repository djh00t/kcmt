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
    let path = std::env::temp_dir().join(format!("kcmt-configure-{label}-{nanos}-{suffix}"));
    fs::create_dir_all(&path).expect("temp dir should be created");
    path
}

fn kcmt_command() -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_kcmt"));
    command.env_clear();
    for key in ["PATH", "HOME", "USER", "TMPDIR", "LANG", "LC_ALL"] {
        if let Ok(value) = std::env::var(key) {
            command.env(key, value);
        }
    }
    command.env("KCMT_DISABLE_KEYCHAIN", "1");
    command
}

#[test]
fn configure_writes_python_compatible_config_file() {
    let config_home = unique_temp_dir("home");
    let repo = unique_temp_dir("repo");

    let output = kcmt_command()
        .env("KCMT_CONFIG_HOME", &config_home)
        .args([
            "--configure",
            "--provider",
            "anthropic",
            "--model",
            "claude-test",
            "--endpoint",
            "https://anthropic.test",
            "--api-key-env",
            "ANTHROPIC_TEST_KEY",
            "--batch-timeout",
            "900",
            "--no-auto-push",
            "--repo-path",
        ])
        .arg(&repo)
        .output()
        .expect("kcmt configure should run");

    assert!(
        output.status.success(),
        "stdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let config_path = config_home.join("config.json");
    let config: serde_json::Value =
        serde_json::from_slice(&fs::read(&config_path).expect("config file")).expect("config json");
    assert_eq!(config["provider"], "anthropic");
    assert_eq!(config["model"], "claude-test");
    assert_eq!(config["llm_endpoint"], "https://anthropic.test");
    assert_eq!(config["api_key_env"], "ANTHROPIC_TEST_KEY");
    assert_eq!(config["auto_push"], false);
    assert_eq!(
        config["providers"]["anthropic"]["api_key_env"],
        "ANTHROPIC_TEST_KEY"
    );
    assert_eq!(config["model_priority"][0]["provider"], "anthropic");
    assert_eq!(config["model_priority"][0]["model"], "claude-test");
}

#[test]
fn configure_all_accepts_noninteractive_overrides() {
    let config_home = unique_temp_dir("home");
    let repo = unique_temp_dir("repo");

    let output = kcmt_command()
        .env("KCMT_CONFIG_HOME", &config_home)
        .args([
            "--configure-all",
            "--provider",
            "openai",
            "--model",
            "gpt-test",
            "--api-key-env",
            "OPENAI_TEST_KEY",
            "--repo-path",
        ])
        .arg(&repo)
        .output()
        .expect("kcmt configure-all should run");

    assert!(
        output.status.success(),
        "stdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let config_path = config_home.join("config.json");
    let config: serde_json::Value =
        serde_json::from_slice(&fs::read(&config_path).expect("config file")).expect("config json");
    assert_eq!(config["provider"], "openai");
    assert_eq!(config["model"], "gpt-test");
    assert_eq!(
        config["providers"]["openai"]["api_key_env"],
        "OPENAI_TEST_KEY"
    );
}

#[test]
fn list_models_prints_supported_provider_defaults() {
    let output = kcmt_command()
        .args(["--list-models"])
        .output()
        .expect("kcmt list-models should run");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("openai"));
    assert!(stdout.contains("gpt-5-mini-2025-08-07"));
    assert!(stdout.contains("anthropic"));
    assert!(stdout.contains("claude-3-5-haiku-latest"));
    assert!(stdout.contains("xai"));
    assert!(stdout.contains("github"));
}

#[test]
fn configure_writes_default_preferences_file() {
    let config_home = unique_temp_dir("home");
    let repo = unique_temp_dir("repo");

    let output = kcmt_command()
        .env("KCMT_CONFIG_HOME", &config_home)
        .args(["--configure", "--repo-path"])
        .arg(&repo)
        .output()
        .expect("kcmt configure should run");

    assert!(
        output.status.success(),
        "stdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let preferences_path = config_home.join("preferences.json");
    let preferences: serde_json::Value =
        serde_json::from_slice(&fs::read(&preferences_path).expect("preferences file"))
            .expect("preferences json");
    assert_eq!(preferences["selection_policy"], "fastest_cheap");
    assert_eq!(preferences["default_prompt_profile"], "conventional");
    assert_eq!(preferences["prompt_profiles"][0]["id"], "conventional");
}

#[test]
fn list_models_debug_prints_structured_provider_json() {
    let output = kcmt_command()
        .args(["--debug", "--list-models"])
        .output()
        .expect("kcmt list-models debug should run");

    assert!(output.status.success());
    let payload: serde_json::Value =
        serde_json::from_slice(&output.stdout).expect("debug model list is json");
    let providers = payload.as_array().expect("provider array");
    assert_eq!(providers.len(), 4);
    assert!(providers.iter().any(|provider| {
        provider["provider"] == "openai" && provider["models"][0]["id"] == "gpt-5-mini-2025-08-07"
    }));
    assert!(providers.iter().any(|provider| {
        provider["provider"] == "github" && provider["models"][0]["api_key_env"] == "GITHUB_TOKEN"
    }));
}

#[test]
fn verify_keys_prints_provider_env_presence() {
    let output = kcmt_command()
        .env("OPENAI_API_KEY", "test-key")
        .args(["--verify-keys"])
        .output()
        .expect("kcmt verify-keys should run");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("API Key Verification"));
    assert!(stdout.contains("openai\tOPENAI_API_KEY\tyes\tOPENAI_API_KEY"));
    assert!(stdout.contains("anthropic\tANTHROPIC_API_KEY\tno\t-"));
    assert!(stdout.contains("xai\tXAI_API_KEY\tno\t-"));
    assert!(stdout.contains("github\tGITHUB_TOKEN\tno\t-"));
}
