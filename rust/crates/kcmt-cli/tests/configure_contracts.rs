use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};
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
    assert_eq!(
        config["providers"]["anthropic"]["keychain_account"],
        "provider/anthropic/default"
    );
    assert_eq!(config["model_priority"][0]["provider"], "anthropic");
    assert_eq!(config["model_priority"][0]["model"], "claude-test");
    let rendered = serde_json::to_string(&config).expect("config serializes");
    assert!(!rendered.contains("secret"));
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
    assert_eq!(
        config["providers"]["openai"]["keychain_account"],
        "provider/openai/default"
    );
    assert!(config["providers"]["openai"].get("api_key").is_none());
}

#[test]
fn configure_all_updates_target_provider_without_clobbering_primary() {
    let config_home = unique_temp_dir("home");
    let repo = unique_temp_dir("repo");
    let config_path = config_home.join("config.json");
    fs::write(
        &config_path,
        serde_json::json!({
            "provider": "openai",
            "model": "gpt-existing",
            "llm_endpoint": "https://openai.existing/v1",
            "api_key_env": "OPENAI_EXISTING_KEY",
            "git_repo_path": repo.to_string_lossy(),
            "max_commit_length": 72,
            "auto_push": false,
            "use_batch": false,
            "batch_model": "gpt-existing",
            "batch_timeout_seconds": 900,
            "providers": {
                "openai": {
                    "name": "OpenAI",
                    "endpoint": "https://openai.existing/v1",
                    "api_key_env": "OPENAI_EXISTING_KEY",
                    "keychain_account": "provider/openai/existing",
                    "preferred_model": "gpt-existing"
                },
                "anthropic": {
                    "name": "Anthropic",
                    "endpoint": "https://anthropic.existing",
                    "api_key_env": "ANTHROPIC_EXISTING_KEY",
                    "preferred_model": "claude-existing"
                }
            },
            "model_priority": [
                {"provider": "openai", "model": "gpt-existing"},
                {"provider": "anthropic", "model": "claude-existing"}
            ]
        })
        .to_string(),
    )
    .expect("seed config");

    let output = kcmt_command()
        .env("KCMT_CONFIG_HOME", &config_home)
        .args([
            "--configure-all",
            "--provider",
            "anthropic",
            "--model",
            "claude-new",
            "--endpoint",
            "https://anthropic.new",
            "--api-key-env",
            "ANTHROPIC_NEW_KEY",
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
    let config: serde_json::Value =
        serde_json::from_slice(&fs::read(&config_path).expect("config file")).expect("config json");
    assert_eq!(config["provider"], "openai");
    assert_eq!(config["model"], "gpt-existing");
    assert_eq!(config["llm_endpoint"], "https://openai.existing/v1");
    assert_eq!(config["api_key_env"], "OPENAI_EXISTING_KEY");
    assert_eq!(config["model_priority"][0]["provider"], "openai");
    assert_eq!(config["model_priority"][1]["provider"], "anthropic");
    assert_eq!(
        config["providers"]["openai"]["keychain_account"],
        "provider/openai/existing"
    );
    assert_eq!(
        config["providers"]["anthropic"]["endpoint"],
        "https://anthropic.new"
    );
    assert_eq!(
        config["providers"]["anthropic"]["api_key_env"],
        "ANTHROPIC_NEW_KEY"
    );
    assert_eq!(
        config["providers"]["anthropic"]["preferred_model"],
        "claude-new"
    );
    assert_eq!(
        config["providers"]["anthropic"]["keychain_account"],
        "provider/anthropic/default"
    );
}

#[test]
fn list_models_prints_supported_provider_defaults() {
    let config_home = unique_temp_dir("home");
    let output = kcmt_command()
        .env("KCMT_CONFIG_HOME", &config_home)
        .args(["--list-models"])
        .output()
        .expect("kcmt list-models should run");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("openai"));
    assert!(stdout.contains("gpt-5.4-mini"));
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
    assert!(preferences["provider_rules"]["openai"].is_object());
    assert!(preferences["provider_rules"]["anthropic"].is_object());
    assert!(preferences["provider_rules"]["xai"].is_object());
    assert!(preferences["provider_rules"]["github"].is_object());
}

#[test]
fn configure_preserves_existing_preferences_while_initializing_provider_rules() {
    let config_home = unique_temp_dir("home");
    let repo = unique_temp_dir("repo");
    fs::write(
        config_home.join("preferences.json"),
        serde_json::json!({
            "schema_version": 1,
            "selection_policy": "best_quality",
            "provider_rules": {
                "openai": {
                    "preset": "pin_exact_model",
                    "value": "gpt-existing",
                    "strict": true
                }
            },
            "prompt_profiles": [
                {
                    "id": "custom",
                    "name": "Custom",
                    "system_instruction": "Use the custom profile.",
                    "user_instruction": "Only output a commit."
                }
            ],
            "default_prompt_profile": "custom",
            "tui": {"last_screen": "providers"},
            "model_cache": {"ttl_seconds": 123},
            "skip_commitizen_install_prompt": true
        })
        .to_string(),
    )
    .expect("seed preferences");

    let output = kcmt_command()
        .env("KCMT_CONFIG_HOME", &config_home)
        .args([
            "--configure",
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
        .expect("kcmt configure should run");

    assert!(
        output.status.success(),
        "stdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let preferences: serde_json::Value =
        serde_json::from_slice(&fs::read(config_home.join("preferences.json")).expect("prefs"))
            .expect("preferences json");
    assert_eq!(preferences["selection_policy"], "best_quality");
    assert_eq!(preferences["default_prompt_profile"], "custom");
    assert_eq!(preferences["tui"]["last_screen"], "providers");
    assert_eq!(preferences["model_cache"]["ttl_seconds"], 123);
    assert_eq!(preferences["skip_commitizen_install_prompt"], true);
    assert_eq!(
        preferences["provider_rules"]["openai"]["preset"],
        "pin_exact_model"
    );
    assert_eq!(preferences["provider_rules"]["openai"]["strict"], true);
    assert!(preferences["provider_rules"]["anthropic"].is_object());
    assert!(preferences["provider_rules"]["xai"].is_object());
    assert!(preferences["provider_rules"]["github"].is_object());
}

#[test]
fn configure_rejects_invalid_credential_combination() {
    let config_home = unique_temp_dir("home");
    let repo = unique_temp_dir("repo");

    let output = kcmt_command()
        .env("KCMT_CONFIG_HOME", &config_home)
        .args([
            "--configure",
            "--provider",
            "anthropic",
            "--endpoint",
            "ANTHROPIC_API_KEY",
            "--api-key-env",
            "https://anthropic.test",
            "--repo-path",
        ])
        .arg(&repo)
        .output()
        .expect("kcmt configure should reject invalid inputs");

    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("endpoint must be an http(s) URL"));
    assert!(!config_home.join("config.json").exists());
}

#[test]
fn list_models_debug_prints_structured_provider_json() {
    let config_home = unique_temp_dir("home");
    let output = kcmt_command()
        .env("KCMT_CONFIG_HOME", &config_home)
        .env("OPENAI_API_KEY", "sk-test-secret")
        .args(["--debug", "--list-models"])
        .output()
        .expect("kcmt list-models debug should run");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(!stdout.contains("sk-test-secret"));
    let payload: serde_json::Value =
        serde_json::from_slice(&output.stdout).expect("debug model list is json");
    let providers = payload.as_array().expect("provider array");
    assert_eq!(providers.len(), 4);
    let openai = providers
        .iter()
        .find(|provider| provider["provider"] == "openai")
        .expect("openai provider");
    assert!(matches!(
        openai["source"].as_str(),
        Some("live" | "static_fallback" | "cache")
    ));
    assert_eq!(openai["display_name"], "OpenAI");
    assert_eq!(openai["endpoint"], "https://api.openai.com/v1");
    assert_eq!(openai["api_key_env"], "OPENAI_API_KEY");
    assert!(openai["error"].is_null() || openai["error"].is_string());
    let openai_model = &openai["models"][0];
    assert!(openai_model["id"].as_str().is_some());
    assert_eq!(openai_model["provider"], "openai");
    assert_eq!(openai_model["endpoint"], "https://api.openai.com/v1");
    assert_eq!(openai_model["api_key_env"], "OPENAI_API_KEY");
    assert!(openai_model["created"].is_null() || openai_model["created"].is_string());
    assert!(openai_model["family"].is_null() || openai_model["family"].is_string());
    assert!(openai_model["code_capable"].is_boolean());

    let anthropic = providers
        .iter()
        .find(|provider| provider["provider"] == "anthropic")
        .expect("anthropic provider");
    assert_eq!(anthropic["source"], "static_fallback");
    assert_eq!(anthropic["models"][0]["family"], "haiku");

    let github = providers
        .iter()
        .find(|provider| provider["provider"] == "github")
        .expect("github provider");
    assert_eq!(github["models"][0]["api_key_env"], "GITHUB_TOKEN");
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

#[test]
fn save_api_key_respects_disabled_keychain_without_printing_secret() {
    let config_home = unique_temp_dir("home");
    let repo = unique_temp_dir("repo");
    let secret = "linux-unavailable-secret";

    let mut child = kcmt_command()
        .env("KCMT_CONFIG_HOME", &config_home)
        .args([
            "--configure",
            "--provider",
            "anthropic",
            "--api-key-stdin",
            "--save-api-key",
            "--repo-path",
        ])
        .arg(&repo)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("kcmt configure should spawn");
    child
        .stdin
        .as_mut()
        .expect("stdin")
        .write_all(secret.as_bytes())
        .expect("secret written");
    let output = child
        .wait_with_output()
        .expect("kcmt configure should exit");

    assert!(!output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("OS keychain access is disabled by KCMT_DISABLE_KEYCHAIN"));
    assert!(!stdout.contains(secret));
    assert!(!stderr.contains(secret));
}
