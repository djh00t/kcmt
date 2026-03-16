//! Configuration loader honoring precedence:
//! CLI overrides > environment > persisted file > defaults.

use std::collections::HashMap;
use std::env;
use std::path::{Component, Path, PathBuf};

use crate::config::persisted::load_persisted_config;
use crate::config::workflow_config::validate_config;
use crate::error::Result;
use crate::model::WorkflowConfig;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConfigSource {
    Cli,
    Environment,
    Persisted,
    Default,
}

#[derive(Debug, Clone)]
pub struct ConfigLayer {
    pub source: ConfigSource,
    pub value: WorkflowConfig,
}

#[derive(Debug, Clone)]
pub struct ConfigLoader {
    pub layers: Vec<ConfigLayer>,
}

#[derive(Debug, Clone, Default)]
pub struct ConfigOverrides {
    pub provider: Option<String>,
    pub model: Option<String>,
    pub endpoint: Option<String>,
    pub api_key_env: Option<String>,
    pub repo_path: Option<PathBuf>,
    pub max_commit_length: Option<usize>,
}

impl ConfigLoader {
    pub fn new() -> Self {
        Self { layers: Vec::new() }
    }

    pub fn with_layer(mut self, source: ConfigSource, value: WorkflowConfig) -> Self {
        self.layers.push(ConfigLayer { source, value });
        self
    }

    pub fn resolve(&self) -> WorkflowConfig {
        self.layers
            .iter()
            .last()
            .map(|layer| layer.value.clone())
            .unwrap_or_default()
    }
}

#[derive(Debug, Clone, Copy)]
struct ProviderDefaults {
    provider: &'static str,
    model: &'static str,
    endpoint: &'static str,
    api_key_env: &'static str,
}

const PROVIDER_DEFAULTS: &[ProviderDefaults] = &[
    ProviderDefaults {
        provider: "openai",
        model: "gpt-5-mini-2025-08-07",
        endpoint: "https://api.openai.com/v1",
        api_key_env: "OPENAI_API_KEY",
    },
    ProviderDefaults {
        provider: "anthropic",
        model: "claude-3-5-haiku-latest",
        endpoint: "https://api.anthropic.com/v1",
        api_key_env: "ANTHROPIC_API_KEY",
    },
    ProviderDefaults {
        provider: "xai",
        model: "grok-code-fast",
        endpoint: "https://api.x.ai/v1",
        api_key_env: "XAI_API_KEY",
    },
    ProviderDefaults {
        provider: "github",
        model: "openai/gpt-4.1-mini",
        endpoint: "https://models.github.ai/inference",
        api_key_env: "GITHUB_TOKEN",
    },
];

const PROVIDER_HINTS: &[(&str, &[&str])] = &[
    ("openai", &["OPENAI", "OPENAI_API", "OA_KEY"]),
    ("anthropic", &["ANTHROPIC", "CLAUDE"]),
    ("xai", &["XAI", "GROK"]),
    ("github", &["GITHUB_TOKEN", "GH_TOKEN", "GH_MODELS"]),
];

pub fn load_config(repo_root: &Path, overrides: &ConfigOverrides) -> Result<WorkflowConfig> {
    let persisted = persisted_config_for(repo_root);
    let detected = detect_available_providers();

    let provider = selected_provider(overrides, persisted.as_ref(), &detected);
    let defaults = provider_defaults(&provider).unwrap_or(PROVIDER_DEFAULTS[0]);

    let model = overrides
        .model
        .clone()
        .or_else(|| env::var("KLINGON_CMT_LLM_MODEL").ok())
        .or_else(|| persisted.as_ref().map(|cfg| cfg.model.clone()))
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| defaults.model.to_string());
    let llm_endpoint = overrides
        .endpoint
        .clone()
        .or_else(|| env::var("KLINGON_CMT_LLM_ENDPOINT").ok())
        .or_else(|| persisted.as_ref().map(|cfg| cfg.llm_endpoint.clone()))
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| defaults.endpoint.to_string());
    let api_key_env = overrides
        .api_key_env
        .clone()
        .or_else(|| persisted.as_ref().map(|cfg| cfg.api_key_env.clone()))
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| selected_api_key_env(&provider, &detected).to_string());
    let git_repo_path = overrides
        .repo_path
        .clone()
        .or_else(|| env::var("KLINGON_CMT_GIT_REPO_PATH").ok().map(PathBuf::from))
        .or_else(|| persisted.as_ref().map(|cfg| PathBuf::from(&cfg.git_repo_path)))
        .unwrap_or_else(|| repo_root.to_path_buf());
    let git_repo_path = normalize_repo_path(repo_root, &git_repo_path);
    let max_commit_length = overrides
        .max_commit_length
        .or_else(|| {
            env::var("KLINGON_CMT_MAX_COMMIT_LENGTH")
                .ok()
                .and_then(|value| value.parse::<usize>().ok())
        })
        .or_else(|| persisted.as_ref().map(|cfg| cfg.max_commit_length))
        .unwrap_or(72);
    let auto_push = persisted.as_ref().map(|cfg| cfg.auto_push).unwrap_or(true);

    let config = WorkflowConfig {
        provider,
        model,
        llm_endpoint,
        api_key_env,
        git_repo_path: git_repo_path.to_string_lossy().to_string(),
        max_commit_length,
        auto_push,
    };
    validate_config(&config)?;
    Ok(config)
}

fn persisted_config_for(repo_root: &Path) -> Option<WorkflowConfig> {
    let path = config_file_path(repo_root);
    load_persisted_config(repo_root, &path).ok()
}

pub fn config_home() -> PathBuf {
    if let Ok(home) = env::var("KCMT_CONFIG_HOME") {
        if !home.trim().is_empty() {
            return PathBuf::from(home);
        }
    }

    if let Ok(home) = env::var("XDG_CONFIG_HOME") {
        if !home.trim().is_empty() {
            return PathBuf::from(home).join("kcmt");
        }
    }

    if let Ok(home) = env::var("HOME") {
        if !home.trim().is_empty() {
            return PathBuf::from(home).join(".config").join("kcmt");
        }
    }

    PathBuf::from(".kcmt")
}

pub fn config_file_path(_repo_root: &Path) -> PathBuf {
    config_home().join("config.json")
}

fn normalize_repo_path(repo_root: &Path, candidate: &Path) -> PathBuf {
    let joined = if candidate.is_absolute() {
        candidate.to_path_buf()
    } else {
        repo_root.join(candidate)
    };

    let mut normalized = PathBuf::new();
    for component in joined.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                normalized.pop();
            }
            other => normalized.push(other.as_os_str()),
        }
    }
    normalized
}

fn provider_defaults(provider: &str) -> Option<ProviderDefaults> {
    PROVIDER_DEFAULTS
        .iter()
        .copied()
        .find(|defaults| defaults.provider == provider)
}

fn detect_available_providers() -> HashMap<String, Vec<String>> {
    let mut detected: HashMap<String, Vec<String>> = HashMap::new();
    let env_values: Vec<String> = env::vars().map(|(key, _)| key).collect();

    for defaults in PROVIDER_DEFAULTS {
        let mut matches = Vec::new();
        if env_values.iter().any(|value| value == defaults.api_key_env) {
            matches.push(defaults.api_key_env.to_string());
        }
        if let Some((_, hints)) = PROVIDER_HINTS.iter().find(|(provider, _)| *provider == defaults.provider) {
            for key in &env_values {
                if hints.iter().any(|hint| key.contains(hint)) && !matches.contains(key) {
                    matches.push(key.clone());
                }
            }
        }
        detected.insert(defaults.provider.to_string(), matches);
    }

    detected
}

fn selected_provider(
    overrides: &ConfigOverrides,
    persisted: Option<&WorkflowConfig>,
    detected: &HashMap<String, Vec<String>>,
) -> String {
    let candidate = overrides
        .provider
        .clone()
        .or_else(|| env::var("KCMT_PROVIDER").ok())
        .or_else(|| persisted.map(|cfg| cfg.provider.clone()))
        .unwrap_or_else(|| auto_select_provider(detected));

    if provider_defaults(&candidate).is_some() {
        candidate
    } else {
        "openai".to_string()
    }
}

fn selected_api_key_env<'a>(
    provider: &str,
    detected: &'a HashMap<String, Vec<String>>,
) -> &'a str {
    let defaults = provider_defaults(provider).unwrap_or(PROVIDER_DEFAULTS[0]);
    detected
        .get(provider)
        .and_then(|values| {
            if values.iter().any(|value| value == defaults.api_key_env) {
                Some(defaults.api_key_env)
            } else {
                values.first().map(String::as_str)
            }
        })
        .unwrap_or(defaults.api_key_env)
}

fn auto_select_provider(detected: &HashMap<String, Vec<String>>) -> String {
    for provider in ["openai", "anthropic", "xai", "github"] {
        if detected
            .get(provider)
            .is_some_and(|values| !values.is_empty())
        {
            return provider.to_string();
        }
    }

    "openai".to_string()
}

#[cfg(test)]
mod tests {
    use super::{config_file_path, config_home, load_config, ConfigOverrides};
    use std::fs;
    use std::path::PathBuf;
    use std::sync::{Mutex, OnceLock};
    use std::time::{SystemTime, UNIX_EPOCH};

    fn env_lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

    fn clear_test_env() {
        for key in [
            "KCMT_CONFIG_HOME",
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "ANTHROPIC_KEY_ALT",
            "XAI_GROK_TOKEN",
            "KCMT_PROVIDER",
            "KLINGON_CMT_LLM_MODEL",
            "KLINGON_CMT_LLM_ENDPOINT",
            "KLINGON_CMT_GIT_REPO_PATH",
            "KLINGON_CMT_MAX_COMMIT_LENGTH",
        ] {
            std::env::remove_var(key);
        }
    }

    fn unique_temp_dir(label: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock should be after unix epoch")
            .as_nanos();
        let path = std::env::temp_dir().join(format!("kcmt-loader-{label}-{nanos}"));
        fs::create_dir_all(&path).expect("temp dir should be created");
        path
    }

    #[test]
    fn load_config_prefers_openai_env() {
        let _guard = env_lock().lock().unwrap_or_else(|err| err.into_inner());
        clear_test_env();
        let repo = unique_temp_dir("openai");
        let cfg_home = unique_temp_dir("cfg-home");
        std::env::set_var("KCMT_CONFIG_HOME", &cfg_home);
        std::env::set_var("OPENAI_API_KEY", "sk-test");
        std::env::set_var("KLINGON_CMT_LLM_MODEL", "gpt-5-x");
        std::env::set_var("KLINGON_CMT_LLM_ENDPOINT", "https://api.openai.test");

        let cfg = load_config(&repo, &ConfigOverrides::default()).expect("config");

        assert_eq!(cfg.provider, "openai");
        assert_eq!(cfg.model, "gpt-5-x");
        assert_eq!(cfg.llm_endpoint, "https://api.openai.test");
        assert_eq!(cfg.api_key_env, "OPENAI_API_KEY");
        assert_eq!(config_file_path(&repo), config_home().join("config.json"));
    }

    #[test]
    fn load_config_upgrades_relative_repo_path_from_persisted_file() {
        let _guard = env_lock().lock().unwrap_or_else(|err| err.into_inner());
        clear_test_env();
        let repo = unique_temp_dir("persisted");
        let cfg_home = unique_temp_dir("cfg-home-persisted");
        std::env::set_var("KCMT_CONFIG_HOME", &cfg_home);
        std::env::remove_var("KCMT_PROVIDER");
        std::env::remove_var("KLINGON_CMT_LLM_MODEL");
        std::env::remove_var("KLINGON_CMT_LLM_ENDPOINT");
        std::env::remove_var("OPENAI_API_KEY");

        let config_path = config_home().join("config.json");
        fs::create_dir_all(config_path.parent().expect("config parent")).expect("config dir");
        fs::write(
            &config_path,
            r#"{
  "provider": "openai",
  "model": "gpt-5-mini",
  "llm_endpoint": "https://api.openai.com/v1",
  "api_key_env": "OPENAI_API_KEY",
  "git_repo_path": ".",
  "max_commit_length": 72
}"#,
        )
        .expect("persisted config");

        let cfg = load_config(&repo, &ConfigOverrides::default()).expect("config");
        assert_eq!(cfg.git_repo_path, repo.to_string_lossy());
        assert_eq!(cfg.model, "gpt-5-mini");
    }

    #[test]
    fn overrides_take_precedence() {
        let _guard = env_lock().lock().unwrap_or_else(|err| err.into_inner());
        clear_test_env();
        let repo = unique_temp_dir("override");
        let overrides = ConfigOverrides {
            provider: Some("xai".to_string()),
            model: Some("grok-code-fast".to_string()),
            endpoint: Some("https://custom.x.ai".to_string()),
            api_key_env: Some("XAI_SECRET".to_string()),
            repo_path: Some(repo.join("nested")),
            max_commit_length: Some(80),
        };

        let cfg = load_config(&repo, &overrides).expect("config");

        assert_eq!(cfg.provider, "xai");
        assert_eq!(cfg.model, "grok-code-fast");
        assert_eq!(cfg.llm_endpoint, "https://custom.x.ai");
        assert_eq!(cfg.api_key_env, "XAI_SECRET");
        assert_eq!(cfg.max_commit_length, 80);
        assert!(cfg.git_repo_path.ends_with("nested"));
    }
}
