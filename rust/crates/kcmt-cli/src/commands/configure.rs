use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

use anyhow::{bail, Result};
use kcmt_core::config::loader::{config_file_path, load_config, ConfigOverrides};
use kcmt_core::model::{ModelPreference, ProviderConfigEntry};
use kcmt_core::preferences::{
    default_keychain_account, load_preferences, preferences_path, Preferences, ProviderRule,
};
use serde_json::json;

use super::model_discovery::{
    default_provider_definitions, discover_model_catalog, ProviderDefinition,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConfigureMode {
    Single,
    All,
}

pub fn run_configure(repo_path: PathBuf, overrides: ConfigOverrides, mode: ConfigureMode) -> i32 {
    match write_config(repo_path, overrides, mode) {
        Ok(path) => {
            println!("Configuration saved to {}", path.display());
            0
        }
        Err(err) => {
            eprintln!("{err}");
            1
        }
    }
}

pub fn run_list_models(debug: bool) -> i32 {
    let preferences = load_preferences().unwrap_or_default();
    let catalog = discover_model_catalog(&preferences);
    if debug {
        match serde_json::to_string_pretty(&catalog) {
            Ok(rendered) => println!("{rendered}"),
            Err(err) => {
                eprintln!("{err}");
                return 1;
            }
        }
        return 0;
    }

    for provider in catalog {
        for model in provider.models {
            println!("{}\t{}\t{}", provider.provider, model.id, provider.endpoint);
        }
    }
    0
}

pub fn run_verify_keys() -> i32 {
    println!("API Key Verification");
    println!("provider\tenv_var\tpresent\tdetected");
    for definition in default_provider_definitions() {
        let present = std::env::var(&definition.api_key_env)
            .ok()
            .map(|value| !value.trim().is_empty())
            .unwrap_or(false);
        let detected = if present {
            definition.api_key_env.as_str()
        } else {
            "-"
        };
        println!(
            "{}\t{}\t{}\t{detected}",
            definition.provider,
            definition.api_key_env,
            if present { "yes" } else { "no" }
        );
    }
    0
}

fn write_config(
    repo_path: PathBuf,
    overrides: ConfigOverrides,
    mode: ConfigureMode,
) -> Result<PathBuf> {
    let definitions = default_provider_definitions();
    validate_overrides(&overrides, &definitions)?;
    let path = config_file_path(&repo_path);
    let persisted_config_exists = path.exists();
    let target_provider = overrides.provider.clone();
    let mut resolved_overrides = overrides.clone();
    if mode == ConfigureMode::All && persisted_config_exists {
        resolved_overrides.provider = None;
        resolved_overrides.model = None;
        resolved_overrides.endpoint = None;
        resolved_overrides.api_key_env = None;
    }

    let cfg = load_config(&repo_path, &resolved_overrides)?;
    let target_provider = target_provider.unwrap_or_else(|| cfg.provider.clone());
    let Some(target_definition) = provider_definition(&definitions, &target_provider) else {
        bail!("unsupported provider '{target_provider}' for configure");
    };

    let mut providers: HashMap<String, ProviderConfigEntry> = cfg.providers.clone();
    for definition in &definitions {
        providers
            .entry(definition.provider.clone())
            .and_modify(|entry| {
                entry
                    .name
                    .get_or_insert_with(|| definition.display_name.clone());
                entry
                    .endpoint
                    .get_or_insert_with(|| definition.endpoint.clone());
                entry
                    .api_key_env
                    .get_or_insert_with(|| definition.api_key_env.clone());
                entry
                    .keychain_account
                    .get_or_insert_with(|| default_keychain_account(&definition.provider));
                entry
                    .preferred_model
                    .get_or_insert_with(|| definition.default_model.clone());
            })
            .or_insert_with(|| ProviderConfigEntry {
                name: Some(definition.display_name.clone()),
                endpoint: Some(definition.endpoint.clone()),
                api_key_env: Some(definition.api_key_env.clone()),
                keychain_account: Some(default_keychain_account(&definition.provider)),
                preferred_model: Some(definition.default_model.clone()),
            });
    }

    if mode == ConfigureMode::All {
        providers
            .entry(target_provider.clone())
            .and_modify(|entry| {
                entry
                    .name
                    .get_or_insert_with(|| target_definition.display_name.clone());
                if let Some(endpoint) = overrides.endpoint.clone() {
                    entry.endpoint = Some(endpoint);
                } else {
                    entry
                        .endpoint
                        .get_or_insert_with(|| target_definition.endpoint.clone());
                }
                if let Some(api_key_env) = overrides.api_key_env.clone() {
                    entry.api_key_env = Some(api_key_env);
                } else {
                    entry
                        .api_key_env
                        .get_or_insert_with(|| target_definition.api_key_env.clone());
                }
                entry
                    .keychain_account
                    .get_or_insert_with(|| default_keychain_account(&target_provider));
                if let Some(model) = overrides.model.clone() {
                    entry.preferred_model = Some(model);
                } else {
                    entry
                        .preferred_model
                        .get_or_insert_with(|| target_definition.default_model.clone());
                }
            })
            .or_insert_with(|| ProviderConfigEntry {
                name: Some(target_definition.display_name.clone()),
                endpoint: Some(
                    overrides
                        .endpoint
                        .clone()
                        .unwrap_or_else(|| target_definition.endpoint.clone()),
                ),
                api_key_env: Some(
                    overrides
                        .api_key_env
                        .clone()
                        .unwrap_or_else(|| target_definition.api_key_env.clone()),
                ),
                keychain_account: Some(default_keychain_account(&target_provider)),
                preferred_model: Some(
                    overrides
                        .model
                        .clone()
                        .unwrap_or_else(|| target_definition.default_model.clone()),
                ),
            });
    }

    if let Some(primary) = providers.get_mut(&cfg.provider) {
        primary.endpoint = Some(cfg.llm_endpoint.clone());
        primary.api_key_env = Some(cfg.api_key_env.clone());
        primary
            .keychain_account
            .get_or_insert_with(|| default_keychain_account(&cfg.provider));
        primary.preferred_model = Some(cfg.model.clone());
    }

    let mut provider = cfg.provider.clone();
    let mut model = cfg.model.clone();
    let mut llm_endpoint = cfg.llm_endpoint.clone();
    let mut api_key_env = cfg.api_key_env.clone();
    if mode == ConfigureMode::All && target_provider == cfg.provider && persisted_config_exists {
        if let Some(value) = overrides.model.clone() {
            model = value;
        }
        if let Some(value) = overrides.endpoint.clone() {
            llm_endpoint = value;
        }
        if let Some(value) = overrides.api_key_env.clone() {
            api_key_env = value;
        }
        provider = target_provider.clone();
        if let Some(primary) = providers.get_mut(&provider) {
            primary.endpoint = Some(llm_endpoint.clone());
            primary.api_key_env = Some(api_key_env.clone());
            primary.preferred_model = Some(model.clone());
        }
    }

    let model_priority = if cfg.model_priority.is_empty() {
        vec![ModelPreference {
            provider: provider.clone(),
            model: model.clone(),
        }]
    } else {
        cfg.model_priority.clone()
    };

    let payload = json!({
        "provider": provider,
        "model": model,
        "llm_endpoint": llm_endpoint,
        "api_key_env": api_key_env,
        "git_repo_path": cfg.git_repo_path,
        "max_commit_length": cfg.max_commit_length,
        "auto_push": cfg.auto_push,
        "providers": providers,
        "model_priority": model_priority,
        "use_batch": cfg.use_batch,
        "batch_model": cfg.batch_model,
        "batch_timeout_seconds": cfg.batch_timeout_seconds
    });
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&path, serde_json::to_string_pretty(&payload)?)?;
    initialize_preferences(&definitions)?;
    Ok(path)
}

fn initialize_preferences(definitions: &[ProviderDefinition]) -> Result<PathBuf> {
    let path = preferences_path();
    let defaults = serde_json::to_value(Preferences::default())?;
    let mut preferences = if path.exists() {
        fs::read_to_string(&path)
            .ok()
            .and_then(|raw| serde_json::from_str::<serde_json::Value>(&raw).ok())
            .filter(|value| value.is_object())
            .unwrap_or_else(|| defaults.clone())
    } else {
        defaults.clone()
    };
    let object = preferences
        .as_object_mut()
        .expect("preferences value is object");
    if let Some(default_object) = defaults.as_object() {
        for (key, value) in default_object {
            object.entry(key.clone()).or_insert_with(|| value.clone());
        }
    }
    if object
        .get("prompt_profiles")
        .and_then(|value| value.as_array())
        .map(|profiles| profiles.is_empty())
        .unwrap_or(true)
    {
        object["prompt_profiles"] = defaults["prompt_profiles"].clone();
    }
    if object
        .get("default_prompt_profile")
        .and_then(|value| value.as_str())
        .map(|value| value.trim().is_empty())
        .unwrap_or(true)
    {
        object["default_prompt_profile"] = defaults["default_prompt_profile"].clone();
    }
    let provider_rules = object
        .entry("provider_rules".to_string())
        .or_insert_with(|| serde_json::json!({}));
    if !provider_rules.is_object() {
        *provider_rules = serde_json::json!({});
    }
    let provider_rules = provider_rules
        .as_object_mut()
        .expect("provider_rules value is object");
    let default_rule = serde_json::to_value(ProviderRule::default())?;
    for definition in definitions {
        provider_rules
            .entry(definition.provider.clone())
            .or_insert_with(|| default_rule.clone());
    }
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&path, serde_json::to_string_pretty(&preferences)?)?;
    Ok(path)
}

fn provider_definition<'a>(
    definitions: &'a [ProviderDefinition],
    provider: &str,
) -> Option<&'a ProviderDefinition> {
    definitions
        .iter()
        .find(|definition| definition.provider == provider)
}

fn validate_overrides(
    overrides: &ConfigOverrides,
    definitions: &[ProviderDefinition],
) -> Result<()> {
    if let Some(provider) = overrides.provider.as_deref() {
        if provider_definition(definitions, provider).is_none() {
            let expected = definitions
                .iter()
                .map(|definition| definition.provider.as_str())
                .collect::<Vec<_>>()
                .join(", ");
            bail!("unsupported provider '{provider}'; expected one of: {expected}");
        }
    }
    if let Some(model) = overrides.model.as_deref() {
        let trimmed = model.trim();
        if trimmed.is_empty() || trimmed.chars().any(char::is_whitespace) {
            bail!("model must be a non-empty provider model id without whitespace");
        }
    }
    if let Some(endpoint) = overrides.endpoint.as_deref() {
        let trimmed = endpoint.trim();
        if !(trimmed.starts_with("https://") || trimmed.starts_with("http://")) {
            bail!(
                "endpoint must be an http(s) URL; use --api-key-env for environment variable names"
            );
        }
    }
    if let Some(api_key_env) = overrides.api_key_env.as_deref() {
        if api_key_env.starts_with("http://") || api_key_env.starts_with("https://") {
            bail!("api key env must be an environment variable name, not a URL");
        }
        let mut chars = api_key_env.chars();
        let Some(first) = chars.next() else {
            bail!("api key env must be a non-empty environment variable name");
        };
        if !(first == '_' || first.is_ascii_alphabetic())
            || !chars.all(|ch| ch == '_' || ch.is_ascii_alphanumeric())
        {
            bail!("api key env must match [A-Za-z_][A-Za-z0-9_]*");
        }
    }
    Ok(())
}
