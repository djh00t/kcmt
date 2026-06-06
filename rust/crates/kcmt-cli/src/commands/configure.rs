use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

use kcmt_core::config::loader::{config_file_path, load_config, ConfigOverrides};
use kcmt_core::model::{ModelPreference, ProviderConfigEntry};
use kcmt_core::preferences::{load_preferences, save_preferences};
use serde_json::json;

use super::model_discovery::{default_provider_definitions, discover_model_catalog};

pub fn run_configure(repo_path: PathBuf, overrides: ConfigOverrides) -> i32 {
    match write_config(repo_path, overrides) {
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

fn write_config(repo_path: PathBuf, overrides: ConfigOverrides) -> anyhow::Result<PathBuf> {
    let cfg = load_config(&repo_path, &overrides)?;
    let mut providers: HashMap<String, ProviderConfigEntry> = cfg.providers.clone();
    for definition in default_provider_definitions() {
        providers
            .entry(definition.provider.to_string())
            .and_modify(|entry| {
                entry
                    .name
                    .get_or_insert_with(|| definition.display_name.to_string());
                entry
                    .endpoint
                    .get_or_insert_with(|| definition.endpoint.to_string());
                entry
                    .api_key_env
                    .get_or_insert_with(|| definition.api_key_env.to_string());
                entry
                    .keychain_account
                    .get_or_insert_with(|| format!("provider/{}/default", definition.provider));
                entry
                    .preferred_model
                    .get_or_insert_with(|| definition.default_model.to_string());
            })
            .or_insert_with(|| ProviderConfigEntry {
                name: Some(definition.display_name.to_string()),
                endpoint: Some(definition.endpoint.to_string()),
                api_key_env: Some(definition.api_key_env.to_string()),
                keychain_account: Some(format!("provider/{}/default", definition.provider)),
                preferred_model: Some(definition.default_model.to_string()),
            });
    }

    if let Some(primary) = providers.get_mut(&cfg.provider) {
        primary.endpoint = Some(cfg.llm_endpoint.clone());
        primary.api_key_env = Some(cfg.api_key_env.clone());
        primary
            .keychain_account
            .get_or_insert_with(|| format!("provider/{}/default", cfg.provider));
        primary.preferred_model = Some(cfg.model.clone());
    }

    let model_priority = if cfg.model_priority.is_empty() {
        vec![ModelPreference {
            provider: cfg.provider.clone(),
            model: cfg.model.clone(),
        }]
    } else {
        cfg.model_priority.clone()
    };

    let payload = json!({
        "provider": cfg.provider,
        "model": cfg.model,
        "llm_endpoint": cfg.llm_endpoint,
        "api_key_env": cfg.api_key_env,
        "git_repo_path": cfg.git_repo_path,
        "max_commit_length": cfg.max_commit_length,
        "auto_push": cfg.auto_push,
        "providers": providers,
        "model_priority": model_priority,
        "use_batch": cfg.use_batch,
        "batch_model": cfg.batch_model,
        "batch_timeout_seconds": cfg.batch_timeout_seconds
    });
    let path = config_file_path(&repo_path);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&path, serde_json::to_string_pretty(&payload)?)?;
    let preferences = load_preferences().unwrap_or_default();
    save_preferences(&preferences)?;
    Ok(path)
}
