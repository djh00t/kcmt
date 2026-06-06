use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::time::Duration;

use kcmt_core::config::loader::{config_file_path, load_config, ConfigOverrides};
use kcmt_core::model::{ModelPreference, ProviderConfigEntry};
use kcmt_core::preferences::{
    default_keychain_account, load_preferences, resolve_credential, save_preferences,
    CredentialRequest, OsKeychainStore,
};
use kcmt_provider::clients::{AnthropicClient, GitHubModelsClient, OpenAiClient, XaiClient};
use kcmt_provider::transport::{AsyncTransport, RetryPolicy};
use serde_json::json;

const PROVIDERS: &[(&str, &str, &str, &str, &str)] = &[
    (
        "openai",
        "OpenAI",
        "gpt-5-mini-2025-08-07",
        "https://api.openai.com/v1",
        "OPENAI_API_KEY",
    ),
    (
        "anthropic",
        "Anthropic",
        "claude-3-5-haiku-latest",
        "https://api.anthropic.com",
        "ANTHROPIC_API_KEY",
    ),
    (
        "xai",
        "X.AI",
        "grok-code-fast",
        "https://api.x.ai/v1",
        "XAI_API_KEY",
    ),
    (
        "github",
        "GitHub Models",
        "openai/gpt-4.1-mini",
        "https://models.github.ai/inference",
        "GITHUB_TOKEN",
    ),
];

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
    let provider_models = resolved_provider_models();
    if debug {
        let payload = PROVIDERS
            .iter()
            .map(|(provider, name, model, endpoint, api_key_env)| {
                let models = provider_models
                    .get(*provider)
                    .cloned()
                    .unwrap_or_else(|| vec![(*model).to_string()]);
                json!({
                    "provider": provider,
                    "display_name": name,
                    "models": models.into_iter().map(|model_id| json!({
                        "id": model_id,
                        "endpoint": endpoint,
                        "api_key_env": api_key_env
                    })).collect::<Vec<_>>()
                })
            })
            .collect::<Vec<_>>();
        match serde_json::to_string_pretty(&payload) {
            Ok(rendered) => println!("{rendered}"),
            Err(err) => {
                eprintln!("{err}");
                return 1;
            }
        }
        return 0;
    }

    for (provider, _name, model, endpoint, _api_key_env) in PROVIDERS {
        let models = provider_models
            .get(*provider)
            .cloned()
            .unwrap_or_else(|| vec![(*model).to_string()]);
        for model in models {
            println!("{provider}\t{model}\t{endpoint}");
        }
    }
    0
}

fn resolved_provider_models() -> HashMap<String, Vec<String>> {
    PROVIDERS
        .iter()
        .map(|(provider, _name, model, endpoint, api_key_env)| {
            let models = live_models_for_provider(provider, endpoint, api_key_env)
                .filter(|models| !models.is_empty())
                .unwrap_or_else(|| vec![(*model).to_string()]);
            ((*provider).to_string(), models)
        })
        .collect()
}

fn live_models_for_provider(
    provider: &str,
    endpoint: &str,
    api_key_env: &str,
) -> Option<Vec<String>> {
    let explicit_provider = std::env::var("KCMT_EXPLICIT_API_KEY_PROVIDER").ok();
    let explicit_secret = if explicit_provider.as_deref().unwrap_or(provider) == provider {
        std::env::var("KCMT_EXPLICIT_API_KEY").ok()
    } else {
        None
    };
    let request = CredentialRequest {
        provider: provider.to_string(),
        explicit_secret,
        keychain_account: Some(default_keychain_account(provider)),
        env_var: api_key_env.to_string(),
    };
    let api_key = resolve_credential(&request, &OsKeychainStore)
        .ok()
        .flatten()?
        .secret;
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .ok()?;
    let transport = AsyncTransport::new(
        Duration::from_secs(5),
        RetryPolicy {
            max_attempts: 1,
            base_backoff: Duration::from_millis(100),
        },
    )
    .ok()?;
    runtime
        .block_on(async {
            match provider {
                "anthropic" => AnthropicClient::list_models(&transport, endpoint, &api_key).await,
                "xai" => XaiClient::list_models(&transport, endpoint, &api_key).await,
                "github" => GitHubModelsClient::list_models(&transport, endpoint, &api_key).await,
                _ => OpenAiClient::list_models(&transport, endpoint, &api_key).await,
            }
        })
        .ok()
        .map(|models| models.into_iter().map(|model| model.id).collect())
}

pub fn run_verify_keys() -> i32 {
    println!("API Key Verification");
    println!("provider\tenv_var\tpresent\tdetected");
    for (provider, _name, _model, _endpoint, api_key_env) in PROVIDERS {
        let present = std::env::var(api_key_env)
            .ok()
            .map(|value| !value.trim().is_empty())
            .unwrap_or(false);
        let detected = if present { *api_key_env } else { "-" };
        println!(
            "{provider}\t{api_key_env}\t{}\t{detected}",
            if present { "yes" } else { "no" }
        );
    }
    0
}

fn write_config(repo_path: PathBuf, overrides: ConfigOverrides) -> anyhow::Result<PathBuf> {
    let cfg = load_config(&repo_path, &overrides)?;
    let mut providers: HashMap<String, ProviderConfigEntry> = cfg.providers.clone();
    for (provider, name, model, endpoint, api_key_env) in PROVIDERS {
        providers
            .entry((*provider).to_string())
            .and_modify(|entry| {
                entry.name.get_or_insert_with(|| (*name).to_string());
                entry
                    .endpoint
                    .get_or_insert_with(|| (*endpoint).to_string());
                entry
                    .api_key_env
                    .get_or_insert_with(|| (*api_key_env).to_string());
                entry
                    .keychain_account
                    .get_or_insert_with(|| format!("provider/{provider}/default"));
                entry
                    .preferred_model
                    .get_or_insert_with(|| (*model).to_string());
            })
            .or_insert_with(|| ProviderConfigEntry {
                name: Some((*name).to_string()),
                endpoint: Some((*endpoint).to_string()),
                api_key_env: Some((*api_key_env).to_string()),
                keychain_account: Some(format!("provider/{provider}/default")),
                preferred_model: Some((*model).to_string()),
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
