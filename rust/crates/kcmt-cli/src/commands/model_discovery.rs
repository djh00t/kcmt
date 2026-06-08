//! Model discovery catalog, cache, and selector integration.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use kcmt_core::config::loader::{config_home, load_config, ConfigOverrides};
use kcmt_core::model::WorkflowConfig;
use kcmt_core::preferences::{
    default_keychain_account, resolve_credential, CredentialRequest, OsKeychainStore, Preferences,
};
use kcmt_core::selector::ModelCandidate;
use kcmt_provider::clients::{
    AnthropicClient, GitHubModelsClient, ListedModel, OpenAiClient, XaiClient,
};
use kcmt_provider::transport::{AsyncTransport, RetryPolicy};
use serde::{Deserialize, Serialize};

const CACHE_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone)]
pub struct ProviderDefinition {
    pub provider: String,
    pub display_name: String,
    pub default_model: String,
    pub endpoint: String,
    pub api_key_env: String,
}

impl ProviderDefinition {
    fn new(
        provider: &str,
        display_name: &str,
        default_model: &str,
        endpoint: &str,
        api_key_env: &str,
    ) -> Self {
        Self {
            provider: provider.to_string(),
            display_name: display_name.to_string(),
            default_model: default_model.to_string(),
            endpoint: endpoint.to_string(),
            api_key_env: api_key_env.to_string(),
        }
    }
}

pub fn default_provider_definitions() -> Vec<ProviderDefinition> {
    vec![
        ProviderDefinition::new(
            "openai",
            "OpenAI",
            "gpt-5.4-mini",
            "https://api.openai.com/v1",
            "OPENAI_API_KEY",
        ),
        ProviderDefinition::new(
            "anthropic",
            "Anthropic",
            "claude-3-5-haiku-latest",
            "https://api.anthropic.com",
            "ANTHROPIC_API_KEY",
        ),
        ProviderDefinition::new(
            "xai",
            "X.AI",
            "grok-code-fast",
            "https://api.x.ai/v1",
            "XAI_API_KEY",
        ),
        ProviderDefinition::new(
            "github",
            "GitHub Models",
            "openai/gpt-4.1-mini",
            "https://models.github.ai/inference",
            "GITHUB_TOKEN",
        ),
    ]
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum DiscoverySource {
    Live,
    Cache,
    StaticFallback,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DiscoveredModel {
    pub id: String,
    pub provider: String,
    pub endpoint: String,
    pub api_key_env: String,
    pub created: Option<String>,
    pub family: Option<String>,
    pub code_capable: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ProviderModelCatalog {
    pub provider: String,
    pub display_name: String,
    pub endpoint: String,
    pub api_key_env: String,
    pub source: DiscoverySource,
    pub error: Option<String>,
    pub models: Vec<DiscoveredModel>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ModelCacheFile {
    schema_version: u32,
    generated_unix_seconds: u64,
    providers: Vec<ProviderModelCatalog>,
}

pub fn discover_model_catalog(preferences: &Preferences) -> Vec<ProviderModelCatalog> {
    discover_model_catalog_for_definitions(&provider_definitions_from_config(), preferences)
}

pub fn catalog_for_config(
    config: &WorkflowConfig,
    preferences: &Preferences,
) -> Vec<ProviderModelCatalog> {
    discover_model_catalog_for_definitions(&provider_definitions(config), preferences)
}

pub fn cached_or_static_catalog_for_config(
    config: &WorkflowConfig,
    preferences: &Preferences,
) -> Vec<ProviderModelCatalog> {
    provider_definitions(config)
        .iter()
        .map(|definition| fallback_catalog(definition, preferences, None))
        .collect()
}

pub fn catalog_to_selector_candidates(catalog: &[ProviderModelCatalog]) -> Vec<ModelCandidate> {
    catalog
        .iter()
        .flat_map(|provider| {
            provider.models.iter().map(|model| ModelCandidate {
                provider: model.provider.clone(),
                id: model.id.clone(),
                family: model.family.clone(),
                code_capable: model.code_capable,
                beta: model.id.to_ascii_lowercase().contains("beta"),
                input_cost_per_million: None,
                output_cost_per_million: None,
                median_latency_ms: None,
                created_at: model.created.clone(),
            })
        })
        .collect()
}

fn discover_model_catalog_for_definitions(
    definitions: &[ProviderDefinition],
    preferences: &Preferences,
) -> Vec<ProviderModelCatalog> {
    let mut catalogs = Vec::new();
    let mut live_catalogs = Vec::new();
    for definition in definitions {
        let live = live_models_for_provider(definition);
        let catalog = match live {
            Ok(models) if !models.is_empty() => {
                let catalog = provider_catalog(definition, DiscoverySource::Live, None, models);
                live_catalogs.push(catalog.clone());
                catalog
            }
            Ok(_) => fallback_catalog(definition, preferences, Some("provider returned no models")),
            Err(err) => fallback_catalog(definition, preferences, Some(err.as_str())),
        };
        catalogs.push(catalog);
    }
    if !live_catalogs.is_empty() {
        let _ = save_model_cache(live_catalogs);
    }
    catalogs
}

fn fallback_catalog(
    definition: &ProviderDefinition,
    preferences: &Preferences,
    error: Option<&str>,
) -> ProviderModelCatalog {
    if let Some(cached) = cached_provider_catalog(definition, preferences) {
        let mut cached = cached;
        cached.source = DiscoverySource::Cache;
        cached.error = error.map(ToOwned::to_owned);
        return cached;
    }
    provider_catalog(
        definition,
        DiscoverySource::StaticFallback,
        error.map(ToOwned::to_owned),
        static_models(definition),
    )
}

fn live_models_for_provider(
    definition: &ProviderDefinition,
) -> Result<Vec<DiscoveredModel>, String> {
    let explicit_provider = std::env::var("KCMT_EXPLICIT_API_KEY_PROVIDER").ok();
    let explicit_secret = explicit_secret_for_provider(
        &definition.provider,
        explicit_provider.as_deref(),
        std::env::var("KCMT_EXPLICIT_API_KEY").ok(),
    );
    let request = CredentialRequest {
        provider: definition.provider.clone(),
        explicit_secret,
        keychain_account: Some(default_keychain_account(&definition.provider)),
        env_var: definition.api_key_env.clone(),
    };
    let api_key = resolve_credential(&request, &OsKeychainStore)
        .map_err(|err| format!("credential lookup failed: {err}"))?
        .map(|credential| credential.secret)
        .ok_or_else(|| format!("missing credential {}", definition.api_key_env))?;
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|err| format!("runtime unavailable: {err}"))?;
    let transport = AsyncTransport::new(
        Duration::from_secs(5),
        RetryPolicy {
            max_attempts: 1,
            base_backoff: Duration::from_millis(100),
        },
    )
    .map_err(|err| format!("transport unavailable: {err}"))?;
    let models = runtime
        .block_on(async {
            match definition.provider.as_str() {
                "anthropic" => {
                    AnthropicClient::list_models(&transport, &definition.endpoint, &api_key).await
                }
                "xai" => XaiClient::list_models(&transport, &definition.endpoint, &api_key).await,
                "github" => {
                    GitHubModelsClient::list_models(&transport, &definition.endpoint, &api_key)
                        .await
                }
                _ => OpenAiClient::list_models(&transport, &definition.endpoint, &api_key).await,
            }
        })
        .map_err(|err| err.to_string())?;
    Ok(models
        .into_iter()
        .map(|model| normalize_model(definition, &model))
        .collect())
}

fn explicit_secret_for_provider(
    provider: &str,
    explicit_provider: Option<&str>,
    explicit_secret: Option<String>,
) -> Option<String> {
    (explicit_provider == Some(provider))
        .then_some(explicit_secret)
        .flatten()
}

fn provider_catalog(
    definition: &ProviderDefinition,
    source: DiscoverySource,
    error: Option<String>,
    models: Vec<DiscoveredModel>,
) -> ProviderModelCatalog {
    ProviderModelCatalog {
        provider: definition.provider.clone(),
        display_name: definition.display_name.clone(),
        endpoint: definition.endpoint.clone(),
        api_key_env: definition.api_key_env.clone(),
        source,
        error,
        models,
    }
}

fn normalize_model(definition: &ProviderDefinition, model: &ListedModel) -> DiscoveredModel {
    DiscoveredModel {
        id: model.id.clone(),
        provider: definition.provider.clone(),
        endpoint: definition.endpoint.clone(),
        api_key_env: definition.api_key_env.clone(),
        created: model.created_at.clone(),
        family: infer_family(&model.id),
        code_capable: infer_code_capable(&model.id),
    }
}

fn static_model(definition: &ProviderDefinition) -> DiscoveredModel {
    normalize_model(
        definition,
        &ListedModel {
            id: definition.default_model.clone(),
            created_at: None,
        },
    )
}

fn static_models(definition: &ProviderDefinition) -> Vec<DiscoveredModel> {
    let mut models = vec![static_model(definition)];
    if let Some(default_definition) = default_provider_definitions()
        .into_iter()
        .find(|candidate| candidate.provider == definition.provider)
    {
        if default_definition.default_model != definition.default_model {
            models.push(normalize_model(
                definition,
                &ListedModel {
                    id: default_definition.default_model,
                    created_at: None,
                },
            ));
        }
    }
    models
}

fn cached_provider_catalog(
    definition: &ProviderDefinition,
    preferences: &Preferences,
) -> Option<ProviderModelCatalog> {
    cached_provider_catalog_from_path(definition, preferences, &model_cache_path())
}

fn cached_provider_catalog_from_path(
    definition: &ProviderDefinition,
    preferences: &Preferences,
    path: &Path,
) -> Option<ProviderModelCatalog> {
    let ttl = preferences.model_cache.ttl_seconds;
    if ttl == 0 {
        return None;
    }
    let cache = load_model_cache_from_path(path)?;
    if cache.schema_version != CACHE_SCHEMA_VERSION {
        return None;
    }
    let now = now_unix_seconds();
    if now.saturating_sub(cache.generated_unix_seconds) > ttl {
        return None;
    }
    cache
        .providers
        .into_iter()
        .find(|provider| provider.provider == definition.provider && !provider.models.is_empty())
}

fn load_model_cache_from_path(path: &Path) -> Option<ModelCacheFile> {
    let raw = fs::read_to_string(path).ok()?;
    serde_json::from_str(&raw).ok()
}

fn save_model_cache(providers: Vec<ProviderModelCatalog>) -> std::io::Result<()> {
    let path = model_cache_path();
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let cache = ModelCacheFile {
        schema_version: CACHE_SCHEMA_VERSION,
        generated_unix_seconds: now_unix_seconds(),
        providers,
    };
    fs::write(path, serde_json::to_string_pretty(&cache)?)
}

fn model_cache_path() -> PathBuf {
    config_home().join("model-cache.json")
}

fn now_unix_seconds() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0)
}

fn provider_definitions_from_config() -> Vec<ProviderDefinition> {
    if !config_home().join("config.json").exists() {
        return default_provider_definitions();
    }
    let config = std::env::current_dir()
        .ok()
        .and_then(|cwd| load_config(&cwd, &ConfigOverrides::default()).ok());
    match config {
        Some(config) => provider_definitions(&config),
        None => default_provider_definitions(),
    }
}

fn provider_definitions(config: &WorkflowConfig) -> Vec<ProviderDefinition> {
    let mut definitions = default_provider_definitions();
    let defaults = default_provider_definitions()
        .iter()
        .map(|definition| (definition.provider.clone(), definition.clone()))
        .collect::<HashMap<_, _>>();
    for definition in &mut definitions {
        if definition.provider == config.provider {
            definition.endpoint = config.llm_endpoint.clone();
            definition.api_key_env = config.api_key_env.clone();
            definition.default_model = config.model.clone();
        }
        if let Some(entry) = config.providers.get(&definition.provider) {
            if let Some(endpoint) = entry
                .endpoint
                .as_ref()
                .filter(|value| !value.trim().is_empty())
            {
                definition.endpoint = endpoint.clone();
            }
            if let Some(api_key_env) = entry
                .api_key_env
                .as_ref()
                .filter(|value| !value.trim().is_empty())
            {
                definition.api_key_env = api_key_env.clone();
            }
            if let Some(model) = entry
                .preferred_model
                .as_ref()
                .filter(|value| !value.trim().is_empty())
            {
                definition.default_model = model.clone();
            }
        }
        if definition.endpoint.is_empty() {
            definition.endpoint = defaults[&definition.provider].endpoint.clone();
        }
        if definition.api_key_env.is_empty() {
            definition.api_key_env = defaults[&definition.provider].api_key_env.clone();
        }
        if definition.default_model.is_empty() {
            definition.default_model = defaults[&definition.provider].default_model.clone();
        }
    }
    definitions
}

fn infer_family(model_id: &str) -> Option<String> {
    let id = model_id.to_ascii_lowercase();
    let family = if id.contains("haiku") {
        "haiku"
    } else if id.contains("sonnet") {
        "sonnet"
    } else if id.contains("opus") {
        "opus"
    } else if id.contains("grok-code") {
        "grok-code"
    } else if id.contains("grok") {
        "grok"
    } else if id.contains("gpt-5") {
        "gpt-5"
    } else if id.contains("gpt-4.1") {
        "gpt-4.1"
    } else if id.contains("o4") {
        "o4"
    } else if id.contains("o3") {
        "o3"
    } else {
        return None;
    };
    Some(family.to_string())
}

fn infer_code_capable(model_id: &str) -> bool {
    let id = model_id.to_ascii_lowercase();
    !id.contains("embedding")
        && !id.contains("audio")
        && !id.contains("image")
        && !id.contains("vision")
        && !id.contains("moderation")
}

#[cfg(test)]
mod tests {
    use std::env;
    use std::fs;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use kcmt_core::preferences::Preferences;

    use super::{
        cached_provider_catalog_from_path, catalog_to_selector_candidates, fallback_catalog,
        provider_definitions, DiscoveredModel, DiscoverySource, ModelCacheFile, ProviderDefinition,
        ProviderModelCatalog, CACHE_SCHEMA_VERSION,
    };

    static COUNTER: AtomicUsize = AtomicUsize::new(0);

    fn temp_config_home() -> std::path::PathBuf {
        let path = env::temp_dir().join(format!(
            "kcmt-model-discovery-test-{}-{}",
            std::process::id(),
            COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = fs::remove_dir_all(&path);
        fs::create_dir_all(&path).expect("temp config home");
        path
    }

    #[test]
    fn static_fallback_keeps_normalized_metadata_and_error() {
        let definition = ProviderDefinition {
            provider: "anthropic".to_string(),
            display_name: "Anthropic".to_string(),
            default_model: "claude-3-5-haiku-latest".to_string(),
            endpoint: "https://api.anthropic.com".to_string(),
            api_key_env: "ANTHROPIC_API_KEY".to_string(),
        };

        let catalog = fallback_catalog(&definition, &Preferences::default(), Some("offline"));

        assert_eq!(catalog.source, DiscoverySource::StaticFallback);
        assert_eq!(catalog.error.as_deref(), Some("offline"));
        assert_eq!(catalog.models[0].provider, "anthropic");
        assert_eq!(catalog.models[0].endpoint, "https://api.anthropic.com");
        assert_eq!(catalog.models[0].api_key_env, "ANTHROPIC_API_KEY");
        assert_eq!(catalog.models[0].family.as_deref(), Some("haiku"));
        assert!(catalog.models[0].code_capable);
    }

    #[test]
    fn explicit_api_key_requires_matching_provider_marker() {
        assert_eq!(
            super::explicit_secret_for_provider("anthropic", None, Some("secret".to_string())),
            None
        );
        assert_eq!(
            super::explicit_secret_for_provider(
                "anthropic",
                Some("openai"),
                Some("secret".to_string())
            ),
            None
        );
        assert_eq!(
            super::explicit_secret_for_provider(
                "anthropic",
                Some("anthropic"),
                Some("secret".to_string())
            )
            .as_deref(),
            Some("secret")
        );
    }

    #[test]
    fn fresh_cache_is_used_before_static_fallback() {
        let config_home = temp_config_home();
        let cache_path = config_home.join("model-cache.json");
        let cache = ModelCacheFile {
            schema_version: CACHE_SCHEMA_VERSION,
            generated_unix_seconds: super::now_unix_seconds(),
            providers: vec![ProviderModelCatalog {
                provider: "openai".to_string(),
                display_name: "OpenAI".to_string(),
                endpoint: "https://cached.example/v1".to_string(),
                api_key_env: "OPENAI_API_KEY".to_string(),
                source: DiscoverySource::Live,
                error: None,
                models: vec![DiscoveredModel {
                    id: "gpt-5-cached".to_string(),
                    provider: "openai".to_string(),
                    endpoint: "https://cached.example/v1".to_string(),
                    api_key_env: "OPENAI_API_KEY".to_string(),
                    created: Some("2026-01-01".to_string()),
                    family: Some("gpt-5".to_string()),
                    code_capable: true,
                }],
            }],
        };
        fs::write(
            &cache_path,
            serde_json::to_string(&cache).expect("cache json"),
        )
        .expect("cache write");

        let definition = super::default_provider_definitions()[0].clone();
        let cached =
            cached_provider_catalog_from_path(&definition, &Preferences::default(), &cache_path)
                .expect("fresh cache");

        assert_eq!(cached.models[0].id, "gpt-5-cached");
    }

    #[test]
    fn expired_cache_falls_back_to_static_models() {
        let config_home = temp_config_home();
        let cache_path = config_home.join("model-cache.json");
        let cache = ModelCacheFile {
            schema_version: CACHE_SCHEMA_VERSION,
            generated_unix_seconds: 1,
            providers: vec![ProviderModelCatalog {
                provider: "openai".to_string(),
                display_name: "OpenAI".to_string(),
                endpoint: "https://cached.example/v1".to_string(),
                api_key_env: "OPENAI_API_KEY".to_string(),
                source: DiscoverySource::Live,
                error: None,
                models: vec![DiscoveredModel {
                    id: "gpt-5-cached".to_string(),
                    provider: "openai".to_string(),
                    endpoint: "https://cached.example/v1".to_string(),
                    api_key_env: "OPENAI_API_KEY".to_string(),
                    created: None,
                    family: Some("gpt-5".to_string()),
                    code_capable: true,
                }],
            }],
        };
        fs::write(
            &cache_path,
            serde_json::to_string(&cache).expect("cache json"),
        )
        .expect("cache write");
        let definition = super::default_provider_definitions()[0].clone();

        let cached =
            cached_provider_catalog_from_path(&definition, &Preferences::default(), &cache_path);
        let static_models = super::static_models(&definition);

        assert!(cached.is_none());
        assert_eq!(static_models[0].id, "gpt-5.4-mini");
    }

    #[test]
    fn selector_candidates_preserve_discovered_metadata() {
        let catalog = vec![ProviderModelCatalog {
            provider: "xai".to_string(),
            display_name: "X.AI".to_string(),
            endpoint: "https://api.x.ai/v1".to_string(),
            api_key_env: "XAI_API_KEY".to_string(),
            source: DiscoverySource::Live,
            error: None,
            models: vec![DiscoveredModel {
                id: "grok-code-fast".to_string(),
                provider: "xai".to_string(),
                endpoint: "https://api.x.ai/v1".to_string(),
                api_key_env: "XAI_API_KEY".to_string(),
                created: Some("2026-01-01".to_string()),
                family: Some("grok-code".to_string()),
                code_capable: true,
            }],
        }];

        let candidates = catalog_to_selector_candidates(&catalog);

        assert_eq!(candidates[0].provider, "xai");
        assert_eq!(candidates[0].id, "grok-code-fast");
        assert_eq!(candidates[0].family.as_deref(), Some("grok-code"));
        assert_eq!(candidates[0].created_at.as_deref(), Some("2026-01-01"));
        assert!(candidates[0].code_capable);
    }

    #[test]
    fn static_fallback_includes_configured_and_builtin_defaults() {
        let definition = ProviderDefinition {
            provider: "anthropic".to_string(),
            display_name: "Anthropic".to_string(),
            default_model: "claude-sonnet-4-20250514".to_string(),
            endpoint: "https://api.anthropic.com".to_string(),
            api_key_env: "ANTHROPIC_API_KEY".to_string(),
        };

        let catalog = fallback_catalog(&definition, &Preferences::default(), Some("offline"));
        let model_ids = catalog
            .models
            .iter()
            .map(|model| model.id.as_str())
            .collect::<Vec<_>>();

        assert_eq!(catalog.source, DiscoverySource::StaticFallback);
        assert!(model_ids.contains(&"claude-sonnet-4-20250514"));
        assert!(model_ids.contains(&"claude-3-5-haiku-latest"));
    }

    #[test]
    fn provider_definitions_honor_persisted_provider_metadata() {
        let config = kcmt_core::model::WorkflowConfig {
            provider: "openai".to_string(),
            model: "gpt-custom".to_string(),
            llm_endpoint: "https://custom.openai.test/v1".to_string(),
            api_key_env: "OPENAI_TEST_KEY".to_string(),
            ..kcmt_core::model::WorkflowConfig::default()
        };

        let definitions = provider_definitions(&config);
        let openai = definitions
            .iter()
            .find(|definition| definition.provider == "openai")
            .expect("openai definition");

        assert_eq!(openai.default_model, "gpt-custom");
        assert_eq!(openai.endpoint, "https://custom.openai.test/v1");
        assert_eq!(openai.api_key_env, "OPENAI_TEST_KEY");
    }
}
