//! User preferences and credential resolution.

use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::config::loader::config_home;
use crate::error::{KcmtError, Result};

pub const PREFERENCES_SCHEMA_VERSION: u32 = 1;
pub const KEYCHAIN_SERVICE: &str = "kcmt";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default)]
pub struct Preferences {
    pub schema_version: u32,
    pub selection_policy: SelectionPolicy,
    pub provider_rules: BTreeMap<String, ProviderRule>,
    pub prompt_profiles: Vec<PromptProfile>,
    pub default_prompt_profile: String,
    pub tui: TuiPreferences,
    pub model_cache: ModelCachePreferences,
}

impl Default for Preferences {
    fn default() -> Self {
        let default_profile = PromptProfile::default_commit();
        Self {
            schema_version: PREFERENCES_SCHEMA_VERSION,
            selection_policy: SelectionPolicy::FastestCheap,
            provider_rules: BTreeMap::new(),
            default_prompt_profile: default_profile.id.clone(),
            prompt_profiles: vec![default_profile],
            tui: TuiPreferences::default(),
            model_cache: ModelCachePreferences::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SelectionPolicy {
    FastestCheap,
    Balanced,
    BestQuality,
}

impl Default for SelectionPolicy {
    fn default() -> Self {
        Self::FastestCheap
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default)]
pub struct ProviderRule {
    pub preset: ProviderRulePreset,
    pub value: Option<String>,
    pub strict: bool,
}

impl Default for ProviderRule {
    fn default() -> Self {
        Self {
            preset: ProviderRulePreset::None,
            value: None,
            strict: false,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ProviderRulePreset {
    None,
    PinExactModel,
    LatestFamily,
    LatestHaiku,
    CheapestCodeModel,
    FastestCodeModel,
    ExcludeBeta,
}

impl Default for ProviderRulePreset {
    fn default() -> Self {
        Self::None
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default)]
pub struct PromptProfile {
    pub id: String,
    pub name: String,
    pub system_instruction: String,
    pub user_instruction: String,
}

impl PromptProfile {
    pub fn default_commit() -> Self {
        Self {
            id: "conventional".to_string(),
            name: "Conventional Commit".to_string(),
            system_instruction: "You generate strictly valid Conventional Commit messages."
                .to_string(),
            user_instruction:
                "Analyze the changes carefully and be specific. Only output the commit message."
                    .to_string(),
        }
    }
}

impl Default for PromptProfile {
    fn default() -> Self {
        Self::default_commit()
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default)]
pub struct TuiPreferences {
    pub last_screen: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default)]
pub struct ModelCachePreferences {
    pub ttl_seconds: u64,
}

impl Default for ModelCachePreferences {
    fn default() -> Self {
        Self {
            ttl_seconds: 86_400,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CredentialRequest {
    pub provider: String,
    pub explicit_secret: Option<String>,
    pub keychain_account: Option<String>,
    pub env_var: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Credential {
    pub secret: String,
    pub source: CredentialSource,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CredentialSource {
    Explicit,
    Keychain,
    Environment,
}

pub trait SecretStore {
    fn get_secret(&self, account: &str) -> std::result::Result<Option<String>, String>;
    fn set_secret(&self, account: &str, secret: &str) -> std::result::Result<(), String>;
}

#[derive(Debug, Clone, Copy, Default)]
pub struct OsKeychainStore;

impl SecretStore for OsKeychainStore {
    fn get_secret(&self, account: &str) -> std::result::Result<Option<String>, String> {
        let entry =
            keyring::Entry::new(KEYCHAIN_SERVICE, account).map_err(|err| err.to_string())?;
        match entry.get_password() {
            Ok(secret) if !secret.trim().is_empty() => Ok(Some(secret)),
            Ok(_) => Ok(None),
            Err(keyring::Error::NoEntry) => Ok(None),
            Err(err) => Err(err.to_string()),
        }
    }

    fn set_secret(&self, account: &str, secret: &str) -> std::result::Result<(), String> {
        let entry =
            keyring::Entry::new(KEYCHAIN_SERVICE, account).map_err(|err| err.to_string())?;
        entry.set_password(secret).map_err(|err| err.to_string())
    }
}

pub fn preferences_path() -> PathBuf {
    config_home().join("preferences.json")
}

pub fn load_preferences() -> Result<Preferences> {
    let path = preferences_path();
    if !path.exists() {
        return Ok(Preferences::default());
    }
    let raw = fs::read_to_string(path)?;
    let mut preferences: Preferences =
        serde_json::from_str(&raw).map_err(|err| KcmtError::Message(err.to_string()))?;
    if preferences.prompt_profiles.is_empty() {
        preferences
            .prompt_profiles
            .push(PromptProfile::default_commit());
    }
    if preferences.default_prompt_profile.trim().is_empty() {
        preferences.default_prompt_profile = preferences.prompt_profiles[0].id.clone();
    }
    Ok(preferences)
}

pub fn save_preferences(preferences: &Preferences) -> Result<PathBuf> {
    let path = preferences_path();
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let rendered = serde_json::to_string_pretty(preferences)
        .map_err(|err| KcmtError::Message(err.to_string()))?;
    fs::write(&path, rendered)?;
    Ok(path)
}

pub fn resolve_credential(
    request: &CredentialRequest,
    store: &dyn SecretStore,
) -> std::result::Result<Option<Credential>, String> {
    if let Some(secret) = request
        .explicit_secret
        .as_ref()
        .map(|value| value.trim())
        .filter(|value| !value.is_empty())
    {
        return Ok(Some(Credential {
            secret: secret.to_string(),
            source: CredentialSource::Explicit,
        }));
    }

    let account = request
        .keychain_account
        .clone()
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| default_keychain_account(&request.provider));
    if let Ok(Some(secret)) = store.get_secret(&account) {
        return Ok(Some(Credential {
            secret,
            source: CredentialSource::Keychain,
        }));
    }

    if let Ok(secret) = env::var(&request.env_var) {
        let secret = secret.trim().to_string();
        if !secret.is_empty() {
            return Ok(Some(Credential {
                secret,
                source: CredentialSource::Environment,
            }));
        }
    }

    Ok(None)
}

pub fn default_keychain_account(provider: &str) -> String {
    format!("provider/{provider}/default")
}

#[cfg(test)]
mod tests {
    use super::{
        resolve_credential, CredentialRequest, CredentialSource, Preferences, SecretStore,
    };
    use std::collections::BTreeMap;
    use std::sync::{Mutex, OnceLock};

    #[derive(Default)]
    struct FakeStore {
        secrets: BTreeMap<String, String>,
    }

    impl SecretStore for FakeStore {
        fn get_secret(&self, account: &str) -> std::result::Result<Option<String>, String> {
            Ok(self.secrets.get(account).cloned())
        }

        fn set_secret(&self, _account: &str, _secret: &str) -> std::result::Result<(), String> {
            Ok(())
        }
    }

    fn env_lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

    #[test]
    fn default_preferences_include_conventional_prompt() {
        let preferences = Preferences::default();

        assert_eq!(
            preferences.selection_policy,
            super::SelectionPolicy::FastestCheap
        );
        assert_eq!(preferences.default_prompt_profile, "conventional");
        assert_eq!(preferences.prompt_profiles.len(), 1);
    }

    #[test]
    fn credential_resolution_prefers_cli_then_keychain_then_env() {
        let _guard = env_lock().lock().unwrap_or_else(|err| err.into_inner());
        std::env::set_var("KCMT_TEST_API_KEY", "env-secret");
        let mut store = FakeStore::default();
        store.secrets.insert(
            "provider/openai/default".to_string(),
            "keychain-secret".to_string(),
        );
        let base = CredentialRequest {
            provider: "openai".to_string(),
            explicit_secret: None,
            keychain_account: None,
            env_var: "KCMT_TEST_API_KEY".to_string(),
        };

        let from_cli = resolve_credential(
            &CredentialRequest {
                explicit_secret: Some("cli-secret".to_string()),
                ..base.clone()
            },
            &store,
        )
        .expect("credential")
        .expect("present");
        assert_eq!(from_cli.source, CredentialSource::Explicit);
        assert_eq!(from_cli.secret, "cli-secret");

        let from_keychain = resolve_credential(&base, &store)
            .expect("credential")
            .expect("present");
        assert_eq!(from_keychain.source, CredentialSource::Keychain);
        assert_eq!(from_keychain.secret, "keychain-secret");

        let from_env = resolve_credential(&base, &FakeStore::default())
            .expect("credential")
            .expect("present");
        assert_eq!(from_env.source, CredentialSource::Environment);
        assert_eq!(from_env.secret, "env-secret");
        std::env::remove_var("KCMT_TEST_API_KEY");
    }
}
