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

impl ProviderRulePreset {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::None => "none",
            Self::PinExactModel => "pin_exact_model",
            Self::LatestFamily => "latest_family",
            Self::LatestHaiku => "latest_haiku",
            Self::CheapestCodeModel => "cheapest_code_model",
            Self::FastestCodeModel => "fastest_code_model",
            Self::ExcludeBeta => "exclude_beta",
        }
    }
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeychainSaveMode {
    PlatformDefault,
    BiometricPreferred,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeychainProtection {
    PlatformDefault,
    BiometricPreferred,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SecretSaveResult {
    pub protection: KeychainProtection,
    pub fallback_reason: Option<String>,
}

impl SecretSaveResult {
    fn platform_default() -> Self {
        Self {
            protection: KeychainProtection::PlatformDefault,
            fallback_reason: None,
        }
    }

    #[cfg(target_os = "macos")]
    fn biometric_preferred() -> Self {
        Self {
            protection: KeychainProtection::BiometricPreferred,
            fallback_reason: None,
        }
    }

    #[cfg(target_os = "macos")]
    fn biometric_fallback(reason: String) -> Self {
        Self {
            protection: KeychainProtection::PlatformDefault,
            fallback_reason: Some(reason),
        }
    }
}

pub trait SecretStore {
    fn get_secret(&self, account: &str) -> std::result::Result<Option<String>, String>;
    fn set_secret(&self, account: &str, secret: &str) -> std::result::Result<(), String>;
    fn set_secret_with_mode(
        &self,
        account: &str,
        secret: &str,
        _mode: KeychainSaveMode,
    ) -> std::result::Result<SecretSaveResult, String> {
        self.set_secret(account, secret)
            .map(|()| SecretSaveResult::platform_default())
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct OsKeychainStore;

#[cfg(any(target_os = "macos", target_os = "windows"))]
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

    fn set_secret_with_mode(
        &self,
        account: &str,
        secret: &str,
        mode: KeychainSaveMode,
    ) -> std::result::Result<SecretSaveResult, String> {
        match mode {
            KeychainSaveMode::PlatformDefault => {
                self.set_secret(account, secret)?;
                Ok(SecretSaveResult::platform_default())
            }
            KeychainSaveMode::BiometricPreferred => {
                self.set_secret_biometric_preferred(account, secret)
            }
        }
    }
}

#[cfg(not(any(target_os = "macos", target_os = "windows")))]
impl SecretStore for OsKeychainStore {
    fn get_secret(&self, _account: &str) -> std::result::Result<Option<String>, String> {
        Ok(None)
    }

    fn set_secret(&self, _account: &str, _secret: &str) -> std::result::Result<(), String> {
        Err("OS keychain is not available on this platform".to_string())
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

    if !keychain_disabled() {
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

pub fn save_keychain_secret(
    store: &dyn SecretStore,
    account: &str,
    secret: &str,
    mode: KeychainSaveMode,
) -> std::result::Result<SecretSaveResult, String> {
    if let Some(reason) = keychain_disabled_reason() {
        return Err(reason.to_string());
    }
    store.set_secret_with_mode(account, secret, mode)
}

fn keychain_disabled() -> bool {
    keychain_disabled_reason().is_some()
}

fn keychain_disabled_reason() -> Option<&'static str> {
    if env_truthy("KCMT_DISABLE_KEYCHAIN") {
        return Some("OS keychain access is disabled by KCMT_DISABLE_KEYCHAIN");
    }
    if env_truthy("KCMT_RUNTIME_BENCHMARK") {
        return Some("OS keychain access is disabled during KCMT_RUNTIME_BENCHMARK");
    }
    None
}

fn env_truthy(key: &str) -> bool {
    env::var(key)
        .ok()
        .map(|value| {
            let normalized = value.trim().to_ascii_lowercase();
            matches!(normalized.as_str(), "1" | "true" | "yes" | "on")
        })
        .unwrap_or(false)
}

pub fn default_keychain_account(provider: &str) -> String {
    format!("provider/{provider}/default")
}

#[cfg(target_os = "macos")]
impl OsKeychainStore {
    fn set_secret_biometric_preferred(
        &self,
        account: &str,
        secret: &str,
    ) -> std::result::Result<SecretSaveResult, String> {
        match set_macos_biometric_secret(account, secret) {
            Ok(()) => Ok(SecretSaveResult::biometric_preferred()),
            Err(err) => {
                self.set_secret(account, secret)?;
                Ok(SecretSaveResult::biometric_fallback(err))
            }
        }
    }
}

#[cfg(target_os = "windows")]
impl OsKeychainStore {
    fn set_secret_biometric_preferred(
        &self,
        account: &str,
        secret: &str,
    ) -> std::result::Result<SecretSaveResult, String> {
        self.set_secret(account, secret)?;
        Ok(SecretSaveResult::platform_default())
    }
}

#[cfg(target_os = "macos")]
fn set_macos_biometric_secret(account: &str, secret: &str) -> std::result::Result<(), String> {
    use security_framework::access_control::{ProtectionMode, SecAccessControl};
    use security_framework::passwords::{
        delete_generic_password, set_generic_password_options, AccessControlOptions,
        PasswordOptions,
    };

    const ERR_SEC_ITEM_NOT_FOUND: i32 = -25300;

    fn biometric_access_control() -> security_framework::base::Result<SecAccessControl> {
        SecAccessControl::create_with_protection(
            Some(ProtectionMode::AccessibleWhenUnlockedThisDeviceOnly),
            (AccessControlOptions::BIOMETRY_ANY
                | AccessControlOptions::OR
                | AccessControlOptions::DEVICE_PASSCODE)
                .bits(),
        )
    }

    fn add_biometric_secret(
        account: &str,
        secret: &str,
        access_control: SecAccessControl,
    ) -> security_framework::base::Result<()> {
        let mut options = PasswordOptions::new_generic_password(KEYCHAIN_SERVICE, account);
        options.set_access_control(access_control);
        set_generic_password_options(secret.as_bytes(), options)
    }

    let access_control = biometric_access_control().map_err(|err| err.to_string())?;
    match delete_generic_password(KEYCHAIN_SERVICE, account) {
        Ok(()) => {}
        Err(err) if err.code() == ERR_SEC_ITEM_NOT_FOUND => {}
        Err(err) => return Err(err.to_string()),
    }
    add_biometric_secret(account, secret, access_control).map_err(|err| err.to_string())
}

#[cfg(test)]
mod tests {
    use super::{
        resolve_credential, save_keychain_secret, CredentialRequest, CredentialSource,
        KeychainProtection, KeychainSaveMode, Preferences, SecretSaveResult, SecretStore,
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

    #[derive(Default)]
    struct CapturingStore {
        modes: Mutex<Vec<KeychainSaveMode>>,
    }

    impl SecretStore for CapturingStore {
        fn get_secret(&self, _account: &str) -> std::result::Result<Option<String>, String> {
            Ok(None)
        }

        fn set_secret(&self, _account: &str, _secret: &str) -> std::result::Result<(), String> {
            Ok(())
        }

        fn set_secret_with_mode(
            &self,
            _account: &str,
            _secret: &str,
            mode: KeychainSaveMode,
        ) -> std::result::Result<SecretSaveResult, String> {
            self.modes
                .lock()
                .unwrap_or_else(|err| err.into_inner())
                .push(mode);
            Ok(SecretSaveResult {
                protection: match mode {
                    KeychainSaveMode::PlatformDefault => KeychainProtection::PlatformDefault,
                    KeychainSaveMode::BiometricPreferred => KeychainProtection::BiometricPreferred,
                },
                fallback_reason: None,
            })
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
        std::env::remove_var("KCMT_DISABLE_KEYCHAIN");
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

    #[test]
    fn credential_resolution_can_disable_keychain_for_hermetic_runs() {
        let _guard = env_lock().lock().unwrap_or_else(|err| err.into_inner());
        std::env::set_var("KCMT_DISABLE_KEYCHAIN", "1");
        std::env::set_var("KCMT_TEST_API_KEY", "env-secret");
        let mut store = FakeStore::default();
        store.secrets.insert(
            "provider/openai/default".to_string(),
            "keychain-secret".to_string(),
        );
        let request = CredentialRequest {
            provider: "openai".to_string(),
            explicit_secret: None,
            keychain_account: None,
            env_var: "KCMT_TEST_API_KEY".to_string(),
        };

        let credential = resolve_credential(&request, &store)
            .expect("credential")
            .expect("present");

        assert_eq!(credential.source, CredentialSource::Environment);
        assert_eq!(credential.secret, "env-secret");
        std::env::remove_var("KCMT_TEST_API_KEY");
        std::env::remove_var("KCMT_DISABLE_KEYCHAIN");
    }

    #[test]
    fn credential_resolution_skips_keychain_in_runtime_benchmark_mode() {
        let _guard = env_lock().lock().unwrap_or_else(|err| err.into_inner());
        std::env::remove_var("KCMT_DISABLE_KEYCHAIN");
        std::env::set_var("KCMT_RUNTIME_BENCHMARK", "1");
        std::env::set_var("KCMT_TEST_API_KEY", "env-secret");
        let mut store = FakeStore::default();
        store.secrets.insert(
            "provider/openai/default".to_string(),
            "keychain-secret".to_string(),
        );
        let request = CredentialRequest {
            provider: "openai".to_string(),
            explicit_secret: None,
            keychain_account: None,
            env_var: "KCMT_TEST_API_KEY".to_string(),
        };

        let credential = resolve_credential(&request, &store)
            .expect("credential")
            .expect("present");

        assert_eq!(credential.source, CredentialSource::Environment);
        assert_eq!(credential.secret, "env-secret");
        std::env::remove_var("KCMT_TEST_API_KEY");
        std::env::remove_var("KCMT_RUNTIME_BENCHMARK");
    }

    #[test]
    fn credential_resolution_uses_configured_keychain_account() {
        let _guard = env_lock().lock().unwrap_or_else(|err| err.into_inner());
        std::env::remove_var("KCMT_DISABLE_KEYCHAIN");
        std::env::remove_var("KCMT_TEST_API_KEY");
        let mut store = FakeStore::default();
        store
            .secrets
            .insert("custom/account".to_string(), "custom-secret".to_string());
        let request = CredentialRequest {
            provider: "openai".to_string(),
            explicit_secret: None,
            keychain_account: Some("custom/account".to_string()),
            env_var: "KCMT_TEST_API_KEY".to_string(),
        };

        let credential = resolve_credential(&request, &store)
            .expect("credential")
            .expect("present");

        assert_eq!(credential.source, CredentialSource::Keychain);
        assert_eq!(credential.secret, "custom-secret");
    }

    #[test]
    fn default_keychain_save_mode_uses_platform_protection() {
        let _guard = env_lock().lock().unwrap_or_else(|err| err.into_inner());
        std::env::remove_var("KCMT_DISABLE_KEYCHAIN");
        let store = FakeStore::default();

        let result = save_keychain_secret(
            &store,
            "provider/openai/default",
            "secret-value",
            KeychainSaveMode::PlatformDefault,
        )
        .expect("saved");

        assert_eq!(result.protection, KeychainProtection::PlatformDefault);
        assert!(result.fallback_reason.is_none());
    }

    #[test]
    fn keychain_save_can_request_biometric_preferred_protection() {
        let _guard = env_lock().lock().unwrap_or_else(|err| err.into_inner());
        std::env::remove_var("KCMT_DISABLE_KEYCHAIN");
        let store = CapturingStore::default();

        let result = save_keychain_secret(
            &store,
            "provider/openai/default",
            "secret-value",
            KeychainSaveMode::BiometricPreferred,
        )
        .expect("saved");

        assert_eq!(result.protection, KeychainProtection::BiometricPreferred);
        assert_eq!(
            store.modes.lock().unwrap_or_else(|err| err.into_inner())[0],
            KeychainSaveMode::BiometricPreferred
        );
    }

    #[test]
    fn keychain_save_respects_disabled_keychain_flag() {
        let _guard = env_lock().lock().unwrap_or_else(|err| err.into_inner());
        std::env::set_var("KCMT_DISABLE_KEYCHAIN", "1");
        let store = CapturingStore::default();

        let result = save_keychain_secret(
            &store,
            "provider/openai/default",
            "secret-value",
            KeychainSaveMode::BiometricPreferred,
        );

        assert_eq!(
            result.expect_err("save should be disabled"),
            "OS keychain access is disabled by KCMT_DISABLE_KEYCHAIN"
        );
        std::env::remove_var("KCMT_DISABLE_KEYCHAIN");
    }

    #[test]
    fn keychain_save_reports_runtime_benchmark_disable_reason() {
        let _guard = env_lock().lock().unwrap_or_else(|err| err.into_inner());
        std::env::remove_var("KCMT_DISABLE_KEYCHAIN");
        std::env::set_var("KCMT_RUNTIME_BENCHMARK", "1");
        let store = CapturingStore::default();

        let result = save_keychain_secret(
            &store,
            "provider/openai/default",
            "secret-value",
            KeychainSaveMode::BiometricPreferred,
        );

        assert_eq!(
            result.expect_err("save should be disabled"),
            "OS keychain access is disabled during KCMT_RUNTIME_BENCHMARK"
        );
        std::env::remove_var("KCMT_RUNTIME_BENCHMARK");
    }
}
