//! Provider profile and resolution helpers.

#[derive(Debug, Clone)]
pub struct ProviderProfile {
    pub provider_id: String,
    pub endpoint: String,
    pub api_key_env: String,
    pub preferred_model: Option<String>,
}

impl ProviderProfile {
    pub fn resolve_model(&self, fallback_model: &str) -> String {
        self.preferred_model
            .clone()
            .unwrap_or_else(|| fallback_model.to_string())
    }
}
