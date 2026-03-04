//! Provider client module namespace.

pub mod anthropic;
pub mod github;
pub mod openai;
pub mod xai;

pub trait ProviderClient {
    fn provider_id(&self) -> &'static str;
}

#[derive(Debug, Default, Clone, Copy)]
pub struct OpenAiClient;

impl ProviderClient for OpenAiClient {
    fn provider_id(&self) -> &'static str {
        "openai"
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub struct AnthropicClient;

impl ProviderClient for AnthropicClient {
    fn provider_id(&self) -> &'static str {
        "anthropic"
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub struct XaiClient;

impl ProviderClient for XaiClient {
    fn provider_id(&self) -> &'static str {
        "xai"
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub struct GitHubModelsClient;

impl ProviderClient for GitHubModelsClient {
    fn provider_id(&self) -> &'static str {
        "github"
    }
}
