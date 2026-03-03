//! Provider error normalization with secret-safe messaging.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProviderFailureKind {
    Timeout,
    RateLimited,
    MalformedResponse,
    Unauthorized,
    Unknown,
}

#[derive(Debug, Clone)]
pub struct NormalizedProviderError {
    pub kind: ProviderFailureKind,
    pub message: String,
}

pub fn normalize_error(raw_message: &str, status: Option<u16>) -> NormalizedProviderError {
    let kind = match status {
        Some(401 | 403) => ProviderFailureKind::Unauthorized,
        Some(429) => ProviderFailureKind::RateLimited,
        Some(500..=599) => ProviderFailureKind::Timeout,
        _ if raw_message.to_lowercase().contains("timeout") => ProviderFailureKind::Timeout,
        _ if raw_message.to_lowercase().contains("parse") => ProviderFailureKind::MalformedResponse,
        _ => ProviderFailureKind::Unknown,
    };

    NormalizedProviderError {
        kind,
        message: "Provider request failed; check provider settings and retry.".to_string(),
    }
}
