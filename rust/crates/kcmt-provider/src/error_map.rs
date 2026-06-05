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
        message: render_provider_error_message(raw_message, status, kind),
    }
}

fn render_provider_error_message(
    raw_message: &str,
    status: Option<u16>,
    kind: ProviderFailureKind,
) -> String {
    let detail = redact_secret_like_values(raw_message)
        .chars()
        .filter(|ch| !ch.is_control() || *ch == '\n' || *ch == '\t')
        .take(1000)
        .collect::<String>();
    let kind_label = match kind {
        ProviderFailureKind::Timeout => "timeout/transient",
        ProviderFailureKind::RateLimited => "rate limited",
        ProviderFailureKind::MalformedResponse => "malformed response",
        ProviderFailureKind::Unauthorized => "unauthorized",
        ProviderFailureKind::Unknown => "request failed",
    };
    match (status, detail.trim().is_empty()) {
        (Some(status), false) => format!("{kind_label} (HTTP {status}): {detail}"),
        (Some(status), true) => format!("{kind_label} (HTTP {status})"),
        (None, false) => format!("{kind_label}: {detail}"),
        (None, true) => "Provider request failed; check provider settings and retry.".to_string(),
    }
}

fn redact_secret_like_values(raw: &str) -> String {
    let mut redacted = raw.to_string();
    for marker in ["sk-", "sk_live_", "sk_test_"] {
        while let Some(start) = redacted.find(marker) {
            let end = redacted[start..]
                .find(|ch: char| ch.is_whitespace() || matches!(ch, '"' | '\'' | ',' | '}'))
                .map(|offset| start + offset)
                .unwrap_or(redacted.len());
            redacted.replace_range(start..end, "[redacted]");
        }
    }
    redacted
}

#[cfg(test)]
mod tests {
    use super::{normalize_error, ProviderFailureKind};

    #[test]
    fn preserves_provider_error_detail_without_secret_values() {
        let error = normalize_error(
            "provider request failed with status 400 Bad Request: {\"error\":{\"message\":\"invalid model\",\"api_key\":\"sk-test-secret\"}}",
            Some(400),
        );

        assert_eq!(error.kind, ProviderFailureKind::Unknown);
        assert!(error.message.contains("HTTP 400"));
        assert!(error.message.contains("invalid model"));
        assert!(!error.message.contains("sk-test-secret"));
        assert!(error.message.contains("[redacted]"));
    }
}
