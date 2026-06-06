//! Provider dispatch bridge used by commit flow.

#[derive(Debug, Clone)]
pub struct ProviderSelection {
    pub provider_id: String,
    pub model: String,
}

pub fn select_provider(preferred: Option<&str>, fallback: &[(&str, &str)]) -> ProviderSelection {
    if let Some(id) = preferred {
        if let Some((provider_id, model)) = fallback.iter().find(|(pid, _)| *pid == id) {
            return ProviderSelection {
                provider_id: (*provider_id).to_string(),
                model: (*model).to_string(),
            };
        }
    }

    let (provider_id, model) = fallback
        .first()
        .copied()
        .unwrap_or(("openai", "gpt-4o-mini"));
    ProviderSelection {
        provider_id: provider_id.to_string(),
        model: model.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::select_provider;

    #[test]
    fn select_provider_returns_preferred_match() {
        let selection = select_provider(
            Some("anthropic"),
            &[("openai", "gpt-4o-mini"), ("anthropic", "claude-sonnet")],
        );

        assert_eq!(selection.provider_id, "anthropic");
        assert_eq!(selection.model, "claude-sonnet");
    }

    #[test]
    fn select_provider_falls_back_when_preferred_missing() {
        let selection = select_provider(
            Some("xai"),
            &[("openai", "gpt-4o-mini"), ("anthropic", "claude-sonnet")],
        );

        assert_eq!(selection.provider_id, "openai");
        assert_eq!(selection.model, "gpt-4o-mini");
    }

    #[test]
    fn select_provider_uses_default_when_fallback_empty() {
        let selection = select_provider(None, &[]);

        assert_eq!(selection.provider_id, "openai");
        assert_eq!(selection.model, "gpt-4o-mini");
    }
}
