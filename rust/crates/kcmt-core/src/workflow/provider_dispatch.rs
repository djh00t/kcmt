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

    let (provider_id, model) = fallback.first().copied().unwrap_or(("openai", "gpt-4o-mini"));
    ProviderSelection {
        provider_id: provider_id.to_string(),
        model: model.to_string(),
    }
}
