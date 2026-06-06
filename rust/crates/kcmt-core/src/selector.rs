//! Model metadata, provider rules, and selection policy.

use std::cmp::Ordering;

use serde::{Deserialize, Serialize};

use crate::model::WorkflowConfig;
use crate::preferences::{Preferences, ProviderRule, ProviderRulePreset, SelectionPolicy};
use crate::telemetry::UsageSummary;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct ModelCandidate {
    pub provider: String,
    pub id: String,
    pub family: Option<String>,
    pub code_capable: bool,
    pub beta: bool,
    pub input_cost_per_million: Option<f64>,
    pub output_cost_per_million: Option<f64>,
    pub median_latency_ms: Option<f64>,
    pub created_at: Option<String>,
}

impl Default for ModelCandidate {
    fn default() -> Self {
        Self {
            provider: String::new(),
            id: String::new(),
            family: None,
            code_capable: true,
            beta: false,
            input_cost_per_million: None,
            output_cost_per_million: None,
            median_latency_ms: None,
            created_at: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ModelSelection {
    pub provider: String,
    pub model: String,
    pub rule_applied: Option<ProviderRulePreset>,
    pub fallback_reason: Option<String>,
}

pub fn select_model(
    config: &WorkflowConfig,
    preferences: &Preferences,
    live_models: &[ModelCandidate],
    telemetry: &UsageSummary,
    available_providers: &[String],
) -> ModelSelection {
    let mut providers = available_providers.to_vec();
    if providers.is_empty() {
        providers.push(config.provider.clone());
    } else if available_providers
        .iter()
        .any(|provider| provider == &config.provider)
        && providers.first() != Some(&config.provider)
    {
        providers.insert(0, config.provider.clone());
    }

    let mut fallback_reason = None;
    for provider in providers {
        let provider_models: Vec<ModelCandidate> = live_models
            .iter()
            .filter(|model| model.provider == provider)
            .cloned()
            .collect();
        let defaults = default_models_for_provider(&provider);
        let candidates = if provider_models.is_empty() {
            defaults
        } else {
            provider_models
        };
        if candidates.is_empty() {
            continue;
        }
        let rule = preferences.provider_rules.get(&provider);
        let (filtered, rule_reason) = apply_provider_rule(&provider, &candidates, rule);
        if let Some(reason) = rule_reason {
            fallback_reason = Some(reason);
        }
        if filtered.is_empty() {
            if rule.is_some_and(|rule| rule.strict) {
                continue;
            }
            let ranked = rank_candidates(&candidates, &preferences.selection_policy, telemetry);
            if let Some(selected) = ranked.first() {
                return ModelSelection {
                    provider,
                    model: selected.id.clone(),
                    rule_applied: rule.map(|rule| rule.preset.clone()),
                    fallback_reason,
                };
            }
        } else {
            let ranked = rank_candidates(&filtered, &preferences.selection_policy, telemetry);
            if let Some(selected) = ranked.first() {
                return ModelSelection {
                    provider,
                    model: selected.id.clone(),
                    rule_applied: rule.map(|rule| rule.preset.clone()),
                    fallback_reason,
                };
            }
        }
    }

    ModelSelection {
        provider: config.provider.clone(),
        model: config.model.clone(),
        rule_applied: None,
        fallback_reason: Some("no selectable live model; using configured model".to_string()),
    }
}

pub fn apply_provider_rule(
    provider: &str,
    candidates: &[ModelCandidate],
    rule: Option<&ProviderRule>,
) -> (Vec<ModelCandidate>, Option<String>) {
    let Some(rule) = rule else {
        return (candidates.to_vec(), None);
    };
    let filtered = match rule.preset {
        ProviderRulePreset::None => candidates.to_vec(),
        ProviderRulePreset::PinExactModel => rule
            .value
            .as_ref()
            .map(|value| {
                candidates
                    .iter()
                    .filter(|candidate| candidate.id == *value)
                    .cloned()
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default(),
        ProviderRulePreset::LatestFamily => {
            let family = rule.value.as_deref().unwrap_or_default();
            newest_matching(candidates, |candidate| {
                candidate
                    .family
                    .as_deref()
                    .is_some_and(|value| value.eq_ignore_ascii_case(family))
                    || candidate
                        .id
                        .to_ascii_lowercase()
                        .contains(&family.to_ascii_lowercase())
            })
        }
        ProviderRulePreset::LatestHaiku => newest_matching(candidates, |candidate| {
            candidate.id.to_ascii_lowercase().contains("haiku")
                || candidate
                    .family
                    .as_deref()
                    .is_some_and(|value| value.eq_ignore_ascii_case("haiku"))
        }),
        ProviderRulePreset::CheapestCodeModel => cheapest(candidates),
        ProviderRulePreset::FastestCodeModel => fastest(candidates),
        ProviderRulePreset::ExcludeBeta => candidates
            .iter()
            .filter(|candidate| {
                !candidate.beta && !candidate.id.to_ascii_lowercase().contains("beta")
            })
            .cloned()
            .collect(),
    };
    let reason = if filtered.is_empty() {
        Some(format!(
            "provider rule {:?} matched no models for {provider}",
            rule.preset
        ))
    } else {
        None
    };
    (filtered, reason)
}

pub fn default_models_for_provider(provider: &str) -> Vec<ModelCandidate> {
    let id = match provider {
        "anthropic" => "claude-3-5-haiku-latest",
        "xai" => "grok-code-fast",
        "github" => "openai/gpt-4.1-mini",
        _ => "gpt-5-mini-2025-08-07",
    };
    vec![ModelCandidate {
        provider: provider.to_string(),
        id: id.to_string(),
        family: infer_family(id),
        code_capable: true,
        beta: id.contains("beta"),
        input_cost_per_million: None,
        output_cost_per_million: None,
        median_latency_ms: None,
        created_at: None,
    }]
}

fn newest_matching(
    candidates: &[ModelCandidate],
    predicate: impl Fn(&ModelCandidate) -> bool,
) -> Vec<ModelCandidate> {
    let mut matches: Vec<ModelCandidate> = candidates
        .iter()
        .filter(|candidate| predicate(candidate))
        .cloned()
        .collect();
    matches.sort_by(|left, right| compare_newest(left, right));
    matches.into_iter().take(1).collect()
}

fn cheapest(candidates: &[ModelCandidate]) -> Vec<ModelCandidate> {
    let mut matches: Vec<ModelCandidate> = candidates
        .iter()
        .filter(|candidate| candidate.code_capable)
        .cloned()
        .collect();
    matches.sort_by(|left, right| compare_cost(left, right));
    matches.into_iter().take(1).collect()
}

fn fastest(candidates: &[ModelCandidate]) -> Vec<ModelCandidate> {
    let mut matches: Vec<ModelCandidate> = candidates
        .iter()
        .filter(|candidate| candidate.code_capable)
        .cloned()
        .collect();
    matches.sort_by(|left, right| compare_latency(left, right));
    matches.into_iter().take(1).collect()
}

fn rank_candidates(
    candidates: &[ModelCandidate],
    policy: &SelectionPolicy,
    telemetry: &UsageSummary,
) -> Vec<ModelCandidate> {
    let mut ranked = candidates.to_vec();
    ranked.sort_by(|left, right| match policy {
        SelectionPolicy::FastestCheap => compare_fastest_cheap(left, right, telemetry),
        SelectionPolicy::Balanced => compare_newest(left, right),
        SelectionPolicy::BestQuality => compare_quality_hint(left, right),
    });
    ranked
}

fn compare_fastest_cheap(
    left: &ModelCandidate,
    right: &ModelCandidate,
    telemetry: &UsageSummary,
) -> Ordering {
    compare_latency_with_telemetry(left, right, telemetry)
        .then_with(|| compare_cost(left, right))
        .then_with(|| compare_quality_hint(left, right))
}

fn compare_latency_with_telemetry(
    left: &ModelCandidate,
    right: &ModelCandidate,
    telemetry: &UsageSummary,
) -> Ordering {
    let left_latency = telemetry
        .latency_for(&left.provider, &left.id)
        .or(left.median_latency_ms)
        .unwrap_or(f64::MAX);
    let right_latency = telemetry
        .latency_for(&right.provider, &right.id)
        .or(right.median_latency_ms)
        .unwrap_or(f64::MAX);
    left_latency
        .partial_cmp(&right_latency)
        .unwrap_or(Ordering::Equal)
}

fn compare_latency(left: &ModelCandidate, right: &ModelCandidate) -> Ordering {
    left.median_latency_ms
        .unwrap_or(f64::MAX)
        .partial_cmp(&right.median_latency_ms.unwrap_or(f64::MAX))
        .unwrap_or(Ordering::Equal)
}

fn compare_cost(left: &ModelCandidate, right: &ModelCandidate) -> Ordering {
    model_cost(left)
        .partial_cmp(&model_cost(right))
        .unwrap_or(Ordering::Equal)
}

fn compare_quality_hint(left: &ModelCandidate, right: &ModelCandidate) -> Ordering {
    right
        .code_capable
        .cmp(&left.code_capable)
        .then_with(|| left.id.cmp(&right.id))
}

fn compare_newest(left: &ModelCandidate, right: &ModelCandidate) -> Ordering {
    right
        .created_at
        .cmp(&left.created_at)
        .then_with(|| right.id.cmp(&left.id))
}

fn model_cost(model: &ModelCandidate) -> f64 {
    model.input_cost_per_million.unwrap_or(1_000_000.0)
        + model.output_cost_per_million.unwrap_or(1_000_000.0)
}

fn infer_family(model_id: &str) -> Option<String> {
    let lower = model_id.to_ascii_lowercase();
    for family in ["haiku", "sonnet", "opus", "gpt", "grok"] {
        if lower.contains(family) {
            return Some(family.to_string());
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::{apply_provider_rule, select_model, ModelCandidate};
    use crate::model::WorkflowConfig;
    use crate::preferences::{Preferences, ProviderRule, ProviderRulePreset};
    use crate::telemetry::{TelemetryAggregate, UsageSummary};

    fn model(
        provider: &str,
        id: &str,
        latency: f64,
        cost: f64,
        created_at: &str,
    ) -> ModelCandidate {
        ModelCandidate {
            provider: provider.to_string(),
            id: id.to_string(),
            family: None,
            code_capable: true,
            beta: id.contains("beta"),
            input_cost_per_million: Some(cost),
            output_cost_per_million: Some(cost),
            median_latency_ms: Some(latency),
            created_at: Some(created_at.to_string()),
        }
    }

    #[test]
    fn latest_haiku_rule_selects_newest_haiku() {
        let candidates = vec![
            model(
                "anthropic",
                "claude-3-haiku-20240101",
                900.0,
                1.0,
                "2024-01-01",
            ),
            model(
                "anthropic",
                "claude-3-5-haiku-20250601",
                950.0,
                1.2,
                "2025-06-01",
            ),
            model(
                "anthropic",
                "claude-3-7-sonnet-20250501",
                800.0,
                2.0,
                "2025-05-01",
            ),
        ];
        let rule = ProviderRule {
            preset: ProviderRulePreset::LatestHaiku,
            value: None,
            strict: false,
        };

        let (selected, reason) = apply_provider_rule("anthropic", &candidates, Some(&rule));

        assert_eq!(reason, None);
        assert_eq!(selected.len(), 1);
        assert_eq!(selected[0].id, "claude-3-5-haiku-20250601");
    }

    #[test]
    fn fastest_cheap_policy_uses_telemetry_latency_first() {
        let config = WorkflowConfig {
            provider: "openai".to_string(),
            model: "slow".to_string(),
            ..WorkflowConfig::default()
        };
        let preferences = Preferences::default();
        let models = vec![
            model("openai", "cheap-slow", 1000.0, 0.1, "2025-01-01"),
            model("openai", "fast-expensive", 100.0, 10.0, "2025-01-01"),
        ];
        let telemetry = UsageSummary {
            aggregates: vec![TelemetryAggregate {
                provider: "openai".to_string(),
                model: "cheap-slow".to_string(),
                selected_rule: None,
                runs: 3,
                successes: 3,
                failures: 0,
                avg_latency_ms: 50.0,
                fallback_count: 0,
                request_count: 3,
            }],
        };

        let selected = select_model(
            &config,
            &preferences,
            &models,
            &telemetry,
            &["openai".to_string()],
        );

        assert_eq!(selected.model, "cheap-slow");
    }

    #[test]
    fn non_strict_unmatched_rule_falls_back_with_reason() {
        let config = WorkflowConfig {
            provider: "anthropic".to_string(),
            model: "claude-3-5-haiku-latest".to_string(),
            ..WorkflowConfig::default()
        };
        let mut preferences = Preferences::default();
        preferences.provider_rules.insert(
            "anthropic".to_string(),
            ProviderRule {
                preset: ProviderRulePreset::PinExactModel,
                value: Some("missing-model".to_string()),
                strict: false,
            },
        );
        let models = vec![model(
            "anthropic",
            "claude-3-5-haiku-latest",
            100.0,
            1.0,
            "2025-01-01",
        )];

        let selected = select_model(
            &config,
            &preferences,
            &models,
            &UsageSummary::default(),
            &["anthropic".to_string()],
        );

        assert_eq!(selected.model, "claude-3-5-haiku-latest");
        assert!(selected
            .fallback_reason
            .as_deref()
            .is_some_and(|reason| reason.contains("matched no models")));
    }

    #[test]
    fn strict_unmatched_rule_skips_to_configured_fallback() {
        let config = WorkflowConfig {
            provider: "anthropic".to_string(),
            model: "configured-fallback".to_string(),
            ..WorkflowConfig::default()
        };
        let mut preferences = Preferences::default();
        preferences.provider_rules.insert(
            "anthropic".to_string(),
            ProviderRule {
                preset: ProviderRulePreset::PinExactModel,
                value: Some("missing-model".to_string()),
                strict: true,
            },
        );
        let models = vec![model(
            "anthropic",
            "claude-3-5-haiku-latest",
            100.0,
            1.0,
            "2025-01-01",
        )];

        let selected = select_model(
            &config,
            &preferences,
            &models,
            &UsageSummary::default(),
            &["anthropic".to_string()],
        );

        assert_eq!(selected.model, "configured-fallback");
        assert!(selected.fallback_reason.is_some());
    }
}
