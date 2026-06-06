//! Local aggregate usage telemetry.

use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::config::loader::config_home;
use crate::error::{KcmtError, Result};

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
#[serde(default)]
pub struct UsageSummary {
    pub aggregates: Vec<TelemetryAggregate>,
}

impl UsageSummary {
    pub fn latency_for(&self, provider: &str, model: &str) -> Option<f64> {
        self.aggregates
            .iter()
            .find(|entry| entry.provider == provider && entry.model == model && entry.runs > 0)
            .map(|entry| entry.avg_latency_ms)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
#[serde(default)]
pub struct TelemetryAggregate {
    pub provider: String,
    pub model: String,
    pub selected_rule: Option<String>,
    pub runs: u64,
    pub successes: u64,
    pub failures: u64,
    pub avg_latency_ms: f64,
    pub fallback_count: u64,
    pub request_count: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TelemetryRunRecord {
    pub provider: String,
    pub model: String,
    pub selected_rule: Option<String>,
    pub success: bool,
    pub latency_ms: f64,
    pub fallback_count: u64,
    pub request_count: u64,
}

pub fn usage_summary_path(repo_path: &Path) -> PathBuf {
    state_dir(repo_path).join("usage_summary.json")
}

pub fn load_usage_summary(repo_path: &Path) -> Result<UsageSummary> {
    let path = usage_summary_path(repo_path);
    if !path.exists() {
        return Ok(UsageSummary::default());
    }
    let raw = fs::read_to_string(path)?;
    serde_json::from_str(&raw).map_err(|err| KcmtError::Message(err.to_string()))
}

pub fn record_usage(repo_path: &Path, record: &TelemetryRunRecord) -> Result<PathBuf> {
    let mut summary = load_usage_summary(repo_path)?;
    let aggregate = summary.aggregates.iter_mut().find(|entry| {
        entry.provider == record.provider
            && entry.model == record.model
            && entry.selected_rule == record.selected_rule
    });
    if let Some(aggregate) = aggregate {
        let previous_runs = aggregate.runs;
        aggregate.runs += 1;
        aggregate.successes += u64::from(record.success);
        aggregate.failures += u64::from(!record.success);
        aggregate.avg_latency_ms = ((aggregate.avg_latency_ms * previous_runs as f64)
            + record.latency_ms)
            / aggregate.runs as f64;
        aggregate.fallback_count += record.fallback_count;
        aggregate.request_count += record.request_count;
    } else {
        summary.aggregates.push(TelemetryAggregate {
            provider: record.provider.clone(),
            model: record.model.clone(),
            selected_rule: record.selected_rule.clone(),
            runs: 1,
            successes: u64::from(record.success),
            failures: u64::from(!record.success),
            avg_latency_ms: record.latency_ms,
            fallback_count: record.fallback_count,
            request_count: record.request_count,
        });
    }
    save_usage_summary(repo_path, &summary)
}

pub fn save_usage_summary(repo_path: &Path, summary: &UsageSummary) -> Result<PathBuf> {
    let path = usage_summary_path(repo_path);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let rendered =
        serde_json::to_string_pretty(summary).map_err(|err| KcmtError::Message(err.to_string()))?;
    fs::write(&path, rendered)?;
    Ok(path)
}

fn state_dir(repo_path: &Path) -> PathBuf {
    config_home().join("repos").join(repo_namespace(repo_path))
}

fn repo_namespace(repo_path: &Path) -> String {
    let normalized = if repo_path.is_absolute() {
        repo_path.to_path_buf()
    } else {
        std::env::current_dir()
            .unwrap_or_else(|_| PathBuf::from("."))
            .join(repo_path)
    };
    let digest = Sha256::digest(normalized.to_string_lossy().as_bytes());
    let digest_hex = format!("{digest:x}");
    let safe_tail = normalized
        .file_name()
        .and_then(|name| name.to_str())
        .map(|raw| {
            raw.chars()
                .map(|ch| {
                    if ch.is_ascii_alphanumeric() || matches!(ch, '.' | '_' | '-') {
                        ch
                    } else {
                        '-'
                    }
                })
                .collect::<String>()
        })
        .filter(|tail| !tail.is_empty())
        .unwrap_or_else(|| "repo".to_string());
    format!("{}-{}", safe_tail, &digest_hex[..8])
}

#[cfg(test)]
mod tests {
    use super::{record_usage, TelemetryRunRecord};
    use std::fs;
    use std::path::PathBuf;
    use std::sync::{Mutex, OnceLock};
    use std::time::{SystemTime, UNIX_EPOCH};

    fn env_lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

    fn unique_temp_dir(label: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock should be after unix epoch")
            .as_nanos();
        let path = std::env::temp_dir().join(format!("kcmt-telemetry-{label}-{nanos}"));
        fs::create_dir_all(&path).expect("temp dir should be created");
        path
    }

    #[test]
    fn aggregate_usage_does_not_store_payload_content() {
        let _guard = env_lock().lock().unwrap_or_else(|err| err.into_inner());
        let config_home = unique_temp_dir("home");
        let repo = unique_temp_dir("repo");
        std::env::set_var("KCMT_CONFIG_HOME", &config_home);

        record_usage(
            &repo,
            &TelemetryRunRecord {
                provider: "anthropic".to_string(),
                model: "claude-3-5-haiku-latest".to_string(),
                selected_rule: Some("latest_haiku".to_string()),
                success: true,
                latency_ms: 42.0,
                fallback_count: 0,
                request_count: 1,
            },
        )
        .expect("record usage");

        let raw = fs::read_to_string(
            config_home
                .join("repos")
                .read_dir()
                .expect("repos dir")
                .next()
                .expect("repo entry")
                .expect("repo entry")
                .path()
                .join("usage_summary.json"),
        )
        .expect("summary");
        assert!(raw.contains("latest_haiku"));
        assert!(!raw.contains("DIFF:"));
        assert!(!raw.contains("Generate a conventional commit"));
        std::env::remove_var("KCMT_CONFIG_HOME");
    }
}
