use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum StageOutcome {
    Completed,
    Failed,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StageTiming {
    pub stage: String,
    pub start_ms: f64,
    pub end_ms: f64,
    pub duration_ms: f64,
    pub outcome: StageOutcome,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub file_path: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

impl StageTiming {
    pub fn new(
        stage: impl Into<String>,
        start_ms: f64,
        end_ms: f64,
        outcome: StageOutcome,
        file_path: Option<String>,
        error: Option<String>,
    ) -> Self {
        Self {
            stage: stage.into(),
            start_ms,
            end_ms,
            duration_ms: end_ms - start_ms,
            outcome,
            file_path,
            error: error.map(redact_sensitive_text),
        }
    }

    pub fn completed(
        stage: impl Into<String>,
        start_ms: f64,
        end_ms: f64,
        file_path: Option<String>,
    ) -> Self {
        Self::new(
            stage,
            start_ms,
            end_ms,
            StageOutcome::Completed,
            file_path,
            None,
        )
    }

    pub fn failed(
        stage: impl Into<String>,
        start_ms: f64,
        end_ms: f64,
        error: impl Into<String>,
    ) -> Self {
        Self::new(
            stage,
            start_ms,
            end_ms,
            StageOutcome::Failed,
            None,
            Some(error.into()),
        )
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowTelemetry {
    pub schema_version: u32,
    pub runtime: String,
    pub corpus_id: String,
    pub stages: Vec<StageTiming>,
}

impl WorkflowTelemetry {
    pub fn new(runtime: impl Into<String>, corpus_id: impl Into<String>) -> Self {
        Self {
            schema_version: 1,
            runtime: runtime.into(),
            corpus_id: corpus_id.into(),
            stages: Vec::new(),
        }
    }

    pub fn record(&mut self, timing: StageTiming) {
        self.stages.push(timing);
    }
}

fn redact_sensitive_text(value: String) -> String {
    value
        .replace("OPENAI_API_KEY", "[redacted]")
        .replace("ANTHROPIC_API_KEY", "[redacted]")
        .replace("XAI_API_KEY", "[redacted]")
        .replace("sk-", "[redacted]-")
}

#[cfg(test)]
mod tests {
    use super::{StageOutcome, StageTiming, WorkflowTelemetry};

    #[test]
    fn telemetry_serializes_required_stage_names_without_secret_values() {
        let mut telemetry = WorkflowTelemetry::new("rust", "synthetic-1000");
        telemetry.record(StageTiming::completed(
            "repo_discovery",
            0.0,
            2.5,
            Some("src/main.rs".to_string()),
        ));
        telemetry.record(StageTiming::failed(
            "llm_wait",
            2.5,
            5.5,
            "provider request failed",
        ));

        let rendered = serde_json::to_string(&telemetry).expect("telemetry should serialize");

        assert!(rendered.contains("\"runtime\":\"rust\""));
        assert!(rendered.contains("\"stage\":\"repo_discovery\""));
        assert!(rendered.contains("\"stage\":\"llm_wait\""));
        assert!(rendered.contains("\"outcome\":\"failed\""));
        assert!(!rendered.contains("OPENAI_API_KEY"));
        assert!(!rendered.contains("sk-"));
    }

    #[test]
    fn stage_timing_reports_duration_from_start_and_end() {
        let timing = StageTiming::new(
            "diff_preparation",
            10.0,
            27.25,
            StageOutcome::Completed,
            None,
            None,
        );

        assert_eq!(timing.duration_ms, 17.25);
    }
}
