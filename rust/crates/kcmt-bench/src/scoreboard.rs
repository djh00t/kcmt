use serde::{Deserialize, Serialize};

use crate::model::{RuntimeBenchmarkResult, RuntimeKind, RuntimeScenarioStatus};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ComparisonStatus {
    Faster,
    Slower,
    Same,
    Missing,
    NotComparable,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoreboardRow {
    pub stage: String,
    pub python_median_ms: Option<f64>,
    pub rust_median_ms: Option<f64>,
    pub delta_ms: Option<f64>,
    pub rust_change_percent: Option<f64>,
    pub quality_impact: String,
    pub comparable: bool,
    pub status: ComparisonStatus,
    pub notes: String,
}

impl ScoreboardRow {
    pub fn comparable(
        stage: impl Into<String>,
        python_median_ms: Option<f64>,
        rust_median_ms: Option<f64>,
        quality_impact: impl Into<String>,
    ) -> Self {
        let delta_ms = match (python_median_ms, rust_median_ms) {
            (Some(python), Some(rust)) => Some(rust - python),
            _ => None,
        };
        let rust_change_percent = match (python_median_ms, rust_median_ms) {
            (Some(python), Some(rust)) if python != 0.0 => Some(((rust - python) / python) * 100.0),
            _ => None,
        };
        let status = match delta_ms {
            Some(delta) if delta < 0.0 => ComparisonStatus::Faster,
            Some(delta) if delta > 0.0 => ComparisonStatus::Slower,
            Some(_) => ComparisonStatus::Same,
            None => ComparisonStatus::Missing,
        };
        Self {
            stage: stage.into(),
            python_median_ms,
            rust_median_ms,
            delta_ms,
            rust_change_percent,
            quality_impact: quality_impact.into(),
            comparable: true,
            status,
            notes: String::new(),
        }
    }

    pub fn not_comparable(
        stage: impl Into<String>,
        python_median_ms: Option<f64>,
        rust_median_ms: Option<f64>,
        notes: impl Into<String>,
    ) -> Self {
        Self {
            stage: stage.into(),
            python_median_ms,
            rust_median_ms,
            delta_ms: None,
            rust_change_percent: None,
            quality_impact: "neutral".to_string(),
            comparable: false,
            status: ComparisonStatus::NotComparable,
            notes: notes.into(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowComparisonScoreboard {
    pub rows: Vec<ScoreboardRow>,
}

impl WorkflowComparisonScoreboard {
    pub fn new(rows: Vec<ScoreboardRow>) -> Self {
        Self { rows }
    }

    pub fn from_results(results: &[RuntimeBenchmarkResult]) -> Self {
        let mut stages = Vec::new();
        for result in results {
            for timing in &result.stage_timings {
                if !stages.iter().any(|stage| stage == &timing.stage) {
                    stages.push(timing.stage.clone());
                }
            }
        }
        stages.sort();

        let rows = stages
            .into_iter()
            .map(|stage| {
                let python = median_stage_duration(results, RuntimeKind::Python, &stage);
                let rust = median_stage_duration(results, RuntimeKind::Rust, &stage);
                if python.is_some() && rust.is_some() {
                    ScoreboardRow::comparable(stage, python, rust, "neutral")
                } else {
                    let notes = match (python, rust) {
                        (Some(_), None) => "Rust stage missing",
                        (None, Some(_)) => "Python stage missing",
                        (None, None) => "No stage timing available",
                        (Some(_), Some(_)) => "",
                    };
                    ScoreboardRow::not_comparable(stage, python, rust, notes)
                }
            })
            .collect();
        Self { rows }
    }
}

fn median_stage_duration(
    results: &[RuntimeBenchmarkResult],
    runtime: RuntimeKind,
    stage: &str,
) -> Option<f64> {
    let mut values: Vec<f64> = results
        .iter()
        .filter(|result| {
            result.runtime == runtime && result.status != RuntimeScenarioStatus::Excluded
        })
        .flat_map(|result| {
            result
                .stage_timings
                .iter()
                .filter(move |timing| timing.stage == stage)
                .map(|timing| timing.duration_ms)
        })
        .collect();
    if values.is_empty() {
        return None;
    }
    values.sort_by(|left, right| left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal));
    let mid = values.len() / 2;
    if values.len() % 2 == 0 {
        Some((values[mid - 1] + values[mid]) / 2.0)
    } else {
        Some(values[mid])
    }
}

#[cfg(test)]
mod tests {
    use crate::model::{RuntimeBenchmarkResult, RuntimeKind, RuntimeScenarioStatus};

    use super::{ComparisonStatus, ScoreboardRow, WorkflowComparisonScoreboard};

    #[test]
    fn scoreboard_marks_rust_faster_with_delta_and_percent_change() {
        let row = ScoreboardRow::comparable("status_scan", Some(100.0), Some(40.0), "neutral");

        assert_eq!(row.delta_ms, Some(-60.0));
        assert_eq!(row.rust_change_percent, Some(-60.0));
        assert_eq!(row.status, ComparisonStatus::Faster);
        assert!(row.comparable);
    }

    #[test]
    fn scoreboard_marks_missing_stage_as_not_comparable() {
        let row = ScoreboardRow::not_comparable(
            "python_tui_probe",
            Some(25.0),
            None,
            "Python wrapper only",
        );

        assert_eq!(row.delta_ms, None);
        assert_eq!(row.rust_change_percent, None);
        assert_eq!(row.status, ComparisonStatus::NotComparable);
        assert!(!row.comparable);
        assert_eq!(row.notes, "Python wrapper only");
    }

    #[test]
    fn scoreboard_contains_runtime_rows_in_order() {
        let scoreboard = WorkflowComparisonScoreboard::new(vec![
            ScoreboardRow::comparable("repo_discovery", Some(8.0), Some(4.0), "neutral"),
            ScoreboardRow::not_comparable(
                "python_tui_probe",
                Some(12.0),
                None,
                "Python wrapper only",
            ),
        ]);

        assert_eq!(scoreboard.rows[0].stage, "repo_discovery");
        assert_eq!(scoreboard.rows[1].stage, "python_tui_probe");
    }

    #[test]
    fn scoreboard_builds_rows_from_runtime_stage_timings() {
        let results = vec![
            RuntimeBenchmarkResult {
                scenario_id: "status".to_string(),
                workflow_contract_id: "status-repo-path".to_string(),
                corpus_id: "corpus".to_string(),
                runtime: RuntimeKind::Python,
                command_label: "status".to_string(),
                iterations: 1,
                status: RuntimeScenarioStatus::Passed,
                wall_time_ms: 100.0,
                median_time_ms: Some(100.0),
                peak_rss_bytes: None,
                exit_code: Some(0),
                failure_reason: None,
                stage_timings: vec![crate::model::RuntimeStageTiming::new(
                    "status_scan",
                    100.0,
                    RuntimeScenarioStatus::Passed,
                    None,
                )],
            },
            RuntimeBenchmarkResult {
                scenario_id: "status".to_string(),
                workflow_contract_id: "status-repo-path".to_string(),
                corpus_id: "corpus".to_string(),
                runtime: RuntimeKind::Rust,
                command_label: "status".to_string(),
                iterations: 1,
                status: RuntimeScenarioStatus::Passed,
                wall_time_ms: 40.0,
                median_time_ms: Some(40.0),
                peak_rss_bytes: None,
                exit_code: Some(0),
                failure_reason: None,
                stage_timings: vec![crate::model::RuntimeStageTiming::new(
                    "status_scan",
                    40.0,
                    RuntimeScenarioStatus::Passed,
                    None,
                )],
            },
        ];

        let scoreboard = WorkflowComparisonScoreboard::from_results(&results);

        assert_eq!(scoreboard.rows.len(), 1);
        assert_eq!(scoreboard.rows[0].status, ComparisonStatus::Faster);
        assert_eq!(scoreboard.rows[0].rust_change_percent, Some(-60.0));
    }
}
