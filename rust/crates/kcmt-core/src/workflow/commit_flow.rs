//! Commit recommendation orchestration for staged and file-scoped flows.

use crate::error::Result;
use crate::git::repo::GitRepository;
use crate::model::CommitRecommendation;
use crate::workflow::changeset::collect_changes;

pub fn recommend_commit(repo: &dyn GitRepository) -> Result<CommitRecommendation> {
    let changes = collect_changes(repo)?;
    let subject = if changes.is_empty() {
        "chore: no relevant changes".to_string()
    } else {
        "chore: update repository changes".to_string()
    };

    Ok(CommitRecommendation {
        subject: subject.clone(),
        body: None,
        raw_message: subject,
        provider_id: "none".to_string(),
        model: "heuristic".to_string(),
        success: true,
        error: None,
    })
}
