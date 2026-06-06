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

#[cfg(test)]
mod tests {
    use super::recommend_commit;
    use crate::error::{KcmtError, Result};
    use crate::git::repo::GitRepository;

    struct StubRepo {
        status: Result<String>,
    }

    impl GitRepository for StubRepo {
        fn staged_files(&self) -> Result<Vec<String>> {
            Ok(Vec::new())
        }

        fn status_porcelain(&self) -> Result<String> {
            match &self.status {
                Ok(value) => Ok(value.clone()),
                Err(KcmtError::Io(err)) => Err(KcmtError::Io(std::io::Error::new(
                    err.kind(),
                    err.to_string(),
                ))),
                Err(KcmtError::CommandFailure {
                    command,
                    status,
                    stderr,
                }) => Err(KcmtError::CommandFailure {
                    command: command.clone(),
                    status: *status,
                    stderr: stderr.clone(),
                }),
                Err(KcmtError::Message(message)) => Err(KcmtError::Message(message.clone())),
            }
        }

        fn status_porcelain_for_path(&self, _file_path: &str) -> Result<String> {
            Ok(String::new())
        }
    }

    #[test]
    fn recommend_commit_returns_no_relevant_changes_for_empty_status() {
        let recommendation = recommend_commit(&StubRepo {
            status: Ok(String::new()),
        })
        .expect("recommendation should succeed");

        assert_eq!(recommendation.subject, "chore: no relevant changes");
        assert_eq!(recommendation.raw_message, recommendation.subject);
        assert_eq!(recommendation.provider_id, "none");
        assert_eq!(recommendation.model, "heuristic");
        assert!(recommendation.success);
        assert!(recommendation.body.is_none());
        assert!(recommendation.error.is_none());
    }

    #[test]
    fn recommend_commit_returns_update_message_for_non_empty_status() {
        let recommendation = recommend_commit(&StubRepo {
            status: Ok(" M tracked.py\n".to_string()),
        })
        .expect("recommendation should succeed");

        assert_eq!(recommendation.subject, "chore: update repository changes");
        assert_eq!(recommendation.raw_message, recommendation.subject);
    }

    #[test]
    fn recommend_commit_propagates_status_error() {
        let err = recommend_commit(&StubRepo {
            status: Err(KcmtError::Message("status failed".to_string())),
        })
        .expect_err("status failure should propagate");

        match err {
            KcmtError::Message(message) => assert_eq!(message, "status failed"),
            other => panic!("expected message error, got {other:?}"),
        }
    }
}
