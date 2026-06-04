//! OpenAI-compatible chat completion client for commit-message generation.

use std::time::Duration;

use anyhow::{anyhow, Result};
use reqwest::header::{HeaderMap, HeaderValue, AUTHORIZATION};
use serde_json::{json, Value};

use crate::transport::{AsyncTransport, RetryPolicy};

#[derive(Clone)]
pub struct OpenAiCommitClient {
    endpoint: String,
    model: String,
    api_key: String,
    transport: AsyncTransport,
}

impl OpenAiCommitClient {
    pub fn new(
        endpoint: impl Into<String>,
        model: impl Into<String>,
        api_key: impl Into<String>,
    ) -> Result<Self> {
        Ok(Self {
            endpoint: endpoint.into(),
            model: model.into(),
            api_key: api_key.into(),
            transport: AsyncTransport::new(
                Duration::from_secs(30),
                RetryPolicy {
                    max_attempts: 3,
                    base_backoff: Duration::from_millis(250),
                },
            )?,
        })
    }

    pub async fn generate_commit_message(&self, diff: &str, context: &str) -> Result<String> {
        let payload = self.chat_completion_payload(diff, context);
        let mut headers = HeaderMap::new();
        let bearer = format!("Bearer {}", self.api_key);
        headers.insert(
            AUTHORIZATION,
            HeaderValue::from_str(&bearer)
                .map_err(|err| anyhow!("invalid OpenAI authorization header: {err}"))?,
        );
        let response = self
            .transport
            .post_json(
                &format!("{}/chat/completions", self.endpoint.trim_end_matches('/')),
                headers,
                &payload,
            )
            .await?;
        parse_chat_completion_message(&response)
    }

    pub fn chat_completion_payload(&self, diff: &str, context: &str) -> Value {
        json!({
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "Generate exactly one Conventional Commit message. Return only the commit message."
                },
                {
                    "role": "user",
                    "content": build_commit_prompt(diff, context)
                }
            ],
            "temperature": 0.2,
            "max_completion_tokens": 256
        })
    }
}

pub fn build_commit_prompt(diff: &str, context: &str) -> String {
    let mut prompt = vec![
        "Generate a conventional commit message for these changes:".to_string(),
        String::new(),
        "DIFF:".to_string(),
        truncate_diff(diff),
    ];
    if !context.trim().is_empty() {
        prompt.extend([String::new(), "CONTEXT:".to_string(), context.to_string()]);
    }
    prompt.extend([
        String::new(),
        "STRICT REQUIREMENTS:".to_string(),
        "- MUST use format: type(scope): description".to_string(),
        "- MUST use a valid Conventional Commit type".to_string(),
        "- MUST keep the first line concise".to_string(),
        "- MUST NOT include markdown fences, labels, or explanations".to_string(),
    ]);
    prompt.join("\n")
}

fn truncate_diff(diff: &str) -> String {
    if diff.len() <= 12_000 {
        return diff.to_string();
    }
    format!("{}\n...\n{}", &diff[..8_000], &diff[diff.len() - 2_000..])
}

pub fn parse_chat_completion_message(response: &Value) -> Result<String> {
    response
        .get("choices")
        .and_then(Value::as_array)
        .and_then(|choices| choices.first())
        .and_then(|choice| choice.get("message"))
        .and_then(|message| message.get("content"))
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|content| !content.is_empty())
        .map(ToOwned::to_owned)
        .ok_or_else(|| anyhow!("OpenAI response did not contain a commit message"))
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::{build_commit_prompt, parse_chat_completion_message, OpenAiCommitClient};

    #[test]
    fn builds_chat_completion_payload_with_model_and_messages() {
        let client = OpenAiCommitClient::new("https://api.openai.com/v1", "gpt-test", "test-key")
            .expect("client");

        let payload =
            client.chat_completion_payload("diff --git a/app.rs b/app.rs", "File: app.rs");

        assert_eq!(payload["model"], "gpt-test");
        assert_eq!(payload["messages"][0]["role"], "system");
        assert_eq!(payload["messages"][1]["role"], "user");
        assert!(payload["messages"][1]["content"]
            .as_str()
            .expect("prompt")
            .contains("STRICT REQUIREMENTS"));
    }

    #[test]
    fn prompt_includes_diff_and_context() {
        let prompt = build_commit_prompt("+pub fn run() {}", "File: src/main.rs");

        assert!(prompt.contains("DIFF:"));
        assert!(prompt.contains("+pub fn run() {}"));
        assert!(prompt.contains("CONTEXT:"));
        assert!(prompt.contains("File: src/main.rs"));
    }

    #[test]
    fn parses_chat_completion_message_content() {
        let response = json!({
            "choices": [{
                "message": {
                    "content": "feat(core): add rust provider"
                }
            }]
        });

        let message = parse_chat_completion_message(&response).expect("message");

        assert_eq!(message, "feat(core): add rust provider");
    }

    #[test]
    fn rejects_missing_chat_completion_content() {
        let response = json!({"choices": [{"message": {}}]});

        assert!(parse_chat_completion_message(&response).is_err());
    }
}
