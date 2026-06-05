//! Provider client module namespace.

pub mod anthropic;
pub mod github;
pub mod openai;
pub mod xai;

use std::collections::BTreeMap;
use std::time::{Duration, Instant};

use anyhow::{anyhow, Result};
use reqwest::header::{HeaderMap, HeaderName, HeaderValue};
use serde_json::{json, Value};

use crate::transport::AsyncTransport;

pub trait ProviderClient {
    fn provider_id(&self) -> &'static str;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProviderMessage {
    pub role: String,
    pub content: String,
}

impl ProviderMessage {
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: "system".to_string(),
            content: content.into(),
        }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: "user".to_string(),
            content: content.into(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ProviderRequest {
    pub url: String,
    pub headers: BTreeMap<String, String>,
    pub payload: Value,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OpenAiBatchJob {
    pub custom_id: String,
    pub messages: Vec<ProviderMessage>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OpenAiBatchResult {
    pub custom_id: String,
    pub content: String,
}

impl ProviderRequest {
    pub fn header_map(&self) -> Result<HeaderMap> {
        let mut headers = HeaderMap::new();
        for (key, value) in &self.headers {
            let name = HeaderName::from_bytes(key.as_bytes())?;
            let value = HeaderValue::from_str(value)?;
            headers.insert(name, value);
        }
        Ok(headers)
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub struct OpenAiClient;

impl ProviderClient for OpenAiClient {
    fn provider_id(&self) -> &'static str {
        "openai"
    }
}

impl OpenAiClient {
    pub fn build_chat_request(
        endpoint: &str,
        api_key: &str,
        model: &str,
        messages: &[ProviderMessage],
    ) -> ProviderRequest {
        build_openai_compatible_request(endpoint, api_key, model, messages)
    }

    pub fn parse_chat_response(payload: &Value) -> Result<String, String> {
        parse_openai_compatible_response(payload)
    }

    pub async fn invoke_chat(
        transport: &AsyncTransport,
        endpoint: &str,
        api_key: &str,
        model: &str,
        messages: &[ProviderMessage],
    ) -> Result<String> {
        let request = Self::build_chat_request(endpoint, api_key, model, messages);
        let response = transport
            .post_json(&request.url, request.header_map()?, &request.payload)
            .await?;
        Self::parse_chat_response(&response).map_err(|err| anyhow!(err))
    }

    pub fn build_batch_jsonl(model: &str, jobs: &[OpenAiBatchJob]) -> String {
        jobs.iter()
            .map(|job| {
                json!({
                    "custom_id": job.custom_id,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": model,
                        "messages": job.messages.iter().map(|message| json!({
                            "role": message.role,
                            "content": message.content,
                        })).collect::<Vec<_>>(),
                        "max_tokens": 512,
                        "temperature": 1
                    }
                })
                .to_string()
            })
            .collect::<Vec<_>>()
            .join("\n")
            + "\n"
    }

    pub fn parse_batch_output(raw_text: &str) -> Result<Vec<OpenAiBatchResult>, String> {
        let mut results = Vec::new();
        for line in raw_text
            .lines()
            .map(str::trim)
            .filter(|line| !line.is_empty())
        {
            let payload: Value =
                serde_json::from_str(line).map_err(|err| format!("invalid batch output: {err}"))?;
            let custom_id = payload
                .get("custom_id")
                .and_then(Value::as_str)
                .ok_or_else(|| "batch output missing custom_id".to_string())?;
            if let Some(error) = payload.get("error").filter(|value| !value.is_null()) {
                return Err(format!("batch response error for {custom_id}: {error}"));
            }
            let response = payload
                .get("response")
                .and_then(Value::as_object)
                .ok_or_else(|| format!("batch output missing response for {custom_id}"))?;
            let status_code = response
                .get("status_code")
                .and_then(Value::as_u64)
                .unwrap_or(200);
            if status_code >= 400 {
                return Err(format!(
                    "batch response error for {custom_id} (status {status_code})"
                ));
            }
            let body = response
                .get("body")
                .ok_or_else(|| format!("batch output missing body for {custom_id}"))?;
            let content = Self::parse_chat_response(body)?;
            results.push(OpenAiBatchResult {
                custom_id: custom_id.to_string(),
                content,
            });
        }
        if results.is_empty() {
            Err("batch output missing expected responses".to_string())
        } else {
            Ok(results)
        }
    }

    pub async fn invoke_batch(
        transport: &AsyncTransport,
        endpoint: &str,
        api_key: &str,
        model: &str,
        jobs: &[OpenAiBatchJob],
        timeout: Duration,
        poll_interval: Duration,
    ) -> Result<Vec<OpenAiBatchResult>> {
        if jobs.is_empty() {
            return Ok(Vec::new());
        }
        let mut headers = BTreeMap::new();
        headers.insert("Authorization".to_string(), format!("Bearer {api_key}"));
        let headers = ProviderRequest {
            url: String::new(),
            headers,
            payload: Value::Null,
        }
        .header_map()?;

        let jsonl = Self::build_batch_jsonl(model, jobs);
        let upload = transport
            .post_multipart_text(
                &format!("{}/files", trim_endpoint(endpoint)),
                headers.clone(),
                "file",
                "kcmt-batch.jsonl",
                jsonl,
                "batch",
            )
            .await?;
        let file_id = upload
            .get("id")
            .and_then(Value::as_str)
            .ok_or_else(|| anyhow!("OpenAI batch file upload missing id"))?;

        let batch = transport
            .post_json(
                &format!("{}/batches", trim_endpoint(endpoint)),
                headers.clone(),
                &json!({
                    "input_file_id": file_id,
                    "endpoint": "/v1/chat/completions",
                    "completion_window": "24h"
                }),
            )
            .await?;
        let batch_id = batch
            .get("id")
            .and_then(Value::as_str)
            .ok_or_else(|| anyhow!("OpenAI batch create missing id"))?
            .to_string();
        let deadline = Instant::now() + timeout;
        let mut current = batch;
        loop {
            let status = current
                .get("status")
                .and_then(Value::as_str)
                .unwrap_or("queued");
            if status == "completed" {
                break;
            }
            if matches!(status, "failed" | "cancelling" | "cancelled" | "expired") {
                return Err(anyhow!("OpenAI batch exited with status {status}"));
            }
            if Instant::now() >= deadline {
                return Err(anyhow!("OpenAI batch did not complete before timeout"));
            }
            tokio::time::sleep(poll_interval).await;
            current = transport
                .get_json(
                    &format!("{}/batches/{batch_id}", trim_endpoint(endpoint)),
                    headers.clone(),
                )
                .await?;
        }
        let output_file_id = current
            .get("output_file_id")
            .and_then(Value::as_str)
            .ok_or_else(|| anyhow!("OpenAI batch completed without output file id"))?;
        let output = transport
            .get_text(
                &format!("{}/files/{output_file_id}/content", trim_endpoint(endpoint)),
                headers,
            )
            .await?;
        Self::parse_batch_output(&output).map_err(|err| anyhow!(err))
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub struct AnthropicClient;

impl ProviderClient for AnthropicClient {
    fn provider_id(&self) -> &'static str {
        "anthropic"
    }
}

impl AnthropicClient {
    pub fn build_messages_request(
        endpoint: &str,
        api_key: &str,
        model: &str,
        system: &str,
        prompt: &str,
    ) -> ProviderRequest {
        let mut headers = BTreeMap::new();
        headers.insert("x-api-key".to_string(), api_key.to_string());
        headers.insert("anthropic-version".to_string(), "2023-06-01".to_string());
        headers.insert("content-type".to_string(), "application/json".to_string());

        ProviderRequest {
            url: format!("{}/v1/messages", trim_endpoint(endpoint)),
            headers,
            payload: json!({
                "model": model,
                "max_output_tokens": 512,
                "messages": [{
                    "role": "user",
                    "content": [{
                        "type": "text",
                        "text": prompt
                    }]
                }],
                "system": system
            }),
        }
    }

    pub fn parse_messages_response(payload: &Value) -> Result<String, String> {
        let content = payload
            .get("content")
            .and_then(Value::as_array)
            .ok_or_else(|| "provider response missing content array".to_string())?;
        let text = content
            .iter()
            .filter_map(|chunk| {
                if chunk.get("type").and_then(Value::as_str) == Some("text") {
                    chunk.get("text").and_then(Value::as_str)
                } else {
                    None
                }
            })
            .filter(|value| !value.trim().is_empty())
            .collect::<Vec<_>>()
            .join("\n");
        if text.is_empty() {
            Err("provider response missing assistant content".to_string())
        } else {
            Ok(text)
        }
    }

    pub async fn invoke_messages(
        transport: &AsyncTransport,
        endpoint: &str,
        api_key: &str,
        model: &str,
        system: &str,
        prompt: &str,
    ) -> Result<String> {
        let request = Self::build_messages_request(endpoint, api_key, model, system, prompt);
        let response = transport
            .post_json(&request.url, request.header_map()?, &request.payload)
            .await?;
        Self::parse_messages_response(&response).map_err(|err| anyhow!(err))
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub struct XaiClient;

impl ProviderClient for XaiClient {
    fn provider_id(&self) -> &'static str {
        "xai"
    }
}

impl XaiClient {
    pub fn build_chat_request(
        endpoint: &str,
        api_key: &str,
        model: &str,
        messages: &[ProviderMessage],
    ) -> ProviderRequest {
        build_openai_compatible_request(endpoint, api_key, model, messages)
    }

    pub fn parse_chat_response(payload: &Value) -> Result<String, String> {
        parse_openai_compatible_response(payload)
    }

    pub async fn invoke_chat(
        transport: &AsyncTransport,
        endpoint: &str,
        api_key: &str,
        model: &str,
        messages: &[ProviderMessage],
    ) -> Result<String> {
        let request = Self::build_chat_request(endpoint, api_key, model, messages);
        let response = transport
            .post_json(&request.url, request.header_map()?, &request.payload)
            .await?;
        Self::parse_chat_response(&response).map_err(|err| anyhow!(err))
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub struct GitHubModelsClient;

impl ProviderClient for GitHubModelsClient {
    fn provider_id(&self) -> &'static str {
        "github"
    }
}

impl GitHubModelsClient {
    pub fn build_chat_request(
        endpoint: &str,
        api_key: &str,
        model: &str,
        messages: &[ProviderMessage],
    ) -> ProviderRequest {
        build_openai_compatible_request(endpoint, api_key, model, messages)
    }

    pub fn parse_chat_response(payload: &Value) -> Result<String, String> {
        parse_openai_compatible_response(payload)
    }

    pub async fn invoke_chat(
        transport: &AsyncTransport,
        endpoint: &str,
        api_key: &str,
        model: &str,
        messages: &[ProviderMessage],
    ) -> Result<String> {
        let request = Self::build_chat_request(endpoint, api_key, model, messages);
        let response = transport
            .post_json(&request.url, request.header_map()?, &request.payload)
            .await?;
        Self::parse_chat_response(&response).map_err(|err| anyhow!(err))
    }
}

fn build_openai_compatible_request(
    endpoint: &str,
    api_key: &str,
    model: &str,
    messages: &[ProviderMessage],
) -> ProviderRequest {
    let mut headers = BTreeMap::new();
    headers.insert("Authorization".to_string(), format!("Bearer {api_key}"));
    headers.insert("content-type".to_string(), "application/json".to_string());

    ProviderRequest {
        url: format!("{}/chat/completions", trim_endpoint(endpoint)),
        headers,
        payload: json!({
            "model": model,
            "messages": messages
                .iter()
                .map(|message| json!({
                    "role": message.role,
                    "content": message.content,
                }))
                .collect::<Vec<_>>(),
            "max_tokens": 512,
            "temperature": 1,
        }),
    }
}

fn parse_openai_compatible_response(payload: &Value) -> Result<String, String> {
    let content = payload
        .get("choices")
        .and_then(Value::as_array)
        .and_then(|choices| choices.first())
        .and_then(|choice| {
            choice
                .get("message")
                .and_then(|message| message.get("content"))
                .or_else(|| choice.get("text"))
        })
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned);

    content.ok_or_else(|| "provider response missing assistant content".to_string())
}

fn trim_endpoint(endpoint: &str) -> &str {
    endpoint.trim_end_matches('/')
}

#[cfg(test)]
mod tests {
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::sync::mpsc;
    use std::thread;
    use std::time::Duration;

    use serde_json::json;

    use crate::transport::{AsyncTransport, RetryPolicy};

    use super::{
        AnthropicClient, GitHubModelsClient, OpenAiBatchJob, OpenAiClient, ProviderMessage,
        XaiClient,
    };

    #[test]
    fn openai_builds_chat_completion_request_and_extracts_content() {
        let request = OpenAiClient::build_chat_request(
            "https://api.openai.com/v1",
            "sk-test",
            "gpt-test",
            &[
                ProviderMessage::system("system"),
                ProviderMessage::user("prompt"),
            ],
        );

        assert_eq!(request.url, "https://api.openai.com/v1/chat/completions");
        assert_eq!(request.headers["Authorization"], "Bearer sk-test");
        assert_eq!(request.payload["model"], "gpt-test");
        assert_eq!(request.payload["messages"][0]["role"], "system");
        assert_eq!(request.payload["messages"][1]["content"], "prompt");
        assert_eq!(request.payload["max_tokens"], 512);

        let content = OpenAiClient::parse_chat_response(&json!({
            "choices": [{"message": {"content": "fix(core): parse response"}}]
        }))
        .expect("openai response should parse");
        assert_eq!(content, "fix(core): parse response");
    }

    #[test]
    fn anthropic_builds_messages_request_and_extracts_text_chunks() {
        let request = AnthropicClient::build_messages_request(
            "https://api.anthropic.com",
            "anthropic-key",
            "claude-test",
            "system",
            "prompt",
        );

        assert_eq!(request.url, "https://api.anthropic.com/v1/messages");
        assert_eq!(request.headers["x-api-key"], "anthropic-key");
        assert_eq!(request.headers["anthropic-version"], "2023-06-01");
        assert_eq!(request.payload["model"], "claude-test");
        assert_eq!(request.payload["system"], "system");
        assert_eq!(
            request.payload["messages"][0]["content"][0]["text"],
            "prompt"
        );

        let content = AnthropicClient::parse_messages_response(&json!({
            "content": [
                {"type": "text", "text": "feat(core): add client"},
                {"type": "tool_use", "name": "ignored"},
                {"type": "text", "text": "Body line"}
            ]
        }))
        .expect("anthropic response should parse");
        assert_eq!(content, "feat(core): add client\nBody line");
    }

    #[test]
    fn xai_and_github_use_openai_compatible_chat_requests() {
        let xai = XaiClient::build_chat_request(
            "https://api.x.ai/v1/",
            "xai-key",
            "grok-test",
            &[ProviderMessage::user("prompt")],
        );
        assert_eq!(xai.url, "https://api.x.ai/v1/chat/completions");
        assert_eq!(xai.headers["Authorization"], "Bearer xai-key");
        assert_eq!(xai.payload["model"], "grok-test");

        let github = GitHubModelsClient::build_chat_request(
            "https://models.github.ai/inference",
            "gh-key",
            "openai/gpt-test",
            &[ProviderMessage::user("prompt")],
        );
        assert_eq!(
            github.url,
            "https://models.github.ai/inference/chat/completions"
        );
        assert_eq!(github.headers["Authorization"], "Bearer gh-key");
        assert_eq!(github.payload["model"], "openai/gpt-test");
    }

    #[test]
    fn missing_provider_content_is_reported_as_parse_error() {
        let err = OpenAiClient::parse_chat_response(&json!({"choices": []}))
            .expect_err("empty choices should fail");

        assert!(err.contains("missing assistant content"));
    }

    #[test]
    fn openai_builds_and_parses_batch_jsonl() {
        let jsonl = OpenAiClient::build_batch_jsonl(
            "gpt-batch",
            &[OpenAiBatchJob {
                custom_id: "alpha.py".to_string(),
                messages: vec![ProviderMessage::user("prompt")],
            }],
        );

        assert!(jsonl.contains(r#""custom_id":"alpha.py""#));
        assert!(jsonl.contains(r#""url":"/v1/chat/completions""#));
        assert!(jsonl.contains(r#""model":"gpt-batch""#));
        assert!(jsonl.contains(r#""content":"prompt""#));

        let results = OpenAiClient::parse_batch_output(
            r#"{"custom_id":"alpha.py","response":{"status_code":200,"body":{"choices":[{"message":{"content":"fix(core): batch alpha"}}]}}}"#,
        )
        .expect("batch output should parse");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].custom_id, "alpha.py");
        assert_eq!(results[0].content, "fix(core): batch alpha");
    }

    #[tokio::test]
    async fn openai_invoke_chat_posts_payload_and_parses_response() {
        let (endpoint, received) = spawn_json_server(
            r#"{"choices":[{"message":{"content":"fix(core): invoke provider"}}]}"#,
        );
        let transport =
            AsyncTransport::new(Duration::from_secs(2), RetryPolicy::default()).unwrap();

        let content = OpenAiClient::invoke_chat(
            &transport,
            &endpoint,
            "sk-test",
            "gpt-test",
            &[ProviderMessage::user("prompt")],
        )
        .await
        .expect("invoke should parse provider response");

        assert_eq!(content, "fix(core): invoke provider");
        let request = received.recv_timeout(Duration::from_secs(2)).unwrap();
        assert!(request.contains("POST /chat/completions HTTP/1.1"));
        assert!(request.contains("authorization: Bearer sk-test"));
        assert!(request.contains(r#""model":"gpt-test""#));
        assert!(request.contains(r#""content":"prompt""#));
    }

    #[tokio::test]
    async fn anthropic_invoke_messages_posts_payload_and_parses_text_chunks() {
        let (endpoint, received) = spawn_json_server(
            r#"{"content":[{"type":"text","text":"feat(core): invoke anthropic"}]}"#,
        );
        let transport =
            AsyncTransport::new(Duration::from_secs(2), RetryPolicy::default()).unwrap();

        let content = AnthropicClient::invoke_messages(
            &transport,
            &endpoint,
            "anthropic-key",
            "claude-test",
            "system",
            "prompt",
        )
        .await
        .expect("invoke should parse provider response");

        assert_eq!(content, "feat(core): invoke anthropic");
        let request = received.recv_timeout(Duration::from_secs(2)).unwrap();
        assert!(request.contains("POST /v1/messages HTTP/1.1"));
        assert!(request.contains("x-api-key: anthropic-key"));
        assert!(request.contains("anthropic-version: 2023-06-01"));
        assert!(request.contains(r#""model":"claude-test""#));
        assert!(request.contains(r#""text":"prompt""#));
    }

    #[tokio::test]
    async fn openai_invoke_batch_uploads_polls_and_downloads_results() {
        let (endpoint, received) = spawn_batch_server();
        let transport =
            AsyncTransport::new(Duration::from_secs(2), RetryPolicy::default()).unwrap();

        let results = OpenAiClient::invoke_batch(
            &transport,
            &endpoint,
            "sk-test",
            "gpt-batch",
            &[
                OpenAiBatchJob {
                    custom_id: "alpha.py".to_string(),
                    messages: vec![ProviderMessage::user("prompt alpha")],
                },
                OpenAiBatchJob {
                    custom_id: "beta.py".to_string(),
                    messages: vec![ProviderMessage::user("prompt beta")],
                },
            ],
            Duration::from_secs(2),
            Duration::from_millis(5),
        )
        .await
        .expect("batch should complete");

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].custom_id, "alpha.py");
        assert_eq!(results[0].content, "fix(core): batch alpha");
        assert_eq!(results[1].custom_id, "beta.py");
        assert_eq!(results[1].content, "fix(core): batch beta");

        let requests: Vec<String> = (0..4)
            .map(|_| received.recv_timeout(Duration::from_secs(2)).unwrap())
            .collect();
        assert!(requests[0].contains("POST /files HTTP/1.1"));
        assert!(requests[0].contains("authorization: Bearer sk-test"));
        assert!(requests[0].contains("alpha.py"));
        assert!(requests[0].contains("prompt beta"));
        assert!(requests[1].contains("POST /batches HTTP/1.1"));
        assert!(requests[1].contains(r#""endpoint":"/v1/chat/completions""#));
        assert!(requests[2].contains("GET /batches/batch_1 HTTP/1.1"));
        assert!(requests[3].contains("GET /files/output_1/content HTTP/1.1"));
    }

    fn spawn_json_server(body: &'static str) -> (String, mpsc::Receiver<String>) {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind mock server");
        let endpoint = format!("http://{}", listener.local_addr().expect("local addr"));
        let (sender, receiver) = mpsc::channel();

        thread::spawn(move || {
            let (mut stream, _) = listener.accept().expect("accept request");
            let mut buffer = [0_u8; 8192];
            let read = stream.read(&mut buffer).expect("read request");
            let request = String::from_utf8_lossy(&buffer[..read]).to_string();
            sender.send(request).expect("send captured request");
            let response = format!(
                "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
                body.len(),
                body
            );
            stream
                .write_all(response.as_bytes())
                .expect("write response");
        });

        (endpoint, receiver)
    }

    fn spawn_batch_server() -> (String, mpsc::Receiver<String>) {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind mock server");
        let endpoint = format!("http://{}", listener.local_addr().expect("local addr"));
        let (sender, receiver) = mpsc::channel();

        thread::spawn(move || {
            let responses = [
                r#"{"id":"file_1"}"#,
                r#"{"id":"batch_1","status":"validating"}"#,
                r#"{"id":"batch_1","status":"completed","output_file_id":"output_1"}"#,
                r#"{"custom_id":"alpha.py","response":{"status_code":200,"body":{"choices":[{"message":{"content":"fix(core): batch alpha"}}]}}}
{"custom_id":"beta.py","response":{"status_code":200,"body":{"choices":[{"message":{"content":"fix(core): batch beta"}}]}}}
"#,
            ];
            for body in responses {
                let (mut stream, _) = listener.accept().expect("accept request");
                let mut buffer = [0_u8; 16384];
                let read = stream.read(&mut buffer).expect("read request");
                let request = String::from_utf8_lossy(&buffer[..read]).to_string();
                sender.send(request).expect("send captured request");
                let response = format!(
                    "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
                    body.len(),
                    body
                );
                stream
                    .write_all(response.as_bytes())
                    .expect("write response");
            }
        });

        (endpoint, receiver)
    }
}
