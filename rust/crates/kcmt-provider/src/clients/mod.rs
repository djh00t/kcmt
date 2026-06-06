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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OpenAiBatchFailure {
    pub custom_id: String,
    pub error: String,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct OpenAiBatchOutput {
    pub results: Vec<OpenAiBatchResult>,
    pub failures: Vec<OpenAiBatchFailure>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ListedModel {
    pub id: String,
    pub created_at: Option<String>,
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

    pub fn build_models_request(endpoint: &str, api_key: &str) -> ProviderRequest {
        build_bearer_models_request(endpoint, api_key)
    }

    pub fn parse_models_response(payload: &Value) -> Result<Vec<ListedModel>, String> {
        parse_openai_compatible_models_response(payload)
    }

    pub async fn list_models(
        transport: &AsyncTransport,
        endpoint: &str,
        api_key: &str,
    ) -> Result<Vec<ListedModel>> {
        let request = Self::build_models_request(endpoint, api_key);
        let response = transport
            .get_json(&request.url, request.header_map()?)
            .await?;
        Self::parse_models_response(&response).map_err(|err| anyhow!(err))
    }

    pub fn build_batch_jsonl(model: &str, jobs: &[OpenAiBatchJob]) -> String {
        let token_limit_key = openai_token_limit_key(model);
        let token_limit = openai_token_limit(model);
        jobs.iter()
            .map(|job| {
                let mut body = json!({
                    "custom_id": job.custom_id,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": model,
                        "messages": job.messages.iter().map(|message| json!({
                            "role": message.role,
                            "content": message.content,
                        })).collect::<Vec<_>>(),
                        "temperature": 1
                    }
                });
                body["body"][token_limit_key] = json!(token_limit);
                body.to_string()
            })
            .collect::<Vec<_>>()
            .join("\n")
            + "\n"
    }

    pub fn parse_batch_output(raw_text: &str) -> Result<OpenAiBatchOutput, String> {
        let mut output = OpenAiBatchOutput::default();
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
                output.failures.push(OpenAiBatchFailure {
                    custom_id: custom_id.to_string(),
                    error: format!(
                        "batch response error for {custom_id}: {}",
                        redact_json(error)
                    ),
                });
                continue;
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
                output.failures.push(OpenAiBatchFailure {
                    custom_id: custom_id.to_string(),
                    error: format!("batch response error for {custom_id} (status {status_code})"),
                });
                continue;
            }
            let body = response
                .get("body")
                .ok_or_else(|| format!("batch output missing body for {custom_id}"))?;
            match Self::parse_chat_response(body) {
                Ok(content) => output.results.push(OpenAiBatchResult {
                    custom_id: custom_id.to_string(),
                    content,
                }),
                Err(error) => output.failures.push(OpenAiBatchFailure {
                    custom_id: custom_id.to_string(),
                    error,
                }),
            }
        }
        if output.results.is_empty() && output.failures.is_empty() {
            Err("batch output missing expected responses".to_string())
        } else {
            Ok(output)
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
    ) -> Result<OpenAiBatchOutput> {
        if jobs.is_empty() {
            return Ok(OpenAiBatchOutput::default());
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

fn redact_json(value: &Value) -> String {
    match value {
        Value::Object(map) => {
            let redacted = map
                .iter()
                .map(|(key, value)| {
                    if key.to_ascii_lowercase().contains("key")
                        || key.to_ascii_lowercase().contains("token")
                        || key.to_ascii_lowercase().contains("secret")
                    {
                        (key.clone(), Value::String("[redacted]".to_string()))
                    } else {
                        (key.clone(), value.clone())
                    }
                })
                .collect::<serde_json::Map<_, _>>();
            Value::Object(redacted).to_string()
        }
        _ => value.to_string(),
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
            url: format!("{}/v1/messages", trim_anthropic_endpoint(endpoint)),
            headers,
            payload: json!({
                "model": model,
                "max_tokens": 512,
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

    pub fn build_models_request(endpoint: &str, api_key: &str) -> ProviderRequest {
        let mut headers = BTreeMap::new();
        headers.insert("x-api-key".to_string(), api_key.to_string());
        headers.insert("anthropic-version".to_string(), "2023-06-01".to_string());
        headers.insert("content-type".to_string(), "application/json".to_string());
        ProviderRequest {
            url: format!("{}/v1/models", trim_anthropic_endpoint(endpoint)),
            headers,
            payload: Value::Null,
        }
    }

    pub fn parse_models_response(payload: &Value) -> Result<Vec<ListedModel>, String> {
        parse_openai_compatible_models_response(payload)
    }

    pub async fn list_models(
        transport: &AsyncTransport,
        endpoint: &str,
        api_key: &str,
    ) -> Result<Vec<ListedModel>> {
        let request = Self::build_models_request(endpoint, api_key);
        let response = transport
            .get_json(&request.url, request.header_map()?)
            .await?;
        Self::parse_models_response(&response).map_err(|err| anyhow!(err))
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

    pub fn build_models_request(endpoint: &str, api_key: &str) -> ProviderRequest {
        build_bearer_models_request(endpoint, api_key)
    }

    pub fn parse_models_response(payload: &Value) -> Result<Vec<ListedModel>, String> {
        parse_openai_compatible_models_response(payload)
    }

    pub async fn list_models(
        transport: &AsyncTransport,
        endpoint: &str,
        api_key: &str,
    ) -> Result<Vec<ListedModel>> {
        let request = Self::build_models_request(endpoint, api_key);
        let response = transport
            .get_json(&request.url, request.header_map()?)
            .await?;
        Self::parse_models_response(&response).map_err(|err| anyhow!(err))
    }

    pub fn build_batch_requests_payload(model: &str, jobs: &[OpenAiBatchJob]) -> Value {
        json!({
            "batch_requests": jobs.iter().map(|job| json!({
                "batch_request_id": job.custom_id,
                "batch_request": {
                    "responses": {
                        "model": model,
                        "input": job.messages.iter().map(|message| json!({
                            "role": message.role,
                            "content": message.content,
                        })).collect::<Vec<_>>()
                    }
                }
            })).collect::<Vec<_>>()
        })
    }

    pub fn parse_batch_results(payload: &Value) -> OpenAiBatchOutput {
        let mut output = OpenAiBatchOutput::default();
        let Some(results) = payload.get("results").and_then(Value::as_array) else {
            return output;
        };
        for result in results {
            let custom_id = result
                .get("batch_request_id")
                .and_then(Value::as_str)
                .unwrap_or("unknown")
                .to_string();
            if let Some(error) = result.get("error_message").and_then(Value::as_str) {
                output.failures.push(OpenAiBatchFailure {
                    custom_id,
                    error: error.to_string(),
                });
                continue;
            }
            let content = result
                .get("batch_result")
                .and_then(|value| value.get("response"))
                .and_then(|value| value.get("chat_get_completion"))
                .or_else(|| result.get("response"))
                .and_then(|value| parse_openai_compatible_response(value).ok());
            match content {
                Some(content) => output
                    .results
                    .push(OpenAiBatchResult { custom_id, content }),
                None => output.failures.push(OpenAiBatchFailure {
                    custom_id,
                    error: "xAI batch response missing assistant content".to_string(),
                }),
            }
        }
        output
    }

    pub async fn invoke_batch(
        transport: &AsyncTransport,
        endpoint: &str,
        api_key: &str,
        model: &str,
        jobs: &[OpenAiBatchJob],
        timeout: Duration,
        poll_interval: Duration,
    ) -> Result<OpenAiBatchOutput> {
        if jobs.is_empty() {
            return Ok(OpenAiBatchOutput::default());
        }
        let mut headers = BTreeMap::new();
        headers.insert("Authorization".to_string(), format!("Bearer {api_key}"));
        headers.insert("content-type".to_string(), "application/json".to_string());
        let headers = ProviderRequest {
            url: String::new(),
            headers,
            payload: Value::Null,
        }
        .header_map()?;

        let batch = transport
            .post_json(
                &format!("{}/batches", trim_endpoint(endpoint)),
                headers.clone(),
                &json!({"name": "kcmt_commit_messages"}),
            )
            .await?;
        let batch_id = batch
            .get("batch_id")
            .or_else(|| batch.get("id"))
            .and_then(Value::as_str)
            .ok_or_else(|| anyhow!("xAI batch create missing batch_id"))?
            .to_string();

        transport
            .post_json(
                &format!("{}/batches/{batch_id}/requests", trim_endpoint(endpoint)),
                headers.clone(),
                &Self::build_batch_requests_payload(model, jobs),
            )
            .await?;

        let deadline = Instant::now() + timeout;
        loop {
            let current = transport
                .get_json(
                    &format!("{}/batches/{batch_id}", trim_endpoint(endpoint)),
                    headers.clone(),
                )
                .await?;
            let pending = current
                .get("state")
                .and_then(|state| state.get("num_pending"))
                .and_then(Value::as_u64);
            let total = current
                .get("state")
                .and_then(|state| state.get("num_requests"))
                .and_then(Value::as_u64);
            if pending == Some(0) && total.unwrap_or(0) > 0 {
                break;
            }
            if matches!(
                current.get("status").and_then(Value::as_str),
                Some("failed" | "cancelled" | "expired")
            ) {
                return Err(anyhow!(
                    "xAI batch exited with status {:?}",
                    current.get("status")
                ));
            }
            if Instant::now() >= deadline {
                return Err(anyhow!("xAI batch did not complete before timeout"));
            }
            tokio::time::sleep(poll_interval).await;
        }

        let mut output = OpenAiBatchOutput::default();
        let mut pagination_token = None::<String>;
        loop {
            let url = match pagination_token.as_deref() {
                Some(token) => format!(
                    "{}/batches/{batch_id}/results?limit=100&pagination_token={token}",
                    trim_endpoint(endpoint)
                ),
                None => format!(
                    "{}/batches/{batch_id}/results?limit=100",
                    trim_endpoint(endpoint)
                ),
            };
            let page = transport.get_json(&url, headers.clone()).await?;
            let page_output = Self::parse_batch_results(&page);
            output.results.extend(page_output.results);
            output.failures.extend(page_output.failures);
            pagination_token = page
                .get("pagination_token")
                .and_then(Value::as_str)
                .filter(|value| !value.is_empty())
                .map(ToOwned::to_owned);
            if pagination_token.is_none() {
                break;
            }
        }
        Ok(output)
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

    pub fn build_models_request(endpoint: &str, api_key: &str) -> ProviderRequest {
        build_bearer_models_request(endpoint, api_key)
    }

    pub fn parse_models_response(payload: &Value) -> Result<Vec<ListedModel>, String> {
        parse_openai_compatible_models_response(payload)
    }

    pub async fn list_models(
        transport: &AsyncTransport,
        endpoint: &str,
        api_key: &str,
    ) -> Result<Vec<ListedModel>> {
        let request = Self::build_models_request(endpoint, api_key);
        let response = transport
            .get_json(&request.url, request.header_map()?)
            .await?;
        Self::parse_models_response(&response).map_err(|err| anyhow!(err))
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

    let mut payload = json!({
        "model": model,
        "messages": messages
            .iter()
            .map(|message| json!({
                "role": message.role,
                "content": message.content,
            }))
            .collect::<Vec<_>>(),
        "temperature": 1,
    });
    payload[openai_token_limit_key(model)] = json!(openai_token_limit(model));

    ProviderRequest {
        url: format!("{}/chat/completions", trim_endpoint(endpoint)),
        headers,
        payload,
    }
}

fn build_bearer_models_request(endpoint: &str, api_key: &str) -> ProviderRequest {
    let mut headers = BTreeMap::new();
    headers.insert("Authorization".to_string(), format!("Bearer {api_key}"));
    headers.insert("content-type".to_string(), "application/json".to_string());
    ProviderRequest {
        url: format!("{}/models", trim_endpoint(endpoint)),
        headers,
        payload: Value::Null,
    }
}

fn openai_token_limit_key(model: &str) -> &'static str {
    let normalized = model.trim().to_ascii_lowercase();
    if normalized.starts_with("gpt-5")
        || normalized.starts_with("o1")
        || normalized.starts_with("o3")
        || normalized.starts_with("o4")
    {
        "max_completion_tokens"
    } else {
        "max_tokens"
    }
}

fn openai_token_limit(model: &str) -> u32 {
    if openai_token_limit_key(model) == "max_completion_tokens" {
        4096
    } else {
        512
    }
}

fn parse_openai_compatible_response(payload: &Value) -> Result<String, String> {
    let first_choice = payload
        .get("choices")
        .and_then(Value::as_array)
        .and_then(|choices| choices.first())
        .ok_or_else(|| "provider response missing choices".to_string())?;

    let content = first_choice
        .get("message")
        .and_then(|message| message.get("content"))
        .or_else(|| first_choice.get("text"))
        .and_then(provider_content_text)
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty());

    content.ok_or_else(|| {
        let finish_reason = first_choice
            .get("finish_reason")
            .and_then(Value::as_str)
            .unwrap_or("unknown");
        format!("provider response missing assistant content (finish_reason={finish_reason})")
    })
}

fn parse_openai_compatible_models_response(payload: &Value) -> Result<Vec<ListedModel>, String> {
    let data = payload
        .get("data")
        .and_then(Value::as_array)
        .or_else(|| payload.get("models").and_then(Value::as_array))
        .ok_or_else(|| "provider models response missing data array".to_string())?;
    let models = data
        .iter()
        .filter_map(|entry| {
            let id = entry.get("id").and_then(Value::as_str)?;
            let created_at = entry
                .get("created_at")
                .or_else(|| entry.get("created"))
                .and_then(|value| {
                    value
                        .as_str()
                        .map(ToOwned::to_owned)
                        .or_else(|| value.as_i64().map(|number| number.to_string()))
                });
            Some(ListedModel {
                id: id.to_string(),
                created_at,
            })
        })
        .collect::<Vec<_>>();
    if models.is_empty() {
        Err("provider models response did not include model ids".to_string())
    } else {
        Ok(models)
    }
}

fn provider_content_text(value: &Value) -> Option<String> {
    match value {
        Value::String(text) => Some(text.clone()),
        Value::Array(items) => {
            let text = items
                .iter()
                .filter_map(|item| {
                    item.get("text")
                        .or_else(|| item.get("content"))
                        .and_then(Value::as_str)
                })
                .filter(|part| !part.trim().is_empty())
                .collect::<Vec<_>>()
                .join("\n");
            (!text.trim().is_empty()).then_some(text)
        }
        _ => None,
    }
}

fn trim_endpoint(endpoint: &str) -> &str {
    endpoint.trim_end_matches('/')
}

fn trim_anthropic_endpoint(endpoint: &str) -> &str {
    trim_endpoint(endpoint).trim_end_matches("/v1")
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
    fn openai_gpt5_uses_max_completion_tokens() {
        let request = OpenAiClient::build_chat_request(
            "https://api.openai.com/v1",
            "sk-test",
            "gpt-5-mini-2025-08-07",
            &[ProviderMessage::user("prompt")],
        );

        assert_eq!(request.payload["max_completion_tokens"], 4096);
        assert!(request.payload.get("max_tokens").is_none());
    }

    #[test]
    fn openai_parses_structured_text_content() {
        let content = OpenAiClient::parse_chat_response(&json!({
            "choices": [{
                "message": {
                    "content": [{"type": "text", "text": "fix(core): parse structured content"}]
                },
                "finish_reason": "stop"
            }]
        }))
        .expect("structured content should parse");

        assert_eq!(content, "fix(core): parse structured content");
    }

    #[test]
    fn openai_missing_content_reports_finish_reason() {
        let err = OpenAiClient::parse_chat_response(&json!({
            "choices": [{"message": {"content": null}, "finish_reason": "length"}]
        }))
        .expect_err("missing content should fail");

        assert!(err.contains("finish_reason=length"));
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
        assert_eq!(request.payload["max_tokens"], 512);
        assert!(request.payload.get("max_output_tokens").is_none());
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
    fn anthropic_endpoint_accepts_legacy_v1_suffix() {
        let request = AnthropicClient::build_messages_request(
            "https://api.anthropic.com/v1",
            "anthropic-key",
            "claude-sonnet-4-20250514",
            "system",
            "prompt",
        );

        assert_eq!(request.url, "https://api.anthropic.com/v1/messages");
    }

    #[test]
    fn provider_models_requests_use_provider_specific_paths_and_headers() {
        let openai = OpenAiClient::build_models_request("https://api.openai.com/v1/", "openai-key");
        assert_eq!(openai.url, "https://api.openai.com/v1/models");
        assert_eq!(openai.headers["Authorization"], "Bearer openai-key");

        let anthropic =
            AnthropicClient::build_models_request("https://api.anthropic.com/v1", "claude-key");
        assert_eq!(anthropic.url, "https://api.anthropic.com/v1/models");
        assert_eq!(anthropic.headers["x-api-key"], "claude-key");
        assert_eq!(anthropic.headers["anthropic-version"], "2023-06-01");
        assert!(!anthropic.headers.contains_key("Authorization"));

        let xai = XaiClient::build_models_request("https://api.x.ai/v1", "xai-key");
        assert_eq!(xai.url, "https://api.x.ai/v1/models");
        assert_eq!(xai.headers["Authorization"], "Bearer xai-key");

        let github = GitHubModelsClient::build_models_request(
            "https://models.github.ai/inference",
            "gh-key",
        );
        assert_eq!(github.url, "https://models.github.ai/inference/models");
        assert_eq!(github.headers["Authorization"], "Bearer gh-key");
    }

    #[test]
    fn provider_models_parsers_normalize_ids_and_created_metadata() {
        let openai = OpenAiClient::parse_models_response(&json!({
            "data": [
                {"id": "gpt-5-mini", "created": 1780716000},
                {"object": "model"},
                {"id": "text-embedding-3-small"}
            ]
        }))
        .expect("openai models should parse");
        assert_eq!(openai.len(), 2);
        assert_eq!(openai[0].id, "gpt-5-mini");
        assert_eq!(openai[0].created_at.as_deref(), Some("1780716000"));

        let anthropic = AnthropicClient::parse_models_response(&json!({
            "models": [{"id": "claude-3-5-haiku-latest", "created_at": "2026-01-01"}]
        }))
        .expect("anthropic models should parse");
        assert_eq!(anthropic[0].id, "claude-3-5-haiku-latest");
        assert_eq!(anthropic[0].created_at.as_deref(), Some("2026-01-01"));

        let xai = XaiClient::parse_models_response(&json!({
            "data": [{"id": "grok-code-fast", "created": "2026-01-02"}]
        }))
        .expect("xai models should parse");
        assert_eq!(xai[0].id, "grok-code-fast");
        assert_eq!(xai[0].created_at.as_deref(), Some("2026-01-02"));

        let github = GitHubModelsClient::parse_models_response(&json!({
            "data": [{"id": "openai/gpt-4.1-mini"}]
        }))
        .expect("github models should parse");
        assert_eq!(github[0].id, "openai/gpt-4.1-mini");
        assert_eq!(github[0].created_at, None);
    }

    #[test]
    fn provider_models_parser_reports_missing_model_ids() {
        let err = OpenAiClient::parse_models_response(&json!({"data": [{"object": "model"}]}))
            .expect_err("missing ids should fail");

        assert!(err.contains("did not include model ids"));
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
    fn xai_builds_responses_batch_requests_payload() {
        let payload = XaiClient::build_batch_requests_payload(
            "grok-code-fast",
            &[OpenAiBatchJob {
                custom_id: "alpha.py".to_string(),
                messages: vec![
                    ProviderMessage::system("system"),
                    ProviderMessage::user("prompt"),
                ],
            }],
        );

        assert_eq!(payload["batch_requests"][0]["batch_request_id"], "alpha.py");
        assert_eq!(
            payload["batch_requests"][0]["batch_request"]["responses"]["model"],
            "grok-code-fast"
        );
        assert_eq!(
            payload["batch_requests"][0]["batch_request"]["responses"]["input"][0]["role"],
            "system"
        );
        assert_eq!(
            payload["batch_requests"][0]["batch_request"]["responses"]["input"][1]["content"],
            "prompt"
        );
    }

    #[test]
    fn xai_parses_batch_results_and_failures() {
        let output = XaiClient::parse_batch_results(&json!({
            "results": [
                {
                    "batch_request_id": "alpha.py",
                    "batch_result": {
                        "response": {
                            "chat_get_completion": {
                                "choices": [{
                                    "message": {"content": "fix(alpha): batch alpha"}
                                }]
                            }
                        }
                    }
                },
                {
                    "batch_request_id": "beta.py",
                    "error_message": "model failed"
                }
            ]
        }));

        assert_eq!(output.results.len(), 1);
        assert_eq!(output.results[0].custom_id, "alpha.py");
        assert_eq!(output.results[0].content, "fix(alpha): batch alpha");
        assert_eq!(output.failures.len(), 1);
        assert_eq!(output.failures[0].custom_id, "beta.py");
        assert_eq!(output.failures[0].error, "model failed");
    }

    #[test]
    fn missing_provider_content_is_reported_as_parse_error() {
        let err = OpenAiClient::parse_chat_response(&json!({"choices": []}))
            .expect_err("empty choices should fail");

        assert!(err.contains("missing choices"));
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

        let output = OpenAiClient::parse_batch_output(
            r#"{"custom_id":"alpha.py","response":{"status_code":200,"body":{"choices":[{"message":{"content":"fix(core): batch alpha"}}]}}}
{"custom_id":"beta.py","response":{"status_code":400,"body":{"error":{"message":"invalid model","api_key":"sk-test"}}}}
{"custom_id":"gamma.py","error":{"message":"rate limited","token":"secret-token"}}"#,
        )
        .expect("batch output should parse");
        assert_eq!(output.results.len(), 1);
        assert_eq!(output.results[0].custom_id, "alpha.py");
        assert_eq!(output.results[0].content, "fix(core): batch alpha");
        assert_eq!(output.failures.len(), 2);
        assert_eq!(output.failures[0].custom_id, "beta.py");
        assert!(output.failures[0].error.contains("status 400"));
        assert_eq!(output.failures[1].custom_id, "gamma.py");
        assert!(output.failures[1].error.contains("[redacted]"));
        assert!(!output.failures[1].error.contains("secret-token"));
    }

    #[test]
    fn openai_batch_gpt5_uses_max_completion_tokens() {
        let jsonl = OpenAiClient::build_batch_jsonl(
            "gpt-5-mini-2025-08-07",
            &[OpenAiBatchJob {
                custom_id: "alpha.py".to_string(),
                messages: vec![ProviderMessage::user("prompt")],
            }],
        );

        assert!(jsonl.contains(r#""max_completion_tokens":4096"#));
        assert!(!jsonl.contains(r#""max_tokens""#));
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

        let output = OpenAiClient::invoke_batch(
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

        assert_eq!(output.results.len(), 2);
        assert!(output.failures.is_empty());
        assert_eq!(output.results[0].custom_id, "alpha.py");
        assert_eq!(output.results[0].content, "fix(core): batch alpha");
        assert_eq!(output.results[1].custom_id, "beta.py");
        assert_eq!(output.results[1].content, "fix(core): batch beta");

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
