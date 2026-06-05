//! Shared async HTTP transport with retry and simple rate-limit backoff handling.

use std::time::Duration;

use anyhow::{anyhow, Result};
use reqwest::header::HeaderMap;
use reqwest::multipart;
use reqwest::Client;
use serde_json::Value;
use tokio::time::sleep;

#[derive(Debug, Clone)]
pub struct RetryPolicy {
    pub max_attempts: usize,
    pub base_backoff: Duration,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            base_backoff: Duration::from_millis(250),
        }
    }
}

#[derive(Clone)]
pub struct AsyncTransport {
    client: Client,
    retry_policy: RetryPolicy,
}

impl AsyncTransport {
    pub fn new(timeout: Duration, retry_policy: RetryPolicy) -> Result<Self> {
        let client = Client::builder().timeout(timeout).build()?;
        Ok(Self {
            client,
            retry_policy,
        })
    }

    pub async fn post_json(&self, url: &str, headers: HeaderMap, payload: &Value) -> Result<Value> {
        let mut attempt = 0usize;
        loop {
            attempt += 1;
            let response = self
                .client
                .post(url)
                .headers(headers.clone())
                .json(payload)
                .send()
                .await;

            match response {
                Ok(resp) if resp.status().is_success() => {
                    let json = resp.json::<Value>().await?;
                    return Ok(json);
                }
                Ok(resp) if attempt < self.retry_policy.max_attempts => {
                    // 429 and 5xx are considered transient in this baseline transport.
                    if resp.status().as_u16() == 429 || resp.status().is_server_error() {
                        sleep(self.retry_policy.base_backoff * attempt as u32).await;
                        continue;
                    }
                    return Err(anyhow!(
                        "provider request failed with status {}",
                        resp.status()
                    ));
                }
                Ok(resp) => {
                    return Err(anyhow!(
                        "provider request failed with status {}",
                        resp.status()
                    ));
                }
                Err(err) if attempt < self.retry_policy.max_attempts => {
                    sleep(self.retry_policy.base_backoff * attempt as u32).await;
                    tracing::warn!("transient transport error (attempt {attempt}): {err}");
                }
                Err(err) => return Err(anyhow!("provider transport error after retries: {err}")),
            }
        }
    }

    pub async fn get_json(&self, url: &str, headers: HeaderMap) -> Result<Value> {
        let mut attempt = 0usize;
        loop {
            attempt += 1;
            let response = self.client.get(url).headers(headers.clone()).send().await;

            match response {
                Ok(resp) if resp.status().is_success() => return Ok(resp.json::<Value>().await?),
                Ok(resp) if attempt < self.retry_policy.max_attempts => {
                    if resp.status().as_u16() == 429 || resp.status().is_server_error() {
                        sleep(self.retry_policy.base_backoff * attempt as u32).await;
                        continue;
                    }
                    return Err(anyhow!(
                        "provider request failed with status {}",
                        resp.status()
                    ));
                }
                Ok(resp) => {
                    return Err(anyhow!(
                        "provider request failed with status {}",
                        resp.status()
                    ));
                }
                Err(err) if attempt < self.retry_policy.max_attempts => {
                    sleep(self.retry_policy.base_backoff * attempt as u32).await;
                    tracing::warn!("transient transport error (attempt {attempt}): {err}");
                }
                Err(err) => return Err(anyhow!("provider transport error after retries: {err}")),
            }
        }
    }

    pub async fn get_text(&self, url: &str, headers: HeaderMap) -> Result<String> {
        let mut attempt = 0usize;
        loop {
            attempt += 1;
            let response = self.client.get(url).headers(headers.clone()).send().await;

            match response {
                Ok(resp) if resp.status().is_success() => return Ok(resp.text().await?),
                Ok(resp) if attempt < self.retry_policy.max_attempts => {
                    if resp.status().as_u16() == 429 || resp.status().is_server_error() {
                        sleep(self.retry_policy.base_backoff * attempt as u32).await;
                        continue;
                    }
                    return Err(anyhow!(
                        "provider request failed with status {}",
                        resp.status()
                    ));
                }
                Ok(resp) => {
                    return Err(anyhow!(
                        "provider request failed with status {}",
                        resp.status()
                    ));
                }
                Err(err) if attempt < self.retry_policy.max_attempts => {
                    sleep(self.retry_policy.base_backoff * attempt as u32).await;
                    tracing::warn!("transient transport error (attempt {attempt}): {err}");
                }
                Err(err) => return Err(anyhow!("provider transport error after retries: {err}")),
            }
        }
    }

    pub async fn post_multipart_text(
        &self,
        url: &str,
        headers: HeaderMap,
        field_name: &str,
        file_name: &str,
        content: String,
        purpose: &str,
    ) -> Result<Value> {
        let mut attempt = 0usize;
        loop {
            attempt += 1;
            let part = multipart::Part::text(content.clone()).file_name(file_name.to_string());
            let form = multipart::Form::new()
                .part(field_name.to_string(), part)
                .text("purpose", purpose.to_string());
            let response = self
                .client
                .post(url)
                .headers(headers.clone())
                .multipart(form)
                .send()
                .await;

            match response {
                Ok(resp) if resp.status().is_success() => return Ok(resp.json::<Value>().await?),
                Ok(resp) if attempt < self.retry_policy.max_attempts => {
                    if resp.status().as_u16() == 429 || resp.status().is_server_error() {
                        sleep(self.retry_policy.base_backoff * attempt as u32).await;
                        continue;
                    }
                    return Err(anyhow!(
                        "provider request failed with status {}",
                        resp.status()
                    ));
                }
                Ok(resp) => {
                    return Err(anyhow!(
                        "provider request failed with status {}",
                        resp.status()
                    ));
                }
                Err(err) if attempt < self.retry_policy.max_attempts => {
                    sleep(self.retry_policy.base_backoff * attempt as u32).await;
                    tracing::warn!("transient transport error (attempt {attempt}): {err}");
                }
                Err(err) => return Err(anyhow!("provider transport error after retries: {err}")),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::sync::mpsc;
    use std::thread;
    use std::time::Duration;

    use reqwest::header::HeaderMap;

    use super::{AsyncTransport, RetryPolicy};

    #[tokio::test]
    async fn post_multipart_text_retries_transient_statuses() {
        let (endpoint, received) = spawn_retrying_multipart_server();
        let transport = AsyncTransport::new(
            Duration::from_secs(2),
            RetryPolicy {
                max_attempts: 2,
                base_backoff: Duration::from_millis(1),
            },
        )
        .expect("transport");

        let response = transport
            .post_multipart_text(
                &endpoint,
                HeaderMap::new(),
                "file",
                "batch.jsonl",
                "{\"prompt\":\"alpha\"}".to_string(),
                "batch",
            )
            .await
            .expect("retry should recover");

        assert_eq!(response["id"], "file_1");
        let requests: Vec<String> = (0..2)
            .map(|_| received.recv_timeout(Duration::from_secs(2)).unwrap())
            .collect();
        assert!(requests
            .iter()
            .all(|request| request.contains("POST /upload HTTP/1.1")));
        assert!(requests
            .iter()
            .all(|request| request.contains("{\"prompt\":\"alpha\"}")));
    }

    fn spawn_retrying_multipart_server() -> (String, mpsc::Receiver<String>) {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind mock server");
        let endpoint = format!(
            "http://{}/upload",
            listener.local_addr().expect("local addr")
        );
        let (sender, receiver) = mpsc::channel();

        thread::spawn(move || {
            for (index, body) in [r#"{"error":"retry"}"#, r#"{"id":"file_1"}"#]
                .into_iter()
                .enumerate()
            {
                let (mut stream, _) = listener.accept().expect("accept request");
                let mut buffer = [0_u8; 16384];
                let read = stream.read(&mut buffer).expect("read request");
                let request = String::from_utf8_lossy(&buffer[..read]).to_string();
                sender.send(request).expect("send captured request");
                let status = if index == 0 {
                    "429 Too Many Requests"
                } else {
                    "200 OK"
                };
                let response = format!(
                    "HTTP/1.1 {status}\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
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
