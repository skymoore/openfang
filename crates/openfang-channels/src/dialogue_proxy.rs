//! Dialogue proxy channel adapter — outbound-only adapter for Dialogue's
//! multi-tenant architecture.
//!
//! In Dialogue, each user gets their own OpenFang pod, but channel connectivity
//! (Telegram, Slack, etc.) is handled by a shared channel-router service.
//! This adapter bridges the gap: when an agent calls `channel_send(channel="telegram", ...)`,
//! the adapter POSTs the message to the channel-router's isolated outbound endpoint,
//! which then delivers it to the user's linked platform account.
//!
//! The adapter is registered under each configured channel name (e.g., "telegram", "slack")
//! so that `channel_send(channel="telegram", ...)` resolves to this adapter transparently.
//!
//! ## Security
//!
//! - Authenticated via the pod's unique API key (`Authorization: Bearer <key>`).
//! - The channel-router verifies the key against the shard manager.
//! - Each pod can only send to its own user's linked accounts.
//! - The outbound endpoint runs on a dedicated port (8021) with no other routes exposed.

use crate::types::{
    ChannelAdapter, ChannelContent, ChannelMessage, ChannelStatus, ChannelType, ChannelUser,
    DeliveryReceipt, DeliveryStatus,
};
use async_trait::async_trait;
use base64::Engine;
use chrono::Utc;
use futures::Stream;
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use tracing::{debug, error, info};
use zeroize::Zeroizing;

/// Outbound-only channel adapter that proxies messages through Dialogue's
/// channel-router service.
pub struct DialogueProxyAdapter {
    /// Channel name this adapter handles (e.g., "telegram", "slack").
    channel_name: String,
    /// URL of the channel-router outbound endpoint (e.g., "http://channel-router:8021/send").
    callback_url: String,
    /// Pod's API key for authentication (zeroized on drop).
    api_key: Zeroizing<String>,
    /// Dialogue user ID — the owner of this pod.
    user_id: String,
    /// HTTP client for outbound requests.
    client: reqwest::Client,
    /// Messages sent counter.
    messages_sent: AtomicU64,
    /// Last delivery receipt from the channel-router (read by kernel after send).
    last_receipt: Mutex<Option<DeliveryReceipt>>,
}

impl DialogueProxyAdapter {
    /// Create a new Dialogue proxy adapter.
    ///
    /// # Arguments
    /// * `channel_name` - Channel name to register as (e.g., "telegram", "slack").
    /// * `callback_url` - URL of the channel-router outbound endpoint.
    /// * `api_key` - Pod's OPENFANG_API_KEY for authentication.
    /// * `user_id` - Dialogue user ID (pod owner).
    pub fn new(
        channel_name: String,
        callback_url: String,
        api_key: String,
        user_id: String,
    ) -> Self {
        Self {
            channel_name,
            callback_url,
            api_key: Zeroizing::new(api_key),
            user_id,
            client: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(30))
                .build()
                .unwrap_or_default(),
            messages_sent: AtomicU64::new(0),
            last_receipt: Mutex::new(None),
        }
    }

    /// Retrieve the last delivery receipt (consumed — returns `None` on second call).
    ///
    /// Called by the kernel after `send()` returns `Ok(())` to get delivery
    /// details for the agent's tool result and the `DeliveryTracker`.
    pub fn take_last_receipt(&self) -> Option<DeliveryReceipt> {
        self.last_receipt
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .take()
    }

    /// Map a channel name string to the corresponding `ChannelType` enum variant.
    fn resolve_channel_type(name: &str) -> ChannelType {
        match name {
            "telegram" => ChannelType::Telegram,
            "discord" => ChannelType::Discord,
            "slack" => ChannelType::Slack,
            "whatsapp" => ChannelType::WhatsApp,
            "signal" => ChannelType::Signal,
            "matrix" => ChannelType::Matrix,
            "email" => ChannelType::Email,
            "teams" => ChannelType::Teams,
            "mattermost" => ChannelType::Mattermost,
            other => ChannelType::Custom(other.to_string()),
        }
    }

    /// Build the JSON payload for the outbound request.
    fn build_payload(
        &self,
        user: &ChannelUser,
        content: &ChannelContent,
        thread_id: Option<&str>,
    ) -> serde_json::Value {
        let mut payload = serde_json::json!({
            "user_id": self.user_id,
            "channel": self.channel_name,
        });

        // Use the platform_id as recipient_id if it differs from the user_id
        // (i.e., the agent specified an explicit recipient).
        let obj = payload.as_object_mut().unwrap();

        if !user.platform_id.is_empty() {
            obj.insert(
                "recipient_id".to_string(),
                serde_json::Value::String(user.platform_id.clone()),
            );
        }

        if let Some(tid) = thread_id {
            if !tid.is_empty() {
                obj.insert(
                    "thread_id".to_string(),
                    serde_json::Value::String(tid.to_string()),
                );
            }
        }

        match content {
            ChannelContent::Text(text) => {
                obj.insert(
                    "message".to_string(),
                    serde_json::Value::String(text.clone()),
                );
            }
            ChannelContent::Image { url, caption } => {
                obj.insert(
                    "image_url".to_string(),
                    serde_json::Value::String(url.clone()),
                );
                if let Some(cap) = caption {
                    obj.insert(
                        "message".to_string(),
                        serde_json::Value::String(cap.clone()),
                    );
                }
            }
            ChannelContent::File { url, filename } => {
                obj.insert(
                    "file_url".to_string(),
                    serde_json::Value::String(url.clone()),
                );
                obj.insert(
                    "filename".to_string(),
                    serde_json::Value::String(filename.clone()),
                );
            }
            ChannelContent::FileData {
                data,
                filename,
                mime_type,
            } => {
                // Base64-encode file data for transport.
                let encoded = base64::engine::general_purpose::STANDARD.encode(data);
                obj.insert(
                    "file_data".to_string(),
                    serde_json::Value::String(encoded),
                );
                obj.insert(
                    "filename".to_string(),
                    serde_json::Value::String(filename.clone()),
                );
                obj.insert(
                    "mime_type".to_string(),
                    serde_json::Value::String(mime_type.clone()),
                );
            }
            ChannelContent::Voice { url, .. } => {
                obj.insert(
                    "file_url".to_string(),
                    serde_json::Value::String(url.clone()),
                );
                obj.insert(
                    "message".to_string(),
                    serde_json::Value::String("[Voice message]".to_string()),
                );
            }
            ChannelContent::Location { lat, lon } => {
                obj.insert(
                    "message".to_string(),
                    serde_json::Value::String(format!("Location: {lat}, {lon}")),
                );
            }
            ChannelContent::Command { name, args } => {
                let text = if args.is_empty() {
                    format!("/{name}")
                } else {
                    format!("/{name} {}", args.join(" "))
                };
                obj.insert("message".to_string(), serde_json::Value::String(text));
            }
        }

        payload
    }

    /// Send the payload to the channel-router outbound endpoint.
    ///
    /// On success, parses the response body for delivery metadata and stores
    /// a `DeliveryReceipt` in `last_receipt` for the kernel to retrieve.
    async fn post_to_channel_router(
        &self,
        payload: serde_json::Value,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let recipient = payload["recipient_id"]
            .as_str()
            .unwrap_or(&self.user_id)
            .to_string();

        let resp = self
            .client
            .post(&self.callback_url)
            .header("Content-Type", "application/json")
            .header("Authorization", format!("Bearer {}", &*self.api_key))
            .json(&payload)
            .send()
            .await?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            error!(
                channel = %self.channel_name,
                status = %status,
                "Dialogue proxy send failed: {body}",
            );

            // Store a failed receipt.
            let receipt = DeliveryReceipt {
                message_id: String::new(),
                channel: self.channel_name.clone(),
                recipient: recipient.clone(),
                status: DeliveryStatus::Failed,
                timestamp: Utc::now(),
                error: Some(format!("{status}: {}", body.chars().take(256).collect::<String>())),
            };
            *self.last_receipt.lock().unwrap_or_else(|e| e.into_inner()) = Some(receipt);

            return Err(format!("Channel-router returned {status}: {body}").into());
        }

        // Parse response body for delivery metadata.
        let response_body: serde_json::Value = resp
            .json()
            .await
            .unwrap_or_else(|_| serde_json::json!({"status": "sent"}));

        let delivery_status = match response_body["status"].as_str() {
            Some("delivered") => DeliveryStatus::Delivered,
            Some("sent") => DeliveryStatus::Sent,
            _ => DeliveryStatus::BestEffort,
        };

        let platform_message_id = response_body["platform_message_id"]
            .as_str()
            .unwrap_or("")
            .to_string();

        let receipt = DeliveryReceipt {
            message_id: platform_message_id,
            channel: self.channel_name.clone(),
            recipient,
            status: delivery_status,
            timestamp: Utc::now(),
            error: None,
        };
        *self.last_receipt.lock().unwrap_or_else(|e| e.into_inner()) = Some(receipt);

        self.messages_sent.fetch_add(1, Ordering::Relaxed);
        debug!(
            channel = %self.channel_name,
            "Dialogue proxy: message sent successfully",
        );

        Ok(())
    }
}

#[async_trait]
impl ChannelAdapter for DialogueProxyAdapter {
    fn name(&self) -> &str {
        &self.channel_name
    }

    fn channel_type(&self) -> ChannelType {
        Self::resolve_channel_type(&self.channel_name)
    }

    /// Returns an empty stream — this adapter is outbound-only.
    /// Inbound messages are handled by the Dialogue channel-router directly.
    async fn start(
        &self,
    ) -> Result<Pin<Box<dyn Stream<Item = ChannelMessage> + Send>>, Box<dyn std::error::Error>>
    {
        info!(
            channel = %self.channel_name,
            callback = %self.callback_url,
            "Dialogue proxy adapter started (outbound-only)",
        );

        // Return an empty stream that never yields.
        // The `futures::stream::pending()` stream stays open forever without items,
        // which is exactly what we need — the bridge manager keeps the adapter alive
        // and we only use `send()` / `send_in_thread()` for outbound messages.
        Ok(Box::pin(futures::stream::pending()))
    }

    async fn send(
        &self,
        user: &ChannelUser,
        content: ChannelContent,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let payload = self.build_payload(user, &content, None);
        self.post_to_channel_router(payload).await
    }

    async fn send_in_thread(
        &self,
        user: &ChannelUser,
        content: ChannelContent,
        thread_id: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let payload = self.build_payload(user, &content, Some(thread_id));
        self.post_to_channel_router(payload).await
    }

    async fn send_typing(&self, _user: &ChannelUser) -> Result<(), Box<dyn std::error::Error>> {
        // Typing indicators are not meaningful for proxy delivery — the channel-router
        // can add its own typing indicator when it receives the message.
        Ok(())
    }

    async fn stop(&self) -> Result<(), Box<dyn std::error::Error>> {
        info!(
            channel = %self.channel_name,
            sent = self.messages_sent.load(Ordering::Relaxed),
            "Dialogue proxy adapter stopped",
        );
        Ok(())
    }

    fn status(&self) -> ChannelStatus {
        ChannelStatus {
            connected: true,
            messages_sent: self.messages_sent.load(Ordering::Relaxed),
            ..Default::default()
        }
    }

    fn take_delivery_receipt(&self) -> Option<DeliveryReceipt> {
        self.take_last_receipt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adapter_name_and_type() {
        let adapter = DialogueProxyAdapter::new(
            "telegram".to_string(),
            "http://localhost:8021/send".to_string(),
            "test-key".to_string(),
            "user-123".to_string(),
        );
        assert_eq!(adapter.name(), "telegram");
        assert_eq!(adapter.channel_type(), ChannelType::Telegram);
    }

    #[test]
    fn test_adapter_slack_type() {
        let adapter = DialogueProxyAdapter::new(
            "slack".to_string(),
            "http://localhost:8021/send".to_string(),
            "test-key".to_string(),
            "user-123".to_string(),
        );
        assert_eq!(adapter.name(), "slack");
        assert_eq!(adapter.channel_type(), ChannelType::Slack);
    }

    #[test]
    fn test_adapter_custom_type() {
        let adapter = DialogueProxyAdapter::new(
            "custom-channel".to_string(),
            "http://localhost:8021/send".to_string(),
            "test-key".to_string(),
            "user-123".to_string(),
        );
        assert_eq!(
            adapter.channel_type(),
            ChannelType::Custom("custom-channel".to_string())
        );
    }

    #[test]
    fn test_build_payload_text() {
        let adapter = DialogueProxyAdapter::new(
            "telegram".to_string(),
            "http://localhost:8021/send".to_string(),
            "key".to_string(),
            "user-abc".to_string(),
        );
        let user = ChannelUser {
            platform_id: "12345".to_string(),
            display_name: "Alice".to_string(),
            openfang_user: None,
        };
        let content = ChannelContent::Text("Hello from agent!".to_string());
        let payload = adapter.build_payload(&user, &content, None);

        assert_eq!(payload["user_id"], "user-abc");
        assert_eq!(payload["channel"], "telegram");
        assert_eq!(payload["recipient_id"], "12345");
        assert_eq!(payload["message"], "Hello from agent!");
        assert!(payload.get("thread_id").is_none());
    }

    #[test]
    fn test_build_payload_with_thread() {
        let adapter = DialogueProxyAdapter::new(
            "slack".to_string(),
            "http://localhost:8021/send".to_string(),
            "key".to_string(),
            "user-abc".to_string(),
        );
        let user = ChannelUser {
            platform_id: "U12345".to_string(),
            display_name: "Bob".to_string(),
            openfang_user: None,
        };
        let content = ChannelContent::Text("Thread reply".to_string());
        let payload = adapter.build_payload(&user, &content, Some("1234567890.123456"));

        assert_eq!(payload["thread_id"], "1234567890.123456");
        assert_eq!(payload["message"], "Thread reply");
    }

    #[test]
    fn test_build_payload_image() {
        let adapter = DialogueProxyAdapter::new(
            "telegram".to_string(),
            "http://localhost:8021/send".to_string(),
            "key".to_string(),
            "user-abc".to_string(),
        );
        let user = ChannelUser {
            platform_id: "12345".to_string(),
            display_name: "Alice".to_string(),
            openfang_user: None,
        };
        let content = ChannelContent::Image {
            url: "https://example.com/chart.png".to_string(),
            caption: Some("Weekly report chart".to_string()),
        };
        let payload = adapter.build_payload(&user, &content, None);

        assert_eq!(payload["image_url"], "https://example.com/chart.png");
        assert_eq!(payload["message"], "Weekly report chart");
    }

    #[test]
    fn test_build_payload_file() {
        let adapter = DialogueProxyAdapter::new(
            "telegram".to_string(),
            "http://localhost:8021/send".to_string(),
            "key".to_string(),
            "user-abc".to_string(),
        );
        let user = ChannelUser {
            platform_id: "12345".to_string(),
            display_name: "Alice".to_string(),
            openfang_user: None,
        };
        let content = ChannelContent::File {
            url: "https://example.com/report.pdf".to_string(),
            filename: "report.pdf".to_string(),
        };
        let payload = adapter.build_payload(&user, &content, None);

        assert_eq!(payload["file_url"], "https://example.com/report.pdf");
        assert_eq!(payload["filename"], "report.pdf");
    }

    #[test]
    fn test_build_payload_file_data() {
        let adapter = DialogueProxyAdapter::new(
            "telegram".to_string(),
            "http://localhost:8021/send".to_string(),
            "key".to_string(),
            "user-abc".to_string(),
        );
        let user = ChannelUser {
            platform_id: "12345".to_string(),
            display_name: "Alice".to_string(),
            openfang_user: None,
        };
        let content = ChannelContent::FileData {
            data: b"hello world".to_vec(),
            filename: "test.txt".to_string(),
            mime_type: "text/plain".to_string(),
        };
        let payload = adapter.build_payload(&user, &content, None);

        assert_eq!(payload["filename"], "test.txt");
        assert_eq!(payload["mime_type"], "text/plain");
        // Verify base64 encoding
        let decoded =
            base64::engine::general_purpose::STANDARD.decode(payload["file_data"].as_str().unwrap());
        assert_eq!(decoded.unwrap(), b"hello world");
    }

    #[test]
    fn test_build_payload_location() {
        let adapter = DialogueProxyAdapter::new(
            "telegram".to_string(),
            "http://localhost:8021/send".to_string(),
            "key".to_string(),
            "user-abc".to_string(),
        );
        let user = ChannelUser {
            platform_id: "12345".to_string(),
            display_name: "Alice".to_string(),
            openfang_user: None,
        };
        let content = ChannelContent::Location {
            lat: 40.7128,
            lon: -74.0060,
        };
        let payload = adapter.build_payload(&user, &content, None);

        assert!(payload["message"].as_str().unwrap().contains("40.7128"));
        assert!(payload["message"].as_str().unwrap().contains("-74.006"));
    }

    #[test]
    fn test_resolve_channel_type() {
        assert_eq!(
            DialogueProxyAdapter::resolve_channel_type("telegram"),
            ChannelType::Telegram
        );
        assert_eq!(
            DialogueProxyAdapter::resolve_channel_type("slack"),
            ChannelType::Slack
        );
        assert_eq!(
            DialogueProxyAdapter::resolve_channel_type("discord"),
            ChannelType::Discord
        );
        assert_eq!(
            DialogueProxyAdapter::resolve_channel_type("email"),
            ChannelType::Email
        );
        assert_eq!(
            DialogueProxyAdapter::resolve_channel_type("unknown"),
            ChannelType::Custom("unknown".to_string())
        );
    }
}
