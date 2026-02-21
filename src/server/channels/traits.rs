//! Cloud Channel Traits
//!
//! Defines the interface for cloud TTS providers

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use crate::server::types::{SynthesisRequest, SynthesisResponse, SpeakerInfo};

/// Cloud channel type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CloudChannelType {
    /// Local inference
    #[serde(rename = "local")]
    Local,
    /// Aliyun (Alibaba Cloud)
    #[serde(rename = "aliyun")]
    Aliyun,
    /// OpenAI
    #[serde(rename = "openai")]
    Openai,
    /// Volcano (火山引擎)
    #[serde(rename = "volcano")]
    Volcano,
    /// Minimax
    #[serde(rename = "minimax")]
    Minimax,
    /// Azure
    #[serde(rename = "azure")]
    Azure,
    /// Google
    #[serde(rename = "google")]
    Google,
    /// AWS Polly
    #[serde(rename = "aws_polly")]
    AwsPolly,
}

impl Default for CloudChannelType {
    fn default() -> Self {
        Self::Openai
    }
}

/// Cloud channel configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CloudChannelConfig {
    /// Channel name
    #[serde(default)]
    pub name: String,
    /// Channel type
    #[serde(rename = "type", default)]
    pub channel_type: CloudChannelType,
    /// API key
    #[serde(default)]
    pub api_key: String,
    /// API secret (if required)
    #[serde(default)]
    pub api_secret: Option<String>,
    /// App ID (if required)
    #[serde(default)]
    pub app_id: Option<String>,
    /// Base URL (if custom)
    #[serde(default)]
    pub base_url: Option<String>,
    /// Available models
    #[serde(default)]
    pub models: Vec<String>,
    /// Default model
    #[serde(default)]
    pub default_model: Option<String>,
    /// Request timeout (seconds)
    #[serde(default = "default_timeout")]
    pub timeout: u64,
    /// Retry attempts
    #[serde(default = "default_retries")]
    pub retries: u32,
}

impl Default for CloudChannelConfig {
    fn default() -> Self {
        Self {
            name: String::new(),
            channel_type: CloudChannelType::Openai,
            api_key: String::new(),
            api_secret: None,
            app_id: None,
            base_url: None,
            models: Vec::new(),
            default_model: None,
            timeout: default_timeout(),
            retries: default_retries(),
        }
    }
}

fn default_timeout() -> u64 {
    30
}

fn default_retries() -> u32 {
    3
}

/// Cloud channel trait
#[async_trait]
pub trait CloudChannel: Send + Sync {
    /// Get channel name
    fn name(&self) -> &str;
    
    /// Get channel type
    fn channel_type(&self) -> CloudChannelType;
    
    /// Synthesize speech
    async fn synthesize(&self, request: &SynthesisRequest) -> Result<SynthesisResponse, String>;
    
    /// List available speakers
    async fn list_speakers(&self) -> Result<Vec<SpeakerInfo>, String>;
    
    /// List available models
    async fn list_models(&self) -> Result<Vec<String>, String>;
    
    /// Get channel configuration
    fn config(&self) -> &CloudChannelConfig;
    
    /// Check if channel is available
    async fn health_check(&self) -> bool {
        true
    }
}

/// Channel factory function
pub type ChannelFactory = Box<dyn FnOnce(&CloudChannelConfig) -> Result<Box<dyn CloudChannel>, String> + Send>;

/// Channel registry entry
pub struct ChannelEntry {
    /// Channel name
    pub name: String,
    /// Channel instance
    pub channel: Box<dyn CloudChannel>,
    /// Is enabled
    pub enabled: bool,
}
