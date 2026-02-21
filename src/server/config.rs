//! Server Configuration
//!
//! Supports Local and Cloud modes with multiple channels

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Server configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    /// Server host
    #[serde(default = "default_host")]
    pub host: String,
    
    /// Server port
    #[serde(default = "default_port")]
    pub port: u16,
    
    /// Server mode
    #[serde(default)]
    pub mode: ServerMode,
    
    /// Local mode configuration
    #[serde(default)]
    pub local: LocalConfig,
    
    /// Cloud mode configuration
    #[serde(default)]
    pub cloud: CloudConfig,
    
    /// Speaker library configuration
    #[serde(default)]
    pub speaker_lib: SpeakerLibConfig,
    
    /// API authentication
    #[serde(default)]
    pub auth: AuthConfig,
    
    /// Logging configuration
    #[serde(default)]
    pub logging: LoggingConfig,
}

/// Server mode
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum ServerMode {
    /// Local inference only
    #[default]
    Local,
    /// Cloud APIs only
    Cloud,
    /// Hybrid mode (local + cloud)
    Hybrid,
}

/// Local mode configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocalConfig {
    /// Enable local mode
    #[serde(default = "default_true")]
    pub enabled: bool,
    
    /// Model checkpoints directory
    #[serde(default = "default_checkpoints_dir")]
    pub checkpoints_dir: PathBuf,
    
    /// Default local engine
    #[serde(default = "default_engine")]
    pub default_engine: String,
    
    /// GPU acceleration
    #[serde(default = "default_true")]
    pub use_gpu: bool,
    
    /// FP16 precision
    #[serde(default)]
    pub use_fp16: bool,
    
    /// Batch size
    #[serde(default = "default_batch_size")]
    pub batch_size: usize,
    
    /// Max concurrent requests
    #[serde(default = "default_max_concurrent")]
    pub max_concurrent: usize,
}

impl Default for LocalConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            checkpoints_dir: default_checkpoints_dir(),
            default_engine: default_engine(),
            use_gpu: true,
            use_fp16: false,
            batch_size: default_batch_size(),
            max_concurrent: default_max_concurrent(),
        }
    }
}

/// Cloud mode configuration
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CloudConfig {
    /// Enable cloud mode
    #[serde(default)]
    pub enabled: bool,
    
    /// Cloud channels
    #[serde(default)]
    pub channels: Vec<ChannelConfig>,
    
    /// Default cloud channel
    #[serde(default)]
    pub default_channel: Option<String>,
}

/// Channel configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChannelConfig {
    /// Channel name
    pub name: String,
    
    /// Channel type
    #[serde(rename = "type")]
    pub channel_type: ChannelTypeConfig,
    
    /// API key
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

/// Channel type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ChannelTypeConfig {
    /// Aliyun (Alibaba Cloud)
    Aliyun,
    /// OpenAI
    Openai,
    /// Volcano (火山引擎)
    Volcano,
    /// Minimax
    Minimax,
    /// Azure
    Azure,
    /// Google
    Google,
    /// AWS Polly
    AwsPolly,
}

/// Speaker library configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeakerLibConfig {
    /// Enable speaker library
    #[serde(default = "default_true")]
    pub enabled: bool,
    
    /// Local speaker library path
    #[serde(default = "default_speaker_lib_dir")]
    pub local_path: PathBuf,
    
    /// Enable cloud speaker library
    #[serde(default)]
    pub cloud_enabled: bool,
    
    /// Auto-sync cloud speakers
    #[serde(default)]
    pub auto_sync: bool,
    
    /// Max speakers in cache
    #[serde(default = "default_max_speakers")]
    pub max_cache_size: usize,
}

impl Default for SpeakerLibConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            local_path: default_speaker_lib_dir(),
            cloud_enabled: false,
            auto_sync: false,
            max_cache_size: default_max_speakers(),
        }
    }
}

/// Authentication configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthConfig {
    /// Enable authentication
    #[serde(default)]
    pub enabled: bool,
    
    /// API key for server access
    #[serde(default)]
    pub api_key: Option<String>,
    
    /// JWT secret (if using JWT)
    #[serde(default)]
    pub jwt_secret: Option<String>,
    
    /// Token expiration (hours)
    #[serde(default = "default_token_expiry")]
    pub token_expiry: u64,
}

impl Default for AuthConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            api_key: None,
            jwt_secret: None,
            token_expiry: default_token_expiry(),
        }
    }
}

/// Logging configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingConfig {
    /// Log level
    #[serde(default = "default_log_level")]
    pub level: String,
    
    /// Log to file
    #[serde(default)]
    pub file: Option<PathBuf>,
    
    /// Log rotation (MB)
    #[serde(default = "default_rotation_size")]
    pub rotation_size: u64,
    
    /// Keep log files
    #[serde(default = "default_keep_files")]
    pub keep_files: usize,
    
    /// Enable access log
    #[serde(default = "default_true")]
    pub access_log: bool,
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            level: default_log_level(),
            file: None,
            rotation_size: default_rotation_size(),
            keep_files: default_keep_files(),
            access_log: true,
        }
    }
}

/// Default values
fn default_host() -> String {
    "0.0.0.0".to_string()
}

fn default_port() -> u16 {
    8080
}

fn default_true() -> bool {
    true
}

fn default_checkpoints_dir() -> PathBuf {
    PathBuf::from("checkpoints")
}

fn default_engine() -> String {
    "indextts2".to_string()
}

fn default_batch_size() -> usize {
    4
}

fn default_max_concurrent() -> usize {
    10
}

fn default_timeout() -> u64 {
    30
}

fn default_retries() -> u32 {
    3
}

fn default_speaker_lib_dir() -> PathBuf {
    PathBuf::from("speaker_library")
}

fn default_max_speakers() -> usize {
    1000
}

fn default_token_expiry() -> u64 {
    24
}

fn default_log_level() -> String {
    "info".to_string()
}

fn default_rotation_size() -> u64 {
    100
}

fn default_keep_files() -> usize {
    5
}

impl ServerConfig {
    /// Load from file
    pub fn load<P: AsRef<std::path::Path>>(path: P) -> Result<Self, Box<dyn std::error::Error>> {
        let content = std::fs::read_to_string(path.as_ref())?;
        let config: Self = serde_yaml::from_str(&content)?;
        Ok(config)
    }
    
    /// Save to file
    pub fn save<P: AsRef<std::path::Path>>(&self, path: P) -> Result<(), Box<dyn std::error::Error>> {
        let content = serde_yaml::to_string(self)?;
        std::fs::write(path.as_ref(), content)?;
        Ok(())
    }
    
    /// Create default config
    pub fn default_local() -> Self {
        Self {
            mode: ServerMode::Local,
            local: LocalConfig::default(),
            cloud: CloudConfig::default(),
            ..Default::default()
        }
    }
    
    /// Create cloud-only config
    pub fn default_cloud() -> Self {
        Self {
            mode: ServerMode::Cloud,
            cloud: CloudConfig {
                enabled: true,
                ..Default::default()
            },
            ..Default::default()
        }
    }
    
    /// Create hybrid config
    pub fn default_hybrid() -> Self {
        Self {
            mode: ServerMode::Hybrid,
            local: LocalConfig::default(),
            cloud: CloudConfig {
                enabled: true,
                ..Default::default()
            },
            ..Default::default()
        }
    }
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self::default_local()
    }
}
