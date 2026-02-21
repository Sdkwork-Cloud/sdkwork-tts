//! Server Types
//!
//! Common types used across the server module

use serde::{Deserialize, Serialize};

/// Synthesis request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SynthesisRequest {
    /// Text to synthesize
    pub text: String,
    
    /// Speaker ID or name
    pub speaker: String,
    
    /// Channel (local or cloud channel name)
    #[serde(default)]
    pub channel: Option<String>,
    
    /// Model (for cloud channels)
    #[serde(default)]
    pub model: Option<String>,
    
    /// Language code
    #[serde(default)]
    pub language: Option<String>,
    
    /// Synthesis parameters
    #[serde(default)]
    pub parameters: SynthesisParameters,
    
    /// Voice design options
    #[serde(default)]
    pub voice_design: Option<VoiceDesignOptions>,
    
    /// Voice clone options
    #[serde(default)]
    pub voice_clone: Option<VoiceCloneOptions>,
    
    /// Output format
    #[serde(default = "default_output_format")]
    pub output_format: AudioFormat,
    
    /// Streaming
    #[serde(default)]
    pub streaming: bool,
}

/// Synthesis parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SynthesisParameters {
    /// Speed multiplier
    #[serde(default = "default_speed")]
    pub speed: f32,
    
    /// Pitch shift (semitones)
    #[serde(default)]
    pub pitch: f32,
    
    /// Volume gain (dB)
    #[serde(default)]
    pub volume: f32,
    
    /// Emotion
    #[serde(default)]
    pub emotion: Option<String>,
    
    /// Emotion intensity (0.0 - 1.0)
    #[serde(default)]
    pub emotion_intensity: f32,
    
    /// Temperature
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    
    /// Top-k
    #[serde(default)]
    pub top_k: Option<usize>,
    
    /// Top-p
    #[serde(default)]
    pub top_p: Option<f32>,
}

impl Default for SynthesisParameters {
    fn default() -> Self {
        Self {
            speed: default_speed(),
            pitch: 0.0,
            volume: 0.0,
            emotion: None,
            emotion_intensity: 1.0,
            temperature: default_temperature(),
            top_k: None,
            top_p: None,
        }
    }
}

/// Voice design options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceDesignOptions {
    /// Voice description (natural language)
    pub description: String,
    
    /// Gender
    #[serde(default)]
    pub gender: Option<Gender>,
    
    /// Age range
    #[serde(default)]
    pub age: Option<AgeRange>,
    
    /// Accent
    #[serde(default)]
    pub accent: Option<String>,
    
    /// Style
    #[serde(default)]
    pub style: Option<String>,
}

/// Voice clone options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceCloneOptions {
    /// Reference audio URL or path
    pub reference_audio: String,
    
    /// Reference text (optional, improves quality)
    #[serde(default)]
    pub reference_text: Option<String>,
    
    /// Clone mode
    #[serde(default = "default_clone_mode")]
    pub mode: CloneMode,
}

/// Gender
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Gender {
    Male,
    Female,
    Neutral,
}

/// Age range
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AgeRange {
    Child,
    Young,
    Adult,
    Senior,
}

/// Clone mode
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CloneMode {
    /// Quick clone (x-vector only)
    Quick,
    /// Full clone (with semantic features)
    Full,
    /// Fine-tune clone (requires more data)
    FineTune,
}

/// Audio format
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum AudioFormat {
    #[default]
    Wav,
    Mp3,
    Flac,
    Opus,
    Aac,
}

/// Synthesis response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SynthesisResponse {
    /// Request ID
    pub request_id: String,
    
    /// Status
    pub status: SynthesisStatus,
    
    /// Audio data (base64 encoded)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub audio: Option<String>,
    
    /// Audio URL (for streaming or large files)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub audio_url: Option<String>,
    
    /// Duration (seconds)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub duration: Option<f32>,
    
    /// Sample rate
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sample_rate: Option<u32>,
    
    /// Format
    #[serde(skip_serializing_if = "Option::is_none")]
    pub format: Option<AudioFormat>,
    
    /// Error message (if failed)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    
    /// Processing time (ms)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub processing_time_ms: Option<u64>,
    
    /// Channel used
    #[serde(skip_serializing_if = "Option::is_none")]
    pub channel: Option<String>,
    
    /// Model used
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
}

/// Synthesis status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SynthesisStatus {
    Success,
    Processing,
    Failed,
    Cancelled,
}

/// Speaker information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeakerInfo {
    /// Speaker ID
    pub id: String,
    
    /// Speaker name
    pub name: String,
    
    /// Description
    #[serde(default)]
    pub description: Option<String>,
    
    /// Gender
    #[serde(default)]
    pub gender: Option<Gender>,
    
    /// Age range
    #[serde(default)]
    pub age: Option<AgeRange>,
    
    /// Languages
    #[serde(default)]
    pub languages: Vec<String>,
    
    /// Source (local or cloud channel)
    #[serde(default)]
    pub source: SpeakerSource,
    
    /// Preview audio URL
    #[serde(default)]
    pub preview_url: Option<String>,
    
    /// Tags
    #[serde(default)]
    pub tags: Vec<String>,
    
    /// Created at
    #[serde(default)]
    pub created_at: Option<String>,
    
    /// Updated at
    #[serde(default)]
    pub updated_at: Option<String>,
}

/// Speaker source
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum SpeakerSource {
    /// Local speaker library
    #[default]
    Local,
    /// Cloud channel
    Cloud {
        /// Channel name
        channel: String,
    },
    /// Custom (user uploaded)
    Custom,
}

/// List speakers response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ListSpeakersResponse {
    /// Total count
    pub total: usize,
    
    /// Speakers
    pub speakers: Vec<SpeakerInfo>,
    
    /// Pagination
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pagination: Option<Pagination>,
}

/// Pagination
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Pagination {
    /// Current page
    pub page: usize,
    
    /// Page size
    pub page_size: usize,
    
    /// Total pages
    pub total_pages: usize,
}

/// Health check response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthResponse {
    /// Status
    pub status: String,
    
    /// Version
    pub version: String,
    
    /// Mode
    pub mode: String,
    
    /// Uptime (seconds)
    pub uptime: u64,
    
    /// Active channels
    #[serde(default)]
    pub channels: Vec<String>,
    
    /// Available speakers
    #[serde(default)]
    pub speaker_count: usize,
}

/// Server statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerStats {
    /// Total requests
    pub total_requests: u64,
    
    /// Successful requests
    pub successful_requests: u64,
    
    /// Failed requests
    pub failed_requests: u64,
    
    /// Average processing time (ms)
    pub avg_processing_time_ms: f32,
    
    /// Active connections
    pub active_connections: usize,
    
    /// Queue size
    pub queue_size: usize,
    
    /// Memory usage (MB)
    pub memory_usage_mb: f32,
    
    /// Uptime (seconds)
    pub uptime: u64,
}

/// Error response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorResponse {
    /// Error code
    pub code: String,
    
    /// Error message
    pub message: String,
    
    /// Details
    #[serde(default)]
    pub details: Option<String>,
    
    /// Request ID
    #[serde(default)]
    pub request_id: Option<String>,
}

/// Default values
fn default_output_format() -> AudioFormat {
    AudioFormat::Wav
}

fn default_speed() -> f32 {
    1.0
}

fn default_temperature() -> f32 {
    0.8
}

fn default_clone_mode() -> CloneMode {
    CloneMode::Full
}
