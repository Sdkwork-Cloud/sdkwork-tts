//! OpenAI TTS Channel Implementation
//!
//! Complete implementation for OpenAI Text-to-Speech API
//! Supports: tts-1, tts-1-hd models with multiple voices

use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::time::Duration;

use crate::server::channels::traits::{CloudChannel, CloudChannelConfig, CloudChannelType};
use crate::server::types::{
    AudioFormat, Gender, SpeakerInfo, SpeakerSource, SynthesisRequest,
    SynthesisResponse, SynthesisStatus,
};

/// OpenAI TTS channel
pub struct OpenAiChannel {
    config: CloudChannelConfig,
    client: Client,
}

/// OpenAI API response structure
#[derive(Debug, Deserialize)]
struct OpenAiErrorResponse {
    error: OpenAiError,
}

#[derive(Debug, Deserialize)]
struct OpenAiError {
    message: String,
    #[serde(rename = "type")]
    error_type: String,
    code: Option<String>,
}

/// OpenAI voices
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpenAiVoice {
    Alloy,
    Echo,
    Fable,
    Onyx,
    Nova,
    Shimmer,
}

impl OpenAiVoice {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Alloy => "alloy",
            Self::Echo => "echo",
            Self::Fable => "fable",
            Self::Onyx => "onyx",
            Self::Nova => "nova",
            Self::Shimmer => "shimmer",
        }
    }
    
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "alloy" => Some(Self::Alloy),
            "echo" => Some(Self::Echo),
            "fable" => Some(Self::Fable),
            "onyx" => Some(Self::Onyx),
            "nova" => Some(Self::Nova),
            "shimmer" => Some(Self::Shimmer),
            _ => None,
        }
    }
    
    pub fn gender(&self) -> Gender {
        match self {
            Self::Alloy => Gender::Neutral,
            Self::Echo => Gender::Male,
            Self::Fable => Gender::Neutral,
            Self::Onyx => Gender::Male,
            Self::Nova => Gender::Female,
            Self::Shimmer => Gender::Female,
        }
    }
}

impl std::str::FromStr for OpenAiVoice {
    type Err = ();
    
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::from_str(s).ok_or(())
    }
}

/// OpenAI models
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpenAiModel {
    Tts1,
    Tts1Hd,
}

impl OpenAiModel {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Tts1 => "tts-1",
            Self::Tts1Hd => "tts-1-hd",
        }
    }
}

impl OpenAiChannel {
    /// Create new OpenAI channel
    pub fn new(config: CloudChannelConfig) -> Result<Self, String> {
        if config.api_key.is_empty() {
            return Err("OpenAI API key is required".to_string());
        }
        
        let client = Client::builder()
            .timeout(Duration::from_secs(config.timeout.max(30)))
            .default_headers({
                let mut headers = reqwest::header::HeaderMap::new();
                headers.insert(
                    "Authorization",
                    format!("Bearer {}", config.api_key).parse()
                        .map_err(|e| format!("Invalid API key: {}", e))?,
                );
                headers.insert(
                    "Content-Type",
                    "application/json".parse().unwrap(),
                );
                headers
            })
            .build()
            .map_err(|e| format!("Failed to create HTTP client: {}", e))?;
        
        Ok(Self {
            config,
            client,
        })
    }
    
    /// Get OpenAI TTS API endpoint
    fn get_endpoint(&self) -> String {
        self.config.base_url.clone()
            .unwrap_or_else(|| "https://api.openai.com/v1/audio/speech".to_string())
    }
    
    /// Map speaker to OpenAI voice
    fn map_speaker_to_voice(&self, speaker: &str) -> OpenAiVoice {
        // Try to match by name
        if let Some(voice) = OpenAiVoice::from_str(speaker) {
            return voice;
        }
        
        // Default to alloy for unknown speakers
        OpenAiVoice::Alloy
    }
    
    /// Map audio format to OpenAI format
    fn map_audio_format(&self, format: AudioFormat) -> &'static str {
        match format {
            AudioFormat::Mp3 => "mp3",
            AudioFormat::Wav => "wav",
            AudioFormat::Opus => "opus",
            AudioFormat::Aac => "aac",
            AudioFormat::Flac => "flac",
        }
    }
    
    /// Create synthesis request body
    fn create_request_body(&self, request: &SynthesisRequest) -> SynthesisBody {
        let voice = self.map_speaker_to_voice(&request.speaker);
        let model = request.model.as_deref()
            .map(|m| if m.contains("hd") { "tts-1-hd" } else { "tts-1" })
            .unwrap_or("tts-1");
        
        SynthesisBody {
            model: model.to_string(),
            input: request.text.clone(),
            voice: voice.as_str().to_string(),
            response_format: self.map_audio_format(request.output_format).to_string(),
            speed: request.parameters.speed.clamp(0.25, 4.0),
        }
    }
}

/// OpenAI TTS request body
#[derive(Debug, Serialize)]
struct SynthesisBody {
    model: String,
    input: String,
    voice: String,
    response_format: String,
    speed: f32,
}

#[async_trait]
impl CloudChannel for OpenAiChannel {
    fn name(&self) -> &str {
        &self.config.name
    }
    
    fn channel_type(&self) -> CloudChannelType {
        CloudChannelType::Openai
    }
    
    async fn synthesize(&self, request: &SynthesisRequest) -> Result<SynthesisResponse, String> {
        let start_time = std::time::Instant::now();
        
        // Create request body
        let body = self.create_request_body(request);
        
        // Make API call
        let response = self.client
            .post(self.get_endpoint())
            .json(&body)
            .send()
            .await
            .map_err(|e| format!("Failed to send request: {}", e))?;
        
        // Check status
        let status = response.status();
        if !status.is_success() {
            let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
            
            // Try to parse as OpenAI error
            if let Ok(error) = serde_json::from_str::<OpenAiErrorResponse>(&error_text) {
                return Err(format!(
                    "OpenAI API error ({}): {} - {}",
                    status, error.error.error_type, error.error.message
                ));
            }
            
            return Err(format!("OpenAI API error ({}): {}", status, error_text));
        }
        
        // Get audio data
        let audio_bytes = response.bytes()
            .await
            .map_err(|e| format!("Failed to get response body: {}", e))?;
        
        // Convert to base64
        let audio_base64 = base64::Engine::encode(
            &base64::engine::general_purpose::STANDARD,
            &audio_bytes
        );
        
        // Estimate duration (OpenAI doesn't provide this, estimate from text length)
        let estimated_duration = (request.text.chars().count() as f32 / 15.0).max(0.5);
        
        Ok(SynthesisResponse {
            request_id: uuid::Uuid::new_v4().to_string(),
            status: SynthesisStatus::Success,
            audio: Some(audio_base64),
            audio_url: None,
            duration: Some(estimated_duration),
            sample_rate: Some(24000),
            format: Some(request.output_format),
            error: None,
            processing_time_ms: Some(start_time.elapsed().as_millis() as u64),
            channel: Some("openai".to_string()),
            model: Some(body.model),
        })
    }
    
    async fn list_speakers(&self) -> Result<Vec<SpeakerInfo>, String> {
        Ok(vec![
            SpeakerInfo {
                id: "alloy".to_string(),
                name: "Alloy".to_string(),
                description: Some("Neutral, versatile voice suitable for most use cases".to_string()),
                gender: Some(Gender::Neutral),
                age: None,
                languages: vec!["en".to_string()],
                source: SpeakerSource::Cloud {
                    channel: "openai".to_string(),
                },
                preview_url: Some("https://cdn.openai.com/speech/samples/alloy.mp3".to_string()),
                tags: vec!["neutral".to_string(), "versatile".to_string()],
                created_at: None,
                updated_at: None,
            },
            SpeakerInfo {
                id: "echo".to_string(),
                name: "Echo".to_string(),
                description: Some("Warm, rounded voice with a friendly tone".to_string()),
                gender: Some(Gender::Male),
                age: Some(crate::server::types::AgeRange::Adult),
                languages: vec!["en".to_string()],
                source: SpeakerSource::Cloud {
                    channel: "openai".to_string(),
                },
                preview_url: Some("https://cdn.openai.com/speech/samples/echo.mp3".to_string()),
                tags: vec!["warm".to_string(), "friendly".to_string()],
                created_at: None,
                updated_at: None,
            },
            SpeakerInfo {
                id: "fable".to_string(),
                name: "Fable".to_string(),
                description: Some("British accent, expressive and animated".to_string()),
                gender: Some(Gender::Neutral),
                age: None,
                languages: vec!["en-GB".to_string()],
                source: SpeakerSource::Cloud {
                    channel: "openai".to_string(),
                },
                preview_url: Some("https://cdn.openai.com/speech/samples/fable.mp3".to_string()),
                tags: vec!["british".to_string(), "expressive".to_string()],
                created_at: None,
                updated_at: None,
            },
            SpeakerInfo {
                id: "onyx".to_string(),
                name: "Onyx".to_string(),
                description: Some("Deep, resonant voice with authority".to_string()),
                gender: Some(Gender::Male),
                age: Some(crate::server::types::AgeRange::Adult),
                languages: vec!["en".to_string()],
                source: SpeakerSource::Cloud {
                    channel: "openai".to_string(),
                },
                preview_url: Some("https://cdn.openai.com/speech/samples/onyx.mp3".to_string()),
                tags: vec!["deep".to_string(), "authoritative".to_string()],
                created_at: None,
                updated_at: None,
            },
            SpeakerInfo {
                id: "nova".to_string(),
                name: "Nova".to_string(),
                description: Some("Bright, enthusiastic voice with energy".to_string()),
                gender: Some(Gender::Female),
                age: Some(crate::server::types::AgeRange::Young),
                languages: vec!["en".to_string()],
                source: SpeakerSource::Cloud {
                    channel: "openai".to_string(),
                },
                preview_url: Some("https://cdn.openai.com/speech/samples/nova.mp3".to_string()),
                tags: vec!["bright".to_string(), "enthusiastic".to_string()],
                created_at: None,
                updated_at: None,
            },
            SpeakerInfo {
                id: "shimmer".to_string(),
                name: "Shimmer".to_string(),
                description: Some("Soft, gentle voice with a calming presence".to_string()),
                gender: Some(Gender::Female),
                age: Some(crate::server::types::AgeRange::Adult),
                languages: vec!["en".to_string()],
                source: SpeakerSource::Cloud {
                    channel: "openai".to_string(),
                },
                preview_url: Some("https://cdn.openai.com/speech/samples/shimmer.mp3".to_string()),
                tags: vec!["soft".to_string(), "gentle".to_string(), "calming".to_string()],
                created_at: None,
                updated_at: None,
            },
        ])
    }
    
    async fn list_models(&self) -> Result<Vec<String>, String> {
        Ok(vec![
            "tts-1".to_string(),
            "tts-1-hd".to_string(),
        ])
    }
    
    fn config(&self) -> &CloudChannelConfig {
        &self.config
    }
    
    async fn health_check(&self) -> bool {
        // Simple health check by making a small request
        let result = self.client
            .get("https://api.openai.com/v1/models")
            .timeout(Duration::from_secs(5))
            .send()
            .await;
        
        result.is_ok_and(|r| r.status().is_success())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_voice_from_str() {
        assert_eq!(OpenAiVoice::from_str("alloy"), Some(OpenAiVoice::Alloy));
        assert_eq!(OpenAiVoice::from_str("ALLOY"), Some(OpenAiVoice::Alloy));
        assert_eq!(OpenAiVoice::from_str("unknown"), None);
    }
    
    #[test]
    fn test_voice_gender() {
        assert_eq!(OpenAiVoice::Alloy.gender(), Gender::Neutral);
        assert_eq!(OpenAiVoice::Echo.gender(), Gender::Male);
        assert_eq!(OpenAiVoice::Nova.gender(), Gender::Female);
    }
}
