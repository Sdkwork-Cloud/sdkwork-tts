//! Google Cloud Text-to-Speech Channel Implementation
//!
//! Complete implementation for Google Cloud TTS API
//! Supports: WaveNet, Neural2, Studio voices with multiple languages

use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::time::Duration;

use crate::server::channels::traits::{CloudChannel, CloudChannelConfig, CloudChannelType};
use crate::server::types::{
    AudioFormat, Gender, SpeakerInfo, SpeakerSource, SynthesisRequest,
    SynthesisResponse as TtsSynthesisResponse, SynthesisStatus,
};

/// Google Cloud TTS channel
pub struct GoogleCloudChannel {
    config: CloudChannelConfig,
    client: Client,
}

/// Google Cloud TTS voices
#[derive(Debug, Clone)]
pub struct GoogleVoice {
    pub id: String,
    pub name: String,
    pub language: String,
    pub gender: Gender,
    pub quality: VoiceQuality,
}

/// Voice quality levels
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VoiceQuality {
    Standard,
    WaveNet,
    Neural2,
    Studio,
}

impl VoiceQuality {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Standard => "STANDARD",
            Self::WaveNet => "WAVENET",
            Self::Neural2 => "NEURAL2",
            Self::Studio => "STUDIO",
        }
    }
}

/// Google Cloud TTS request body
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct SynthesisBody {
    input: TextInput,
    voice: VoiceSelectionParams,
    audio_config: AudioConfig,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct TextInput {
    text: String,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct VoiceSelectionParams {
    language_code: String,
    name: String,
    ssml_gender: String,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct AudioConfig {
    audio_encoding: String,
    sample_rate_hertz: i32,
    speaking_rate: f32,
    pitch: f32,
    volume_gain_db: f32,
}

/// Google Cloud TTS response
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GoogleSynthesisResponse {
    audio_content: String,
}

impl GoogleCloudChannel {
    /// Create new Google Cloud channel
    pub fn new(config: CloudChannelConfig) -> Result<Self, String> {
        if config.api_key.is_empty() {
            return Err("Google Cloud API key is required".to_string());
        }
        
        let client = Client::builder()
            .timeout(Duration::from_secs(config.timeout.max(30)))
            .default_headers({
                let mut headers = reqwest::header::HeaderMap::new();
                headers.insert(
                    "X-Goog-Api-Key",
                    config.api_key.parse()
                        .map_err(|e| format!("Invalid API key: {}", e))?,
                );
                headers.insert(
                    "Content-Type",
                    "application/json; charset=utf-8".parse().unwrap(),
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
    
    /// Get Google Cloud TTS API endpoint
    fn get_endpoint(&self, project_id: &str) -> String {
        self.config.base_url.clone()
            .unwrap_or_else(|| format!(
                "https://texttospeech.googleapis.com/v1/projects/{}/locations/global/text:synthesize",
                project_id
            ))
    }
    
    /// Map speaker to Google voice
    fn map_speaker_to_voice(&self, speaker: &str) -> GoogleVoice {
        // Try to match by name
        let speaker_lower = speaker.to_lowercase();
        
        // Common Google voices
        if speaker_lower.contains("en-us") || speaker_lower.contains("english") {
            if speaker_lower.contains("female") || speaker_lower.contains("f") {
                return GoogleVoice {
                    id: "en-US-Neural2-F".to_string(),
                    name: "en-US-Neural2-F".to_string(),
                    language: "en-US".to_string(),
                    gender: Gender::Female,
                    quality: VoiceQuality::Neural2,
                };
            } else {
                return GoogleVoice {
                    id: "en-US-Neural2-D".to_string(),
                    name: "en-US-Neural2-D".to_string(),
                    language: "en-US".to_string(),
                    gender: Gender::Male,
                    quality: VoiceQuality::Neural2,
                };
            }
        }
        
        // Default voice
        GoogleVoice {
            id: "en-US-Neural2-F".to_string(),
            name: "en-US-Neural2-F".to_string(),
            language: "en-US".to_string(),
            gender: Gender::Female,
            quality: VoiceQuality::Neural2,
        }
    }
    
    /// Map audio format to Google encoding
    fn map_audio_format(&self, format: AudioFormat) -> (&'static str, i32) {
        match format {
            AudioFormat::Wav => ("LINEAR16", 24000),
            AudioFormat::Mp3 => ("MP3", 24000),
            AudioFormat::Flac => ("FLAC", 24000),
            AudioFormat::Opus => ("OGG_OPUS", 24000),
            AudioFormat::Aac => ("MP3", 24000), // AAC not directly supported, use MP3
        }
    }
    
    /// Get project ID from config
    fn get_project_id(&self) -> String {
        self.config.app_id.clone()
            .unwrap_or_else(|| "default".to_string())
    }
}

#[async_trait]
impl CloudChannel for GoogleCloudChannel {
    fn name(&self) -> &str {
        &self.config.name
    }
    
    fn channel_type(&self) -> CloudChannelType {
        CloudChannelType::Google
    }
    
    async fn synthesize(&self, request: &SynthesisRequest) -> Result<TtsSynthesisResponse, String> {
        let start_time = std::time::Instant::now();
        
        // Map speaker to voice
        let voice = self.map_speaker_to_voice(&request.speaker);
        
        // Map audio format
        let (encoding, sample_rate) = self.map_audio_format(request.output_format);
        
        // Create request body
        let body = SynthesisBody {
            input: TextInput {
                text: request.text.clone(),
            },
            voice: VoiceSelectionParams {
                language_code: request.language.clone().unwrap_or_else(|| "en-US".to_string()),
                name: voice.name.clone(),
                ssml_gender: match voice.gender {
                    Gender::Female => "FEMALE".to_string(),
                    Gender::Male => "MALE".to_string(),
                    Gender::Neutral => "NEUTRAL".to_string(),
                },
            },
            audio_config: AudioConfig {
                audio_encoding: encoding.to_string(),
                sample_rate_hertz: sample_rate,
                speaking_rate: request.parameters.speed.clamp(0.25, 4.0),
                pitch: request.parameters.pitch.clamp(-20.0, 20.0),
                volume_gain_db: request.parameters.volume.clamp(-96.0, 16.0),
            },
        };
        
        // Make API call
        let project_id = self.get_project_id();
        let endpoint = self.get_endpoint(&project_id);
        
        let response = self.client
            .post(&endpoint)
            .json(&body)
            .send()
            .await
            .map_err(|e| format!("Failed to send request: {}", e))?;
        
        // Check status
        let status = response.status();
        if !status.is_success() {
            let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
            return Err(format!("Google Cloud TTS API error ({}): {}", status, error_text));
        }
        
        // Parse response
        let synthesis_result: GoogleSynthesisResponse = response.json()
            .await
            .map_err(|e| format!("Failed to parse response: {}", e))?;
        
        // Estimate duration
        let estimated_duration = (request.text.chars().count() as f32 / 15.0).max(0.5);
        
        Ok(TtsSynthesisResponse {
            request_id: uuid::Uuid::new_v4().to_string(),
            status: SynthesisStatus::Success,
            audio: Some(synthesis_result.audio_content),
            audio_url: None,
            duration: Some(estimated_duration),
            sample_rate: Some(sample_rate as u32),
            format: Some(request.output_format),
            error: None,
            processing_time_ms: Some(start_time.elapsed().as_millis() as u64),
            channel: Some("google".to_string()),
            model: Some(format!("{}-{}", voice.name, voice.quality.as_str())),
        })
    }
    
    async fn list_speakers(&self) -> Result<Vec<SpeakerInfo>, String> {
        // Return a selection of popular Google Cloud voices
        Ok(vec![
            SpeakerInfo {
                id: "en-US-Neural2-F".to_string(),
                name: "en-US-Neural2-F".to_string(),
                description: Some("High-quality female voice with natural intonation".to_string()),
                gender: Some(Gender::Female),
                age: None,
                languages: vec!["en-US".to_string()],
                source: SpeakerSource::Cloud {
                    channel: "google".to_string(),
                },
                preview_url: None,
                tags: vec!["neural2".to_string(), "female".to_string()],
                created_at: None,
                updated_at: None,
            },
            SpeakerInfo {
                id: "en-US-Neural2-D".to_string(),
                name: "en-US-Neural2-D".to_string(),
                description: Some("High-quality male voice with natural intonation".to_string()),
                gender: Some(Gender::Male),
                age: None,
                languages: vec!["en-US".to_string()],
                source: SpeakerSource::Cloud {
                    channel: "google".to_string(),
                },
                preview_url: None,
                tags: vec!["neural2".to_string(), "male".to_string()],
                created_at: None,
                updated_at: None,
            },
            SpeakerInfo {
                id: "en-GB-Neural2-A".to_string(),
                name: "en-GB-Neural2-A".to_string(),
                description: Some("British English female voice".to_string()),
                gender: Some(Gender::Female),
                age: None,
                languages: vec!["en-GB".to_string()],
                source: SpeakerSource::Cloud {
                    channel: "google".to_string(),
                },
                preview_url: None,
                tags: vec!["neural2".to_string(), "british".to_string()],
                created_at: None,
                updated_at: None,
            },
            SpeakerInfo {
                id: "zh-CN-Neural2-A".to_string(),
                name: "zh-CN-Neural2-A".to_string(),
                description: Some("Mandarin Chinese female voice".to_string()),
                gender: Some(Gender::Female),
                age: None,
                languages: vec!["zh-CN".to_string()],
                source: SpeakerSource::Cloud {
                    channel: "google".to_string(),
                },
                preview_url: None,
                tags: vec!["neural2".to_string(), "chinese".to_string()],
                created_at: None,
                updated_at: None,
            },
            SpeakerInfo {
                id: "ja-JP-Neural2-B".to_string(),
                name: "ja-JP-Neural2-B".to_string(),
                description: Some("Japanese female voice".to_string()),
                gender: Some(Gender::Female),
                age: None,
                languages: vec!["ja-JP".to_string()],
                source: SpeakerSource::Cloud {
                    channel: "google".to_string(),
                },
                preview_url: None,
                tags: vec!["neural2".to_string(), "japanese".to_string()],
                created_at: None,
                updated_at: None,
            },
        ])
    }
    
    async fn list_models(&self) -> Result<Vec<String>, String> {
        Ok(vec![
            "Standard".to_string(),
            "WaveNet".to_string(),
            "Neural2".to_string(),
            "Studio".to_string(),
        ])
    }
    
    fn config(&self) -> &CloudChannelConfig {
        &self.config
    }
    
    async fn health_check(&self) -> bool {
        // Simple health check by listing voices
        let project_id = self.get_project_id();
        let endpoint = format!(
            "https://texttospeech.googleapis.com/v1/projects/{}/locations/global/voices",
            project_id
        );
        
        let result = self.client
            .get(&endpoint)
            .timeout(Duration::from_secs(5))
            .send()
            .await;
        
        result.is_ok_and(|r| r.status().is_success())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::server::channels::CloudChannelType;
    
    #[test]
    fn test_audio_format_mapping() {
        let config = CloudChannelConfig {
            name: "google".to_string(),
            channel_type: CloudChannelType::Google,
            api_key: "test".to_string(),
            api_secret: None,
            app_id: Some("test-project".to_string()),
            base_url: None,
            models: vec![],
            default_model: None,
            timeout: 30,
            retries: 3,
        };
        let channel = GoogleCloudChannel::new(config).unwrap();
        
        assert_eq!(channel.map_audio_format(AudioFormat::Wav), ("LINEAR16", 24000));
        assert_eq!(channel.map_audio_format(AudioFormat::Mp3), ("MP3", 24000));
    }
}
