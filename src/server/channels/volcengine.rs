//! Volcengine (火山引擎) TTS Channel Implementation
//!
//! Complete implementation for Volcengine Text-to-Speech API
//! Supports: Multi-language voices with high quality neural synthesis

use async_trait::async_trait;
use chrono::Utc;
use hmac::{Hmac, Mac};
use reqwest::{Client, StatusCode};
use serde::{Deserialize, Serialize};
use sha2::Sha256;
use std::time::Duration;
use uuid::Uuid;

use crate::server::channels::traits::{CloudChannel, CloudChannelConfig, CloudChannelType};
use crate::server::types::{
    AudioFormat, Gender, SpeakerInfo, SpeakerSource, SynthesisRequest,
    SynthesisResponse, SynthesisStatus,
};

type HmacSha256 = Hmac<Sha256>;

/// Volcengine TTS channel
pub struct VolcengineChannel {
    config: CloudChannelConfig,
    client: Client,
}

/// Volcengine API endpoints
const VOLCENGINE_HOST: &str = "openspeech.bytedance.com";
const VOLCENGINE_SERVICE: &str = "openspeech";
const VOLCENGINE_REGION: &str = "cn-north-1";

/// Volcengine request body
#[derive(Debug, Serialize)]
#[serde(rename_all = "PascalCase")]
struct SynthesisBody {
    app: AppInfo,
    user: UserInfo,
    audio: AudioInfo,
    request: RequestInfo,
}

#[derive(Debug, Serialize)]
struct AppInfo {
    appid: String,
    token: String,
    cluster: String,
}

#[derive(Debug, Serialize)]
struct UserInfo {
    uid: String,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "PascalCase")]
struct AudioInfo {
    voice_type: String,
    encoding: String,
    compression_rate: i32,
    rate: i32,
    speed_ratio: f32,
    volume_ratio: f32,
    pitch_ratio: f32,
    event: String,
    text: String,
    text_type: String,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "PascalCase")]
struct RequestInfo {
    reqid: String,
    timestamp: String,
}

/// Volcengine response
#[derive(Debug, Deserialize)]
struct VolcengineResponse {
    code: i32,
    message: String,
    data: Option<VolcengineData>,
}

#[derive(Debug, Deserialize)]
struct VolcengineData {
    payload: String,
}

impl VolcengineChannel {
    /// Create new Volcengine channel
    pub fn new(config: CloudChannelConfig) -> Result<Self, String> {
        if config.api_key.is_empty() {
            return Err("Volcengine Access Key is required".to_string());
        }
        if config.api_secret.is_none() {
            return Err("Volcengine Secret Key is required".to_string());
        }
        
        let client = Client::builder()
            .timeout(Duration::from_secs(config.timeout.max(30)))
            .default_headers({
                let mut headers = reqwest::header::HeaderMap::new();
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
    
    /// Get Volcengine API endpoint
    fn get_endpoint(&self) -> String {
        self.config.base_url.clone()
            .unwrap_or_else(|| format!("https://{}.{}/api/v1", VOLCENGINE_HOST, VOLCENGINE_SERVICE))
    }
    
    /// Generate signature for Volcengine API
    fn generate_signature(&self, method: &str, path: &str, _query: &str, body: &str, timestamp: &str) -> String {
        let access_key = &self.config.api_key;
        let secret_key = self.config.api_secret.as_ref().unwrap();
        
        // Create canonical request
        let canonical_request = format!(
            "{}\n{}\n{}\nhost:{}\nx-date:{}\n\nhost;x-date\n{}",
            method,
            path,
            "", // query
            VOLCENGINE_HOST,
            timestamp,
            self.sha256_hex(body)
        );
        
        // Create string to sign
        let credential_scope = format!("{}/{}/{}/request", 
            Self::get_date(timestamp), VOLCENGINE_REGION, VOLCENGINE_SERVICE);
        
        let string_to_sign = format!(
            "HMAC-SHA256\n{}\n{}\n{}",
            timestamp,
            credential_scope,
            self.sha256_hex(&canonical_request)
        );
        
        // Calculate signature
        let k_date = self.hmac_sha256(secret_key.as_bytes(), Self::get_date(timestamp).as_bytes());
        let k_region = self.hmac_sha256(&k_date, VOLCENGINE_REGION.as_bytes());
        let k_service = self.hmac_sha256(&k_region, VOLCENGINE_SERVICE.as_bytes());
        let k_signing = self.hmac_sha256(&k_service, b"request");
        
        let signature = self.hmac_sha256_hex(&k_signing, string_to_sign.as_bytes());
        
        format!("HMAC-SHA256 Credential={}/{}, SignedHeaders=host;x-date, Signature={}",
            access_key, credential_scope, signature)
    }
    
    /// Get date from timestamp
    fn get_date(timestamp: &str) -> String {
        // Use first 8 characters (YYYYMMDD)
        if timestamp.len() >= 8 {
            timestamp[..8].to_string()
        } else {
            timestamp.to_string()
        }
    }
    
    /// SHA256 hash
    fn sha256_hex(&self, data: &str) -> String {
        use sha2::Digest;
        let hash = sha2::Sha256::digest(data.as_bytes());
        hex::encode(hash)
    }
    
    /// HMAC-SHA256
    fn hmac_sha256(&self, key: &[u8], data: &[u8]) -> Vec<u8> {
        let mut mac = HmacSha256::new_from_slice(key).expect("HMAC can take key of any size");
        mac.update(data);
        mac.finalize().into_bytes().to_vec()
    }
    
    /// HMAC-SHA256 hex
    fn hmac_sha256_hex(&self, key: &[u8], data: &[u8]) -> String {
        hex::encode(self.hmac_sha256(key, data))
    }
    
    /// Map speaker to Volcengine voice
    fn map_speaker_to_voice(&self, speaker: &str) -> String {
        let speaker_lower = speaker.to_lowercase();
        
        // Common Volcengine voices
        if speaker_lower.contains("zh") || speaker_lower.contains("chinese") {
            if speaker_lower.contains("female") || speaker_lower.contains("f") {
                return "BV001_streaming".to_string(); // Chinese female
            } else {
                return "BV002_streaming".to_string(); // Chinese male
            }
        }
        
        if speaker_lower.contains("en") || speaker_lower.contains("english") {
            if speaker_lower.contains("female") || speaker_lower.contains("f") {
                return "BV005_streaming".to_string(); // English female
            } else {
                return "BV006_streaming".to_string(); // English male
            }
        }
        
        // Default voice
        "BV001_streaming".to_string()
    }
    
    /// Map audio format to Volcengine encoding
    fn map_audio_format(&self, format: AudioFormat) -> &'static str {
        match format {
            AudioFormat::Wav => "wav",
            AudioFormat::Mp3 => "mp3",
            AudioFormat::Flac => "flac",
            AudioFormat::Opus => "ogg",
            AudioFormat::Aac => "aac",
        }
    }
    
    /// Get app ID from config
    fn get_app_id(&self) -> String {
        self.config.app_id.clone()
            .unwrap_or_else(|| "default".to_string())
    }
}

#[async_trait]
impl CloudChannel for VolcengineChannel {
    fn name(&self) -> &str {
        &self.config.name
    }
    
    fn channel_type(&self) -> CloudChannelType {
        CloudChannelType::Volcano
    }
    
    async fn synthesize(&self, request: &SynthesisRequest) -> Result<SynthesisResponse, String> {
        let start_time = std::time::Instant::now();
        
        // Generate timestamp
        let timestamp = Utc::now().format("%Y%m%dT%H%M%SZ").to_string();
        
        // Map speaker to voice
        let voice_type = self.map_speaker_to_voice(&request.speaker);
        
        // Map audio format
        let encoding = self.map_audio_format(request.output_format);
        
        // Create request body
        let body = SynthesisBody {
            app: AppInfo {
                appid: self.get_app_id(),
                token: self.config.api_key.clone(),
                cluster: "volcano_tts".to_string(),
            },
            user: UserInfo {
                uid: Uuid::new_v4().to_string(),
            },
            audio: AudioInfo {
                voice_type,
                encoding: encoding.to_string(),
                compression_rate: 1,
                rate: 24000,
                speed_ratio: request.parameters.speed.clamp(0.5, 2.0),
                volume_ratio: request.parameters.volume.clamp(0.1, 3.0),
                pitch_ratio: request.parameters.pitch.clamp(0.5, 2.0),
                event: "Start".to_string(),
                text: request.text.clone(),
                text_type: "plain".to_string(),
            },
            request: RequestInfo {
                reqid: Uuid::new_v4().to_string(),
                timestamp: timestamp.clone(),
            },
        };
        
        // Serialize body
        let body_json = serde_json::to_string(&body)
            .map_err(|e| format!("Failed to serialize request: {}", e))?;
        
        // Generate signature
        let path = "/api/v1";
        let query = "Action=SubmitTask&Version=2020-12-03";
        let signature = self.generate_signature("POST", path, query, &body_json, &timestamp);
        
        // Make API call
        let endpoint = format!("https://{}{}", VOLCENGINE_HOST, path);
        
        let response = self.client
            .post(&endpoint)
            .query(&[
                ("Action", "SubmitTask"),
                ("Version", "2020-12-03"),
            ])
            .header("Authorization", &signature)
            .header("X-Date", &timestamp)
            .body(body_json)
            .send()
            .await
            .map_err(|e| format!("Failed to send request: {}", e))?;
        
        // Check status
        let status = response.status();
        if !status.is_success() {
            let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
            return Err(format!("Volcengine API error ({}): {}", status, error_text));
        }
        
        // Parse response
        let volcano_response: VolcengineResponse = response.json()
            .await
            .map_err(|e| format!("Failed to parse response: {}", e))?;
        
        if volcano_response.code != 0 {
            return Err(format!("Volcengine API error: {}", volcano_response.message));
        }
        
        let audio_content = volcano_response.data
            .map(|d| d.payload)
            .unwrap_or_default();
        
        // Estimate duration
        let estimated_duration = (request.text.chars().count() as f32 / 15.0).max(0.5);
        
        Ok(SynthesisResponse {
            request_id: Uuid::new_v4().to_string(),
            status: SynthesisStatus::Success,
            audio: Some(audio_content),
            audio_url: None,
            duration: Some(estimated_duration),
            sample_rate: Some(24000),
            format: Some(request.output_format),
            error: None,
            processing_time_ms: Some(start_time.elapsed().as_millis() as u64),
            channel: Some("volcano".to_string()),
            model: Some("volcano_tts".to_string()),
        })
    }
    
    async fn list_speakers(&self) -> Result<Vec<SpeakerInfo>, String> {
        Ok(vec![
            SpeakerInfo {
                id: "BV001_streaming".to_string(),
                name: "中文女声".to_string(),
                description: Some("温柔知性的中文女声".to_string()),
                gender: Some(Gender::Female),
                age: None,
                languages: vec!["zh-CN".to_string()],
                source: SpeakerSource::Cloud {
                    channel: "volcano".to_string(),
                },
                preview_url: None,
                tags: vec!["chinese".to_string(), "female".to_string()],
                created_at: None,
                updated_at: None,
            },
            SpeakerInfo {
                id: "BV002_streaming".to_string(),
                name: "中文男声".to_string(),
                description: Some("沉稳磁性的中文男声".to_string()),
                gender: Some(Gender::Male),
                age: None,
                languages: vec!["zh-CN".to_string()],
                source: SpeakerSource::Cloud {
                    channel: "volcano".to_string(),
                },
                preview_url: None,
                tags: vec!["chinese".to_string(), "male".to_string()],
                created_at: None,
                updated_at: None,
            },
            SpeakerInfo {
                id: "BV005_streaming".to_string(),
                name: "English Female".to_string(),
                description: Some("Natural English female voice".to_string()),
                gender: Some(Gender::Female),
                age: None,
                languages: vec!["en-US".to_string()],
                source: SpeakerSource::Cloud {
                    channel: "volcano".to_string(),
                },
                preview_url: None,
                tags: vec!["english".to_string(), "female".to_string()],
                created_at: None,
                updated_at: None,
            },
            SpeakerInfo {
                id: "BV006_streaming".to_string(),
                name: "English Male".to_string(),
                description: Some("Natural English male voice".to_string()),
                gender: Some(Gender::Male),
                age: None,
                languages: vec!["en-US".to_string()],
                source: SpeakerSource::Cloud {
                    channel: "volcano".to_string(),
                },
                preview_url: None,
                tags: vec!["english".to_string(), "male".to_string()],
                created_at: None,
                updated_at: None,
            },
        ])
    }
    
    async fn list_models(&self) -> Result<Vec<String>, String> {
        Ok(vec![
            "volcano_tts".to_string(),
            "volcano_tts_premium".to_string(),
        ])
    }
    
    fn config(&self) -> &CloudChannelConfig {
        &self.config
    }
    
    async fn health_check(&self) -> bool {
        // Simple health check
        let timestamp = Utc::now().format("%Y%m%d%H%M%S").to_string();

        let result = self.client
            .get(format!("https://{}", VOLCENGINE_HOST))
            .header("X-Date", &timestamp)
            .timeout(Duration::from_secs(5))
            .send()
            .await;

        result.is_ok_and(|r| r.status().is_success() || r.status() == StatusCode::NOT_FOUND)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::server::channels::CloudChannelType;
    
    #[test]
    fn test_speaker_mapping() {
        let config = CloudChannelConfig {
            name: "volcengine".to_string(),
            channel_type: CloudChannelType::Volcano,
            api_key: "test".to_string(),
            api_secret: Some("test".to_string()),
            app_id: None,
            base_url: None,
            models: vec![],
            default_model: None,
            timeout: 30,
            retries: 3,
        };
        let channel = VolcengineChannel::new(config).unwrap();
        
        assert_eq!(channel.map_speaker_to_voice("zh_female"), "BV001_streaming");
        assert_eq!(channel.map_speaker_to_voice("en_male"), "BV006_streaming");
    }
}
