//! Qwen3-TTS Engine Adapter
//!
//! This module provides a TtsEngine implementation for Qwen3-TTS,
//! Alibaba's powerful speech generation model series.
//!
//! ## Features
//! - 10 major languages (Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, Spanish, Italian)
//! - Ultra-low latency streaming (as low as 97ms)
//! - Voice cloning from 3-second audio
//! - Voice design from natural language descriptions
//! - Instruction-based voice control
//!
//! ## Models
//! - Qwen3-TTS-12Hz-1.7B-VoiceDesign
//! - Qwen3-TTS-12Hz-1.7B-CustomVoice
//! - Qwen3-TTS-12Hz-1.7B-Base
//! - Qwen3-TTS-12Hz-0.6B-CustomVoice
//! - Qwen3-TTS-12Hz-0.6B-Base

use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;

use crate::core::error::{Result, TtsError};
use crate::engine::traits::{
    TtsEngine, TtsEngineInfo, EngineType, EngineFeature,
    SynthesisRequest, SynthesisResult, AudioChunk, StreamingCallback,
    SpeakerInfo, EmotionInfo, LanguageSupport, EngineCapabilities, ResourceUsage,
};
use crate::engine::config::EngineConfig;
use crate::models::qwen3_tts::{
    Qwen3TtsModel, QwenConfig, QwenModelVariant,
};

/// Adapter variant for engine trait
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdapterVariant {
    VoiceDesign17B,
    CustomVoice17B,
    Base17B,
    CustomVoice06B,
    Base06B,
}

impl AdapterVariant {
    pub fn supports_voice_design(&self) -> bool {
        matches!(self, Self::VoiceDesign17B)
    }

    pub fn supports_custom_voice(&self) -> bool {
        matches!(self, Self::CustomVoice17B | Self::CustomVoice06B)
    }

    pub fn supports_voice_cloning(&self) -> bool {
        matches!(self, Self::Base17B | Self::Base06B)
    }

    pub fn to_internal(&self) -> QwenModelVariant {
        match self {
            Self::VoiceDesign17B => QwenModelVariant::VoiceDesign17B,
            Self::CustomVoice17B => QwenModelVariant::CustomVoice17B,
            Self::Base17B => QwenModelVariant::Base17B,
            Self::CustomVoice06B => QwenModelVariant::CustomVoice06B,
            Self::Base06B => QwenModelVariant::Base06B,
        }
    }
}

/// Qwen3-TTS engine adapter
pub struct Qwen3TtsEngine {
    /// Engine information
    info: TtsEngineInfo,
    /// Underlying Qwen3-TTS model
    model: Option<Arc<Mutex<Qwen3TtsModel>>>,
    /// Configuration
    config: Option<EngineConfig>,
    /// Resource usage tracking
    resource_usage: ResourceUsage,
    /// Model variant
    model_variant: AdapterVariant,
}

impl Qwen3TtsEngine {
    /// Create a new Qwen3-TTS engine with default variant
    pub fn new() -> Self {
        Self::with_variant(AdapterVariant::CustomVoice17B)
    }

    /// Create with specific model variant
    pub fn with_variant(variant: AdapterVariant) -> Self {
        let mut features = vec![
            EngineFeature::MultiSpeaker,
            EngineFeature::Streaming,
            EngineFeature::MultiLanguage,
        ];

        if variant.supports_voice_design() {
            features.push(EngineFeature::EmotionControl);
        }

        if variant.supports_voice_cloning() {
            features.push(EngineFeature::ZeroShotCloning);
            features.push(EngineFeature::ReferenceEncoding);
        }

        Self {
            info: TtsEngineInfo {
                id: "qwen3-tts".to_string(),
                name: "Qwen3-TTS".to_string(),
                version: "1.0.0".to_string(),
                description: "Alibaba's powerful speech generation model with multi-language support".to_string(),
                author: "Alibaba Cloud Qwen Team".to_string(),
                license: "Apache-2.0".to_string(),
                repository: Some("https://github.com/QwenLM/Qwen3-TTS".to_string()),
                engine_type: EngineType::Autoregressive,
                features,
            },
            model: None,
            config: None,
            resource_usage: ResourceUsage::default(),
            model_variant: variant,
        }
    }

    /// Get model variant
    pub fn variant(&self) -> AdapterVariant {
        self.model_variant
    }

    /// Check if model is loaded
    pub fn is_model_loaded(&self) -> bool {
        self.model.is_some()
    }
}

impl Default for Qwen3TtsEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl TtsEngine for Qwen3TtsEngine {
    fn info(&self) -> &TtsEngineInfo {
        &self.info
    }

    async fn initialize(&mut self, config: &EngineConfig) -> Result<()> {
        let model_dir = &config.model_dir;

        // Check for model files
        let model_safetensors = model_dir.join("model.safetensors");
        let config_json = model_dir.join("config.json");

        if !model_safetensors.exists() && !config_json.exists() {
            return Err(TtsError::ModelLoad {
                message: format!("Qwen3-TTS model files not found in {:?}", model_dir),
                component: "Qwen3TTS".to_string(),
                path: Some(model_dir.to_path_buf()),
            });
        }

        // Create Qwen config
        let qwen_config = QwenConfig {
            variant: self.model_variant.to_internal(),
            use_gpu: config.device.use_gpu,
            use_bf16: config.device.use_gpu,
            use_flash_attn: false,
            device_id: 0,
            verbose: false,
        };

        // Initialize model
        let model = Qwen3TtsModel::new(qwen_config)?;

        self.model = Some(Arc::new(Mutex::new(model)));
        self.config = Some(config.clone());
        self.resource_usage.loaded_models = 1;

        Ok(())
    }

    fn is_ready(&self) -> bool {
        self.model.is_some()
    }

    fn get_speakers(&self) -> Result<Vec<SpeakerInfo>> {
        // Return preset speakers for CustomVoice models
        let speakers = vec![
            SpeakerInfo {
                id: "serena".to_string(),
                name: "Serena".to_string(),
                language: "zh".to_string(),
                gender: Some("female".to_string()),
                description: Some("Warm, gentle young female voice (Chinese)".to_string()),
                preview_audio: None,
            },
            SpeakerInfo {
                id: "vivian".to_string(),
                name: "Vivian".to_string(),
                language: "zh".to_string(),
                gender: Some("female".to_string()),
                description: Some("Bright, slightly husky young female voice (Chinese)".to_string()),
                preview_audio: None,
            },
            SpeakerInfo {
                id: "uncle_fu".to_string(),
                name: "Uncle Fu".to_string(),
                language: "zh".to_string(),
                gender: Some("male".to_string()),
                description: Some("Deep, mellow mature male voice (Chinese)".to_string()),
                preview_audio: None,
            },
            SpeakerInfo {
                id: "ryan".to_string(),
                name: "Ryan".to_string(),
                language: "en".to_string(),
                gender: Some("male".to_string()),
                description: Some("Dynamic, rhythmic male voice (English)".to_string()),
                preview_audio: None,
            },
            SpeakerInfo {
                id: "aiden".to_string(),
                name: "Aiden".to_string(),
                language: "en".to_string(),
                gender: Some("male".to_string()),
                description: Some("Sunny, mid-frequency clear American male voice (English)".to_string()),
                preview_audio: None,
            },
            SpeakerInfo {
                id: "ono_anna".to_string(),
                name: "Ono Anna".to_string(),
                language: "ja".to_string(),
                gender: Some("female".to_string()),
                description: Some("Playful Japanese female voice".to_string()),
                preview_audio: None,
            },
            SpeakerInfo {
                id: "sohee".to_string(),
                name: "Sohee".to_string(),
                language: "ko".to_string(),
                gender: Some("female".to_string()),
                description: Some("Warm, emotional Korean female voice".to_string()),
                preview_audio: None,
            },
        ];

        Ok(speakers)
    }

    fn get_emotions(&self) -> Result<Vec<EmotionInfo>> {
        Ok(vec![
            EmotionInfo {
                id: "neutral".to_string(),
                name: "Neutral".to_string(),
                description: Some("Neutral emotion".to_string()),
                intensity_range: (0.0, 1.0),
                preview_audio: None,
            },
            EmotionInfo {
                id: "happy".to_string(),
                name: "Happy".to_string(),
                description: Some("Happy, cheerful emotion".to_string()),
                intensity_range: (0.0, 1.0),
                preview_audio: None,
            },
            EmotionInfo {
                id: "sad".to_string(),
                name: "Sad".to_string(),
                description: Some("Sad, melancholic emotion".to_string()),
                intensity_range: (0.0, 1.0),
                preview_audio: None,
            },
            EmotionInfo {
                id: "angry".to_string(),
                name: "Angry".to_string(),
                description: Some("Angry, aggressive emotion".to_string()),
                intensity_range: (0.0, 1.0),
                preview_audio: None,
            },
            EmotionInfo {
                id: "surprised".to_string(),
                name: "Surprised".to_string(),
                description: Some("Surprised, amazed emotion".to_string()),
                intensity_range: (0.0, 1.0),
                preview_audio: None,
            },
        ])
    }

    async fn synthesize(&self, request: &SynthesisRequest) -> Result<SynthesisResult> {
        let model_arc = self.model.as_ref().ok_or_else(|| TtsError::Inference {
            stage: crate::core::error::InferenceStage::GptGeneration,
            message: "Qwen3-TTS model not loaded".to_string(),
            recoverable: false,
        })?;

        let model = model_arc.lock().map_err(|_| TtsError::Internal {
            message: "Failed to lock model".to_string(),
            location: Some("Qwen3TtsEngine::synthesize".to_string()),
        })?;

        let start = std::time::Instant::now();

        // Use placeholder synthesis
        let result = model.synthesize(&request.text, None)?;

        let processing_time = start.elapsed();

        Ok(SynthesisResult {
            audio: result.audio,
            sample_rate: result.sample_rate,
            duration: result.duration,
            processing_time_ms: processing_time.as_millis() as u64,
            rtf: result.rtf,
            tokens: result.tokens,
            mel_spectrogram: None,
            speaker_embedding: None,
            metadata: HashMap::new(),
        })
    }

    async fn synthesize_streaming(
        &self,
        request: &SynthesisRequest,
        callback: StreamingCallback,
    ) -> Result<()> {
        // For now, use non-streaming synthesis and chunk the output
        let result = self.synthesize(request).await?;

        let chunk_size = 4096;
        let total_chunks = (result.audio.len() + chunk_size - 1) / chunk_size;

        for (i, chunk_start) in (0..result.audio.len()).step_by(chunk_size).enumerate() {
            let chunk_end = (chunk_start + chunk_size).min(result.audio.len());
            let chunk_samples = result.audio[chunk_start..chunk_end].to_vec();

            let chunk = AudioChunk {
                samples: chunk_samples,
                sample_rate: result.sample_rate,
                index: i,
                is_final: i == total_chunks - 1,
                timestamp_ms: (chunk_start as f64 / result.sample_rate as f64 * 1000.0) as u64,
            };

            callback(chunk)?;
        }

        Ok(())
    }

    async fn load_model(&mut self, _model_path: &Path) -> Result<()> {
        let qwen_config = QwenConfig {
            variant: self.model_variant.to_internal(),
            use_gpu: true,
            use_bf16: true,
            use_flash_attn: false,
            device_id: 0,
            verbose: false,
        };

        let model = Qwen3TtsModel::new(qwen_config)?;

        self.model = Some(Arc::new(Mutex::new(model)));
        self.resource_usage.loaded_models = 1;

        Ok(())
    }

    async fn unload_model(&mut self) -> Result<()> {
        self.model = None;
        self.resource_usage.loaded_models = 0;
        Ok(())
    }

    fn supported_languages(&self) -> Vec<LanguageSupport> {
        vec![
            LanguageSupport {
                code: "zh".to_string(),
                name: "Chinese".to_string(),
                native_name: Some("中文".to_string()),
                quality: 1.0,
                native: true,
            },
            LanguageSupport {
                code: "en".to_string(),
                name: "English".to_string(),
                native_name: Some("English".to_string()),
                quality: 1.0,
                native: true,
            },
            LanguageSupport {
                code: "ja".to_string(),
                name: "Japanese".to_string(),
                native_name: Some("日本語".to_string()),
                quality: 1.0,
                native: true,
            },
            LanguageSupport {
                code: "ko".to_string(),
                name: "Korean".to_string(),
                native_name: Some("한국어".to_string()),
                quality: 1.0,
                native: true,
            },
            LanguageSupport {
                code: "de".to_string(),
                name: "German".to_string(),
                native_name: Some("Deutsch".to_string()),
                quality: 1.0,
                native: true,
            },
            LanguageSupport {
                code: "fr".to_string(),
                name: "French".to_string(),
                native_name: Some("Français".to_string()),
                quality: 1.0,
                native: true,
            },
            LanguageSupport {
                code: "ru".to_string(),
                name: "Russian".to_string(),
                native_name: Some("Русский".to_string()),
                quality: 1.0,
                native: true,
            },
            LanguageSupport {
                code: "pt".to_string(),
                name: "Portuguese".to_string(),
                native_name: Some("Português".to_string()),
                quality: 1.0,
                native: true,
            },
            LanguageSupport {
                code: "es".to_string(),
                name: "Spanish".to_string(),
                native_name: Some("Español".to_string()),
                quality: 1.0,
                native: true,
            },
            LanguageSupport {
                code: "it".to_string(),
                name: "Italian".to_string(),
                native_name: Some("Italiano".to_string()),
                quality: 1.0,
                native: true,
            },
        ]
    }

    fn capabilities(&self) -> EngineCapabilities {
        EngineCapabilities {
            max_text_length: 4000,
            max_audio_duration: 300.0,
            sample_rates: vec![24000],
            streaming: true,
            batch_processing: true,
            min_reference_duration: 3.0,
            max_reference_duration: 30.0,
            typical_rtf: 0.1, // Qwen3-TTS is very fast
        }
    }

    fn resource_usage(&self) -> ResourceUsage {
        self.resource_usage.clone()
    }
}

/// Register Qwen3-TTS engine with the global registry
pub fn register_engine() -> Result<()> {
    let registry = crate::engine::global_registry();

    let info = TtsEngineInfo {
        id: "qwen3-tts".to_string(),
        name: "Qwen3-TTS".to_string(),
        version: "1.0.0".to_string(),
        description: "Alibaba's powerful speech generation model with multi-language support".to_string(),
        author: "Alibaba Cloud Qwen Team".to_string(),
        license: "Apache-2.0".to_string(),
        repository: Some("https://github.com/QwenLM/Qwen3-TTS".to_string()),
        engine_type: EngineType::Autoregressive,
        features: vec![
            EngineFeature::ZeroShotCloning,
            EngineFeature::MultiSpeaker,
            EngineFeature::EmotionControl,
            EngineFeature::Streaming,
            EngineFeature::MultiLanguage,
            EngineFeature::ReferenceEncoding,
        ],
    };

    registry.register_lazy("qwen3-tts", info, || {
        Ok(Box::new(Qwen3TtsEngine::new()))
    })?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_engine_info() {
        let engine = Qwen3TtsEngine::new();
        let info = engine.info();

        assert_eq!(info.id, "qwen3-tts");
        assert_eq!(info.name, "Qwen3-TTS");
        assert!(!info.features.is_empty());
    }

    #[test]
    fn test_engine_not_ready() {
        let engine = Qwen3TtsEngine::new();
        assert!(!engine.is_ready());
    }

    #[test]
    fn test_variant_features() {
        let variant = AdapterVariant::VoiceDesign17B;
        assert!(variant.supports_voice_design());
        assert!(!variant.supports_voice_cloning());

        let variant = AdapterVariant::Base17B;
        assert!(variant.supports_voice_cloning());
        assert!(!variant.supports_voice_design());
    }

    #[test]
    fn test_supported_languages() {
        let engine = Qwen3TtsEngine::new();
        let languages = engine.supported_languages();

        assert_eq!(languages.len(), 10);
        assert!(languages.iter().any(|l| l.code == "zh"));
        assert!(languages.iter().any(|l| l.code == "en"));
        assert!(languages.iter().any(|l| l.code == "ja"));
        assert!(languages.iter().any(|l| l.code == "ko"));
    }

    #[test]
    fn test_capabilities() {
        let engine = Qwen3TtsEngine::new();
        let caps = engine.capabilities();

        assert!(caps.streaming);
        assert!(caps.batch_processing);
        assert_eq!(caps.sample_rates, vec![24000]);
    }

    #[test]
    fn test_speakers() {
        let engine = Qwen3TtsEngine::new();
        let speakers = engine.get_speakers().unwrap();

        assert!(speakers.len() >= 7);
        assert!(speakers.iter().any(|s| s.id == "vivian"));
        assert!(speakers.iter().any(|s| s.id == "ryan"));
    }
}
