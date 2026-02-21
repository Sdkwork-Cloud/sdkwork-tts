//! IndexTTS2 Engine Adapter
//!
//! This module provides a TtsEngine implementation wrapper for IndexTTS2,
//! enabling it to be used through the unified TTS framework.

use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;

use crate::core::error::{Result, TtsError};
use crate::inference::{IndexTTS2, InferenceConfig};
use crate::engine::traits::{
    TtsEngine, TtsEngineInfo, EngineType, EngineFeature,
    SynthesisRequest, SynthesisResult, AudioChunk, StreamingCallback,
    SpeakerInfo, EmotionInfo, LanguageSupport, EngineCapabilities, ResourceUsage,
};
use crate::engine::config::EngineConfig;

/// IndexTTS2 engine adapter
pub struct IndexTTS2Engine {
    /// Engine information
    info: TtsEngineInfo,
    /// Underlying IndexTTS2 instance (wrapped in Mutex for interior mutability)
    tts: Option<Arc<Mutex<IndexTTS2>>>,
    /// Configuration
    config: Option<EngineConfig>,
    /// Resource usage tracking
    resource_usage: ResourceUsage,
}

impl IndexTTS2Engine {
    /// Create a new IndexTTS2 engine
    pub fn new() -> Self {
        Self {
            info: TtsEngineInfo {
                id: "indextts2".to_string(),
                name: "IndexTTS2".to_string(),
                version: env!("CARGO_PKG_VERSION").to_string(),
                description: "Bilibili's Industrial-Level Controllable and Efficient Zero-Shot Text-To-Speech System".to_string(),
                author: "Bilibili".to_string(),
                license: "MIT".to_string(),
                repository: Some("https://github.com/index-tts/index-tts".to_string()),
                engine_type: EngineType::FlowMatching,
                features: vec![
                    EngineFeature::ZeroShotCloning,
                    EngineFeature::MultiSpeaker,
                    EngineFeature::EmotionControl,
                    EngineFeature::Streaming,
                    EngineFeature::ReferenceEncoding,
                    EngineFeature::TextEmotion,
                ],
            },
            tts: None,
            config: None,
            resource_usage: ResourceUsage::default(),
        }
    }

    /// Create with existing IndexTTS2 instance
    pub fn with_instance(tts: IndexTTS2) -> Self {
        let mut engine = Self::new();
        engine.tts = Some(Arc::new(Mutex::new(tts)));
        engine.resource_usage.loaded_models = 1;
        engine
    }

    /// Convert synthesis params to inference config
    fn params_to_config(params: &crate::engine::traits::SynthesisParams) -> InferenceConfig {
        InferenceConfig {
            temperature: params.temperature,
            top_k: params.top_k,
            top_p: params.top_p,
            flow_steps: params.diffusion_steps,
            cfg_rate: params.cfg_rate,
            repetition_penalty: params.repetition_penalty,
            ..Default::default()
        }
    }
}

impl Default for IndexTTS2Engine {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl TtsEngine for IndexTTS2Engine {
    fn info(&self) -> &TtsEngineInfo {
        &self.info
    }

    async fn initialize(&mut self, config: &EngineConfig) -> Result<()> {
        let config_path = config.model_dir.join("config.yaml");
        
        if !config_path.exists() {
            return Err(TtsError::Config {
                message: format!("Config file not found: {:?}", config_path),
                path: Some(config_path.clone()),
            });
        }

        let mut inference_config = InferenceConfig::default();
        inference_config.use_gpu = config.device.use_gpu;

        let mut tts = IndexTTS2::with_config(&config_path, inference_config)?;
        
        let model_dir = &config.model_dir;
        tts.load_weights(model_dir)?;

        self.tts = Some(Arc::new(Mutex::new(tts)));
        self.config = Some(config.clone());
        self.resource_usage.loaded_models = 1;

        Ok(())
    }

    fn is_ready(&self) -> bool {
        self.tts.is_some()
    }

    fn get_speakers(&self) -> Result<Vec<SpeakerInfo>> {
        Ok(vec![])
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
                description: Some("Happy emotion".to_string()),
                intensity_range: (0.0, 1.0),
                preview_audio: None,
            },
            EmotionInfo {
                id: "sad".to_string(),
                name: "Sad".to_string(),
                description: Some("Sad emotion".to_string()),
                intensity_range: (0.0, 1.0),
                preview_audio: None,
            },
            EmotionInfo {
                id: "angry".to_string(),
                name: "Angry".to_string(),
                description: Some("Angry emotion".to_string()),
                intensity_range: (0.0, 1.0),
                preview_audio: None,
            },
        ])
    }

    async fn synthesize(&self, request: &SynthesisRequest) -> Result<SynthesisResult> {
        let tts_arc = self.tts.as_ref().ok_or_else(|| TtsError::Inference {
            stage: crate::core::error::InferenceStage::GptGeneration,
            message: "Engine not initialized".to_string(),
            recoverable: false,
        })?;

        let speaker_path = match &request.speaker {
            crate::engine::traits::SpeakerReference::AudioPath(path) => path.clone(),
            crate::engine::traits::SpeakerReference::Id(id) => {
                return Err(TtsError::Validation {
                    message: format!("Speaker ID '{}' not supported, use audio path", id),
                    field: Some("speaker".to_string()),
                });
            }
            crate::engine::traits::SpeakerReference::AudioSamples { .. } => {
                return Err(TtsError::Validation {
                    message: "Audio samples not supported, use audio path".to_string(),
                    field: Some("speaker".to_string()),
                });
            }
            crate::engine::traits::SpeakerReference::Embedding(_) => {
                return Err(TtsError::Validation {
                    message: "Speaker embedding not supported, use audio path".to_string(),
                    field: Some("speaker".to_string()),
                });
            }
        };

        let start = std::time::Instant::now();

        let emotion_audio: Option<std::path::PathBuf> = request.emotion.as_ref().and_then(|e| e.reference_audio.clone());
        
        let mut tts = tts_arc.lock().map_err(|_| TtsError::Internal {
            message: "Failed to lock engine".to_string(),
            location: None,
        })?;
        
        let result = tts.infer_with_emotion(
            &request.text,
            &speaker_path,
            emotion_audio.as_ref(),
        )?;

        let processing_time = start.elapsed();
        let duration = result.duration();
        let rtf = if processing_time.as_secs_f64() > 0.0 {
            (duration as f64 / processing_time.as_secs_f64()) as f32
        } else {
            0.0
        };

        Ok(SynthesisResult {
            audio: result.audio.clone(),
            sample_rate: 22050,
            duration,
            processing_time_ms: processing_time.as_millis() as u64,
            rtf,
            tokens: Some(result.mel_codes.clone()),
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

    async fn load_model(&mut self, model_path: &Path) -> Result<()> {
        let config_path = model_path.join("config.yaml");
        
        if !config_path.exists() {
            return Err(TtsError::Config {
                message: format!("Config file not found: {:?}", config_path),
                path: Some(config_path.clone()),
            });
        }

        let inference_config = InferenceConfig::default();
        let mut tts = IndexTTS2::with_config(&config_path, inference_config)?;
        tts.load_weights(model_path)?;

        self.tts = Some(Arc::new(Mutex::new(tts)));
        self.resource_usage.loaded_models = 1;

        Ok(())
    }

    async fn unload_model(&mut self) -> Result<()> {
        self.tts = None;
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
                quality: 0.95,
                native: true,
            },
            LanguageSupport {
                code: "ja".to_string(),
                name: "Japanese".to_string(),
                native_name: Some("日本語".to_string()),
                quality: 0.85,
                native: false,
            },
        ]
    }

    fn capabilities(&self) -> EngineCapabilities {
        EngineCapabilities {
            max_text_length: 1000,
            max_audio_duration: 60.0,
            sample_rates: vec![22050, 16000],
            streaming: true,
            batch_processing: false,
            min_reference_duration: 3.0,
            max_reference_duration: 30.0,
            typical_rtf: 0.8,
        }
    }

    fn resource_usage(&self) -> ResourceUsage {
        self.resource_usage.clone()
    }
}

/// Register IndexTTS2 engine with the global registry
pub fn register_engine() -> Result<()> {
    let registry = crate::engine::global_registry();
    
    let info = TtsEngineInfo {
        id: "indextts2".to_string(),
        name: "IndexTTS2".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        description: "Bilibili's Industrial-Level Controllable and Efficient Zero-Shot Text-To-Speech System".to_string(),
        author: "Bilibili".to_string(),
        license: "MIT".to_string(),
        repository: Some("https://github.com/index-tts/index-tts".to_string()),
        engine_type: EngineType::FlowMatching,
        features: vec![
            EngineFeature::ZeroShotCloning,
            EngineFeature::MultiSpeaker,
            EngineFeature::EmotionControl,
            EngineFeature::Streaming,
            EngineFeature::ReferenceEncoding,
            EngineFeature::TextEmotion,
        ],
    };

    registry.register_lazy("indextts2", info, || {
        Ok(Box::new(IndexTTS2Engine::new()))
    })?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_engine_info() {
        let engine = IndexTTS2Engine::new();
        let info = engine.info();
        
        assert_eq!(info.id, "indextts2");
        assert_eq!(info.name, "IndexTTS2");
        assert!(!info.features.is_empty());
    }

    #[test]
    fn test_engine_not_ready() {
        let engine = IndexTTS2Engine::new();
        assert!(!engine.is_ready());
    }

    #[test]
    fn test_supported_languages() {
        let engine = IndexTTS2Engine::new();
        let languages = engine.supported_languages();
        
        assert!(!languages.is_empty());
        assert!(languages.iter().any(|l| l.code == "zh"));
        assert!(languages.iter().any(|l| l.code == "en"));
    }

    #[test]
    fn test_capabilities() {
        let engine = IndexTTS2Engine::new();
        let caps = engine.capabilities();
        
        assert!(caps.streaming);
        assert!(caps.max_text_length > 0);
    }
}
