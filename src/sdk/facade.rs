//! SDK Facade - Unified high-level API for TTS operations
//!
//! This module provides the main `Sdk` struct which serves as the
//! primary entry point for third-party applications.
//!
//! # Example
//!
//! ```rust,ignore
//! use sdkwork_tts::sdk::Sdk;
//!
//! // Initialize SDK
//! let sdk = Sdk::new_default()?;
//!
//! // Simple synthesis (blocking)
//! let audio = sdk.synthesize("Hello world", "speaker.wav")?;
//! audio.save("output.wav")?;
//!
//! // Or use async API
//! tokio::spawn(async move {
//!     let audio = sdk.synthesize_async("Hello world", "speaker.wav").await?;
//! });
//!
//! // Or use the builder API
//! let audio = sdk.synthesis()
//!     .text("Hello world")
//!     .speaker("speaker.wav")
//!     .temperature(0.8)
//!     .build()?;
//! ```

use std::sync::{Arc, RwLock};
use std::path::Path;
use crate::core::{EventBus, MetricsCollector};
use crate::engine::EngineRegistry;
use crate::engine::traits::{SynthesisRequest, SpeakerReference};
use super::config::SdkConfig;
use super::types::{SdkStats, SynthesisOptions, SpeakerRef, AudioData, EngineInfo};
use super::error::{SdkError, Result as SdkResult};

/// Main SDK facade for third-party integration
///
/// # Example
///
/// ```rust,ignore
/// use sdkwork_tts::sdk::Sdk;
///
/// // Initialize SDK
/// let sdk = Sdk::new_default()?;
///
/// // Simple synthesis
/// let audio = sdk.synthesize("Hello world", "speaker.wav")?;
/// audio.save("output.wav")?;
///
/// // Or use the builder API
/// let audio = sdk.synthesis()
///     .text("Hello world")
///     .speaker("speaker.wav")
///     .temperature(0.8)
///     .build()?;
/// ```
pub struct Sdk {
    config: SdkConfig,
    registry: Arc<EngineRegistry>,
    event_bus: Option<Arc<EventBus>>,
    metrics: Option<Arc<MetricsCollector>>,
    stats: Arc<RwLock<SdkStats>>,
}

impl Sdk {
    /// Create new SDK with config and registry
    pub fn new(config: SdkConfig, registry: EngineRegistry) -> SdkResult<Self> {
        let event_bus = if config.event_logging {
            Some(Arc::new(EventBus::new()))
        } else {
            None
        };

        let metrics = if config.metrics {
            Some(Arc::new(MetricsCollector::new()))
        } else {
            None
        };

        Ok(Self {
            config,
            registry: Arc::new(registry),
            event_bus,
            metrics,
            stats: Arc::new(RwLock::new(SdkStats::default())),
        })
    }

    /// Create SDK with default settings (CPU, indextts2)
    pub fn new_default() -> SdkResult<Self> {
        Self::new(SdkConfig::default(), EngineRegistry::new())
    }

    /// Create SDK optimized for CPU usage
    pub fn cpu() -> SdkResult<Self> {
        Self::new(SdkConfig::cpu(), EngineRegistry::new())
    }

    /// Create SDK optimized for GPU usage
    pub fn gpu() -> SdkResult<Self> {
        Self::new(SdkConfig::gpu(), EngineRegistry::new())
    }

    /// Get SDK configuration
    pub fn config(&self) -> &SdkConfig {
        &self.config
    }

    /// Get event bus (if enabled)
    pub fn event_bus(&self) -> Option<&EventBus> {
        self.event_bus.as_ref().map(|e| e.as_ref())
    }

    /// Get metrics collector (if enabled)
    pub fn metrics(&self) -> Option<&MetricsCollector> {
        self.metrics.as_ref().map(|m| m.as_ref())
    }

    /// Get SDK statistics
    pub fn stats(&self) -> SdkStats {
        self.stats.read().unwrap().clone()
    }

    /// List available engines
    pub fn list_engines(&self) -> SdkResult<Vec<EngineInfo>> {
        let engines = self.registry.list_engines()
            .map_err(|e| SdkError::Internal { message: e.to_string() })?;

        Ok(engines.into_iter().map(|e| EngineInfo {
            id: e.id,
            name: e.name,
            version: e.version,
            languages: vec![], // TODO: Get from engine
            features: e.features.iter().map(|f| format!("{:?}", f)).collect(),
            available: true,
        }).collect())
    }

    /// Get default engine ID
    pub fn default_engine(&self) -> &str {
        &self.config.default_engine
    }

    /// Simple synthesis with text and speaker path (blocking)
    ///
    /// This is a convenience method for simple use cases.
    /// For better control, use `synthesize_with_options` or `synthesis` builder.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let audio = sdk.synthesize("Hello world", "speaker.wav")?;
    /// audio.save("output.wav")?;
    /// ```
    pub fn synthesize(&self, text: &str, speaker: &str) -> SdkResult<AudioData> {
        let options = SynthesisOptions::new(text, speaker);
        self.synthesize_with_options(&options)
    }

    /// Async synthesis with text and speaker path
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let audio = sdk.synthesize_async("Hello world", "speaker.wav").await?;
    /// ```
    pub async fn synthesize_async(&self, text: &str, speaker: &str) -> SdkResult<AudioData> {
        let options = SynthesisOptions::new(text, speaker);
        self.synthesize_with_options_async(&options).await
    }

    /// Synthesis with options (blocking)
    ///
    /// This method blocks the current thread until synthesis is complete.
    /// For non-blocking operation, use `synthesize_with_options_async`.
    pub fn synthesize_with_options(&self, options: &SynthesisOptions) -> SdkResult<AudioData> {
        // Try to get current runtime handle
        match tokio::runtime::Handle::try_current() {
            Ok(rt) => {
                // We're already in a tokio runtime, use block_in_place
                tokio::task::block_in_place(|| {
                    rt.block_on(self.synthesize_with_options_async(options))
                })
            }
            Err(_) => {
                // No runtime, create a new one
                let runtime = tokio::runtime::Runtime::new()
                    .map_err(|e| SdkError::Internal { message: e.to_string() })?;
                runtime.block_on(self.synthesize_with_options_async(options))
            }
        }
    }

    /// Async synthesis with options
    ///
    /// This is the main synthesis method. All other synthesis methods
    /// are convenience wrappers around this one.
    pub async fn synthesize_with_options_async(&self, options: &SynthesisOptions) -> SdkResult<AudioData> {
        // Update stats
        {
            let mut stats = self.stats.write().unwrap();
            stats.total_synthesis += 1;
        }

        // Get engine
        let engine = self.registry.get_engine(&self.config.default_engine)
            .map_err(|e| SdkError::Engine {
                engine_id: self.config.default_engine.clone(),
                error: e.to_string(),
            })?;

        // Build synthesis request
        let request = self.build_synthesis_request(options)?;

        // Execute synthesis
        let result = engine.synthesize(&request).await
            .map_err(|e| SdkError::Synthesis {
                message: "Synthesis failed".to_string(),
                details: Some(e.to_string()),
            })?;

        // Update stats
        {
            let mut stats = self.stats.write().unwrap();
            stats.successful_synthesis += 1;
            stats.total_audio_secs += result.duration as f64;
            stats.total_processing_secs += result.processing_time_ms as f64 / 1000.0;
        }

        Ok(AudioData::new(result.audio, result.sample_rate))
    }

    /// Build synthesis request from options
    fn build_synthesis_request(&self, options: &SynthesisOptions) -> SdkResult<SynthesisRequest> {
        let speaker_ref = match &options.speaker {
            SpeakerRef::AudioPath(path) => SpeakerReference::AudioPath(path.clone()),
            SpeakerRef::SpeakerId(id) => SpeakerReference::Id(id.clone()),
            SpeakerRef::Default => SpeakerReference::Id("default".to_string()),
            SpeakerRef::VoiceClone { audio_path, text: _, strength: _ } => {
                SpeakerReference::AudioPath(audio_path.clone())
            }
            SpeakerRef::VoiceDesign { description, preset: _ } => {
                SpeakerReference::Id(description.clone())
            }
        };

        Ok(SynthesisRequest {
            text: options.text.clone(),
            speaker: speaker_ref,
            emotion: None,
            params: crate::engine::traits::SynthesisParams {
                temperature: options.params.temperature,
                top_k: options.params.top_k,
                top_p: options.params.top_p,
                repetition_penalty: options.params.repetition_penalty,
                ..Default::default()
            },
            output_format: Default::default(),
            request_id: None,
        })
    }

    /// Create synthesis builder
    pub fn synthesis(&self) -> SynthesisBuilder<'_> {
        SynthesisBuilder::new(self)
    }

    /// Save audio to file
    pub fn save_audio(&self, audio: &AudioData, path: impl AsRef<Path>) -> SdkResult<()> {
        use hound::{WavSpec, WavWriter};

        let spec = WavSpec {
            channels: audio.channels,
            sample_rate: audio.sample_rate,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };

        let mut writer = WavWriter::create(path.as_ref(), spec)
            .map_err(|e| SdkError::Audio {
                operation: "save".to_string(),
                message: e.to_string(),
            })?;

        for sample in audio.to_i16() {
            writer.write_sample(sample)
                .map_err(|e| SdkError::Audio {
                    operation: "write".to_string(),
                    message: e.to_string(),
                })?;
        }

        writer.finalize()
            .map_err(|e| SdkError::Audio {
                operation: "finalize".to_string(),
                message: e.to_string(),
            })?;

        Ok(())
    }

    /// Get engine registry
    pub fn registry(&self) -> &EngineRegistry {
        &self.registry
    }
}

/// Synthesis builder for fluent API
pub struct SynthesisBuilder<'a> {
    sdk: &'a Sdk,
    text: String,
    speaker: SpeakerRef,
    emotion: Option<super::types::EmotionRef>,
    language: Option<String>,
    temperature: f32,
    top_k: usize,
    top_p: f32,
    output_path: Option<std::path::PathBuf>,
}

impl<'a> SynthesisBuilder<'a> {
    /// Create new builder
    pub fn new(sdk: &'a Sdk) -> Self {
        Self {
            sdk,
            text: String::new(),
            speaker: SpeakerRef::Default,
            emotion: None,
            language: None,
            temperature: 0.8,
            top_k: 50,
            top_p: 0.95,
            output_path: None,
        }
    }

    /// Set text to synthesize
    pub fn text(mut self, text: impl Into<String>) -> Self {
        self.text = text.into();
        self
    }

    /// Set speaker reference
    pub fn speaker(mut self, speaker: impl Into<SpeakerRef>) -> Self {
        self.speaker = speaker.into();
        self
    }

    /// Set emotion
    pub fn emotion(mut self, emotion: impl Into<super::types::EmotionRef>, intensity: f32) -> Self {
        self.emotion = Some(emotion.into());
        let _ = intensity; // TODO: Use intensity
        self
    }

    /// Set language
    pub fn language(mut self, language: impl Into<String>) -> Self {
        self.language = Some(language.into());
        self
    }

    /// Set temperature
    pub fn temperature(mut self, temp: f32) -> Self {
        self.temperature = temp.clamp(0.0, 1.0);
        self
    }

    /// Set top-k
    pub fn top_k(mut self, k: usize) -> Self {
        self.top_k = k;
        self
    }

    /// Set top-p
    pub fn top_p(mut self, p: f32) -> Self {
        self.top_p = p.clamp(0.0, 1.0);
        self
    }

    /// Set output path
    pub fn output(mut self, path: impl Into<std::path::PathBuf>) -> Self {
        self.output_path = Some(path.into());
        self
    }

    /// Build and execute synthesis
    pub fn build(self) -> SdkResult<AudioData> {
        let options = SynthesisOptions {
            text: self.text,
            speaker: self.speaker,
            emotion: self.emotion,
            language: self.language,
            params: crate::sdk::types::GenerationParams {
                temperature: self.temperature,
                top_k: self.top_k,
                top_p: self.top_p,
                ..Default::default()
            },
            output_path: self.output_path,
        };

        self.sdk.synthesize_with_options(&options)
    }

    /// Build and save to file
    pub fn save(self, path: impl AsRef<Path>) -> SdkResult<()> {
        let sdk = self.sdk;
        let audio = self.build()?;
        sdk.save_audio(&audio, path)?;
        Ok(())
    }
}
