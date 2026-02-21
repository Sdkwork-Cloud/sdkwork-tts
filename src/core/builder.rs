//! Builder pattern implementations for IndexTTS2
//!
//! Provides ergonomic APIs for constructing complex objects
//! with sensible defaults and validation.

use std::path::{Path, PathBuf};

use super::error::{Result, TtsError};
use crate::inference::{IndexTTS2, InferenceConfig};

/// Builder for creating IndexTTS2 instances
///
/// # Example
/// ```rust,ignore
/// let tts = TtsBuilder::new("checkpoints/config.yaml")
///     .with_gpu(true)
///     .with_memory_limit(4 * 1024 * 1024 * 1024) // 4GB
///     .with_temperature(0.8)
///     .with_flow_steps(25)
///     .build()?;
/// ```
pub struct TtsBuilder {
    config_path: PathBuf,
    use_gpu: bool,
    memory_limit: Option<usize>,
    gpu_memory_limit: Option<usize>,
    inference_config: InferenceConfig,
    enable_metrics: bool,
    enable_resource_tracking: bool,
    lazy_load: bool,
}

impl TtsBuilder {
    /// Create a new TTS builder
    pub fn new<P: AsRef<Path>>(config_path: P) -> Self {
        Self {
            config_path: config_path.as_ref().to_path_buf(),
            use_gpu: false,
            memory_limit: None,
            gpu_memory_limit: None,
            inference_config: InferenceConfig::default(),
            enable_metrics: true,
            enable_resource_tracking: true,
            lazy_load: false,
        }
    }

    /// Use GPU if available
    pub fn with_gpu(mut self, enable: bool) -> Self {
        self.use_gpu = enable;
        self
    }

    /// Set memory limit in bytes
    pub fn with_memory_limit(mut self, bytes: usize) -> Self {
        self.memory_limit = Some(bytes);
        self
    }

    /// Set GPU memory limit in bytes
    pub fn with_gpu_memory_limit(mut self, bytes: usize) -> Self {
        self.gpu_memory_limit = Some(bytes);
        self
    }

    /// Set generation temperature
    pub fn with_temperature(mut self, temp: f32) -> Self {
        self.inference_config.temperature = temp.clamp(0.0, 2.0);
        self
    }

    /// Set top-k sampling
    pub fn with_top_k(mut self, top_k: usize) -> Self {
        self.inference_config.top_k = top_k;
        self
    }

    /// Set top-p (nucleus) sampling
    pub fn with_top_p(mut self, top_p: f32) -> Self {
        self.inference_config.top_p = top_p.clamp(0.0, 1.0);
        self
    }

    /// Set number of flow matching steps
    pub fn with_flow_steps(mut self, steps: usize) -> Self {
        self.inference_config.flow_steps = steps.max(1);
        self
    }

    /// Set classifier-free guidance rate
    pub fn with_cfg_rate(mut self, rate: f32) -> Self {
        self.inference_config.cfg_rate = rate.max(0.0);
        self
    }

    /// Set emotion alpha blending
    pub fn with_emotion_alpha(mut self, alpha: f32) -> Self {
        self.inference_config.emotion_alpha = alpha.clamp(0.0, 1.0);
        self
    }

    /// Enable/disable metrics collection
    pub fn with_metrics(mut self, enable: bool) -> Self {
        self.enable_metrics = enable;
        self
    }

    /// Enable/disable resource tracking
    pub fn with_resource_tracking(mut self, enable: bool) -> Self {
        self.enable_resource_tracking = enable;
        self
    }

    /// Enable lazy loading of models
    pub fn with_lazy_load(mut self, enable: bool) -> Self {
        self.lazy_load = enable;
        self
    }

    /// Set maximum mel tokens
    pub fn with_max_mel_tokens(mut self, max: usize) -> Self {
        self.inference_config.max_mel_tokens = max;
        self
    }

    /// Set repetition penalty
    pub fn with_repetition_penalty(mut self, penalty: f32) -> Self {
        self.inference_config.repetition_penalty = penalty.max(1.0);
        self
    }

    /// Enable de-rumble filter
    pub fn with_de_rumble(mut self, enable: bool) -> Self {
        self.inference_config.de_rumble = enable;
        self
    }

    /// Set de-rumble cutoff frequency
    pub fn with_de_rumble_cutoff(mut self, hz: f32) -> Self {
        self.inference_config.de_rumble_cutoff_hz = hz.max(20.0);
        self
    }

    /// Build the IndexTTS2 instance
    pub fn build(self) -> Result<IndexTTS2> {
        // Validate config path
        if !self.config_path.exists() {
            return Err(TtsError::Config {
                message: format!("Config file not found: {:?}", self.config_path),
                path: Some(self.config_path.clone()),
            });
        }

        // Update inference config with device preference
        let mut inference_config = self.inference_config;
        inference_config.use_gpu = self.use_gpu;

        // Build the TTS instance
        let mut tts = IndexTTS2::with_config(&self.config_path, inference_config)?;

        // Load weights unless lazy loading is enabled
        if !self.lazy_load {
            let model_dir = self.config_path.parent()
                .unwrap_or(Path::new("."));
            tts.load_weights(model_dir)?;
        }

        Ok(tts)
    }
}

/// Builder for creating InferenceConfig
///
/// # Example
/// ```rust,ignore
/// let config = TtsConfigBuilder::new()
///     .temperature(0.8)
///     .flow_steps(25)
///     .gpu(true)
///     .build();
/// ```
pub struct TtsConfigBuilder {
    config: InferenceConfig,
}

impl TtsConfigBuilder {
    /// Create a new config builder with defaults
    pub fn new() -> Self {
        Self {
            config: InferenceConfig::default(),
        }
    }

    /// Create a new config builder for high quality
    pub fn high_quality() -> Self {
        Self {
            config: InferenceConfig {
                temperature: 0.7,
                top_k: 0,
                top_p: 0.95,
                flow_steps: 50,
                cfg_rate: 0.7,
                ..Default::default()
            },
        }
    }

    /// Create a new config builder for fast inference
    pub fn fast() -> Self {
        Self {
            config: InferenceConfig {
                temperature: 0.9,
                top_k: 20,
                top_p: 0.9,
                flow_steps: 10,
                cfg_rate: 0.0,
                ..Default::default()
            },
        }
    }

    /// Create a new config builder for streaming
    pub fn streaming() -> Self {
        Self {
            config: InferenceConfig {
                temperature: 0.8,
                top_k: 0,
                top_p: 1.0,
                flow_steps: 15,
                cfg_rate: 0.0,
                max_mel_tokens: 500, // Shorter segments for streaming
                ..Default::default()
            },
        }
    }

    /// Set temperature
    pub fn temperature(mut self, value: f32) -> Self {
        self.config.temperature = value.clamp(0.0, 2.0);
        self
    }

    /// Set top-k
    pub fn top_k(mut self, value: usize) -> Self {
        self.config.top_k = value;
        self
    }

    /// Set top-p
    pub fn top_p(mut self, value: f32) -> Self {
        self.config.top_p = value.clamp(0.0, 1.0);
        self
    }

    /// Set flow steps
    pub fn flow_steps(mut self, value: usize) -> Self {
        self.config.flow_steps = value.max(1);
        self
    }

    /// Set CFG rate
    pub fn cfg_rate(mut self, value: f32) -> Self {
        self.config.cfg_rate = value.max(0.0);
        self
    }

    /// Set emotion alpha
    pub fn emotion_alpha(mut self, value: f32) -> Self {
        self.config.emotion_alpha = value.clamp(0.0, 1.0);
        self
    }

    /// Set emotion vector
    pub fn emotion_vector(mut self, vector: Vec<f32>) -> Self {
        self.config.emotion_vector = Some(vector);
        self
    }

    /// Enable emotion from text
    pub fn use_emotion_from_text(mut self, enable: bool) -> Self {
        self.config.use_emo_text = enable;
        self
    }

    /// Set emotion text
    pub fn emotion_text(mut self, text: impl Into<String>) -> Self {
        self.config.emotion_text = Some(text.into());
        self
    }

    /// Set max mel tokens
    pub fn max_mel_tokens(mut self, value: usize) -> Self {
        self.config.max_mel_tokens = value;
        self
    }

    /// Set repetition penalty
    pub fn repetition_penalty(mut self, value: f32) -> Self {
        self.config.repetition_penalty = value.max(1.0);
        self
    }

    /// Use GPU
    pub fn gpu(mut self, enable: bool) -> Self {
        self.config.use_gpu = enable;
        self
    }

    /// Enable de-rumble
    pub fn de_rumble(mut self, enable: bool) -> Self {
        self.config.de_rumble = enable;
        self
    }

    /// Set de-rumble cutoff
    pub fn de_rumble_cutoff(mut self, hz: f32) -> Self {
        self.config.de_rumble_cutoff_hz = hz;
        self
    }

    /// Enable verbose weights
    pub fn verbose_weights(mut self, enable: bool) -> Self {
        self.config.verbose_weights = enable;
        self
    }

    /// Build the configuration
    pub fn build(self) -> InferenceConfig {
        self.config
    }
}

impl Default for TtsConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Preset configurations for common use cases
pub mod presets {
    use super::*;

    /// High quality preset (slower, better quality)
    pub fn high_quality() -> InferenceConfig {
        TtsConfigBuilder::high_quality().build()
    }

    /// Fast preset (faster, lower quality)
    pub fn fast() -> InferenceConfig {
        TtsConfigBuilder::fast().build()
    }

    /// Streaming preset (optimized for real-time)
    pub fn streaming() -> InferenceConfig {
        TtsConfigBuilder::streaming().build()
    }

    /// Balanced preset (default)
    pub fn balanced() -> InferenceConfig {
        InferenceConfig::default()
    }

    /// Low latency preset (minimal delay)
    pub fn low_latency() -> InferenceConfig {
        InferenceConfig {
            temperature: 0.9,
            top_k: 10,
            top_p: 0.85,
            flow_steps: 5,
            cfg_rate: 0.0,
            max_mel_tokens: 200,
            ..Default::default()
        }
    }

    /// Maximum quality preset (slowest, best quality)
    pub fn maximum_quality() -> InferenceConfig {
        InferenceConfig {
            temperature: 0.6,
            top_k: 0,
            top_p: 0.98,
            flow_steps: 100,
            cfg_rate: 1.0,
            repetition_penalty: 1.2,
            ..Default::default()
        }
    }
}

/// Validation utilities
pub mod validation {
    use super::*;

    /// Validate a configuration
    pub fn validate_config(config: &InferenceConfig) -> Result<()> {
        // Validate temperature
        if config.temperature < 0.0 || config.temperature > 2.0 {
            return Err(TtsError::Validation {
                message: format!(
                    "Temperature must be between 0.0 and 2.0, got {}",
                    config.temperature
                ),
                field: Some("temperature".to_string()),
            });
        }

        // Validate top-p
        if config.top_p < 0.0 || config.top_p > 1.0 {
            return Err(TtsError::Validation {
                message: format!("Top-p must be between 0.0 and 1.0, got {}", config.top_p),
                field: Some("top_p".to_string()),
            });
        }

        // Validate flow steps
        if config.flow_steps == 0 {
            return Err(TtsError::Validation {
                message: "Flow steps must be at least 1".to_string(),
                field: Some("flow_steps".to_string()),
            });
        }

        // Validate emotion vector if provided
        if let Some(ref vector) = config.emotion_vector {
            if vector.len() != 8 {
                return Err(TtsError::Validation {
                    message: format!(
                        "Emotion vector must have 8 elements, got {}",
                        vector.len()
                    ),
                    field: Some("emotion_vector".to_string()),
                });
            }
        }

        Ok(())
    }

    /// Validate model path
    pub fn validate_model_path<P: AsRef<Path>>(path: P) -> Result<PathBuf> {
        let path = path.as_ref();

        if !path.exists() {
            return Err(TtsError::Io {
                message: format!("Model path does not exist: {:?}", path),
                path: Some(path.to_path_buf()),
            });
        }

        if !path.is_dir() {
            return Err(TtsError::Io {
                message: format!("Model path is not a directory: {:?}", path),
                path: Some(path.to_path_buf()),
            });
        }

        Ok(path.to_path_buf())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tts_builder_chain() {
        // This would normally need a real config file
        // Just testing the builder pattern compiles correctly
        let builder = TtsBuilder::new("config.yaml")
            .with_gpu(true)
            .with_temperature(0.8)
            .with_flow_steps(25)
            .with_memory_limit(4 * 1024 * 1024 * 1024);

        // Verify builder state
        assert!(builder.use_gpu);
        assert_eq!(builder.inference_config.temperature, 0.8);
        assert_eq!(builder.inference_config.flow_steps, 25);
        assert_eq!(builder.memory_limit, Some(4 * 1024 * 1024 * 1024));
    }

    #[test]
    fn test_config_builder() {
        let config = TtsConfigBuilder::new()
            .temperature(0.8)
            .flow_steps(25)
            .gpu(true)
            .emotion_alpha(0.5)
            .build();

        assert_eq!(config.temperature, 0.8);
        assert_eq!(config.flow_steps, 25);
        assert!(config.use_gpu);
        assert_eq!(config.emotion_alpha, 0.5);
    }

    #[test]
    fn test_config_presets() {
        let high_quality = presets::high_quality();
        assert_eq!(high_quality.flow_steps, 50);
        assert!(high_quality.cfg_rate > 0.0);

        let fast = presets::fast();
        assert_eq!(fast.flow_steps, 10);
        assert_eq!(fast.cfg_rate, 0.0);

        let streaming = presets::streaming();
        assert_eq!(streaming.flow_steps, 15);
        assert_eq!(streaming.max_mel_tokens, 500);
    }

    #[test]
    fn test_validation() {
        let valid_config = InferenceConfig::default();
        assert!(validation::validate_config(&valid_config).is_ok());

        let mut invalid_config = InferenceConfig::default();
        invalid_config.temperature = 3.0;
        assert!(validation::validate_config(&invalid_config).is_err());
    }

    #[test]
    fn test_config_clamping() {
        let config = TtsConfigBuilder::new()
            .temperature(5.0) // Will be clamped to 2.0
            .top_p(1.5) // Will be clamped to 1.0
            .build();

        assert_eq!(config.temperature, 2.0);
        assert_eq!(config.top_p, 1.0);
    }
}
