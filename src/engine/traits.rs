//! Core traits for TTS engine abstraction
//!
//! These traits define the contract for all TTS engines,
//! enabling a unified interface for different implementations.

use std::collections::HashMap;
use std::path::Path;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::core::error::{Result, TtsError};
use super::config::EngineConfig;

/// Core trait for all TTS engines
///
/// This trait defines the unified interface that all TTS engines must implement.
/// It follows the Interface Segregation Principle and enables polymorphic
/// treatment of different TTS implementations.
#[async_trait]
pub trait TtsEngine: Send + Sync {
    /// Get engine information
    fn info(&self) -> &TtsEngineInfo;

    /// Initialize the engine with configuration
    async fn initialize(&mut self, config: &EngineConfig) -> Result<()>;

    /// Check if the engine is ready for synthesis
    fn is_ready(&self) -> bool;

    /// Get available speakers
    fn get_speakers(&self) -> Result<Vec<SpeakerInfo>>;

    /// Get available emotions/styles
    fn get_emotions(&self) -> Result<Vec<EmotionInfo>>;

    /// Synthesize speech from text
    async fn synthesize(&self, request: &SynthesisRequest) -> Result<SynthesisResult>;

    /// Synthesize speech with streaming output
    async fn synthesize_streaming(
        &self,
        request: &SynthesisRequest,
        callback: StreamingCallback,
    ) -> Result<()>;

    /// Load a custom model from path
    async fn load_model(&mut self, model_path: &Path) -> Result<()>;

    /// Unload the current model
    async fn unload_model(&mut self) -> Result<()>;

    /// Get supported languages
    fn supported_languages(&self) -> Vec<LanguageSupport>;

    /// Get engine capabilities
    fn capabilities(&self) -> EngineCapabilities;

    /// Get current resource usage
    fn resource_usage(&self) -> ResourceUsage;

    /// Warm up the engine (pre-load models, allocate memory)
    async fn warmup(&mut self) -> Result<()> {
        Ok(())
    }

    /// Reset engine state
    fn reset(&mut self) -> Result<()> {
        Ok(())
    }
}

/// Streaming callback type
pub type StreamingCallback = std::sync::Arc<dyn Fn(AudioChunk) -> Result<()> + Send + Sync>;

/// Engine information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TtsEngineInfo {
    /// Unique engine identifier
    pub id: String,
    /// Human-readable name
    pub name: String,
    /// Engine version
    pub version: String,
    /// Engine description
    pub description: String,
    /// Author/organization
    pub author: String,
    /// License
    pub license: String,
    /// Repository URL
    pub repository: Option<String>,
    /// Engine type
    pub engine_type: EngineType,
    /// Supported features
    pub features: Vec<EngineFeature>,
}

/// Engine type classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EngineType {
    /// Autoregressive models (GPT-based, etc.)
    Autoregressive,
    /// Diffusion-based models
    Diffusion,
    /// Flow-based models
    FlowMatching,
    /// VAE-based models (VITS, etc.)
    Variational,
    /// Hybrid models
    Hybrid,
    /// Neural vocoder only
    Vocoder,
}

/// Engine features
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EngineFeature {
    /// Zero-shot voice cloning
    ZeroShotCloning,
    /// Multi-speaker support
    MultiSpeaker,
    /// Emotion/style control
    EmotionControl,
    /// Streaming synthesis
    Streaming,
    /// Multi-language support
    MultiLanguage,
    /// Prosody control
    ProsodyControl,
    /// Speed control
    SpeedControl,
    /// Pitch control
    PitchControl,
    /// Reference audio encoding
    ReferenceEncoding,
    /// Text-based emotion extraction
    TextEmotion,
}

/// Engine capabilities
#[derive(Debug, Clone, Default)]
pub struct EngineCapabilities {
    /// Maximum text length
    pub max_text_length: usize,
    /// Maximum audio duration in seconds
    pub max_audio_duration: f32,
    /// Supported sample rates
    pub sample_rates: Vec<u32>,
    /// Supports streaming
    pub streaming: bool,
    /// Supports batch processing
    pub batch_processing: bool,
    /// Minimum reference audio duration
    pub min_reference_duration: f32,
    /// Maximum reference audio duration
    pub max_reference_duration: f32,
    /// Real-time factor (lower is better)
    pub typical_rtf: f32,
}

/// Language support information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LanguageSupport {
    /// ISO 639-1 language code
    pub code: String,
    /// Language name
    pub name: String,
    /// Native name
    pub native_name: Option<String>,
    /// Quality level (0.0 - 1.0)
    pub quality: f32,
    /// Supports this language natively
    pub native: bool,
}

/// Resource usage information
#[derive(Debug, Clone, Default)]
pub struct ResourceUsage {
    /// CPU memory usage in bytes
    pub cpu_memory: usize,
    /// GPU memory usage in bytes
    pub gpu_memory: usize,
    /// Model size in bytes
    pub model_size: usize,
    /// Number of loaded models
    pub loaded_models: usize,
    /// Inference count
    pub inference_count: u64,
    /// Total inference time
    pub total_inference_time_ms: u64,
}

/// Synthesis request
#[derive(Debug, Clone)]
pub struct SynthesisRequest {
    /// Text to synthesize
    pub text: String,
    /// Speaker reference (ID or audio path)
    pub speaker: SpeakerReference,
    /// Emotion/style specification
    pub emotion: Option<EmotionSpec>,
    /// Synthesis parameters
    pub params: SynthesisParams,
    /// Output format
    pub output_format: OutputFormat,
    /// Request ID for tracking
    pub request_id: Option<String>,
}

/// Speaker reference
#[derive(Debug, Clone)]
pub enum SpeakerReference {
    /// Built-in speaker ID
    Id(String),
    /// Reference audio file path
    AudioPath(std::path::PathBuf),
    /// Reference audio samples
    AudioSamples {
        samples: Vec<f32>,
        sample_rate: u32,
    },
    /// Speaker embedding vector
    Embedding(Vec<f32>),
}

/// Emotion specification
#[derive(Debug, Clone)]
pub struct EmotionSpec {
    /// Emotion name or ID
    pub name: Option<String>,
    /// Emotion vector (model-specific)
    pub vector: Option<Vec<f32>>,
    /// Emotion intensity (0.0 - 1.0)
    pub intensity: f32,
    /// Reference audio for emotion extraction
    pub reference_audio: Option<std::path::PathBuf>,
    /// Text for emotion extraction
    pub reference_text: Option<String>,
}

/// Synthesis parameters
#[derive(Debug, Clone)]
pub struct SynthesisParams {
    /// Sampling temperature
    pub temperature: f32,
    /// Top-k sampling
    pub top_k: usize,
    /// Top-p (nucleus) sampling
    pub top_p: f32,
    /// Repetition penalty
    pub repetition_penalty: f32,
    /// Speed multiplier
    pub speed: f32,
    /// Pitch shift in semitones
    pub pitch_shift: f32,
    /// Energy/energy variance
    pub energy: f32,
    /// Denoising strength (for diffusion models)
    pub denoising_strength: f32,
    /// Number of diffusion steps
    pub diffusion_steps: usize,
    /// CFG (classifier-free guidance) rate
    pub cfg_rate: f32,
    /// Seed for reproducibility
    pub seed: Option<u64>,
    /// Custom parameters (engine-specific)
    pub custom: HashMap<String, String>,
}

impl Default for SynthesisParams {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_k: 50,
            top_p: 0.95,
            repetition_penalty: 1.0,
            speed: 1.0,
            pitch_shift: 0.0,
            energy: 1.0,
            denoising_strength: 0.6,
            diffusion_steps: 25,
            cfg_rate: 0.0,
            seed: None,
            custom: HashMap::new(),
        }
    }
}

/// Output format specification
#[derive(Debug, Clone)]
pub struct OutputFormat {
    /// Sample rate
    pub sample_rate: u32,
    /// Number of channels
    pub channels: u16,
    /// Bit depth
    pub bit_depth: u16,
    /// Audio format
    pub format: AudioFormat,
    /// Apply post-processing
    pub post_process: PostProcessOptions,
}

impl Default for OutputFormat {
    fn default() -> Self {
        Self {
            sample_rate: 22050,
            channels: 1,
            bit_depth: 16,
            format: AudioFormat::Wav,
            post_process: PostProcessOptions::default(),
        }
    }
}

/// Audio format
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AudioFormat {
    Wav,
    Mp3,
    Ogg,
    Flac,
    Raw,
}

/// Post-processing options
#[derive(Debug, Clone, Default)]
pub struct PostProcessOptions {
    /// Normalize audio volume
    pub normalize: bool,
    /// Remove DC offset
    pub remove_dc: bool,
    /// Apply high-pass filter (de-rumble)
    pub high_pass_filter: Option<f32>,
    /// Apply low-pass filter
    pub low_pass_filter: Option<f32>,
    /// Trim silence
    pub trim_silence: bool,
    /// Silence threshold in dB
    pub silence_threshold_db: f32,
}

/// Synthesis result
#[derive(Debug, Clone)]
pub struct SynthesisResult {
    /// Audio samples
    pub audio: Vec<f32>,
    /// Sample rate
    pub sample_rate: u32,
    /// Audio duration in seconds
    pub duration: f32,
    /// Processing time in milliseconds
    pub processing_time_ms: u64,
    /// Real-time factor
    pub rtf: f32,
    /// Generated tokens/codes (if available)
    pub tokens: Option<Vec<u32>>,
    /// Mel spectrogram (if available)
    pub mel_spectrogram: Option<Vec<f32>>,
    /// Speaker embedding used
    pub speaker_embedding: Option<Vec<f32>>,
    /// Engine-specific metadata
    pub metadata: HashMap<String, String>,
}

impl SynthesisResult {
    /// Save audio to file
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        crate::audio::AudioOutput::save(&self.audio, self.sample_rate, path)
            .map_err(|e| TtsError::Audio {
                message: format!("Failed to save audio: {}", e),
                operation: crate::core::error::AudioOperation::Saving,
            })
    }

    /// Get audio as bytes in WAV format
    pub fn to_wav_bytes(&self) -> Result<Vec<u8>> {
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: self.sample_rate,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        
        let mut buffer = std::io::Cursor::new(Vec::new());
        {
            let mut writer = hound::WavWriter::new(&mut buffer, spec)
                .map_err(|e| TtsError::Audio {
                    message: format!("Failed to create WAV writer: {}", e),
                    operation: crate::core::error::AudioOperation::Saving,
                })?;
            
            for &sample in &self.audio {
                let sample_i16 = (sample.clamp(-1.0, 1.0) * 32767.0) as i16;
                writer.write_sample(sample_i16)
                    .map_err(|e| TtsError::Audio {
                        message: format!("Failed to write sample: {}", e),
                        operation: crate::core::error::AudioOperation::Saving,
                    })?;
            }
            
            writer.finalize()
                .map_err(|e| TtsError::Audio {
                    message: format!("Failed to finalize WAV: {}", e),
                    operation: crate::core::error::AudioOperation::Saving,
                })?;
        }
        
        Ok(buffer.into_inner())
    }
}

/// Audio chunk for streaming
#[derive(Debug, Clone)]
pub struct AudioChunk {
    /// Audio samples
    pub samples: Vec<f32>,
    /// Sample rate
    pub sample_rate: u32,
    /// Chunk index
    pub index: usize,
    /// Is this the final chunk
    pub is_final: bool,
    /// Timestamp in milliseconds
    pub timestamp_ms: u64,
}

/// Device configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceConfig {
    /// Use GPU if available
    pub use_gpu: bool,
    /// GPU device ID
    pub gpu_id: usize,
    /// Use mixed precision
    pub mixed_precision: bool,
    /// Number of threads for CPU
    pub num_threads: usize,
}

impl Default for DeviceConfig {
    fn default() -> Self {
        Self {
            use_gpu: false,
            gpu_id: 0,
            mixed_precision: false,
            num_threads: 4,
        }
    }
}

/// Memory configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryConfig {
    /// Maximum CPU memory in bytes (0 = unlimited)
    pub max_cpu_memory: usize,
    /// Maximum GPU memory in bytes (0 = unlimited)
    pub max_gpu_memory: usize,
    /// Enable memory pooling
    pub enable_pooling: bool,
    /// Idle timeout for model unloading (seconds)
    pub idle_timeout_secs: u64,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            max_cpu_memory: 0,
            max_gpu_memory: 0,
            enable_pooling: true,
            idle_timeout_secs: 300,
        }
    }
}

/// Performance configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceConfig {
    /// Enable caching
    pub enable_cache: bool,
    /// Cache size in bytes
    pub cache_size: usize,
    /// Batch size for batch processing
    pub batch_size: usize,
    /// Enable async processing
    pub async_processing: bool,
    /// Number of worker threads
    pub worker_threads: usize,
}

impl Default for PerformanceConfig {
    fn default() -> Self {
        Self {
            enable_cache: true,
            cache_size: 1024 * 1024 * 1024, // 1GB
            batch_size: 1,
            async_processing: true,
            worker_threads: 4,
        }
    }
}

/// Speaker information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeakerInfo {
    /// Speaker ID
    pub id: String,
    /// Speaker name
    pub name: String,
    /// Language
    pub language: String,
    /// Gender
    pub gender: Option<String>,
    /// Description
    pub description: Option<String>,
    /// Preview audio path
    pub preview_audio: Option<std::path::PathBuf>,
}

/// Emotion information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmotionInfo {
    /// Emotion ID
    pub id: String,
    /// Emotion name
    pub name: String,
    /// Description
    pub description: Option<String>,
    /// Intensity range
    pub intensity_range: (f32, f32),
    /// Preview audio path
    pub preview_audio: Option<std::path::PathBuf>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_synthesis_params_default() {
        let params = SynthesisParams::default();
        assert_eq!(params.temperature, 1.0);
        assert_eq!(params.top_k, 50);
        assert_eq!(params.speed, 1.0);
    }

    #[test]
    fn test_output_format_default() {
        let format = OutputFormat::default();
        assert_eq!(format.sample_rate, 22050);
        assert_eq!(format.channels, 1);
        assert_eq!(format.bit_depth, 16);
    }

    #[test]
    fn test_engine_config_default() {
        let config = EngineConfig::default();
        assert!(!config.device.use_gpu);
        assert!(config.performance.enable_cache);
    }
}
