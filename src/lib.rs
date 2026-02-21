//! # SDKWork-TTS - Unified TTS Framework
//!
//! A high-performance, extensible Text-to-Speech framework supporting multiple engines
//! including IndexTTS2, Fish-Speech, Qwen3-TTS and more.
//!
//! ## Features
//!
//! - **Multi-Engine Support**: Unified API for Qwen3-TTS (default), IndexTTS2, Fish-Speech, and extensible to more engines
//! - **Zero-shot Voice Cloning**: Clone voices from reference audio
//! - **Emotion Control**: Fine-grained emotion and style control
//! - **Streaming Synthesis**: Real-time audio streaming
//! - **GPU Acceleration**: CUDA and Metal support via Candle
//! - **Production Ready**: Comprehensive error handling, metrics, and resource management
//!
//! ## Quick Start (Third-Party Integration)
//!
//! For third-party applications, use the simplified SDK API:
//!
//! ```rust,ignore
//! use sdkwork_tts::sdk::{Sdk, SdkBuilder};
//!
//! // Initialize SDK (default engine is Qwen3-TTS)
//! let sdk = SdkBuilder::new()
//!     .gpu()
//!     .with_default_engines()
//!     .build()?;
//!
//! // Simple synthesis
//! let audio = sdk.synthesize("Hello world", "speaker.wav")?;
//! sdk.save_audio(&audio, "output.wav")?;
//!
//! // Or use fluent builder
//! sdk.synthesis()
//!     .text("Hello world")
//!     .speaker("speaker.wav")
//!     .temperature(0.8)
//!     .save("output.wav")?;
//! ```
//!
//! ## Using IndexTTS2 Directly
//!
//! ```rust,ignore
//! use sdkwork_tts::IndexTTS2;
//!
//! let tts = IndexTTS2::new("checkpoints/config.yaml")?;
//! let audio = tts.infer("Hello, world!", "voice.wav")?;
//! audio.save("output.wav")?;
//! ```
//!
//! ## Using Qwen3-TTS
//!
//! ```rust,ignore
//! use sdkwork_tts::Qwen3Tts;
//!
//! let tts = Qwen3Tts::new()?;
//! let audio = tts.infer("Hello, world!", "voice.wav")?;
//! audio.save("output.wav")?;
//! ```
//!
//! ## Engine Registry
//!
//! ```rust,ignore
//! use sdkwork_tts::engine::{EngineRegistry, TtsEngine};
//!
//! // List available engines
//! let engines = registry.list_engines()?;
//!
//! // Get a specific engine
//! let engine = registry.get_engine("qwen3-tts")?;
//! ```
//!
//! ## Supported Engines
//!
//! | Engine | Status | Features |
//! |--------|--------|----------|
//! | **Qwen3-TTS** (default) | ✅ Stable | 10 languages, 97ms latency, voice design |
//! | IndexTTS2 | ✅ Stable | Zero-shot cloning, emotion control |
//! | Fish-Speech | 🚧 Adapter | Multi-language, streaming |

// Allow dead code for infrastructure that may be used in the future
#![allow(dead_code)]
// Allow missing docs for internal implementation
#![allow(missing_docs)]
#![allow(rustdoc::missing_crate_level_docs)]

pub mod audio;
pub mod config;
pub mod core;
pub mod debug;
pub mod emotion;
pub mod engine;
pub mod hub;
pub mod inference;
pub mod models;
pub mod sdk;
pub mod server;
pub mod streaming;
pub mod text;
pub mod utils;
pub mod voice;

// Re-exports for convenience
pub use config::ModelConfig;
pub use inference::{IndexTTS2, InferenceConfig, InferenceResult};
pub use inference::{
    get_hf_cache_dir, get_modelscope_cache_dir,
    find_model, get_model_or_error, get_default_indextts2_path,
    list_cached_models, CachedModel,
    resolve_model_id, DEFAULT_MODEL_ID, MODEL_ALIASES,
};
// Qwen3-TTS exports
pub use inference::{Qwen3Tts, QwenInferenceConfig, QwenInferenceResult, QwenModelVariant};
// Qwen3-TTS model exports
pub use models::qwen3_tts::{
    Qwen3TtsModel, QwenConfig, QwenModelVariant as QwenModelVariantExt,
    Language, Speaker, QwenSynthesisResult,
};
// Audio exports
pub use audio::{SpeakerEmbeddingCache, SpeakerCacheConfig, SpeakerCacheStats};
pub use debug::{WeightDiagnostics, ComponentReport};

// Core framework re-exports
pub use core::{
    error::{TtsError, Result, ErrorContext, ResultExt},
    error_ext::{ErrorCode, ErrorSeverity, RecoveryStrategy, ExtendedErrorInfo},
    traits::{
        ModelComponent, Encoder, Decoder, Synthesizer,
        Configurable, Loadable, Initializable, Preprocessor, Postprocessor,
    },
    resource::{ResourceManager, ResourceHandle, ResourceStatistics},
    metrics::{MetricsCollector, PerformanceMetrics, TimingInfo},
    metrics_export::{
        MetricsExporter, PrometheusExporter, JsonExporter,
        CsvExporter, ConsoleExporter, MultiExporter,
    },
    event_bus::{EventBus, Event, EventHandler, events},
    plugin::{Plugin, PluginRegistry, PluginContext},
    builder::{TtsBuilder, TtsConfigBuilder, presets, validation},
    config_center::{ConfigCenter, ConfigValue, ConfigStats},
};

// Engine re-exports
pub use engine::{
    TtsEngine, TtsEngineInfo, SynthesisRequest, SynthesisResult,
    EngineRegistry, EngineFactory, EngineConfig,
    ProcessingPipeline, PipelineStage,
    SpeakerManager, SpeakerInfo,
    EmotionManager, EmotionInfo,
    IndexTTS2Engine,
    FishSpeechEngine,
    Qwen3TtsEngine,
    init_engines,
};

// SDK re-exports (for third-party integration)
pub use sdk::{
    Sdk, SdkBuilder, SdkConfig, SdkError,
    SynthesisOptions, SpeakerRef, EmotionRef, AudioData, AudioChunk,
    EngineInfo, SdkStats,
};

// Voice module re-exports
pub use voice::{
    VoiceCloneConfig, VoiceClonePrompt, VoiceCloningManager, VoiceCloneBuilder,
    VoiceDesignConfig, VoiceDesignManager, VoicePresets,
    Gender, AgeRange, Timbre, PitchLevel, EmotionStyle,
};

// Streaming module re-exports
pub use streaming::{
    StreamingConfig, StreamChunk, StreamHandle, StreamSynthesizer, StreamingStats,
    LanguageDetection, SupportedLanguage,
};

// Emotion module re-exports
pub use emotion::{
    EmotionConfig, EmotionType, ProsodyConfig,
    EmotionInstructionParser, EmotionPresets as EmotionPresets,
    EmotionVectorBuilder,
};

/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Framework name
pub const FRAMEWORK_NAME: &str = "SDKWork-TTS";

/// Default sample rate for output audio (22050 Hz)
pub const DEFAULT_SAMPLE_RATE: u32 = 22050;

/// Maximum text tokens per segment
pub const MAX_TEXT_TOKENS_PER_SEGMENT: usize = 120;
