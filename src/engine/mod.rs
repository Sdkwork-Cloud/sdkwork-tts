//! TTS Engine Abstraction Layer
//!
//! This module provides a unified framework for multiple TTS engines,
//! enabling easy integration of various open-source TTS models.
//!
//! # Supported Engines
//! - **IndexTTS2** (Bilibili) - Fully implemented, stable
//! - **Fish-Speech** - Adapter implemented, model integration in progress
//! - **Qwen3-TTS** (Alibaba) - Adapter implemented, supports 10 languages
//!
//! # Planned Support
//! - GPT-SoVITS
//! - ChatTTS
//! - CosyVoice (Alibaba)
//! - F5-TTS / E2-TTS
//! - VITS / VITS2
//! - StyleTTS2
//! - Bark (Suno)
//!
//! # Architecture
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                    Application Layer                        │
//! │              (CLI, API, Streaming, Batch)                   │
//! ├─────────────────────────────────────────────────────────────┤
//! │                    Unified TTS API                          │
//! │  ┌─────────────────────────────────────────────────────┐   │
//! │  │              TtsEngine Trait                        │   │
//! │  │  - synthesize()    - synthesize_streaming()         │   │
//! │  │  - get_speakers()  - get_emotions()                 │   │
//! │  │  - load_model()    - unload_model()                 │   │
//! │  └─────────────────────────────────────────────────────┘   │
//! ├─────────────────────────────────────────────────────────────┤
//! │                    Engine Registry                          │
//! │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐     │
//! │  │ IndexTTS │ │   Fish   │ │  Qwen3   │ │  Future  │     │
//! │  │    2     │ │  Speech  │ │   TTS    │ │  Engines │     │
//! │  └──────────┘ └──────────┘ └──────────┘ └──────────┘     │
//! ├─────────────────────────────────────────────────────────────┤
//! │                    Processing Pipeline                      │
//! │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐     │
//! │  │  Text    │ │ Speaker  │ │  Audio   │ │  Output  │     │
//! │  │Processor │ │ Encoder  │ │Processor │ │ Handler  │     │
//! │  └──────────┘ └──────────┘ └──────────┘ └──────────┘     │
//! ├─────────────────────────────────────────────────────────────┤
//! │                    Core Infrastructure                      │
//! │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐     │
//! │  │  Error   │ │ Resource │ │ Metrics  │ │  Config  │     │
//! │  │ Handling │ │ Manager  │ │ Collector│ │  System  │     │
//! │  └──────────┘ └──────────┘ └──────────┘ └──────────┘     │
//! └─────────────────────────────────────────────────────────────┘
//! ```

pub mod traits;
pub mod registry;
pub mod pipeline;
pub mod config;
pub mod speaker;
pub mod emotion;
pub mod indextts2_adapter;
pub mod fish_speech_adapter;
pub mod qwen3_tts_adapter;

pub use traits::{TtsEngine, TtsEngineInfo, SynthesisRequest, SynthesisResult, StreamingCallback, AudioChunk};
pub use registry::{EngineRegistry, EngineFactory, global_registry};
pub use pipeline::{ProcessingPipeline, PipelineStage};
pub use config::{EngineConfig, EngineConfigBuilder};
pub use speaker::{SpeakerManager, SpeakerInfo};
pub use emotion::{EmotionManager, EmotionInfo};
pub use indextts2_adapter::{IndexTTS2Engine, register_engine as register_indextts2};
pub use fish_speech_adapter::{FishSpeechEngine, register_engine as register_fish_speech};
pub use qwen3_tts_adapter::{Qwen3TtsEngine, AdapterVariant, register_engine as register_qwen3_tts};

/// Initialize all built-in engines
pub fn init_engines() -> crate::core::error::Result<()> {
    register_indextts2()?;
    register_fish_speech()?;
    register_qwen3_tts()?;
    Ok(())
}
