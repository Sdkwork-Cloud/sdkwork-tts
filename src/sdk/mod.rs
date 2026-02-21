//! SDK Module - Unified API for third-party integration
//!
//! This module provides a simplified, high-level API for integrating
//! SDKWork-TTS into third-party applications.
//!
//! # Quick Start
//!
//! ```rust,ignore
//! use sdkwork_tts::sdk::{SdkBuilder, SdkConfig};
//!
//! // Create SDK with defaults
//! let sdk = SdkBuilder::new()
//!     .with_default_engines()
//!     .build()?;
//!
//! // Simple synthesis
//! let result = sdk.synthesize("Hello world", "speaker.wav")?;
//! result.save("output.wav")?;
//! ```
//!
//! # Advanced Usage
//!
//! ```rust,ignore
//! use sdkwork_tts::sdk::*;
//!
//! // Configure SDK
//! let config = SdkConfig::builder()
//!     .gpu(true)
//!     .default_engine("indextts2")
//!     .memory_limit(4 * 1024 * 1024 * 1024)
//!     .build();
//!
//! // Build SDK
//! let sdk = SdkBuilder::from_config(config)
//!     .with_all_engines()
//!     .with_event_logging(true)
//!     .with_metrics(true)
//!     .build()?;
//!
//! // Use synthesis builder
//! let audio = sdk.synthesis()
//!     .text("Hello world")
//!     .speaker("voice.wav")
//!     .emotion("happy", 0.8)
//!     .temperature(0.8)
//!     .build()?;
//! ```

pub mod config;
pub mod builder;
pub mod facade;
pub mod types;
pub mod error;

pub use config::{SdkConfig, SdkConfigBuilder};
pub use builder::SdkBuilder;
pub use facade::Sdk;
pub use types::*;
pub use error::SdkError;

/// Re-export commonly used types
pub use crate::engine::traits::{SynthesisResult, SynthesisRequest, SpeakerReference, EmotionSpec};
pub use crate::core::{EventBus, MetricsCollector};
