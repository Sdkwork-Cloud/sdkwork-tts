//! TTS Server Module
//!
//! A production-ready TTS server supporting:
//! - Local mode (IndexTTS2, Qwen3-TTS, Fish-Speech)
//! - Cloud mode (Aliyun, OpenAI, Volcano, Minimax)
//! - Speaker library (local and cloud)
//! - REST API compatible with mainstream TTS standards
//! - Voice design and voice cloning APIs

pub mod server_core;
pub mod config;
pub mod routes;
pub mod speaker_lib;
pub mod types;
pub mod channels;
pub mod middleware;

pub use server_core::TtsServer;
pub use config::ServerConfig;
pub use speaker_lib::SpeakerLibrary;
pub use types::*;
pub use channels::{CloudChannel, CloudChannelType, CloudChannelConfig, ChannelRegistry};
pub use middleware::{MetricsState, ServerMetrics, MetricsSummary};
