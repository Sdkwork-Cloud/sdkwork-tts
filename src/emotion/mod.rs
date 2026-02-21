//! Emotion Control Module
//!
//! Provides comprehensive emotion control capabilities:
//! - 16 emotion types with intensity control
//! - Multi-dimensional emotion vectors
//! - Natural language emotion instructions
//! - Prosody control (pitch, speed, energy)
//! - Text-based emotion extraction

pub mod enhanced;

pub use enhanced::{
    EmotionConfig, EmotionType, ProsodyConfig,
    EmotionInstructionParser, EmotionPresets, EmotionVectorBuilder,
};
