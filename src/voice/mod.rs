//! Voice Module
//!
//! Provides advanced voice capabilities:
//! - Voice cloning from reference audio
//! - Voice design through natural language descriptions
//! - Preset voices for common use cases
//! - Voice blending and interpolation

pub mod cloning;
pub mod design;

pub use cloning::{
    VoiceCloneConfig, VoiceClonePrompt, VoiceCloningManager,
    VoiceCloneBuilder,
};
pub use design::{
    VoiceDesignConfig, VoiceDesignManager, VoicePresets,
    Gender, AgeRange, Timbre, PitchLevel, EmotionStyle,
};

/// Re-export for convenience
pub use crate::engine::traits::SpeakerReference;
