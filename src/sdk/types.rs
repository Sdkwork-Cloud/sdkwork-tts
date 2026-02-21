//! SDK types for third-party integration
//!
//! Provides simplified, user-friendly types for SDK operations.

use std::path::PathBuf;

/// SDK configuration for synthesis
#[derive(Debug, Clone)]
pub struct SynthesisOptions {
    /// Text to synthesize
    pub text: String,
    /// Speaker reference (path or ID)
    pub speaker: SpeakerRef,
    /// Emotion name or vector
    pub emotion: Option<EmotionRef>,
    /// Language code (for multi-language engines)
    pub language: Option<String>,
    /// Generation parameters
    pub params: GenerationParams,
    /// Output path (optional, for direct save)
    pub output_path: Option<PathBuf>,
}

/// Speaker reference
#[derive(Debug, Clone)]
pub enum SpeakerRef {
    /// Path to audio file
    AudioPath(PathBuf),
    /// Built-in speaker ID
    SpeakerId(String),
    /// Default speaker
    Default,
    /// Voice clone from reference audio
    VoiceClone {
        /// Reference audio path
        audio_path: PathBuf,
        /// Reference text (optional)
        text: Option<String>,
        /// Clone strength
        strength: f32,
    },
    /// Voice design from description
    VoiceDesign {
        /// Voice description
        description: String,
        /// Preset name (optional)
        preset: Option<String>,
    },
}

/// Emotion reference
#[derive(Debug, Clone)]
pub enum EmotionRef {
    /// Named emotion
    Name(String),
    /// Emotion vector
    Vector(Vec<f32>),
    /// Emotion from audio
    AudioPath(PathBuf),
    /// Emotion from text
    Text(String),
}

/// Generation parameters
#[derive(Debug, Clone)]
pub struct GenerationParams {
    /// Temperature (0.0 - 1.0)
    pub temperature: f32,
    /// Top-k sampling (0 = disabled)
    pub top_k: usize,
    /// Top-p sampling (0.0 - 1.0)
    pub top_p: f32,
    /// Repetition penalty
    pub repetition_penalty: f32,
    /// Flow matching steps
    pub flow_steps: usize,
    /// CFG rate
    pub cfg_rate: f32,
    /// Apply de-rumble filter
    pub de_rumble: bool,
    /// De-rumble cutoff frequency
    pub de_rumble_cutoff_hz: f32,
}

impl Default for GenerationParams {
    fn default() -> Self {
        Self {
            temperature: 0.8,
            top_k: 50,
            top_p: 0.95,
            repetition_penalty: 1.1,
            flow_steps: 25,
            cfg_rate: 0.7,
            de_rumble: true,
            de_rumble_cutoff_hz: 180.0,
        }
    }
}

impl SynthesisOptions {
    /// Create new synthesis options with text and speaker
    pub fn new(text: impl Into<String>, speaker: impl Into<SpeakerRef>) -> Self {
        Self {
            text: text.into(),
            speaker: speaker.into(),
            emotion: None,
            language: None,
            params: GenerationParams::default(),
            output_path: None,
        }
    }

    /// Set emotion
    pub fn with_emotion(mut self, emotion: impl Into<EmotionRef>) -> Self {
        self.emotion = Some(emotion.into());
        self
    }

    /// Set language
    pub fn with_language(mut self, language: impl Into<String>) -> Self {
        self.language = Some(language.into());
        self
    }

    /// Set temperature
    pub fn with_temperature(mut self, temp: f32) -> Self {
        self.params.temperature = temp.clamp(0.0, 1.0);
        self
    }

    /// Set output path
    pub fn with_output(mut self, path: impl Into<PathBuf>) -> Self {
        self.output_path = Some(path.into());
        self
    }
}

impl From<&str> for SpeakerRef {
    fn from(s: &str) -> Self {
        SpeakerRef::AudioPath(PathBuf::from(s))
    }
}

impl From<PathBuf> for SpeakerRef {
    fn from(p: PathBuf) -> Self {
        SpeakerRef::AudioPath(p)
    }
}

impl From<(&str, &str, f32)> for SpeakerRef {
    fn from((audio, text, strength): (&str, &str, f32)) -> Self {
        SpeakerRef::VoiceClone {
            audio_path: PathBuf::from(audio),
            text: Some(text.to_string()),
            strength: strength.clamp(0.0, 1.0),
        }
    }
}

impl From<String> for SpeakerRef {
    fn from(s: String) -> Self {
        SpeakerRef::VoiceDesign {
            description: s,
            preset: None,
        }
    }
}

impl From<&str> for EmotionRef {
    fn from(s: &str) -> Self {
        EmotionRef::Name(s.to_string())
    }
}

impl From<Vec<f32>> for EmotionRef {
    fn from(v: Vec<f32>) -> Self {
        EmotionRef::Vector(v)
    }
}

/// Audio data container
#[derive(Debug, Clone)]
pub struct AudioData {
    /// Audio samples (normalized f32)
    pub samples: Vec<f32>,
    /// Sample rate
    pub sample_rate: u32,
    /// Duration in seconds
    pub duration_secs: f32,
    /// Number of channels
    pub channels: u16,
}

impl AudioData {
    /// Create new audio data
    pub fn new(samples: Vec<f32>, sample_rate: u32) -> Self {
        let duration_secs = samples.len() as f32 / sample_rate as f32;
        Self {
            samples,
            sample_rate,
            duration_secs,
            channels: 1,
        }
    }

    /// Get samples as i16 (for WAV output)
    pub fn to_i16(&self) -> Vec<i16> {
        self.samples
            .iter()
            .map(|&s| (s.clamp(-1.0, 1.0) * 32767.0) as i16)
            .collect()
    }

    /// Get RMS volume
    pub fn rms(&self) -> f32 {
        let sum: f32 = self.samples.iter().map(|&s| s * s).sum();
        (sum / self.samples.len() as f32).sqrt()
    }

    /// Get peak amplitude
    pub fn peak(&self) -> f32 {
        self.samples.iter().fold(0.0f32, |acc, &s| acc.max(s.abs()))
    }
}

/// Streaming audio chunk
#[derive(Debug, Clone)]
pub struct AudioChunk {
    /// Chunk samples
    pub samples: Vec<f32>,
    /// Sample rate
    pub sample_rate: u32,
    /// Chunk index
    pub index: usize,
    /// Is final chunk
    pub is_final: bool,
    /// Timestamp in milliseconds
    pub timestamp_ms: u64,
}

/// Engine information
#[derive(Debug, Clone)]
pub struct EngineInfo {
    /// Engine ID
    pub id: String,
    /// Engine name
    pub name: String,
    /// Engine version
    pub version: String,
    /// Supported languages
    pub languages: Vec<String>,
    /// Features
    pub features: Vec<String>,
    /// Is available
    pub available: bool,
}

/// SDK statistics
#[derive(Debug, Clone, Default)]
pub struct SdkStats {
    /// Total synthesis calls
    pub total_synthesis: u64,
    /// Successful synthesis calls
    pub successful_synthesis: u64,
    /// Failed synthesis calls
    pub failed_synthesis: u64,
    /// Total audio generated (seconds)
    pub total_audio_secs: f64,
    /// Total processing time (seconds)
    pub total_processing_secs: f64,
    /// Average RTF
    pub avg_rtf: f64,
    /// Loaded engines
    pub loaded_engines: usize,
}

impl SdkStats {
    /// Get success rate
    pub fn success_rate(&self) -> f64 {
        if self.total_synthesis == 0 {
            0.0
        } else {
            self.successful_synthesis as f64 / self.total_synthesis as f64
        }
    }

    /// Get average synthesis time
    pub fn avg_synthesis_time_ms(&self) -> f64 {
        if self.total_synthesis == 0 {
            0.0
        } else {
            self.total_processing_secs * 1000.0 / self.total_synthesis as f64
        }
    }
}
