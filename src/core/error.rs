//! Structured error handling for IndexTTS2
//!
//! Provides a hierarchical error type system with rich context,
//! error chaining, and user-friendly error messages.

use std::fmt;
use std::path::PathBuf;
use thiserror::Error;

/// Result type alias with TtsError
pub type Result<T> = std::result::Result<T, TtsError>;

/// Error context for additional information
#[derive(Debug, Clone, Default)]
pub struct ErrorContext {
    /// Operation that caused the error
    pub operation: Option<String>,
    /// Component where the error occurred
    pub component: Option<String>,
    /// User-friendly suggestion
    pub suggestion: Option<String>,
    /// Additional key-value pairs
    pub metadata: Vec<(String, String)>,
}

impl ErrorContext {
    /// Create a new error context
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the operation
    pub fn with_operation(mut self, operation: impl Into<String>) -> Self {
        self.operation = Some(operation.into());
        self
    }

    /// Set the component
    pub fn with_component(mut self, component: impl Into<String>) -> Self {
        self.component = Some(component.into());
        self
    }

    /// Set a suggestion
    pub fn with_suggestion(mut self, suggestion: impl Into<String>) -> Self {
        self.suggestion = Some(suggestion.into());
        self
    }

    /// Add metadata
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.push((key.into(), value.into()));
        self
    }
}

/// Main error type for IndexTTS2
///
/// Note: source field is intentionally omitted for now to reduce boilerplate.
/// Use ErrorContext for additional error information.
#[derive(Error, Debug, Clone)]
pub enum TtsError {
    /// Configuration errors
    #[error("Configuration error: {message}")]
    Config {
        message: String,
        path: Option<PathBuf>,
    },

    /// Model loading errors
    #[error("Model loading error in {component}: {message}")]
    ModelLoad {
        message: String,
        component: String,
        path: Option<PathBuf>,
    },

    /// Inference errors
    #[error("Inference error in {stage}: {message}")]
    Inference {
        stage: InferenceStage,
        message: String,
        recoverable: bool,
    },

    /// Audio processing errors
    #[error("Audio processing error ({operation}): {message}")]
    Audio {
        message: String,
        operation: AudioOperation,
    },

    /// Text processing errors
    #[error("Text processing error ({operation}): {message}")]
    Text {
        message: String,
        operation: TextOperation,
    },

    /// Resource management errors
    #[error("Resource error ({resource_type}): {message}")]
    Resource {
        message: String,
        resource_type: ResourceType,
    },

    /// Device/GPU errors
    #[error("Device error ({device_type}): {message}")]
    Device {
        message: String,
        device_type: String,
    },

    /// Validation errors
    #[error("Validation error: {message}")]
    Validation {
        message: String,
        field: Option<String>,
    },

    /// I/O errors
    #[error("I/O error: {message}")]
    Io {
        message: String,
        path: Option<PathBuf>,
    },

    /// Timeout errors
    #[error("Operation timeout: {message} ({duration_ms}ms)")]
    Timeout {
        message: String,
        duration_ms: u64,
    },

    /// Internal/bug errors
    #[error("Internal error: {message}")]
    Internal {
        message: String,
        location: Option<String>,
    },
}

/// Inference pipeline stages
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InferenceStage {
    TextNormalization,
    Tokenization,
    SpeakerEncoding,
    EmotionEncoding,
    SemanticEncoding,
    GptGeneration,
    FlowMatching,
    Vocoding,
    AudioOutput,
}

impl fmt::Display for InferenceStage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            InferenceStage::TextNormalization => write!(f, "text normalization"),
            InferenceStage::Tokenization => write!(f, "tokenization"),
            InferenceStage::SpeakerEncoding => write!(f, "speaker encoding"),
            InferenceStage::EmotionEncoding => write!(f, "emotion encoding"),
            InferenceStage::SemanticEncoding => write!(f, "semantic encoding"),
            InferenceStage::GptGeneration => write!(f, "GPT generation"),
            InferenceStage::FlowMatching => write!(f, "flow matching"),
            InferenceStage::Vocoding => write!(f, "vocoding"),
            InferenceStage::AudioOutput => write!(f, "audio output"),
        }
    }
}

/// Audio operation types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AudioOperation {
    Loading,
    Resampling,
    MelSpectrogram,
    Saving,
    Mixing,
}

impl fmt::Display for AudioOperation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AudioOperation::Loading => write!(f, "loading"),
            AudioOperation::Resampling => write!(f, "resampling"),
            AudioOperation::MelSpectrogram => write!(f, "mel spectrogram computation"),
            AudioOperation::Saving => write!(f, "saving"),
            AudioOperation::Mixing => write!(f, "mixing"),
        }
    }
}

/// Text operation types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextOperation {
    Normalization,
    Tokenization,
    Segmentation,
    Phonemization,
}

impl fmt::Display for TextOperation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TextOperation::Normalization => write!(f, "normalization"),
            TextOperation::Tokenization => write!(f, "tokenization"),
            TextOperation::Segmentation => write!(f, "segmentation"),
            TextOperation::Phonemization => write!(f, "phonemization"),
        }
    }
}

/// Resource types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResourceType {
    Model,
    Memory,
    FileHandle,
    GpuMemory,
    ThreadPool,
}

impl fmt::Display for ResourceType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ResourceType::Model => write!(f, "model"),
            ResourceType::Memory => write!(f, "memory"),
            ResourceType::FileHandle => write!(f, "file handle"),
            ResourceType::GpuMemory => write!(f, "GPU memory"),
            ResourceType::ThreadPool => write!(f, "thread pool"),
        }
    }
}

/// Extension trait for adding context to errors
pub trait ResultExt<T> {
    /// Add context to an error
    fn with_context<F>(self, f: F) -> Result<T>
    where
        F: FnOnce() -> String;

    /// Add a simple message context
    fn context(self, msg: impl Into<String>) -> Result<T>;
}

impl<T, E> ResultExt<T> for std::result::Result<T, E>
where
    E: std::error::Error + Send + Sync + 'static,
{
    fn with_context<F>(self, f: F) -> Result<T>
    where
        F: FnOnce() -> String,
    {
        self.map_err(|e| TtsError::Internal {
            message: format!("{}: {}", f(), e),
            location: None,
        })
    }

    fn context(self, msg: impl Into<String>) -> Result<T> {
        self.map_err(|e| TtsError::Internal {
            message: format!("{}: {}", msg.into(), e),
            location: None,
        })
    }
}

/// Convert from anyhow::Error
impl From<anyhow::Error> for TtsError {
    fn from(err: anyhow::Error) -> Self {
        TtsError::Internal {
            message: err.to_string(),
            location: None,
        }
    }
}

/// Convert from std::io::Error
impl From<std::io::Error> for TtsError {
    fn from(err: std::io::Error) -> Self {
        TtsError::Io {
            message: err.to_string(),
            path: None,
        }
    }
}

/// Convert from candle_core::Error
impl From<candle_core::Error> for TtsError {
    fn from(err: candle_core::Error) -> Self {
        TtsError::Internal {
            message: format!("Tensor operation failed: {}", err),
            location: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = TtsError::Config {
            message: "Invalid sample rate".to_string(),
            path: Some(PathBuf::from("config.yaml")),
        };
        assert!(err.to_string().contains("Configuration error"));
        assert!(err.to_string().contains("Invalid sample rate"));
    }

    #[test]
    fn test_inference_stage_display() {
        assert_eq!(
            InferenceStage::GptGeneration.to_string(),
            "GPT generation"
        );
    }

    #[test]
    fn test_error_context_builder() {
        let ctx = ErrorContext::new()
            .with_operation("loading model")
            .with_component("GPT")
            .with_suggestion("Check model path");
        
        assert_eq!(ctx.operation, Some("loading model".to_string()));
        assert_eq!(ctx.component, Some("GPT".to_string()));
        assert_eq!(ctx.suggestion, Some("Check model path".to_string()));
    }
}
