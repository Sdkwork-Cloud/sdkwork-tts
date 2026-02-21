//! SDK Error types
//!
//! Provides user-friendly error types for SDK operations.

use std::fmt;
use crate::core::error::TtsError;

/// SDK error result type
pub type Result<T> = std::result::Result<T, SdkError>;

/// SDK error types for third-party integration
#[derive(Debug)]
pub enum SdkError {
    /// SDK not initialized
    NotInitialized {
        component: String,
    },
    /// Invalid configuration
    InvalidConfig {
        field: String,
        message: String,
    },
    /// Engine error
    Engine {
        engine_id: String,
        error: String,
    },
    /// Synthesis error
    Synthesis {
        message: String,
        details: Option<String>,
    },
    /// Audio processing error
    Audio {
        operation: String,
        message: String,
    },
    /// Resource error
    Resource {
        resource_type: String,
        message: String,
    },
    /// Internal error
    Internal {
        message: String,
    },
}

impl fmt::Display for SdkError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SdkError::NotInitialized { component } => {
                write!(f, "SDK component '{}' not initialized", component)
            }
            SdkError::InvalidConfig { field, message } => {
                write!(f, "Invalid configuration for '{}': {}", field, message)
            }
            SdkError::Engine { engine_id, error } => {
                write!(f, "Engine '{}' error: {}", engine_id, error)
            }
            SdkError::Synthesis { message, details } => {
                write!(f, "Synthesis error: {}", message)?;
                if let Some(d) = details {
                    write!(f, " ({})", d)?;
                }
                Ok(())
            }
            SdkError::Audio { operation, message } => {
                write!(f, "Audio {} failed: {}", operation, message)
            }
            SdkError::Resource { resource_type, message } => {
                write!(f, "Resource '{}' error: {}", resource_type, message)
            }
            SdkError::Internal { message } => {
                write!(f, "Internal error: {}", message)
            }
        }
    }
}

impl std::error::Error for SdkError {}

/// Convert from TtsError
impl From<TtsError> for SdkError {
    fn from(err: TtsError) -> Self {
        match err {
            TtsError::Config { message, path } => SdkError::InvalidConfig {
                field: "config".to_string(),
                message: if let Some(p) = path {
                    format!("{}: {:?}", message, p)
                } else {
                    message
                },
            },
            TtsError::ModelLoad { component, message, .. } => SdkError::Engine {
                engine_id: component,
                error: message,
            },
            TtsError::Inference { stage, message, .. } => SdkError::Synthesis {
                message: format!("Failed at stage: {}", stage),
                details: Some(message),
            },
            TtsError::Audio { operation, message } => SdkError::Audio {
                operation: format!("{:?}", operation),
                message,
            },
            TtsError::Resource { resource_type, message } => SdkError::Resource {
                resource_type: format!("{:?}", resource_type),
                message,
            },
            _ => SdkError::Internal {
                message: err.to_string(),
            },
        }
    }
}

/// Convert from anyhow::Error
impl From<anyhow::Error> for SdkError {
    fn from(err: anyhow::Error) -> Self {
        SdkError::Internal {
            message: err.to_string(),
        }
    }
}
