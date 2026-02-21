//! Channel Error Handling
//!
//! Unified error types for all TTS channels

use std::fmt;

/// Channel error types
#[derive(Debug, Clone)]
pub enum ChannelError {
    /// Authentication failed
    Authentication {
        channel: String,
        message: String,
    },
    /// Request failed
    Request {
        channel: String,
        message: String,
        status_code: Option<u16>,
    },
    /// Response parsing failed
    Response {
        channel: String,
        message: String,
    },
    /// Configuration error
    Configuration {
        channel: String,
        message: String,
    },
    /// Rate limit exceeded
    RateLimit {
        channel: String,
        retry_after: Option<u64>,
    },
    /// Service unavailable
    Unavailable {
        channel: String,
        message: String,
    },
    /// Internal error
    Internal {
        channel: String,
        message: String,
    },
}

impl fmt::Display for ChannelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Authentication { channel, message } => {
                write!(f, "[{}] Authentication failed: {}", channel, message)
            }
            Self::Request { channel, message, status_code } => {
                match status_code {
                    Some(code) => write!(f, "[{}] Request failed ({}): {}", channel, code, message),
                    None => write!(f, "[{}] Request failed: {}", channel, message),
                }
            }
            Self::Response { channel, message } => {
                write!(f, "[{}] Response parsing failed: {}", channel, message)
            }
            Self::Configuration { channel, message } => {
                write!(f, "[{}] Configuration error: {}", channel, message)
            }
            Self::RateLimit { channel, retry_after } => {
                match retry_after {
                    Some(secs) => write!(f, "[{}] Rate limit exceeded, retry after {} seconds", channel, secs),
                    None => write!(f, "[{}] Rate limit exceeded", channel),
                }
            }
            Self::Unavailable { channel, message } => {
                write!(f, "[{}] Service unavailable: {}", channel, message)
            }
            Self::Internal { channel, message } => {
                write!(f, "[{}] Internal error: {}", channel, message)
            }
        }
    }
}

impl std::error::Error for ChannelError {}

impl ChannelError {
    /// Create authentication error
    pub fn auth(channel: impl Into<String>, message: impl Into<String>) -> Self {
        Self::Authentication {
            channel: channel.into(),
            message: message.into(),
        }
    }
    
    /// Create request error
    pub fn request(
        channel: impl Into<String>,
        message: impl Into<String>,
        status_code: Option<u16>,
    ) -> Self {
        Self::Request {
            channel: channel.into(),
            message: message.into(),
            status_code,
        }
    }
    
    /// Create response error
    pub fn response(channel: impl Into<String>, message: impl Into<String>) -> Self {
        Self::Response {
            channel: channel.into(),
            message: message.into(),
        }
    }
    
    /// Create configuration error
    pub fn config(channel: impl Into<String>, message: impl Into<String>) -> Self {
        Self::Configuration {
            channel: channel.into(),
            message: message.into(),
        }
    }
    
    /// Create rate limit error
    pub fn rate_limit(channel: impl Into<String>, retry_after: Option<u64>) -> Self {
        Self::RateLimit {
            channel: channel.into(),
            retry_after,
        }
    }
    
    /// Create unavailable error
    pub fn unavailable(channel: impl Into<String>, message: impl Into<String>) -> Self {
        Self::Unavailable {
            channel: channel.into(),
            message: message.into(),
        }
    }
    
    /// Create internal error
    pub fn internal(channel: impl Into<String>, message: impl Into<String>) -> Self {
        Self::Internal {
            channel: channel.into(),
            message: message.into(),
        }
    }
    
    /// Get channel name
    pub fn channel(&self) -> &str {
        match self {
            Self::Authentication { channel, .. } => channel,
            Self::Request { channel, .. } => channel,
            Self::Response { channel, .. } => channel,
            Self::Configuration { channel, .. } => channel,
            Self::RateLimit { channel, .. } => channel,
            Self::Unavailable { channel, .. } => channel,
            Self::Internal { channel, .. } => channel,
        }
    }
    
    /// Get error message
    pub fn message(&self) -> &str {
        match self {
            Self::Authentication { message, .. } => message,
            Self::Request { message, .. } => message,
            Self::Response { message, .. } => message,
            Self::Configuration { message, .. } => message,
            Self::RateLimit { .. } => "Rate limit exceeded",
            Self::Unavailable { message, .. } => message,
            Self::Internal { message, .. } => message,
        }
    }
    
    /// Check if error is retryable
    pub fn is_retryable(&self) -> bool {
        match self {
            Self::Request { status_code, .. } => {
                // 5xx errors are retryable
                status_code.is_some_and(|code| code >= 500)
            }
            Self::Unavailable { .. } => true,
            Self::RateLimit { .. } => true,
            _ => false,
        }
    }
}

/// Channel result type
pub type ChannelResult<T> = Result<T, ChannelError>;

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_error_display() {
        let err = ChannelError::auth("openai", "Invalid API key");
        assert!(err.to_string().contains("openai"));
        assert!(err.to_string().contains("Authentication failed"));
    }
    
    #[test]
    fn test_error_channel() {
        let err = ChannelError::request("google", "Timeout", Some(504));
        assert_eq!(err.channel(), "google");
    }
    
    #[test]
    fn test_error_retryable() {
        let err = ChannelError::request("aliyun", "Server error", Some(503));
        assert!(err.is_retryable());
        
        let err = ChannelError::auth("openai", "Invalid key");
        assert!(!err.is_retryable());
    }
}
