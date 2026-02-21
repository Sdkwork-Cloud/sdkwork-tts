//! Enhanced error handling with error codes and recovery strategies
//!
//! This module extends the basic error handling with:
//! - Error codes for programmatic handling
//! - Error recovery strategies
//! - Error categorization and grouping
//! - Error event broadcasting

use std::fmt;

use super::error::{TtsError, AudioOperation, TextOperation};

/// Error severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ErrorSeverity {
    /// Informational - operation can continue
    Info,
    /// Warning - operation completed with caveats
    Warning,
    /// Error - operation failed but recoverable
    Error,
    /// Critical - operation failed and not recoverable
    Critical,
    /// Fatal - system state compromised
    Fatal,
}

impl fmt::Display for ErrorSeverity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ErrorSeverity::Info => write!(f, "INFO"),
            ErrorSeverity::Warning => write!(f, "WARNING"),
            ErrorSeverity::Error => write!(f, "ERROR"),
            ErrorSeverity::Critical => write!(f, "CRITICAL"),
            ErrorSeverity::Fatal => write!(f, "FATAL"),
        }
    }
}

/// Error codes for programmatic handling
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ErrorCode {
    // Configuration errors (1000-1999)
    ConfigNotFound = 1001,
    ConfigInvalidFormat = 1002,
    ConfigInvalidValue = 1003,
    ConfigMissingField = 1004,
    
    // Model loading errors (2000-2999)
    ModelNotFound = 2001,
    ModelInvalidFormat = 2002,
    ModelMissingWeights = 2003,
    ModelVersionMismatch = 2004,
    ModelLoadTimeout = 2005,
    
    // Inference errors (3000-3999)
    InferenceFailed = 3001,
    InferenceTimeout = 3002,
    InferenceInvalidInput = 3003,
    InferenceOutputInvalid = 3004,
    InferenceStoppedEarly = 3005,
    
    // Audio errors (4000-4999)
    AudioNotFound = 4001,
    AudioInvalidFormat = 4002,
    AudioDecodeFailed = 4003,
    AudioEncodeFailed = 4004,
    AudioPlayFailed = 4005,
    
    // Text errors (5000-5999)
    TextInvalidEncoding = 5001,
    TextTokenizationFailed = 5002,
    TextTooLong = 5003,
    TextEmpty = 5004,
    
    // Resource errors (6000-6999)
    ResourceExhausted = 6001,
    ResourceNotFound = 6002,
    ResourceLocked = 6003,
    ResourceLeak = 6004,
    
    // Device errors (7000-7999)
    DeviceNotFound = 7001,
    DeviceOutOfMemory = 7002,
    DeviceComputeFailed = 7003,
    DeviceDriverError = 7004,
    
    // Validation errors (8000-8999)
    ValidationFailed = 8001,
    ValidationRangeError = 8002,
    ValidationError = 8003,
    
    // Timeout errors (9000-9999)
    TimeoutExceeded = 9001,
    TimeoutWaitingForResource = 9002,
    TimeoutWaitingForLock = 9003,
    
    // Internal errors (10000-10999)
    InternalError = 10001,
    InternalInvariant = 10002,
    InternalUnreachable = 10003,
    
    // Unknown
    Unknown = 99999,
}

impl fmt::Display for ErrorCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "E{:05}", *self as usize)
    }
}

impl ErrorCode {
    /// Get the severity for this error code
    pub fn severity(&self) -> ErrorSeverity {
        match self {
            ErrorCode::ConfigNotFound => ErrorSeverity::Warning,
            ErrorCode::ConfigInvalidFormat | ErrorCode::ConfigInvalidValue | ErrorCode::ConfigMissingField => ErrorSeverity::Error,
            
            ErrorCode::ModelNotFound | ErrorCode::ModelInvalidFormat | ErrorCode::ModelMissingWeights => ErrorSeverity::Error,
            ErrorCode::ModelVersionMismatch => ErrorSeverity::Warning,
            ErrorCode::ModelLoadTimeout => ErrorSeverity::Error,
            
            ErrorCode::InferenceFailed | ErrorCode::InferenceInvalidInput | ErrorCode::InferenceOutputInvalid => ErrorSeverity::Error,
            ErrorCode::InferenceTimeout | ErrorCode::InferenceStoppedEarly => ErrorSeverity::Warning,
            
            ErrorCode::AudioNotFound | ErrorCode::AudioInvalidFormat | ErrorCode::AudioDecodeFailed | ErrorCode::AudioEncodeFailed => ErrorSeverity::Error,
            ErrorCode::AudioPlayFailed => ErrorSeverity::Warning,
            
            ErrorCode::TextInvalidEncoding | ErrorCode::TextTokenizationFailed => ErrorSeverity::Error,
            ErrorCode::TextTooLong => ErrorSeverity::Warning,
            ErrorCode::TextEmpty => ErrorSeverity::Error,
            
            ErrorCode::ResourceExhausted | ErrorCode::ResourceNotFound => ErrorSeverity::Critical,
            ErrorCode::ResourceLocked => ErrorSeverity::Warning,
            ErrorCode::ResourceLeak => ErrorSeverity::Error,
            
            ErrorCode::DeviceNotFound | ErrorCode::DeviceOutOfMemory | ErrorCode::DeviceComputeFailed | ErrorCode::DeviceDriverError => ErrorSeverity::Critical,
            
            ErrorCode::ValidationFailed | ErrorCode::ValidationRangeError | ErrorCode::ValidationError => ErrorSeverity::Error,
            
            ErrorCode::TimeoutExceeded | ErrorCode::TimeoutWaitingForResource | ErrorCode::TimeoutWaitingForLock => ErrorSeverity::Warning,
            
            ErrorCode::InternalError | ErrorCode::InternalInvariant | ErrorCode::InternalUnreachable => ErrorSeverity::Fatal,
            
            ErrorCode::Unknown => ErrorSeverity::Error,
        }
    }
    
    /// Get a human-readable description
    pub fn description(&self) -> &'static str {
        match self {
            ErrorCode::ConfigNotFound => "Configuration file not found",
            ErrorCode::ConfigInvalidFormat => "Invalid configuration format",
            ErrorCode::ConfigInvalidValue => "Invalid configuration value",
            ErrorCode::ConfigMissingField => "Missing required configuration field",
            
            ErrorCode::ModelNotFound => "Model file not found",
            ErrorCode::ModelInvalidFormat => "Invalid model format",
            ErrorCode::ModelMissingWeights => "Model missing required weights",
            ErrorCode::ModelVersionMismatch => "Model version mismatch",
            ErrorCode::ModelLoadTimeout => "Model loading timed out",
            
            ErrorCode::InferenceFailed => "Inference operation failed",
            ErrorCode::InferenceTimeout => "Inference operation timed out",
            ErrorCode::InferenceInvalidInput => "Invalid input for inference",
            ErrorCode::InferenceOutputInvalid => "Invalid inference output",
            ErrorCode::InferenceStoppedEarly => "Inference stopped before completion",
            
            ErrorCode::AudioNotFound => "Audio file not found",
            ErrorCode::AudioInvalidFormat => "Invalid audio format",
            ErrorCode::AudioDecodeFailed => "Audio decoding failed",
            ErrorCode::AudioEncodeFailed => "Audio encoding failed",
            ErrorCode::AudioPlayFailed => "Audio playback failed",
            
            ErrorCode::TextInvalidEncoding => "Invalid text encoding",
            ErrorCode::TextTokenizationFailed => "Text tokenization failed",
            ErrorCode::TextTooLong => "Text exceeds maximum length",
            ErrorCode::TextEmpty => "Text is empty",
            
            ErrorCode::ResourceExhausted => "System resources exhausted",
            ErrorCode::ResourceNotFound => "Resource not found",
            ErrorCode::ResourceLocked => "Resource is locked",
            ErrorCode::ResourceLeak => "Resource leak detected",
            
            ErrorCode::DeviceNotFound => "Device not found",
            ErrorCode::DeviceOutOfMemory => "Device out of memory",
            ErrorCode::DeviceComputeFailed => "Device computation failed",
            ErrorCode::DeviceDriverError => "Device driver error",
            
            ErrorCode::ValidationFailed => "Validation failed",
            ErrorCode::ValidationRangeError => "Value out of valid range",
            ErrorCode::ValidationError => "Validation error",
            
            ErrorCode::TimeoutExceeded => "Operation timeout exceeded",
            ErrorCode::TimeoutWaitingForResource => "Timeout waiting for resource",
            ErrorCode::TimeoutWaitingForLock => "Timeout waiting for lock",
            
            ErrorCode::InternalError => "Internal error occurred",
            ErrorCode::InternalInvariant => "Internal invariant violated",
            ErrorCode::InternalUnreachable => "Unreachable code executed",
            
            ErrorCode::Unknown => "Unknown error",
        }
    }
}

/// Error recovery strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecoveryStrategy {
    /// Retry the operation
    Retry {
        /// Maximum retry attempts
        max_attempts: usize,
        /// Delay between retries in milliseconds
        delay_ms: u64,
        /// Backoff multiplier (as integer percentage, e.g., 200 = 2.0x)
        backoff_percent: u32,
    },
    /// Fallback to alternative
    Fallback {
        /// Fallback operation name
        fallback_name: &'static str,
    },
    /// Degrade gracefully
    Degrade {
        /// Degraded quality level
        quality_level: QualityLevel,
    },
    /// Skip and continue
    Skip,
    /// Abort immediately
    Abort,
    /// Ask user for input
    AskUser,
}

/// Quality level for degradation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QualityLevel {
    Low,
    Medium,
    High,
    Maximum,
}

/// Extended error information
#[derive(Debug, Clone)]
pub struct ExtendedErrorInfo {
    /// Error code
    pub code: ErrorCode,
    /// Severity level
    pub severity: ErrorSeverity,
    /// Operation being performed
    pub operation: Option<String>,
    /// Component where error occurred
    pub component: Option<String>,
    /// Input that caused the error
    pub input_summary: Option<String>,
    /// Suggested fix
    pub suggestion: Option<String>,
    /// Recovery strategy
    pub recovery: Option<RecoveryStrategy>,
    /// Related documentation link
    pub doc_link: Option<String>,
    /// Related error codes
    pub related_codes: Vec<ErrorCode>,
    /// Timestamp in milliseconds since epoch
    pub timestamp_ms: u64,
    /// Thread ID
    pub thread_id: Option<u64>,
    /// Call stack (if available)
    pub call_stack: Option<String>,
}

impl Default for ExtendedErrorInfo {
    fn default() -> Self {
        Self {
            code: ErrorCode::Unknown,
            severity: ErrorSeverity::Error,
            operation: None,
            component: None,
            input_summary: None,
            suggestion: None,
            recovery: None,
            doc_link: None,
            related_codes: Vec::new(),
            timestamp_ms: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64,
            thread_id: None,
            call_stack: None,
        }
    }
}

impl ExtendedErrorInfo {
    /// Create new extended error info with code
    pub fn new(code: ErrorCode) -> Self {
        Self {
            code,
            severity: code.severity(),
            ..Default::default()
        }
    }

    /// Add operation context
    pub fn with_operation(mut self, op: impl Into<String>) -> Self {
        self.operation = Some(op.into());
        self
    }

    /// Add component context
    pub fn with_component(mut self, component: impl Into<String>) -> Self {
        self.component = Some(component.into());
        self
    }

    /// Add input summary
    pub fn with_input(mut self, input: impl Into<String>) -> Self {
        self.input_summary = Some(input.into());
        self
    }

    /// Add suggestion
    pub fn with_suggestion(mut self, suggestion: impl Into<String>) -> Self {
        self.suggestion = Some(suggestion.into());
        self
    }

    /// Add recovery strategy
    pub fn with_recovery(mut self, recovery: RecoveryStrategy) -> Self {
        self.recovery = Some(recovery);
        self
    }

    /// Add documentation link
    pub fn with_doc(mut self, link: impl Into<String>) -> Self {
        self.doc_link = Some(link.into());
        self
    }

    /// Add related error codes
    pub fn with_related_codes(mut self, codes: Vec<ErrorCode>) -> Self {
        self.related_codes = codes;
        self
    }
}

impl fmt::Display for ExtendedErrorInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}] {}: {}", self.code, self.severity, self.code.description())?;
        
        if let Some(ref op) = self.operation {
            write!(f, " (operation: {})", op)?;
        }
        
        if let Some(ref component) = self.component {
            write!(f, " (component: {})", component)?;
        }
        
        if let Some(ref suggestion) = self.suggestion {
            write!(f, "\n  Suggestion: {}", suggestion)?;
        }
        
        if let Some(ref doc) = self.doc_link {
            write!(f, "\n  Documentation: {}", doc)?;
        }
        
        Ok(())
    }
}

/// Convert TtsError to ExtendedErrorInfo
pub fn to_extended_error(err: &TtsError) -> ExtendedErrorInfo {
    match err {
        TtsError::Config { message, path } => {
            let mut info = ExtendedErrorInfo::new(ErrorCode::ConfigInvalidFormat)
                .with_input(message.clone());
            if let Some(p) = path {
                info = info.with_input(format!("{}: {:?}", message, p));
            }
            info
        }

        TtsError::ModelLoad { message, component, path: _ } => {
            let code = if message.contains("not found") || message.contains("missing") {
                ErrorCode::ModelNotFound
            } else if message.contains("format") || message.contains("invalid") {
                ErrorCode::ModelInvalidFormat
            } else if message.contains("version") {
                ErrorCode::ModelVersionMismatch
            } else {
                ErrorCode::ModelLoadTimeout
            };
            
            ExtendedErrorInfo::new(code)
                .with_component(component.clone())
                .with_input(message.clone())
                .with_suggestion("Check model path and ensure weights are downloaded")
        }
        
        TtsError::Inference { stage, message, recoverable } => {
            let code = if *recoverable {
                ErrorCode::InferenceStoppedEarly
            } else {
                ErrorCode::InferenceFailed
            };
            
            ExtendedErrorInfo::new(code)
                .with_operation(format!("{}", stage))
                .with_input(message.clone())
                .with_recovery(RecoveryStrategy::Retry {
                    max_attempts: 3,
                    delay_ms: 100,
                    backoff_percent: 200,
                })
        }
        
        TtsError::Audio { message, operation } => {
            let code = match operation {
                AudioOperation::Loading => ErrorCode::AudioNotFound,
                AudioOperation::Resampling | AudioOperation::MelSpectrogram => ErrorCode::AudioDecodeFailed,
                AudioOperation::Saving => ErrorCode::AudioEncodeFailed,
                AudioOperation::Mixing => ErrorCode::AudioPlayFailed,
            };
            
            ExtendedErrorInfo::new(code)
                .with_operation(format!("audio {}", operation))
                .with_input(message.clone())
        }
        
        TtsError::Text { message, operation } => {
            let code = match operation {
                TextOperation::Normalization | TextOperation::Segmentation => ErrorCode::TextInvalidEncoding,
                TextOperation::Tokenization => ErrorCode::TextTokenizationFailed,
                TextOperation::Phonemization => ErrorCode::TextTokenizationFailed,
            };
            
            ExtendedErrorInfo::new(code)
                .with_operation(format!("text {}", operation))
                .with_input(message.clone())
        }
        
        TtsError::Resource { message, resource_type } => {
            let code = if message.contains("exhausted") || message.contains("limit") {
                ErrorCode::ResourceExhausted
            } else if message.contains("not found") {
                ErrorCode::ResourceNotFound
            } else if message.contains("locked") {
                ErrorCode::ResourceLocked
            } else {
                ErrorCode::ResourceLeak
            };
            
            ExtendedErrorInfo::new(code)
                .with_component(format!("{}", resource_type))
                .with_input(message.clone())
        }
        
        TtsError::Device { message, device_type } => {
            let code = if message.contains("not found") || message.contains("unavailable") {
                ErrorCode::DeviceNotFound
            } else if message.contains("memory") || message.contains("OOM") {
                ErrorCode::DeviceOutOfMemory
            } else {
                ErrorCode::DeviceComputeFailed
            };
            
            ExtendedErrorInfo::new(code)
                .with_component(device_type.clone())
                .with_input(message.clone())
        }
        
        TtsError::Validation { message, field } => {
            let code = if message.contains("range") || message.contains("between") {
                ErrorCode::ValidationRangeError
            } else {
                ErrorCode::ValidationFailed
            };
            
            let mut info = ExtendedErrorInfo::new(code)
                .with_input(message.clone());
            
            if let Some(f) = field {
                info = info.with_component(f.clone());
            }
            
            info
        }
        
        TtsError::Io { message, path } => {
            let mut info = ExtendedErrorInfo::new(ErrorCode::ConfigNotFound)
                .with_input(message.clone());
            
            if let Some(p) = path {
                info = info.with_input(format!("{}: {:?}", message, p));
            }
            
            info
        }
        
        TtsError::Timeout { message, duration_ms } => {
            ExtendedErrorInfo::new(ErrorCode::TimeoutExceeded)
                .with_input(message.clone())
                .with_suggestion(format!("Consider increasing timeout beyond {}ms", duration_ms))
        }
        
        TtsError::Internal { message, location } => {
            let mut info = ExtendedErrorInfo::new(ErrorCode::InternalError)
                .with_input(message.clone());
            
            if let Some(loc) = location {
                info = info.with_component(loc.clone());
            }
            
            info
        }
    }
}

/// Error event for broadcasting
#[derive(Debug, Clone)]
pub struct ErrorEvent {
    /// Error information
    pub error: ExtendedErrorInfo,
    /// Original error
    pub original: String,
    /// Context data
    pub context: std::collections::HashMap<String, String>,
}

impl ErrorEvent {
    /// Create new error event
    pub fn new(error: ExtendedErrorInfo, original: impl Into<String>) -> Self {
        Self {
            error,
            original: original.into(),
            context: std::collections::HashMap::new(),
        }
    }

    /// Add context data
    pub fn with_context(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.context.insert(key.into(), value.into());
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_code_display() {
        assert_eq!(ErrorCode::ConfigNotFound.to_string(), "E01001");
        assert_eq!(ErrorCode::Unknown.to_string(), "E99999");
    }

    #[test]
    fn test_error_code_severity() {
        assert_eq!(ErrorCode::ConfigNotFound.severity(), ErrorSeverity::Warning);
        assert_eq!(ErrorCode::InternalError.severity(), ErrorSeverity::Fatal);
        assert_eq!(ErrorCode::Unknown.severity(), ErrorSeverity::Error);
    }

    #[test]
    fn test_extended_error_info() {
        let info = ExtendedErrorInfo::new(ErrorCode::ModelNotFound)
            .with_component("GPT")
            .with_suggestion("Check model path");

        assert_eq!(info.code, ErrorCode::ModelNotFound);
        assert_eq!(info.component, Some("GPT".to_string()));
        assert_eq!(info.suggestion, Some("Check model path".to_string()));
    }

    #[test]
    fn test_extended_error_display() {
        let info = ExtendedErrorInfo::new(ErrorCode::ConfigNotFound);
        let display = format!("{}", info);

        assert!(display.contains("E01001"));
        assert!(display.contains("Configuration file not found"));
    }
}

// ============================================================================
// Recovery Strategy Executor
// ============================================================================

/// Recovery action result
#[derive(Debug, Clone)]
pub struct RecoveryResult<T> {
    /// Whether recovery was successful
    pub success: bool,
    /// The recovered value (if any)
    pub value: Option<T>,
    /// Attempts made
    pub attempts: u32,
    /// Total time spent in milliseconds
    pub total_time_ms: u64,
    /// Error that caused recovery (if any)
    pub error: Option<String>,
}

impl<T> RecoveryResult<T> {
    /// Create a success result
    pub fn success(value: T, attempts: u32, time_ms: u64) -> Self {
        Self {
            success: true,
            value: Some(value),
            attempts,
            total_time_ms: time_ms,
            error: None,
        }
    }

    /// Create a failure result
    pub fn failure(error: impl Into<String>, attempts: u32, time_ms: u64) -> Self {
        Self {
            success: false,
            value: None,
            attempts,
            total_time_ms: time_ms,
            error: Some(error.into()),
        }
    }
}

/// Recovery strategy executor
///
/// Provides automated error recovery with configurable strategies.
///
/// # Example
///
/// ```rust,ignore
/// let executor = RecoveryExecutor::new();
///
/// // Retry with exponential backoff
/// let result = executor.retry_with_backoff(
///     || load_model(),
///     3,  // max attempts
///     100, // initial delay ms
///     2.0, // backoff multiplier
/// )?;
///
/// // Fallback to alternative
/// let result = executor.with_fallback(
///     || primary_operation(),
///     || fallback_operation(),
/// )?;
/// ```
pub struct RecoveryExecutor {
    /// Default max retries
    pub default_max_retries: u32,
    /// Default initial delay in milliseconds
    pub default_initial_delay_ms: u64,
    /// Default backoff multiplier
    pub default_backoff_multiplier: f64,
}

impl Default for RecoveryExecutor {
    fn default() -> Self {
        Self::new()
    }
}

impl RecoveryExecutor {
    /// Create a new recovery executor with defaults
    pub fn new() -> Self {
        Self {
            default_max_retries: 3,
            default_initial_delay_ms: 100,
            default_backoff_multiplier: 2.0,
        }
    }

    /// Retry an operation with exponential backoff
    ///
    /// # Arguments
    ///
    /// * `operation` - The operation to retry
    /// * `max_attempts` - Maximum number of attempts
    /// * `initial_delay_ms` - Initial delay in milliseconds
    /// * `backoff_multiplier` - Multiplier for each subsequent delay
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let result = executor.retry_with_backoff(
    ///     || fetch_data(),
    ///     3,
    ///     100,
    ///     2.0,
    /// )?;
    /// ```
    pub fn retry_with_backoff<T, E, F>(
        &self,
        mut operation: F,
        max_attempts: u32,
        initial_delay_ms: u64,
        backoff_multiplier: f64,
    ) -> RecoveryResult<T>
    where
        F: FnMut() -> std::result::Result<T, E>,
        E: std::fmt::Display,
    {
        use std::{thread, time::Duration};
        
        let start = std::time::Instant::now();
        let mut delay = initial_delay_ms;

        for attempt in 1..=max_attempts {
            match operation() {
                Ok(value) => {
                    return RecoveryResult::success(
                        value,
                        attempt,
                        start.elapsed().as_millis() as u64,
                    );
                }
                Err(e) => {
                    if attempt < max_attempts {
                        tracing::warn!(
                            "Attempt {} failed: {}. Retrying in {}ms...",
                            attempt,
                            e,
                            delay
                        );
                        thread::sleep(Duration::from_millis(delay));
                        delay = (delay as f64 * backoff_multiplier) as u64;
                    } else {
                        return RecoveryResult::failure(
                            e.to_string(),
                            attempt,
                            start.elapsed().as_millis() as u64,
                        );
                    }
                }
            }
        }

        unreachable!()
    }

    /// Retry an operation with default settings
    pub fn retry<T, E, F>(&self, operation: F) -> RecoveryResult<T>
    where
        F: FnMut() -> std::result::Result<T, E>,
        E: std::fmt::Display,
    {
        self.retry_with_backoff(
            operation,
            self.default_max_retries,
            self.default_initial_delay_ms,
            self.default_backoff_multiplier,
        )
    }

    /// Try primary operation, fallback to alternative on failure
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let result = executor.with_fallback(
    ///     || load_from_gpu(),
    ///     || load_from_cpu(),
    /// )?;
    /// ```
    pub fn with_fallback<T, E, F, G>(
        &self,
        mut primary: F,
        mut fallback: G,
    ) -> RecoveryResult<T>
    where
        F: FnMut() -> std::result::Result<T, E>,
        G: FnMut() -> std::result::Result<T, E>,
        E: std::fmt::Display,
    {
        let start = std::time::Instant::now();

        match primary() {
            Ok(value) => {
                return RecoveryResult::success(value, 1, start.elapsed().as_millis() as u64);
            }
            Err(e) => {
                tracing::warn!("Primary operation failed: {}. Trying fallback...", e);
                
                match fallback() {
                    Ok(value) => {
                        return RecoveryResult::success(value, 2, start.elapsed().as_millis() as u64);
                    }
                    Err(e) => {
                        return RecoveryResult::failure(
                            format!("Primary: {}, Fallback: {}", e, e),
                            2,
                            start.elapsed().as_millis() as u64,
                        );
                    }
                }
            }
        }
    }

    /// Degrade operation quality on failure
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let result = executor.degrade_on_error(
    ///     || synthesize_high_quality(),
    ///     || synthesize_medium_quality(),
    ///     || synthesize_low_quality(),
    /// )?;
    /// ```
    pub fn degrade_on_error<T, E, F>(
        &self,
        operations: Vec<F>,
    ) -> RecoveryResult<T>
    where
        F: FnMut() -> std::result::Result<T, E>,
        E: std::fmt::Display,
    {
        let start = std::time::Instant::now();
        let len = operations.len();

        for (i, mut op) in operations.into_iter().enumerate() {
            match op() {
                Ok(value) => {
                    return RecoveryResult::success(
                        value,
                        (i + 1) as u32,
                        start.elapsed().as_millis() as u64,
                    );
                }
                Err(e) => {
                    tracing::warn!("Quality level {} failed: {}", i + 1, e);
                }
            }
        }

        RecoveryResult::failure(
            "All quality levels failed",
            len as u32,
            start.elapsed().as_millis() as u64,
        )
    }
}
