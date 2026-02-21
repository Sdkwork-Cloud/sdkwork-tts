//! Core abstractions and framework-level components
//!
//! This module provides the foundational abstractions that make SDKWork-TTS
//! a flexible, extensible, and production-ready TTS framework.
//!
//! # Modules
//!
//! - `error`: Structured error handling with error codes and recovery strategies
//! - `error_ext`: Extended error handling with severity levels and event broadcasting
//! - `traits`: Core trait definitions for all components
//! - `resource`: Resource lifecycle management
//! - `metrics`: Performance metrics collection and monitoring
//! - `metrics_export`: Multi-format metrics export (Prometheus, JSON, CSV, Console)
//! - `event_bus`: Publish-subscribe event system for decoupled communication
//! - `plugin`: Plugin system for extensible architecture
//! - `builder`: Builder patterns for ergonomic object construction

pub mod error;
pub mod error_ext;
pub mod traits;
pub mod resource;
pub mod metrics;
pub mod metrics_export;
pub mod event_bus;
pub mod plugin;
pub mod builder;
pub mod config_center;

// Error handling exports
pub use error::{TtsError, Result, ErrorContext, ResourceType};
pub use error_ext::{
    ErrorCode, ErrorSeverity, RecoveryStrategy, QualityLevel,
    ExtendedErrorInfo, ErrorEvent, to_extended_error,
    RecoveryExecutor, RecoveryResult,
};
// Config center exports
pub use config_center::{ConfigCenter, ConfigValue, ConfigEntry, ConfigStats};

// Core trait exports
pub use traits::{
    ModelComponent,
    Encoder,
    Decoder,
    Synthesizer,
    Configurable,
    Loadable,
    Initializable,
    Preprocessor,
    Postprocessor,
    Cacheable,
    Tunable,
    Batched,
    Observable,
    DeviceAffinity,
    Versioned,
    ParameterSchema, ParameterType,
    ComparisonResult, Recommendation,
    PostprocessOptions, OutputFormat,
};

// Resource management exports
pub use resource::{
    ResourceManager, ResourceHandle, ResourceId,
    ResourceMetadata, ResourceState, ResourceStatistics,
};

// Metrics exports
pub use metrics::{
    MetricsCollector, PerformanceMetrics, TimingInfo,
    TimerStats, HistogramStats, MetricsReport, TimerReport,
};

// Metrics export formats
pub use metrics_export::{
    MetricsExporter,
    PrometheusExporter,
    JsonExporter,
    CsvExporter,
    ConsoleExporter,
    MultiExporter,
};

// Event bus exports
pub use event_bus::{
    EventBus, Event, EventHandler, SubscriptionId,
    PublishResult, ChannelStats, EventRecord,
    events,
};

// Plugin system exports
pub use plugin::{
    Plugin, PluginContext, PluginMetadata, PluginRegistry,
    PluginStats, PluginBuilder,
};

// Builder exports
pub use builder::{TtsBuilder, TtsConfigBuilder, presets, validation};
