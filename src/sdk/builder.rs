//! SDK Builder
//!
//! Provides fluent API for SDK initialization.

use crate::core::error::Result as CoreResult;
use crate::engine::{EngineRegistry, init_engines};
use super::config::SdkConfig;
use super::facade::Sdk;

/// SDK Builder for fluent SDK initialization
pub struct SdkBuilder {
    config: SdkConfig,
    init_engines: bool,
    event_bus: bool,
    metrics: bool,
}

impl SdkBuilder {
    /// Create new builder with default config
    pub fn new() -> Self {
        Self {
            config: SdkConfig::default(),
            init_engines: false,
            event_bus: false,
            metrics: true,
        }
    }

    /// Create builder from config
    pub fn from_config(config: SdkConfig) -> Self {
        Self {
            metrics: config.metrics,
            event_bus: config.event_logging,
            config,
            init_engines: false,
        }
    }

    /// Use GPU if available
    pub fn gpu(mut self) -> Self {
        self.config.gpu = true;
        self
    }

    /// Use CPU only
    pub fn cpu(mut self) -> Self {
        self.config.gpu = false;
        self
    }

    /// Set default engine
    pub fn default_engine(mut self, engine: impl Into<String>) -> Self {
        self.config.default_engine = engine.into();
        self
    }

    /// Set memory limit
    pub fn memory_limit(mut self, bytes: usize) -> Self {
        self.config.memory_limit = bytes;
        self
    }

    /// Set log level
    pub fn log_level(mut self, level: impl Into<String>) -> Self {
        self.config.log_level = level.into();
        self
    }

    /// Set model directory
    pub fn model_dir(mut self, path: impl Into<std::path::PathBuf>) -> Self {
        self.config.model_dir = Some(path.into());
        self
    }

    /// Initialize built-in engines
    pub fn with_default_engines(mut self) -> Self {
        self.init_engines = true;
        self
    }

    /// Initialize all available engines
    pub fn with_all_engines(mut self) -> Self {
        self.init_engines = true;
        self
    }

    /// Enable event bus
    pub fn with_event_logging(mut self, enable: bool) -> Self {
        self.event_bus = enable;
        self.config.event_logging = enable;
        self
    }

    /// Enable metrics collection
    pub fn with_metrics(mut self, enable: bool) -> Self {
        self.metrics = enable;
        self.config.metrics = enable;
        self
    }

    /// Build the SDK
    pub fn build(self) -> CoreResult<Sdk> {
        // Initialize logging
        self.init_logging();

        // Create engine registry
        let registry = EngineRegistry::new();

        // Initialize engines if requested
        if self.init_engines {
            init_engines()?;
        }

        // Create SDK instance
        Sdk::new(self.config, registry).map_err(|e| crate::core::error::TtsError::Internal {
            message: e.to_string(),
            location: None,
        })
    }

    /// Initialize logging based on config
    fn init_logging(&self) {
        use tracing_subscriber::{fmt, EnvFilter};

        let filter = EnvFilter::try_from_default_env()
            .unwrap_or_else(|_| EnvFilter::new(&self.config.log_level));

        let _ = fmt()
            .with_target(false)
            .with_thread_ids(false)
            .with_file(false)
            .with_line_number(false)
            .with_env_filter(filter)
            .try_init();
    }
}

impl Default for SdkBuilder {
    fn default() -> Self {
        Self::new()
    }
}
