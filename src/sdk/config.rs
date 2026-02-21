//! SDK Configuration
//!
//! Provides simplified configuration for SDK initialization.

use std::path::PathBuf;

/// SDK configuration
#[derive(Debug, Clone)]
pub struct SdkConfig {
    /// Use GPU if available
    pub gpu: bool,
    /// Default engine ID
    pub default_engine: String,
    /// Memory limit in bytes (0 = unlimited)
    pub memory_limit: usize,
    /// GPU memory limit in bytes (0 = unlimited)
    pub gpu_memory_limit: usize,
    /// Enable metrics collection
    pub metrics: bool,
    /// Enable event logging
    pub event_logging: bool,
    /// Log level (trace, debug, info, warn, error)
    pub log_level: String,
    /// Model cache directory
    pub cache_dir: Option<PathBuf>,
    /// Model directory
    pub model_dir: Option<PathBuf>,
    /// Number of worker threads
    pub worker_threads: usize,
    /// Enable lazy loading
    pub lazy_load: bool,
}

impl Default for SdkConfig {
    fn default() -> Self {
        Self {
            gpu: false,
            default_engine: "qwen3-tts".to_string(),
            memory_limit: 0,
            gpu_memory_limit: 0,
            metrics: true,
            event_logging: false,
            log_level: "info".to_string(),
            cache_dir: None,
            model_dir: None,
            worker_threads: 4,
            lazy_load: false,
        }
    }
}

impl SdkConfig {
    /// Create configuration builder
    pub fn builder() -> SdkConfigBuilder {
        SdkConfigBuilder::new()
    }

    /// Create config for CPU-only usage
    pub fn cpu() -> Self {
        Self {
            gpu: false,
            ..Default::default()
        }
    }

    /// Create config for GPU usage
    pub fn gpu() -> Self {
        Self {
            gpu: true,
            ..Default::default()
        }
    }

    /// Create config for high-quality synthesis
    pub fn high_quality() -> Self {
        Self {
            gpu: true,
            ..Default::default()
        }
    }

    /// Create config for fast synthesis
    pub fn fast() -> Self {
        Self {
            gpu: true,
            lazy_load: true,
            ..Default::default()
        }
    }
}

/// SDK configuration builder
pub struct SdkConfigBuilder {
    config: SdkConfig,
}

impl SdkConfigBuilder {
    /// Create new builder with defaults
    pub fn new() -> Self {
        Self {
            config: SdkConfig::default(),
        }
    }

    /// Enable/disable GPU
    pub fn gpu(mut self, enable: bool) -> Self {
        self.config.gpu = enable;
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

    /// Set GPU memory limit
    pub fn gpu_memory_limit(mut self, bytes: usize) -> Self {
        self.config.gpu_memory_limit = bytes;
        self
    }

    /// Enable/disable metrics
    pub fn metrics(mut self, enable: bool) -> Self {
        self.config.metrics = enable;
        self
    }

    /// Enable/disable event logging
    pub fn event_logging(mut self, enable: bool) -> Self {
        self.config.event_logging = enable;
        self
    }

    /// Set log level
    pub fn log_level(mut self, level: impl Into<String>) -> Self {
        self.config.log_level = level.into();
        self
    }

    /// Set cache directory
    pub fn cache_dir(mut self, path: impl Into<PathBuf>) -> Self {
        self.config.cache_dir = Some(path.into());
        self
    }

    /// Set model directory
    pub fn model_dir(mut self, path: impl Into<PathBuf>) -> Self {
        self.config.model_dir = Some(path.into());
        self
    }

    /// Set worker threads
    pub fn worker_threads(mut self, threads: usize) -> Self {
        self.config.worker_threads = threads;
        self
    }

    /// Enable/disable lazy loading
    pub fn lazy_load(mut self, enable: bool) -> Self {
        self.config.lazy_load = enable;
        self
    }

    /// Build configuration
    pub fn build(self) -> SdkConfig {
        self.config
    }
}

impl Default for SdkConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}
