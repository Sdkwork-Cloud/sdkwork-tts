//! Engine configuration management
//!
//! Provides flexible configuration for TTS engines.

use std::collections::HashMap;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use super::traits::{DeviceConfig, MemoryConfig, PerformanceConfig};

/// Engine configuration
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct EngineConfig {
    /// Engine ID
    pub engine_id: String,
    /// Model directory
    pub model_dir: PathBuf,
    /// Device configuration
    #[serde(default)]
    pub device: DeviceConfig,
    /// Memory settings
    #[serde(default)]
    pub memory: MemoryConfig,
    /// Performance settings
    #[serde(default)]
    pub performance: PerformanceConfig,
    /// Engine-specific settings
    #[serde(default)]
    pub engine_specific: HashMap<String, String>,
}

/// Engine configuration builder
pub struct EngineConfigBuilder {
    config: EngineConfig,
}

impl EngineConfigBuilder {
    /// Create a new configuration builder
    pub fn new(engine_id: impl Into<String>) -> Self {
        Self {
            config: EngineConfig {
                engine_id: engine_id.into(),
                ..Default::default()
            },
        }
    }

    /// Set model directory
    pub fn model_dir(mut self, path: impl Into<PathBuf>) -> Self {
        self.config.model_dir = path.into();
        self
    }

    /// Enable GPU
    pub fn use_gpu(mut self, enable: bool) -> Self {
        self.config.device.use_gpu = enable;
        self
    }

    /// Set GPU device ID
    pub fn gpu_id(mut self, id: usize) -> Self {
        self.config.device.gpu_id = id;
        self
    }

    /// Enable mixed precision
    pub fn mixed_precision(mut self, enable: bool) -> Self {
        self.config.device.mixed_precision = enable;
        self
    }

    /// Set number of CPU threads
    pub fn num_threads(mut self, threads: usize) -> Self {
        self.config.device.num_threads = threads;
        self
    }

    /// Set maximum CPU memory
    pub fn max_cpu_memory(mut self, bytes: usize) -> Self {
        self.config.memory.max_cpu_memory = bytes;
        self
    }

    /// Set maximum GPU memory
    pub fn max_gpu_memory(mut self, bytes: usize) -> Self {
        self.config.memory.max_gpu_memory = bytes;
        self
    }

    /// Enable memory pooling
    pub fn enable_pooling(mut self, enable: bool) -> Self {
        self.config.memory.enable_pooling = enable;
        self
    }

    /// Set idle timeout
    pub fn idle_timeout(mut self, seconds: u64) -> Self {
        self.config.memory.idle_timeout_secs = seconds;
        self
    }

    /// Enable caching
    pub fn enable_cache(mut self, enable: bool) -> Self {
        self.config.performance.enable_cache = enable;
        self
    }

    /// Set cache size
    pub fn cache_size(mut self, bytes: usize) -> Self {
        self.config.performance.cache_size = bytes;
        self
    }

    /// Set batch size
    pub fn batch_size(mut self, size: usize) -> Self {
        self.config.performance.batch_size = size;
        self
    }

    /// Enable async processing
    pub fn async_processing(mut self, enable: bool) -> Self {
        self.config.performance.async_processing = enable;
        self
    }

    /// Set worker threads
    pub fn worker_threads(mut self, threads: usize) -> Self {
        self.config.performance.worker_threads = threads;
        self
    }

    /// Add engine-specific setting
    pub fn custom(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.config.engine_specific.insert(key.into(), value.into());
        self
    }

    /// Build the configuration
    pub fn build(self) -> EngineConfig {
        self.config
    }
}

/// Configuration file format
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigFile {
    /// Configuration version
    pub version: String,
    /// Default engine
    pub default_engine: Option<String>,
    /// Engine configurations
    pub engines: HashMap<String, EngineConfigFile>,
    /// Global settings
    pub global: GlobalConfig,
}

/// Engine configuration in file
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineConfigFile {
    /// Engine type
    pub engine_type: String,
    /// Model path
    pub model_path: PathBuf,
    /// Device settings
    #[serde(default)]
    pub device: DeviceConfig,
    /// Memory settings
    #[serde(default)]
    pub memory: MemoryConfig,
    /// Performance settings
    #[serde(default)]
    pub performance: PerformanceConfig,
    /// Engine-specific settings
    #[serde(default)]
    pub custom: HashMap<String, toml::Value>,
}

/// Global configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlobalConfig {
    /// Log level
    #[serde(default = "default_log_level")]
    pub log_level: String,
    /// Cache directory
    #[serde(default)]
    pub cache_dir: Option<PathBuf>,
    /// Temp directory
    #[serde(default)]
    pub temp_dir: Option<PathBuf>,
    /// Max concurrent requests
    #[serde(default = "default_max_concurrent")]
    pub max_concurrent_requests: usize,
}

fn default_log_level() -> String {
    "info".to_string()
}

fn default_max_concurrent() -> usize {
    4
}

impl Default for GlobalConfig {
    fn default() -> Self {
        Self {
            log_level: default_log_level(),
            cache_dir: None,
            temp_dir: None,
            max_concurrent_requests: default_max_concurrent(),
        }
    }
}

impl ConfigFile {
    /// Load configuration from file
    pub fn load(path: impl AsRef<std::path::Path>) -> crate::core::error::Result<Self> {
        let content = std::fs::read_to_string(path.as_ref())
            .map_err(|e| crate::core::error::TtsError::Io {
                message: format!("Failed to read config file: {}", e),
                path: Some(path.as_ref().to_path_buf()),
            })?;

        let config: Self = toml::from_str(&content)
            .map_err(|e| crate::core::error::TtsError::Config {
                message: format!("Failed to parse config file: {}", e),
                path: Some(path.as_ref().to_path_buf()),
            })?;

        Ok(config)
    }

    /// Save configuration to file
    pub fn save(&self, path: impl AsRef<std::path::Path>) -> crate::core::error::Result<()> {
        let content = toml::to_string_pretty(self)
            .map_err(|e| crate::core::error::TtsError::Config {
                message: format!("Failed to serialize config: {}", e),
                path: None,
            })?;

        std::fs::write(path.as_ref(), content)
            .map_err(|e| crate::core::error::TtsError::Io {
                message: format!("Failed to write config file: {}", e),
                path: Some(path.as_ref().to_path_buf()),
            })?;

        Ok(())
    }

    /// Create default configuration
    pub fn default_config() -> Self {
        let mut engines = HashMap::new();
        
        engines.insert("indextts2".to_string(), EngineConfigFile {
            engine_type: "indextts2".to_string(),
            model_path: PathBuf::from("checkpoints"),
            device: DeviceConfig::default(),
            memory: MemoryConfig::default(),
            performance: PerformanceConfig::default(),
            custom: HashMap::new(),
        });

        Self {
            version: "1.0".to_string(),
            default_engine: Some("indextts2".to_string()),
            engines,
            global: GlobalConfig::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_builder() {
        let config = EngineConfigBuilder::new("test_engine")
            .use_gpu(true)
            .gpu_id(0)
            .num_threads(8)
            .build();

        assert_eq!(config.engine_id, "test_engine");
        assert!(config.device.use_gpu);
        assert_eq!(config.device.gpu_id, 0);
        assert_eq!(config.device.num_threads, 8);
    }

    #[test]
    fn test_default_config() {
        let config = ConfigFile::default_config();
        assert_eq!(config.version, "1.0");
        assert!(config.default_engine.is_some());
        assert!(config.engines.contains_key("indextts2"));
    }
}
