//! Configuration Center
//!
//! Provides centralized configuration management with:
//! - Hot reload support
//! - Type-safe configuration values
//! - Configuration watchers
//! - Default values and validation

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

use dashmap::DashMap;
use serde::{Deserialize, Serialize};

use crate::core::error::{Result, TtsError};

/// Configuration value types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConfigValue {
    /// String value
    String(String),
    /// Integer value
    Integer(i64),
    /// Float value
    Float(f64),
    /// Boolean value
    Boolean(bool),
    /// List of values
    List(Vec<ConfigValue>),
    /// Nested configuration
    Table(HashMap<String, ConfigValue>),
}

impl ConfigValue {
    /// Get as string
    pub fn as_str(&self) -> Option<&str> {
        match self {
            ConfigValue::String(s) => Some(s),
            _ => None,
        }
    }

    /// Get as integer
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            ConfigValue::Integer(i) => Some(*i),
            _ => None,
        }
    }

    /// Get as float
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            ConfigValue::Float(f) => Some(*f),
            ConfigValue::Integer(i) => Some(*i as f64),
            _ => None,
        }
    }

    /// Get as boolean
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            ConfigValue::Boolean(b) => Some(*b),
            _ => None,
        }
    }
}

/// Configuration watcher callback
pub type ConfigWatcher = Arc<dyn Fn(&str, &ConfigValue) + Send + Sync>;

/// Configuration entry with metadata
#[derive(Clone)]
pub struct ConfigEntry {
    /// Configuration value
    pub value: ConfigValue,
    /// When this entry was last modified
    pub modified_at: Instant,
    /// Source file path (if any)
    pub source: Option<PathBuf>,
    /// Whether this entry can be hot-reloaded
    pub hot_reloadable: bool,
}

impl ConfigEntry {
    /// Create a new config entry
    pub fn new(value: ConfigValue) -> Self {
        Self {
            value,
            modified_at: Instant::now(),
            source: None,
            hot_reloadable: true,
        }
    }

    /// Create from file
    pub fn from_file(value: ConfigValue, path: &Path) -> Self {
        Self {
            value,
            modified_at: Instant::now(),
            source: Some(path.to_path_buf()),
            hot_reloadable: true,
        }
    }

    /// Mark as non-reloadable
    pub fn with_hot_reload(mut self, enabled: bool) -> Self {
        self.hot_reloadable = enabled;
        self
    }
}

/// Configuration center for centralized management
///
/// # Example
///
/// ```rust,ignore
/// let config = ConfigCenter::new();
///
/// // Set configuration
/// config.set("tts.temperature", ConfigValue::Float(0.8))?;
/// config.set("tts.max_tokens", ConfigValue::Integer(120))?;
///
/// // Get configuration
/// let temp = config.get_float("tts.temperature").unwrap_or(0.7);
///
/// // Watch for changes
/// config.watch("tts.temperature", Arc::new(|key, value| {
///     println!("Config changed: {} = {:?}", key, value);
/// }));
///
/// // Load from file
/// config.load_from_file("config.yaml")?;
/// ```
pub struct ConfigCenter {
    /// Configuration storage
    entries: DashMap<String, ConfigEntry>,
    /// Watchers for configuration changes
    watchers: DashMap<String, Vec<ConfigWatcher>>,
    /// Global watchers (notified for all changes)
    global_watchers: RwLock<Vec<ConfigWatcher>>,
    /// Configuration file paths
    file_paths: RwLock<Vec<PathBuf>>,
    /// Auto-reload enabled
    auto_reload: RwLock<bool>,
    /// Auto-reload interval
    reload_interval: RwLock<Duration>,
    /// Last reload time
    last_reload: RwLock<Instant>,
}

impl Default for ConfigCenter {
    fn default() -> Self {
        Self::new()
    }
}

impl ConfigCenter {
    /// Create a new configuration center
    pub fn new() -> Self {
        Self {
            entries: DashMap::new(),
            watchers: DashMap::new(),
            global_watchers: RwLock::new(Vec::new()),
            file_paths: RwLock::new(Vec::new()),
            auto_reload: RwLock::new(false),
            reload_interval: RwLock::new(Duration::from_secs(10)),
            last_reload: RwLock::new(Instant::now()),
        }
    }

    /// Create with default TTS configuration
    pub fn with_defaults() -> Self {
        let center = Self::new();
        
        // Set default TTS configuration
        let _ = center.set("tts.temperature", ConfigValue::Float(0.8));
        let _ = center.set("tts.top_k", ConfigValue::Integer(50));
        let _ = center.set("tts.top_p", ConfigValue::Float(0.95));
        let _ = center.set("tts.repetition_penalty", ConfigValue::Float(1.05));
        let _ = center.set("tts.flow_steps", ConfigValue::Integer(25));
        let _ = center.set("tts.flow_cfg_rate", ConfigValue::Float(0.7));
        let _ = center.set("tts.max_tokens", ConfigValue::Integer(120));
        let _ = center.set("tts.sample_rate", ConfigValue::Integer(22050));
        
        center
    }

    /// Set a configuration value
    pub fn set(&self, key: &str, value: ConfigValue) -> Result<()> {
        let entry = ConfigEntry::new(value.clone());
        self.entries.insert(key.to_string(), entry);
        
        // Notify watchers
        self.notify_watchers(key, &value);
        
        Ok(())
    }

    /// Get a configuration value
    pub fn get(&self, key: &str) -> Option<ConfigValue> {
        self.entries.get(key).map(|e| e.value.clone())
    }

    /// Get as string
    pub fn get_string(&self, key: &str) -> Option<String> {
        self.get(key).and_then(|v| match v {
            ConfigValue::String(s) => Some(s),
            _ => None,
        })
    }

    /// Get as integer
    pub fn get_i64(&self, key: &str) -> Option<i64> {
        self.get(key).and_then(|v| v.as_i64())
    }

    /// Get as float
    pub fn get_f64(&self, key: &str) -> Option<f64> {
        self.get(key).and_then(|v| v.as_f64())
    }

    /// Get as boolean
    pub fn get_bool(&self, key: &str) -> Option<bool> {
        self.get(key).and_then(|v| v.as_bool())
    }

    /// Get with default value
    pub fn get_or<T, F>(&self, key: &str, default: T, extractor: F) -> T
    where
        F: Fn(&ConfigValue) -> Option<T>,
    {
        self.get(key).and_then(|v| extractor(&v)).unwrap_or(default)
    }

    /// Get float with default
    pub fn get_float_or(&self, key: &str, default: f64) -> f64 {
        self.get_or(key, default, |v| v.as_f64())
    }

    /// Get integer with default
    pub fn get_int_or(&self, key: &str, default: i64) -> i64 {
        self.get_or(key, default, |v| v.as_i64())
    }

    /// Get string with default
    pub fn get_string_or(&self, key: &str, default: &str) -> String {
        self.get_or(key, default.to_string(), |v| v.as_str().map(|s| s.to_string()))
    }

    /// Check if key exists
    pub fn contains(&self, key: &str) -> bool {
        self.entries.contains_key(key)
    }

    /// Remove a configuration key
    pub fn remove(&self, key: &str) -> Option<ConfigEntry> {
        self.entries.remove(key).map(|(_, v)| v)
    }

    /// Clear all configuration
    pub fn clear(&self) {
        self.entries.clear();
    }

    /// Load configuration from YAML file
    pub fn load_from_file(&self, path: &Path) -> Result<()> {
        use std::fs;
        
        let content = fs::read_to_string(path)
            .map_err(|e| TtsError::Io { 
                message: format!("Failed to read config file: {}", e), 
                path: Some(path.to_path_buf()) 
            })?;
        
        let parsed: serde_yaml::Value = serde_yaml::from_str(&content)
            .map_err(|e| TtsError::Config { 
                message: format!("Failed to parse YAML: {}", e), 
                path: Some(path.to_path_buf()) 
            })?;
        
        let values = yaml_to_config_value(&parsed);
        
        // Flatten nested structure with dot notation
        flatten_config_value("", &values, &mut |key, value| {
            let entry = ConfigEntry::from_file(value, path);
            self.entries.insert(key.to_string(), entry);
        });
        
        // Track file path
        self.file_paths.write().unwrap().push(path.to_path_buf());
        
        Ok(())
    }

    /// Save configuration to YAML file
    /// Note: This is a simplified implementation. Full implementation would
    /// properly handle nested structures and type conversions.
    pub fn save_to_file(&self, _path: &Path) -> Result<()> {
        // TODO: Implement proper YAML serialization
        Ok(())
    }

    /// Watch a configuration key for changes
    pub fn watch(&self, key: &str, watcher: ConfigWatcher) {
        self.watchers
            .entry(key.to_string())
            .or_default()
            .push(watcher);
    }

    /// Add a global watcher (notified for all changes)
    pub fn watch_global(&self, watcher: ConfigWatcher) {
        self.global_watchers.write().unwrap().push(watcher);
    }

    /// Enable/disable auto-reload
    pub fn set_auto_reload(&self, enabled: bool) {
        *self.auto_reload.write().unwrap() = enabled;
    }

    /// Set auto-reload interval
    pub fn set_reload_interval(&self, interval: Duration) {
        *self.reload_interval.write().unwrap() = interval;
    }

    /// Check if auto-reload is needed
    pub fn should_reload(&self) -> bool {
        if !*self.auto_reload.read().unwrap() {
            return false;
        }
        
        let last_reload = *self.last_reload.read().unwrap();
        let interval = *self.reload_interval.read().unwrap();
        
        last_reload.elapsed() >= interval
    }

    /// Reload configuration from files
    pub fn reload(&self) -> Result<()> {
        let paths = self.file_paths.read().unwrap().clone();
        
        for path in paths {
            if path.exists() {
                self.load_from_file(&path)?;
            }
        }
        
        *self.last_reload.write().unwrap() = Instant::now();
        
        Ok(())
    }

    /// Get all configuration keys
    pub fn keys(&self) -> Vec<String> {
        self.entries.iter().map(|e| e.key().clone()).collect()
    }

    /// Get configuration statistics
    pub fn stats(&self) -> ConfigStats {
        ConfigStats {
            total_entries: self.entries.len(),
            file_count: self.file_paths.read().unwrap().len(),
            watcher_count: self.watchers.iter().map(|e| e.value().len()).sum(),
            global_watcher_count: self.global_watchers.read().unwrap().len(),
            auto_reload_enabled: *self.auto_reload.read().unwrap(),
            last_reload_elapsed: self.last_reload.read().unwrap().elapsed().as_secs(),
        }
    }

    /// Notify watchers of a change
    fn notify_watchers(&self, key: &str, value: &ConfigValue) {
        // Notify key-specific watchers
        if let Some(watchers) = self.watchers.get(key) {
            for watcher in watchers.iter() {
                watcher(key, value);
            }
        }
        
        // Notify global watchers
        for watcher in self.global_watchers.read().unwrap().iter() {
            watcher(key, value);
        }
    }
}

/// Configuration statistics
#[derive(Debug, Clone)]
pub struct ConfigStats {
    /// Total configuration entries
    pub total_entries: usize,
    /// Number of loaded files
    pub file_count: usize,
    /// Number of key-specific watchers
    pub watcher_count: usize,
    /// Number of global watchers
    pub global_watcher_count: usize,
    /// Whether auto-reload is enabled
    pub auto_reload_enabled: bool,
    /// Seconds since last reload
    pub last_reload_elapsed: u64,
}

/// Convert YAML value to ConfigValue
fn yaml_to_config_value(value: &serde_yaml::Value) -> ConfigValue {
    match value {
        serde_yaml::Value::String(s) => ConfigValue::String(s.clone()),
        serde_yaml::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                ConfigValue::Integer(i)
            } else if let Some(f) = n.as_f64() {
                ConfigValue::Float(f)
            } else {
                ConfigValue::Integer(0)
            }
        }
        serde_yaml::Value::Bool(b) => ConfigValue::Boolean(*b),
        serde_yaml::Value::Sequence(seq) => {
            ConfigValue::List(seq.iter().map(yaml_to_config_value).collect())
        }
        serde_yaml::Value::Mapping(map) => {
            let mut table = HashMap::new();
            for (k, v) in map {
                if let Some(key) = k.as_str() {
                    table.insert(key.to_string(), yaml_to_config_value(v));
                }
            }
            ConfigValue::Table(table)
        }
        serde_yaml::Value::Null => ConfigValue::String(String::new()),
        serde_yaml::Value::Tagged(_) => ConfigValue::String(String::new()),
    }
}

/// Convert ConfigValue map to YAML value
fn config_map_to_yaml(map: &HashMap<String, ConfigValue>) -> serde_yaml::Value {
    let mut result = serde_yaml::Mapping::new();
    
    for (k, v) in map {
        let key = serde_yaml::Value::String(k.clone());
        let value = match v {
            ConfigValue::String(s) => serde_yaml::Value::String(s.clone()),
            ConfigValue::Integer(i) => serde_yaml::Value::Number((*i).into()),
            ConfigValue::Float(f) => serde_yaml::Value::Number((*f).into()),
            ConfigValue::Boolean(b) => serde_yaml::Value::Bool(*b),
            ConfigValue::List(list) => {
                serde_yaml::Value::Sequence(list.iter().map(config_value_to_yaml).collect())
            }
            ConfigValue::Table(table) => config_map_to_yaml(table),
        };
        result.insert(key, value);
    }
    
    serde_yaml::Value::Mapping(result)
}

fn config_value_to_yaml(value: &ConfigValue) -> serde_yaml::Value {
    match value {
        ConfigValue::String(s) => serde_yaml::Value::String(s.clone()),
        ConfigValue::Integer(i) => serde_yaml::Value::Number((*i).into()),
        ConfigValue::Float(f) => serde_yaml::Value::Number((*f).into()),
        ConfigValue::Boolean(b) => serde_yaml::Value::Bool(*b),
        ConfigValue::List(list) => {
            serde_yaml::Value::Sequence(list.iter().map(config_value_to_yaml).collect())
        }
        ConfigValue::Table(table) => config_map_to_yaml(table),
    }
}

/// Flatten nested config with dot notation
fn flatten_config_value(prefix: &str, value: &ConfigValue, callback: &mut dyn FnMut(&str, ConfigValue)) {
    match value {
        ConfigValue::Table(table) => {
            for (k, v) in table {
                let new_prefix = if prefix.is_empty() {
                    k.clone()
                } else {
                    format!("{}.{}", prefix, k)
                };
                flatten_config_value(&new_prefix, v, callback);
            }
        }
        _ => callback(prefix, value.clone()),
    }
}

/// Insert value into nested structure
fn insert_nested(map: &mut HashMap<String, ConfigValue>, parts: &[&str], value: ConfigValue) {
    if parts.is_empty() {
        return;
    }
    
    if parts.len() == 1 {
        map.insert(parts[0].to_string(), value);
        return;
    }
    
    let nested = map
        .entry(parts[0].to_string())
        .or_insert_with(|| ConfigValue::Table(HashMap::new()));
    
    if let ConfigValue::Table(ref mut inner) = nested {
        insert_nested(inner, &parts[1..], value);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_center_basic() {
        let config = ConfigCenter::new();
        
        config.set("test.key", ConfigValue::String("value".to_string())).unwrap();
        
        assert!(config.contains("test.key"));
        assert_eq!(config.get_string("test.key"), Some("value".to_string()));
    }

    #[test]
    fn test_config_center_defaults() {
        let config = ConfigCenter::with_defaults();
        
        assert!(config.contains("tts.temperature"));
        assert_eq!(config.get_float_or("tts.temperature", 0.0), 0.8);
        assert_eq!(config.get_int_or("tts.top_k", 0), 50);
    }

    #[test]
    fn test_config_center_watchers() {
        let config = ConfigCenter::new();
        let notified = Arc::new(RwLock::new(false));
        
        let notified_clone = Arc::clone(&notified);
        config.watch("test.key", Arc::new(move |_, _| {
            *notified_clone.write().unwrap() = true;
        }));
        
        config.set("test.key", ConfigValue::Integer(42)).unwrap();
        
        assert!(*notified.read().unwrap());
    }

    #[test]
    fn test_config_stats() {
        let config = ConfigCenter::with_defaults();
        let stats = config.stats();
        
        assert!(stats.total_entries > 0);
        assert!(!stats.auto_reload_enabled);
    }
}
