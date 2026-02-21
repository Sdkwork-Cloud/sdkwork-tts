//! Plugin System for extensible TTS framework
//!
//! Provides a flexible plugin architecture:
//! - Dynamic plugin loading and unloading
//! - Plugin lifecycle management
//! - Plugin dependency resolution
//! - Plugin communication channels

use std::any::Any;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};

use crate::core::error::{Result, TtsError};

/// Plugin trait for all framework plugins
pub trait Plugin: Send + Sync {
    /// Plugin unique identifier
    fn plugin_id(&self) -> &'static str;
    
    /// Plugin name
    fn name(&self) -> &'static str;
    
    /// Plugin version
    fn version(&self) -> &'static str;
    
    /// Plugin description
    fn description(&self) -> &'static str;
    
    /// Plugin author
    fn author(&self) -> &'static str {
        "Unknown"
    }
    
    /// Plugin dependencies (other plugin IDs)
    fn dependencies(&self) -> &[&'static str] {
        &[]
    }
    
    /// Initialize the plugin
    fn initialize(&mut self, _ctx: &PluginContext) -> Result<()> {
        Ok(())
    }
    
    /// Shutdown the plugin
    fn shutdown(&mut self) -> Result<()> {
        Ok(())
    }
    
    /// Check if plugin is enabled
    fn is_enabled(&self) -> bool {
        true
    }
    
    /// Enable/disable plugin
    fn set_enabled(&mut self, enabled: bool) {
        let _ = enabled;
    }
    
    /// Get plugin configuration
    fn config(&self) -> Option<&dyn Any> {
        None
    }
    
    /// Set plugin configuration
    fn set_config(&mut self, _config: &dyn Any) -> Result<()> {
        Ok(())
    }
}

/// Plugin metadata
#[derive(Debug, Clone)]
pub struct PluginMetadata {
    /// Plugin ID
    pub id: String,
    /// Plugin name
    pub name: String,
    /// Plugin version
    pub version: String,
    /// Plugin description
    pub description: String,
    /// Plugin author
    pub author: String,
    /// Dependencies
    pub dependencies: Vec<String>,
    /// Plugin file path (if applicable)
    pub path: Option<PathBuf>,
    /// Is plugin loaded
    pub loaded: bool,
    /// Is plugin enabled
    pub enabled: bool,
}

/// Plugin context for plugin-framework communication
pub struct PluginContext {
    /// Plugin directory
    pub plugin_dir: PathBuf,
    /// Configuration directory
    pub config_dir: PathBuf,
    /// Cache directory
    pub cache_dir: PathBuf,
    /// Shared data store
    data: RwLock<HashMap<String, Box<dyn Any + Send + Sync>>>,
}

impl PluginContext {
    /// Create new plugin context
    pub fn new(plugin_dir: PathBuf, config_dir: PathBuf, cache_dir: PathBuf) -> Self {
        Self {
            plugin_dir,
            config_dir,
            cache_dir,
            data: RwLock::new(HashMap::new()),
        }
    }
    
    /// Get shared data
    pub fn get<T: Any + Send + Sync>(&self, key: &str) -> bool {
        let data = self.data.read().unwrap();
        if let Some(boxed) = data.get(key) {
            return boxed.is::<T>();
        }
        false
    }
    
    /// Get shared data value
    pub fn get_value<T: Any + Send + Sync + Clone>(&self, key: &str) -> Option<T> {
        let data = self.data.read().unwrap();
        let boxed = data.get(key)?;
        boxed.downcast_ref::<T>().cloned()
    }
    
    /// Set shared data
    pub fn set<T: Any + Send + Sync>(&self, key: impl Into<String>, value: T) {
        let mut data = self.data.write().unwrap();
        data.insert(key.into(), Box::new(value));
    }
    
    /// Remove shared data
    pub fn remove(&self, key: &str) -> bool {
        let mut data = self.data.write().unwrap();
        data.remove(key).is_some()
    }
}

/// Plugin registry for managing plugins
pub struct PluginRegistry {
    /// Registered plugins
    plugins: RwLock<HashMap<String, Arc<RwLock<dyn Plugin>>>>,
    /// Plugin metadata cache
    metadata: RwLock<HashMap<String, PluginMetadata>>,
    /// Plugin directory
    plugin_dir: PathBuf,
    /// Context for plugins
    context: Arc<PluginContext>,
}

impl PluginRegistry {
    /// Create new plugin registry
    pub fn new<P: AsRef<Path>>(plugin_dir: P) -> Result<Self> {
        let plugin_dir = plugin_dir.as_ref().to_path_buf();
        
        if !plugin_dir.exists() {
            std::fs::create_dir_all(&plugin_dir).map_err(|e| TtsError::Io {
                message: format!("Failed to create plugin directory: {}", e),
                path: Some(plugin_dir.clone()),
            })?;
        }
        
        let config_dir = dirs::config_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("sdkwork-tts");
        
        let cache_dir = dirs::cache_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("sdkwork-tts");
        
        let context = Arc::new(PluginContext::new(
            plugin_dir.clone(),
            config_dir,
            cache_dir,
        ));
        
        Ok(Self {
            plugins: RwLock::new(HashMap::new()),
            metadata: RwLock::new(HashMap::new()),
            plugin_dir,
            context,
        })
    }
    
    /// Register a plugin
    pub fn register<P: Plugin + 'static>(&self, plugin: P) -> Result<()> {
        let metadata = PluginMetadata {
            id: plugin.plugin_id().to_string(),
            name: plugin.name().to_string(),
            version: plugin.version().to_string(),
            description: plugin.description().to_string(),
            author: plugin.author().to_string(),
            dependencies: plugin.dependencies().iter().map(|s| s.to_string()).collect(),
            path: None,
            loaded: false,
            enabled: plugin.is_enabled(),
        };
        
        // Check dependencies
        for dep in &metadata.dependencies {
            let plugins = self.plugins.read().unwrap();
            if !plugins.contains_key(dep) {
                return Err(TtsError::Validation {
                    message: format!("Plugin '{}' requires dependency '{}'", metadata.name, dep),
                    field: Some("dependencies".to_string()),
                });
            }
        }
        
        let plugin_arc = Arc::new(RwLock::new(plugin));
        
        // Initialize plugin
        {
            let mut plugin_write = plugin_arc.write().unwrap();
            plugin_write.initialize(&self.context)?;
        }
        
        self.plugins.write().unwrap()
            .insert(metadata.id.clone(), plugin_arc);
        
        self.metadata.write().unwrap()
            .insert(metadata.id.clone(), metadata);
        
        Ok(())
    }
    
    /// Unregister a plugin
    pub fn unregister(&self, plugin_id: &str) -> Result<()> {
        // Check if other plugins depend on this one
        {
            let metadata = self.metadata.read().unwrap();
            for (id, meta) in metadata.iter() {
                if meta.dependencies.contains(&plugin_id.to_string()) {
                    return Err(TtsError::Validation {
                        message: format!("Cannot unregister '{}': plugin '{}' depends on it", 
                            plugin_id, id),
                        field: Some("plugin_id".to_string()),
                    });
                }
            }
        }
        
        // Shutdown and remove plugin
        if let Some(plugin) = self.plugins.write().unwrap().remove(plugin_id) {
            let mut plugin_write = plugin.write().unwrap();
            plugin_write.shutdown()?;
        }
        
        self.metadata.write().unwrap().remove(plugin_id);
        
        Ok(())
    }
    
    /// Get a plugin by ID
    pub fn get_plugin(&self, plugin_id: &str) -> Option<Arc<RwLock<dyn Plugin>>> {
        self.plugins.read().unwrap().get(plugin_id).cloned()
    }
    
    /// Get plugin metadata
    pub fn get_metadata(&self, plugin_id: &str) -> Option<PluginMetadata> {
        self.metadata.read().unwrap().get(plugin_id).cloned()
    }
    
    /// List all registered plugins
    pub fn list_plugins(&self) -> Vec<PluginMetadata> {
        self.metadata.read().unwrap().values().cloned().collect()
    }
    
    /// Enable a plugin
    pub fn enable_plugin(&self, plugin_id: &str) -> Result<()> {
        if let Some(plugin) = self.plugins.read().unwrap().get(plugin_id) {
            let mut plugin_write = plugin.write().unwrap();
            plugin_write.set_enabled(true);
        }
        
        if let Some(meta) = self.metadata.write().unwrap().get_mut(plugin_id) {
            meta.enabled = true;
        }
        
        Ok(())
    }
    
    /// Disable a plugin
    pub fn disable_plugin(&self, plugin_id: &str) -> Result<()> {
        if let Some(plugin) = self.plugins.read().unwrap().get(plugin_id) {
            let mut plugin_write = plugin.write().unwrap();
            plugin_write.set_enabled(false);
        }
        
        if let Some(meta) = self.metadata.write().unwrap().get_mut(plugin_id) {
            meta.enabled = false;
        }
        
        Ok(())
    }
    
    /// Check if plugin is enabled
    pub fn is_plugin_enabled(&self, plugin_id: &str) -> bool {
        self.metadata.read().unwrap()
            .get(plugin_id)
            .map(|m| m.enabled)
            .unwrap_or(false)
    }
    
    /// Get plugin context
    pub fn context(&self) -> &Arc<PluginContext> {
        &self.context
    }
    
    /// Get plugin directory
    pub fn plugin_dir(&self) -> &Path {
        &self.plugin_dir
    }
    
    /// Get registry statistics
    pub fn stats(&self) -> PluginStats {
        let metadata = self.metadata.read().unwrap();
        
        let total = metadata.len();
        let loaded = metadata.values().filter(|m| m.loaded).count();
        let enabled = metadata.values().filter(|m| m.enabled).count();
        
        PluginStats {
            total_plugins: total,
            loaded_plugins: loaded,
            enabled_plugins: enabled,
            disabled_plugins: total - enabled,
        }
    }
}

/// Plugin statistics
#[derive(Debug, Clone)]
pub struct PluginStats {
    /// Total number of registered plugins
    pub total_plugins: usize,
    /// Number of loaded plugins
    pub loaded_plugins: usize,
    /// Number of enabled plugins
    pub enabled_plugins: usize,
    /// Number of disabled plugins
    pub disabled_plugins: usize,
}

/// Plugin builder for fluent plugin registration
pub struct PluginBuilder {
    id: String,
    name: String,
    version: String,
    description: String,
    author: String,
    dependencies: Vec<String>,
}

impl PluginBuilder {
    /// Create new plugin builder
    pub fn new(id: impl Into<String>, name: impl Into<String>, version: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            version: version.into(),
            description: String::new(),
            author: String::new(),
            dependencies: Vec::new(),
        }
    }
    
    /// Set description
    pub fn description(mut self, desc: impl Into<String>) -> Self {
        self.description = desc.into();
        self
    }
    
    /// Set author
    pub fn author(mut self, author: impl Into<String>) -> Self {
        self.author = author.into();
        self
    }
    
    /// Add dependency
    pub fn with_dependency(mut self, dep: impl Into<String>) -> Self {
        self.dependencies.push(dep.into());
        self
    }
    
    /// Add multiple dependencies
    pub fn with_dependencies(mut self, deps: Vec<String>) -> Self {
        self.dependencies.extend(deps);
        self
    }
    
    /// Build plugin metadata
    pub fn build(self) -> PluginMetadata {
        PluginMetadata {
            id: self.id,
            name: self.name,
            version: self.version,
            description: self.description,
            author: self.author,
            dependencies: self.dependencies,
            path: None,
            loaded: false,
            enabled: true,
        }
    }
}

/// Macro for implementing Plugin trait
#[macro_export]
macro_rules! impl_plugin {
    ($name:ident, $id:expr, $name_str:expr, $version:expr, $desc:expr) => {
        impl $crate::core::plugin::Plugin for $name {
            fn plugin_id(&self) -> &'static str {
                $id
            }
            
            fn name(&self) -> &'static str {
                $name_str
            }
            
            fn version(&self) -> &'static str {
                $version
            }
            
            fn description(&self) -> &'static str {
                $desc
            }
        }
    };
}

/// Example plugin for testing
#[cfg(test)]
pub struct ExamplePlugin {
    enabled: bool,
    initialized: bool,
}

#[cfg(test)]
impl ExamplePlugin {
    pub fn new() -> Self {
        Self {
            enabled: true,
            initialized: false,
        }
    }
}

#[cfg(test)]
impl Plugin for ExamplePlugin {
    fn plugin_id(&self) -> &'static str {
        "example"
    }
    
    fn name(&self) -> &'static str {
        "Example Plugin"
    }
    
    fn version(&self) -> &'static str {
        "1.0.0"
    }
    
    fn description(&self) -> &'static str {
        "An example plugin for testing"
    }
    
    fn initialize(&mut self, _ctx: &PluginContext) -> Result<()> {
        self.initialized = true;
        Ok(())
    }
    
    fn shutdown(&mut self) -> Result<()> {
        self.initialized = false;
        Ok(())
    }
    
    fn is_enabled(&self) -> bool {
        self.enabled
    }
    
    fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    
    #[test]
    fn test_plugin_registry() {
        let temp_dir = std::env::temp_dir().join("plugin_test");
        let registry = PluginRegistry::new(&temp_dir).unwrap();
        
        let plugin = ExamplePlugin::new();
        registry.register(plugin).unwrap();
        
        assert!(registry.is_plugin_enabled("example"));
        assert_eq!(registry.stats().total_plugins, 1);
        
        registry.disable_plugin("example").unwrap();
        assert!(!registry.is_plugin_enabled("example"));
        
        registry.unregister("example").unwrap();
        assert_eq!(registry.stats().total_plugins, 0);
        
        let _ = std::fs::remove_dir_all(&temp_dir);
    }
    
    #[test]
    fn test_plugin_builder() {
        let metadata = PluginBuilder::new("test", "Test Plugin", "1.0.0")
            .description("A test plugin")
            .author("Test Author")
            .with_dependency("base")
            .build();
        
        assert_eq!(metadata.id, "test");
        assert_eq!(metadata.name, "Test Plugin");
        assert_eq!(metadata.version, "1.0.0");
        assert_eq!(metadata.description, "A test plugin");
        assert_eq!(metadata.author, "Test Author");
        assert_eq!(metadata.dependencies, vec!["base"]);
    }
    
    #[test]
    fn test_plugin_context() {
        let ctx = PluginContext::new(
            PathBuf::from("/plugins"),
            PathBuf::from("/config"),
            PathBuf::from("/cache"),
        );
        
        ctx.set("test_key", 42i32);
        
        let exists = ctx.get::<i32>("test_key");
        assert!(exists);
        
        let value = ctx.get_value::<i32>("test_key");
        assert_eq!(value, Some(42));
        
        ctx.remove("test_key");
        assert!(!ctx.get::<i32>("test_key"));
    }
}
