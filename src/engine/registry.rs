//! Engine Registry for managing multiple TTS engines
//!
//! The registry provides a central point for registering, discovering,
//! and instantiating TTS engines.

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use crate::core::error::{Result, TtsError};
use super::traits::{TtsEngine, TtsEngineInfo};
use super::config::EngineConfig;

/// Factory function type for creating engines
pub type EngineFactory = Arc<dyn Fn() -> Result<Box<dyn TtsEngine>> + Send + Sync>;

/// Engine registry for managing TTS engines
pub struct EngineRegistry {
    /// Registered engine factories
    factories: RwLock<HashMap<String, EngineFactory>>,
    /// Engine information cache
    info_cache: RwLock<HashMap<String, TtsEngineInfo>>,
    /// Loaded engine instances
    instances: RwLock<HashMap<String, Arc<dyn TtsEngine>>>,
    /// Default engine ID
    default_engine: RwLock<Option<String>>,
}

impl EngineRegistry {
    /// Create a new engine registry
    pub fn new() -> Self {
        Self {
            factories: RwLock::new(HashMap::new()),
            info_cache: RwLock::new(HashMap::new()),
            instances: RwLock::new(HashMap::new()),
            default_engine: RwLock::new(None),
        }
    }

    /// Register an engine factory
    pub fn register<F>(&self, factory: F) -> Result<()>
    where
        F: Fn() -> Result<Box<dyn TtsEngine>> + Send + Sync + 'static,
    {
        let engine = factory()?;
        let info = engine.info().clone();
        let engine_id = info.id.clone();

        self.factories.write().map_err(|_| TtsError::Internal {
            message: "Failed to acquire write lock on factories".to_string(),
            location: Some("EngineRegistry::register".to_string()),
        })?.insert(engine_id.clone(), Arc::new(factory));

        self.info_cache.write().map_err(|_| TtsError::Internal {
            message: "Failed to acquire write lock on info cache".to_string(),
            location: Some("EngineRegistry::register".to_string()),
        })?.insert(engine_id, info);

        Ok(())
    }

    /// Register an engine with lazy initialization
    pub fn register_lazy<F>(&self, id: &str, info: TtsEngineInfo, factory: F) -> Result<()>
    where
        F: Fn() -> Result<Box<dyn TtsEngine>> + Send + Sync + 'static,
    {
        self.factories.write().map_err(|_| TtsError::Internal {
            message: "Failed to acquire write lock on factories".to_string(),
            location: Some("EngineRegistry::register_lazy".to_string()),
        })?.insert(id.to_string(), Arc::new(factory));

        self.info_cache.write().map_err(|_| TtsError::Internal {
            message: "Failed to acquire write lock on info cache".to_string(),
            location: Some("EngineRegistry::register_lazy".to_string()),
        })?.insert(id.to_string(), info);

        Ok(())
    }

    /// Unregister an engine
    pub fn unregister(&self, id: &str) -> Result<()> {
        self.factories.write().map_err(|_| TtsError::Internal {
            message: "Failed to acquire write lock on factories".to_string(),
            location: Some("EngineRegistry::unregister".to_string()),
        })?.remove(id);

        self.info_cache.write().map_err(|_| TtsError::Internal {
            message: "Failed to acquire write lock on info cache".to_string(),
            location: Some("EngineRegistry::unregister".to_string()),
        })?.remove(id);

        self.instances.write().map_err(|_| TtsError::Internal {
            message: "Failed to acquire write lock on instances".to_string(),
            location: Some("EngineRegistry::unregister".to_string()),
        })?.remove(id);

        Ok(())
    }

    /// Get list of registered engines
    pub fn list_engines(&self) -> Result<Vec<TtsEngineInfo>> {
        let cache = self.info_cache.read().map_err(|_| TtsError::Internal {
            message: "Failed to acquire read lock on info cache".to_string(),
            location: Some("EngineRegistry::list_engines".to_string()),
        })?;

        Ok(cache.values().cloned().collect())
    }

    /// Check if an engine is registered
    pub fn is_registered(&self, id: &str) -> bool {
        self.info_cache.read()
            .map(|cache| cache.contains_key(id))
            .unwrap_or(false)
    }

    /// Get engine info by ID
    pub fn get_info(&self, id: &str) -> Result<Option<TtsEngineInfo>> {
        let cache = self.info_cache.read().map_err(|_| TtsError::Internal {
            message: "Failed to acquire read lock on info cache".to_string(),
            location: Some("EngineRegistry::get_info".to_string()),
        })?;

        Ok(cache.get(id).cloned())
    }

    /// Create or get an engine instance
    pub fn get_engine(&self, id: &str) -> Result<Arc<dyn TtsEngine>> {
        {
            let instances = self.instances.read().map_err(|_| TtsError::Internal {
                message: "Failed to acquire read lock on instances".to_string(),
                location: Some("EngineRegistry::get_engine".to_string()),
            })?;
            
            if let Some(engine) = instances.get(id) {
                return Ok(Arc::clone(engine));
            }
        }

        let engine_id = id.to_string();
        let factory = {
            let factories = self.factories.read().map_err(|_| TtsError::Internal {
                message: "Failed to acquire read lock on factories".to_string(),
                location: Some("EngineRegistry::get_engine".to_string()),
            })?;
            
            Arc::clone(factories.get(&engine_id).ok_or_else(|| TtsError::Config {
                message: format!("Engine '{}' not found", id),
                path: None,
            })?)
        };

        let engine: Box<dyn TtsEngine> = factory()?;
        let engine = Arc::from(engine);

        self.instances.write().map_err(|_| TtsError::Internal {
            message: "Failed to acquire write lock on instances".to_string(),
            location: Some("EngineRegistry::get_engine".to_string()),
        })?.insert(id.to_string(), Arc::clone(&engine));

        Ok(engine)
    }

    /// Initialize an engine with configuration
    pub async fn initialize_engine(&self, id: &str, _config: &EngineConfig) -> Result<()> {
        let _engine = self.get_engine(id)?;
        
        Ok(())
    }

    /// Unload an engine
    pub fn unload_engine(&self, id: &str) -> Result<()> {
        self.instances.write().map_err(|_| TtsError::Internal {
            message: "Failed to acquire write lock on instances".to_string(),
            location: Some("EngineRegistry::unload_engine".to_string()),
        })?.remove(id);

        Ok(())
    }

    /// Set the default engine
    pub fn set_default(&self, id: &str) -> Result<()> {
        if !self.is_registered(id) {
            return Err(TtsError::Config {
                message: format!("Cannot set default: engine '{}' not registered", id),
                path: None,
            });
        }

        let mut default = self.default_engine.write().map_err(|_| TtsError::Internal {
            message: "Failed to acquire write lock on default engine".to_string(),
            location: Some("EngineRegistry::set_default".to_string()),
        })?;

        *default = Some(id.to_string());
        Ok(())
    }

    /// Get the default engine ID
    pub fn get_default_id(&self) -> Option<String> {
        self.default_engine.read()
            .map(|d| d.clone())
            .unwrap_or(None)
    }

    /// Get the default engine
    pub fn get_default_engine(&self) -> Result<Arc<dyn TtsEngine>> {
        let default_id = self.get_default_id().ok_or_else(|| TtsError::Config {
            message: "No default engine set".to_string(),
            path: None,
        })?;

        self.get_engine(&default_id)
    }

    /// Clear all registered engines
    pub fn clear(&self) -> Result<()> {
        self.factories.write().map_err(|_| TtsError::Internal {
            message: "Failed to acquire write lock on factories".to_string(),
            location: Some("EngineRegistry::clear".to_string()),
        })?.clear();

        self.info_cache.write().map_err(|_| TtsError::Internal {
            message: "Failed to acquire write lock on info cache".to_string(),
            location: Some("EngineRegistry::clear".to_string()),
        })?.clear();

        self.instances.write().map_err(|_| TtsError::Internal {
            message: "Failed to acquire write lock on instances".to_string(),
            location: Some("EngineRegistry::clear".to_string()),
        })?.clear();

        Ok(())
    }

    /// Get registry statistics
    pub fn stats(&self) -> RegistryStats {
        let registered = self.info_cache.read()
            .map(|c| c.len())
            .unwrap_or(0);
        let loaded = self.instances.read()
            .map(|i| i.len())
            .unwrap_or(0);

        RegistryStats {
            registered_engines: registered,
            loaded_engines: loaded,
            default_engine: self.get_default_id(),
        }
    }
}

impl Default for EngineRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Registry statistics
#[derive(Debug, Clone)]
pub struct RegistryStats {
    /// Number of registered engines
    pub registered_engines: usize,
    /// Number of loaded engines
    pub loaded_engines: usize,
    /// Default engine ID
    pub default_engine: Option<String>,
}

/// Global engine registry
static REGISTRY: once_cell::sync::Lazy<EngineRegistry> = 
    once_cell::sync::Lazy::new(EngineRegistry::new);

/// Get the global engine registry
pub fn global_registry() -> &'static EngineRegistry {
    &REGISTRY
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_registry_new() {
        let registry = EngineRegistry::new();
        let stats = registry.stats();
        assert_eq!(stats.registered_engines, 0);
        assert_eq!(stats.loaded_engines, 0);
    }

    #[test]
    fn test_registry_list_empty() {
        let registry = EngineRegistry::new();
        let engines = registry.list_engines().unwrap();
        assert!(engines.is_empty());
    }
}
