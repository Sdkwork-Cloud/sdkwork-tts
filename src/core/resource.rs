//! Resource management for IndexTTS2
//!
//! Provides unified resource lifecycle management including:
//! - Model weight loading/unloading
//! - Memory pool management
//! - GPU memory tracking
//! - File handle pooling

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, Weak};
use std::time::{Duration, Instant};

use candle_core::Device;

use super::error::{Result, TtsError, ResourceType};

/// Resource handle for tracked resources
#[derive(Debug, Clone)]
pub struct ResourceHandle {
    /// Resource ID
    pub id: ResourceId,
    /// Resource type
    pub resource_type: ResourceType,
    /// Resource name
    pub name: String,
    /// Reference count
    ref_count: Arc<Mutex<usize>>,
    /// Last access time
    last_accessed: Arc<Mutex<Instant>>,
}

/// Resource ID type
pub type ResourceId = u64;

impl ResourceHandle {
    /// Create a new resource handle
    pub fn new(id: ResourceId, resource_type: ResourceType, name: impl Into<String>) -> Self {
        Self {
            id,
            resource_type,
            name: name.into(),
            ref_count: Arc::new(Mutex::new(1)),
            last_accessed: Arc::new(Mutex::new(Instant::now())),
        }
    }

    /// Increment reference count
    pub fn acquire(&self) {
        if let Ok(mut count) = self.ref_count.lock() {
            *count += 1;
        }
        if let Ok(mut time) = self.last_accessed.lock() {
            *time = Instant::now();
        }
    }

    /// Decrement reference count
    pub fn release(&self) -> usize {
        let count = if let Ok(mut count) = self.ref_count.lock() {
            *count = count.saturating_sub(1);
            *count
        } else {
            0
        };
        count
    }

    /// Get current reference count
    pub fn ref_count(&self) -> usize {
        self.ref_count.lock().map(|c| *c).unwrap_or(0)
    }

    /// Get last access time
    pub fn last_accessed(&self) -> Instant {
        self.last_accessed.lock().map(|t| *t).unwrap_or_else(|_| Instant::now())
    }

    /// Check if resource is idle (no references)
    pub fn is_idle(&self) -> bool {
        self.ref_count() == 0
    }

    /// Check if resource has been idle for longer than duration
    pub fn is_idle_for(&self, duration: Duration) -> bool {
        self.is_idle() && self.last_accessed().elapsed() > duration
    }
}

/// Resource manager for centralized resource lifecycle management
pub struct ResourceManager {
    /// Device for tensor operations
    device: Device,
    /// Tracked resources
    resources: Arc<Mutex<HashMap<ResourceId, ResourceEntry>>>,
    /// Resource ID counter
    next_id: Arc<Mutex<u64>>,
    /// Memory limit in bytes (0 = unlimited)
    memory_limit: usize,
    /// Current memory usage
    current_memory: Arc<Mutex<usize>>,
    /// GPU memory limit (if applicable)
    gpu_memory_limit: Option<usize>,
    /// Current GPU memory usage
    current_gpu_memory: Arc<Mutex<usize>>,
    /// Idle timeout for automatic unloading
    idle_timeout: Duration,
    /// Last cleanup time
    last_cleanup: Arc<Mutex<Instant>>,
}

/// Resource entry in the manager
#[derive(Debug)]
struct ResourceEntry {
    handle: Weak<ResourceHandle>,
    metadata: ResourceMetadata,
    state: ResourceState,
}

/// Resource metadata
#[derive(Debug, Clone)]
pub struct ResourceMetadata {
    /// Resource type
    pub resource_type: ResourceType,
    /// Resource name
    pub name: String,
    /// File path (if applicable)
    pub path: Option<PathBuf>,
    /// Memory size in bytes
    pub memory_size: usize,
    /// GPU memory size in bytes
    pub gpu_memory_size: usize,
    /// Creation time
    pub created_at: Instant,
    /// Load time (if loaded)
    pub loaded_at: Option<Instant>,
    /// Load duration
    pub load_duration: Option<Duration>,
}

/// Resource state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResourceState {
    /// Resource is registered but not loaded
    Registered,
    /// Resource is currently loading
    Loading,
    /// Resource is loaded and ready
    Loaded,
    /// Resource is unloaded but cached
    Unloaded,
    /// Resource has been evicted
    Evicted,
    /// Resource has failed to load
    Failed,
}

impl ResourceManager {
    /// Create a new resource manager
    pub fn new(device: Device) -> Self {
        Self {
            device,
            resources: Arc::new(Mutex::new(HashMap::new())),
            next_id: Arc::new(Mutex::new(1)),
            memory_limit: 0,
            current_memory: Arc::new(Mutex::new(0)),
            gpu_memory_limit: None,
            current_gpu_memory: Arc::new(Mutex::new(0)),
            idle_timeout: Duration::from_secs(300), // 5 minutes
            last_cleanup: Arc::new(Mutex::new(Instant::now())),
        }
    }

    /// Set memory limit
    pub fn with_memory_limit(mut self, limit_bytes: usize) -> Self {
        self.memory_limit = limit_bytes;
        self
    }

    /// Set GPU memory limit
    pub fn with_gpu_memory_limit(mut self, limit_bytes: usize) -> Self {
        self.gpu_memory_limit = Some(limit_bytes);
        self
    }

    /// Set idle timeout
    pub fn with_idle_timeout(mut self, timeout: Duration) -> Self {
        self.idle_timeout = timeout;
        self
    }

    /// Register a new resource
    pub fn register(
        &self,
        resource_type: ResourceType,
        name: impl Into<String>,
        path: Option<PathBuf>,
        memory_size: usize,
    ) -> Result<Arc<ResourceHandle>> {
        let id = self.allocate_id();
        let name = name.into();
        
        let handle = Arc::new(ResourceHandle::new(id, resource_type, &name));
        
        let metadata = ResourceMetadata {
            resource_type,
            name,
            path,
            memory_size,
            gpu_memory_size: 0,
            created_at: Instant::now(),
            loaded_at: None,
            load_duration: None,
        };

        let entry = ResourceEntry {
            handle: Arc::downgrade(&handle),
            metadata,
            state: ResourceState::Registered,
        };

        if let Ok(mut resources) = self.resources.lock() {
            resources.insert(id, entry);
        }

        Ok(handle)
    }

    /// Mark resource as loaded
    pub fn mark_loaded(&self, handle: &ResourceHandle, load_duration: Duration) -> Result<()> {
        if let Ok(mut resources) = self.resources.lock() {
            if let Some(entry) = resources.get_mut(&handle.id) {
                entry.state = ResourceState::Loaded;
                entry.metadata.loaded_at = Some(Instant::now());
                entry.metadata.load_duration = Some(load_duration);
                
                // Update memory tracking
                if let Ok(mut mem) = self.current_memory.lock() {
                    *mem += entry.metadata.memory_size;
                }
                if let Ok(mut gpu_mem) = self.current_gpu_memory.lock() {
                    *gpu_mem += entry.metadata.gpu_memory_size;
                }
            }
        }
        Ok(())
    }

    /// Mark resource as failed
    pub fn mark_failed(&self, handle: &ResourceHandle, error: &str) -> Result<()> {
        if let Ok(mut resources) = self.resources.lock() {
            if let Some(entry) = resources.get_mut(&handle.id) {
                entry.state = ResourceState::Failed;
                tracing::error!(
                    "Resource '{}' failed to load: {}",
                    entry.metadata.name,
                    error
                );
            }
        }
        Ok(())
    }

    /// Unload a resource to free memory
    pub fn unload(&self, handle: &ResourceHandle) -> Result<()> {
        if handle.ref_count() > 0 {
            return Err(TtsError::Resource {
                message: format!(
                    "Cannot unload resource '{}' with {} active references",
                    handle.name,
                    handle.ref_count()
                ),
                resource_type: handle.resource_type,
            });
        }

        if let Ok(mut resources) = self.resources.lock() {
            if let Some(entry) = resources.get_mut(&handle.id) {
                if entry.state == ResourceState::Loaded {
                    entry.state = ResourceState::Unloaded;
                    
                    // Update memory tracking
                    if let Ok(mut mem) = self.current_memory.lock() {
                        *mem = mem.saturating_sub(entry.metadata.memory_size);
                    }
                    if let Ok(mut gpu_mem) = self.current_gpu_memory.lock() {
                        *gpu_mem = gpu_mem.saturating_sub(entry.metadata.gpu_memory_size);
                    }

                    tracing::info!(
                        "Unloaded resource '{}' (freed {} bytes)",
                        entry.metadata.name,
                        entry.metadata.memory_size
                    );
                }
            }
        }
        Ok(())
    }

    /// Get resource metadata
    pub fn get_metadata(&self, handle: &ResourceHandle) -> Option<ResourceMetadata> {
        if let Ok(resources) = self.resources.lock() {
            resources.get(&handle.id).map(|e| e.metadata.clone())
        } else {
            None
        }
    }

    /// Get resource state
    pub fn get_state(&self, handle: &ResourceHandle) -> Option<ResourceState> {
        if let Ok(resources) = self.resources.lock() {
            resources.get(&handle.id).map(|e| e.state)
        } else {
            None
        }
    }

    /// Check if resource is loaded
    pub fn is_loaded(&self, handle: &ResourceHandle) -> bool {
        self.get_state(handle) == Some(ResourceState::Loaded)
    }

    /// Get current memory usage
    pub fn current_memory_usage(&self) -> usize {
        self.current_memory.lock().map(|m| *m).unwrap_or(0)
    }

    /// Get current GPU memory usage
    pub fn current_gpu_memory_usage(&self) -> usize {
        self.current_gpu_memory.lock().map(|m| *m).unwrap_or(0)
    }

    /// Get memory limit
    pub fn memory_limit(&self) -> usize {
        self.memory_limit
    }

    /// Check if memory limit is exceeded
    pub fn is_memory_limit_exceeded(&self) -> bool {
        if self.memory_limit == 0 {
            return false;
        }
        self.current_memory_usage() > self.memory_limit
    }

    /// Clean up idle resources
    pub fn cleanup_idle(&self) -> Result<usize> {
        let mut unloaded = 0;
        
        if let Ok(mut resources) = self.resources.lock() {
            let to_unload: Vec<ResourceId> = resources
                .iter()
                .filter(|(_, entry)| {
                    if let Some(handle) = entry.handle.upgrade() {
                        handle.is_idle_for(self.idle_timeout) && entry.state == ResourceState::Loaded
                    } else {
                        true // Weak reference expired
                    }
                })
                .map(|(id, _)| *id)
                .collect();

            for id in to_unload {
                if let Some(entry) = resources.get_mut(&id) {
                    if entry.state == ResourceState::Loaded {
                        entry.state = ResourceState::Unloaded;
                        
                        if let Ok(mut mem) = self.current_memory.lock() {
                            *mem = mem.saturating_sub(entry.metadata.memory_size);
                        }
                        if let Ok(mut gpu_mem) = self.current_gpu_memory.lock() {
                            *gpu_mem = gpu_mem.saturating_sub(entry.metadata.gpu_memory_size);
                        }
                        
                        unloaded += 1;
                        tracing::debug!(
                            "Auto-unloaded idle resource '{}'",
                            entry.metadata.name
                        );
                    }
                }
            }
        }

        if let Ok(mut last) = self.last_cleanup.lock() {
            *last = Instant::now();
        }

        Ok(unloaded)
    }

    /// Get all resource statistics
    pub fn get_statistics(&self) -> ResourceStatistics {
        let mut stats = ResourceStatistics::default();
        
        if let Ok(resources) = self.resources.lock() {
            for (_, entry) in resources.iter() {
                stats.total_resources += 1;
                
                match entry.state {
                    ResourceState::Registered => stats.registered += 1,
                    ResourceState::Loading => stats.loading += 1,
                    ResourceState::Loaded => {
                        stats.loaded += 1;
                        stats.total_memory += entry.metadata.memory_size;
                        stats.total_gpu_memory += entry.metadata.gpu_memory_size;
                    }
                    ResourceState::Unloaded => stats.unloaded += 1,
                    ResourceState::Evicted => stats.evicted += 1,
                    ResourceState::Failed => stats.failed += 1,
                }
            }
        }

        stats.current_memory = self.current_memory_usage();
        stats.current_gpu_memory = self.current_gpu_memory_usage();
        stats.memory_limit = self.memory_limit;
        stats.gpu_memory_limit = self.gpu_memory_limit.unwrap_or(0);
        
        stats
    }

    /// Allocate a new resource ID
    fn allocate_id(&self) -> ResourceId {
        if let Ok(mut id) = self.next_id.lock() {
            let current = *id;
            *id += 1;
            current
        } else {
            0
        }
    }
}

/// Resource statistics
#[derive(Debug, Clone, Default)]
pub struct ResourceStatistics {
    /// Total number of resources
    pub total_resources: usize,
    /// Number of registered resources
    pub registered: usize,
    /// Number of loading resources
    pub loading: usize,
    /// Number of loaded resources
    pub loaded: usize,
    /// Number of unloaded resources
    pub unloaded: usize,
    /// Number of evicted resources
    pub evicted: usize,
    /// Number of failed resources
    pub failed: usize,
    /// Current memory usage
    pub current_memory: usize,
    /// Total memory of loaded resources
    pub total_memory: usize,
    /// Current GPU memory usage
    pub current_gpu_memory: usize,
    /// Total GPU memory of loaded resources
    pub total_gpu_memory: usize,
    /// Memory limit
    pub memory_limit: usize,
    /// GPU memory limit
    pub gpu_memory_limit: usize,
}

impl ResourceStatistics {
    /// Get memory usage percentage
    pub fn memory_usage_percent(&self) -> f64 {
        if self.memory_limit == 0 {
            0.0
        } else {
            (self.current_memory as f64 / self.memory_limit as f64) * 100.0
        }
    }

    /// Get GPU memory usage percentage
    pub fn gpu_memory_usage_percent(&self) -> f64 {
        if self.gpu_memory_limit == 0 {
            0.0
        } else {
            (self.current_gpu_memory as f64 / self.gpu_memory_limit as f64) * 100.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resource_handle() {
        let handle = ResourceHandle::new(1, ResourceType::Model, "test_model");
        
        assert_eq!(handle.id, 1);
        assert_eq!(handle.name, "test_model");
        assert_eq!(handle.ref_count(), 1);
        assert!(!handle.is_idle());
        
        handle.acquire();
        assert_eq!(handle.ref_count(), 2);
        
        handle.release();
        assert_eq!(handle.ref_count(), 1);
        
        handle.release();
        assert_eq!(handle.ref_count(), 0);
        assert!(handle.is_idle());
    }

    #[test]
    fn test_resource_manager() {
        let manager = ResourceManager::new(Device::Cpu);
        
        let handle = manager.register(
            ResourceType::Model,
            "gpt_model",
            Some(PathBuf::from("model.safetensors")),
            1024 * 1024 * 100, // 100 MB
        ).unwrap();
        
        assert_eq!(handle.name, "gpt_model");
        assert!(!manager.is_loaded(&handle));
        
        // Mark as loaded
        manager.mark_loaded(&handle, Duration::from_secs(1)).unwrap();
        assert!(manager.is_loaded(&handle));
        assert_eq!(manager.current_memory_usage(), 1024 * 1024 * 100);
        
        // Unload
        handle.release(); // Release initial reference
        manager.unload(&handle).unwrap();
        assert!(!manager.is_loaded(&handle));
        assert_eq!(manager.current_memory_usage(), 0);
    }

    #[test]
    fn test_resource_statistics() {
        let stats = ResourceStatistics {
            current_memory: 500,
            memory_limit: 1000,
            current_gpu_memory: 250,
            gpu_memory_limit: 500,
            ..Default::default()
        };
        
        assert_eq!(stats.memory_usage_percent(), 50.0);
        assert_eq!(stats.gpu_memory_usage_percent(), 50.0);
    }
}
