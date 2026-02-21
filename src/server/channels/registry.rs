//! Channel Registry
//!
//! Manages registered cloud channels

use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use crate::server::channels::traits::{CloudChannel, ChannelEntry};

/// Channel registry
pub struct ChannelRegistry {
    channels: Arc<RwLock<HashMap<String, ChannelEntry>>>,
}

impl ChannelRegistry {
    /// Create new channel registry
    pub fn new() -> Self {
        Self {
            channels: Arc::new(RwLock::new(HashMap::new())),
        }
    }
    
    /// Register a channel
    pub fn register(&self, channel: Box<dyn CloudChannel>, enabled: bool) -> Result<(), String> {
        let name = channel.name().to_string();
        
        let mut channels = self.channels.write().map_err(|e| e.to_string())?;
        
        if channels.contains_key(&name) {
            return Err(format!("Channel {} already registered", name));
        }
        
        channels.insert(name.clone(), ChannelEntry {
            name,
            channel,
            enabled,
        });
        
        Ok(())
    }
    
    /// Unregister a channel
    pub fn unregister(&self, name: &str) -> Result<(), String> {
        let mut channels = self.channels.write().map_err(|e| e.to_string())?;
        
        if !channels.contains_key(name) {
            return Err(format!("Channel {} not found", name));
        }
        
        channels.remove(name);
        
        Ok(())
    }
    
    /// Get a channel by name (returns channel name if exists)
    pub fn has_channel(&self, name: &str) -> bool {
        let channels = self.channels.read().unwrap();
        channels.contains_key(name)
    }
    
    /// List all channels
    pub fn list_channels(&self) -> Vec<String> {
        let channels = self.channels.read().unwrap();
        channels.keys().cloned().collect()
    }
    
    /// Get channel count
    pub fn count(&self) -> usize {
        let channels = self.channels.read().unwrap();
        channels.len()
    }
}

impl Default for ChannelRegistry {
    fn default() -> Self {
        Self::new()
    }
}
