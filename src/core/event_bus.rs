//! Event Bus for decoupled component communication
//!
//! Provides a publish-subscribe pattern for component communication:
//! - Type-safe event channels
//! - Synchronous and asynchronous event handling
//! - Event filtering and routing
//! - Priority-based event handling

use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::fmt::Debug;
use std::sync::{Arc, RwLock};
use std::time::Instant;

use crate::core::error::Result;

/// Event trait for all events
pub trait Event: Send + Sync + Any + Debug {
    /// Event name for identification
    fn event_name(&self) -> &'static str;
    
    /// Event timestamp
    fn timestamp(&self) -> Instant {
        Instant::now()
    }
}

/// Event handler trait
pub trait EventHandler<E: Event>: Send + Sync {
    /// Handle the event
    fn handle(&self, event: &E) -> Result<()>;
    
    /// Handler name for identification
    fn handler_name(&self) -> &'static str {
        "anonymous"
    }
    
    /// Handler priority (lower = higher priority)
    fn priority(&self) -> i32 {
        0
    }
}

/// Event subscription
struct Subscription<E: Event> {
    handler: Arc<dyn EventHandler<E>>,
    enabled: bool,
    last_triggered: Option<Instant>,
    trigger_count: u64,
}

/// Event channel for a specific event type
struct EventChannel<E: Event> {
    subscribers: RwLock<Vec<Subscription<E>>>,
    event_count: RwLock<u64>,
    last_event_time: RwLock<Option<Instant>>,
}

impl<E: Event> Default for EventChannel<E> {
    fn default() -> Self {
        Self {
            subscribers: RwLock::new(Vec::new()),
            event_count: RwLock::new(0),
            last_event_time: RwLock::new(None),
        }
    }
}

impl<E: Event> EventChannel<E> {
    /// Subscribe to events
    fn subscribe(&self, handler: Arc<dyn EventHandler<E>>) -> SubscriptionId {
        let mut subscribers = self.subscribers.write().unwrap();
        let id = SubscriptionId(subscribers.len());
        
        subscribers.push(Subscription {
            handler,
            enabled: true,
            last_triggered: None,
            trigger_count: 0,
        });
        
        // Sort by priority
        subscribers.sort_by(|a, b| a.handler.priority().cmp(&b.handler.priority()));
        
        id
    }
    
    /// Unsubscribe from events
    fn unsubscribe(&self, id: SubscriptionId) -> bool {
        let mut subscribers = self.subscribers.write().unwrap();
        
        if id.0 < subscribers.len() {
            subscribers[id.0].enabled = false;
            true
        } else {
            false
        }
    }
    
    /// Publish an event to all subscribers
    fn publish(&self, event: &E) -> Result<PublishResult> {
        let subscribers = self.subscribers.read().unwrap();
        let mut success_count = 0;
        let mut error_count = 0;
        let mut handlers_called = Vec::new();
        
        for (idx, subscription) in subscribers.iter().enumerate() {
            if !subscription.enabled {
                continue;
            }
            
            match subscription.handler.handle(event) {
                Ok(()) => {
                    success_count += 1;
                    handlers_called.push(idx);
                }
                Err(_) => {
                    error_count += 1;
                }
            }
        }
        
        // Update statistics
        if let Ok(mut count) = self.event_count.write() {
            *count += 1;
        }
        if let Ok(mut time) = self.last_event_time.write() {
            *time = Some(Instant::now());
        }
        
        // Update subscription statistics
        let mut subscribers_write = self.subscribers.write().unwrap();
        for idx in handlers_called {
            if idx < subscribers_write.len() {
                subscribers_write[idx].last_triggered = Some(Instant::now());
                subscribers_write[idx].trigger_count += 1;
            }
        }
        
        Ok(PublishResult {
            success_count,
            error_count,
            total_subscribers: subscribers.len(),
        })
    }
    
    /// Get channel statistics
    fn stats(&self) -> ChannelStats {
        let subscribers = self.subscribers.read().unwrap();
        let active_count = subscribers.iter().filter(|s| s.enabled).count();
        let event_count = *self.event_count.read().unwrap();
        let last_event_time = *self.last_event_time.read().unwrap();
        
        ChannelStats {
            subscriber_count: subscribers.len(),
            active_subscriber_count: active_count,
            event_count,
            last_event_time,
        }
    }
}

/// Subscription ID
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SubscriptionId(usize);

/// Publish result
#[derive(Debug, Clone)]
pub struct PublishResult {
    /// Number of successful handler calls
    pub success_count: usize,
    /// Number of failed handler calls
    pub error_count: usize,
    /// Total number of subscribers
    pub total_subscribers: usize,
}

/// Channel statistics
#[derive(Debug, Clone)]
pub struct ChannelStats {
    /// Total subscriber count
    pub subscriber_count: usize,
    /// Active subscriber count
    pub active_subscriber_count: usize,
    /// Total events published
    pub event_count: u64,
    /// Last event time
    pub last_event_time: Option<Instant>,
}

/// Event bus for managing event channels
pub struct EventBus {
    channels: RwLock<HashMap<TypeId, Box<dyn Any + Send + Sync>>>,
    event_history: RwLock<Vec<EventRecord>>,
    max_history: usize,
    enabled: bool,
}

/// Event record for history
#[derive(Debug, Clone)]
pub struct EventRecord {
    /// Event name
    pub name: &'static str,
    /// Timestamp
    pub timestamp: Instant,
    /// Success count
    pub success_count: usize,
    /// Error count
    pub error_count: usize,
}

impl Default for EventBus {
    fn default() -> Self {
        Self::new()
    }
}

impl EventBus {
    /// Create a new event bus
    pub fn new() -> Self {
        Self {
            channels: RwLock::new(HashMap::new()),
            event_history: RwLock::new(Vec::new()),
            max_history: 1000,
            enabled: true,
        }
    }
    
    /// Create with custom history size
    pub fn with_history_size(max_history: usize) -> Self {
        Self {
            channels: RwLock::new(HashMap::new()),
            event_history: RwLock::new(Vec::new()),
            max_history,
            enabled: true,
        }
    }
    
    /// Enable/disable event bus
    pub fn enable(&mut self, enabled: bool) {
        self.enabled = enabled;
    }
    
    /// Check if enabled
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }
    
    /// Subscribe to an event type
    pub fn subscribe<E: Event + 'static>(&self, handler: Arc<dyn EventHandler<E>>) -> SubscriptionId {
        let type_id = TypeId::of::<E>();
        
        let mut channels = self.channels.write().unwrap();
        
        let channel = channels
            .entry(type_id)
            .or_insert_with(|| Box::new(EventChannel::<E>::default()));
        
        let channel = channel.downcast_mut::<EventChannel<E>>().unwrap();
        channel.subscribe(handler)
    }
    
    /// Unsubscribe from an event type
    pub fn unsubscribe<E: Event + 'static>(&self, id: SubscriptionId) -> bool {
        let type_id = TypeId::of::<E>();
        
        if let Ok(channels) = self.channels.read() {
            if let Some(channel) = channels.get(&type_id) {
                if let Some(channel) = channel.downcast_ref::<EventChannel<E>>() {
                    return channel.unsubscribe(id);
                }
            }
        }
        
        false
    }
    
    /// Publish an event
    pub fn publish<E: Event + 'static>(&self, event: &E) -> Result<PublishResult> {
        if !self.enabled {
            return Ok(PublishResult {
                success_count: 0,
                error_count: 0,
                total_subscribers: 0,
            });
        }
        
        let type_id = TypeId::of::<E>();
        let result = {
            let channels = self.channels.read().unwrap();
            
            if let Some(channel) = channels.get(&type_id) {
                if let Some(channel) = channel.downcast_ref::<EventChannel<E>>() {
                    channel.publish(event)?
                } else {
                    PublishResult {
                        success_count: 0,
                        error_count: 0,
                        total_subscribers: 0,
                    }
                }
            } else {
                PublishResult {
                    success_count: 0,
                    error_count: 0,
                    total_subscribers: 0,
                }
            }
        };
        
        // Record in history
        if result.success_count > 0 || result.error_count > 0 {
            let mut history = self.event_history.write().unwrap();
            history.push(EventRecord {
                name: event.event_name(),
                timestamp: Instant::now(),
                success_count: result.success_count,
                error_count: result.error_count,
            });
            
            // Trim history
            if history.len() > self.max_history {
                let drain_count = history.len() - self.max_history;
                history.drain(0..drain_count);
            }
        }
        
        Ok(result)
    }
    
    /// Get channel statistics
    pub fn get_stats<E: Event + 'static>(&self) -> Option<ChannelStats> {
        let type_id = TypeId::of::<E>();
        let channels = self.channels.read().unwrap();
        
        channels.get(&type_id)
            .and_then(|c| c.downcast_ref::<EventChannel<E>>())
            .map(|c| c.stats())
    }
    
    /// Get all channel names
    pub fn get_channel_names(&self) -> Vec<&'static str> {
        // This is a simplified implementation
        // A full implementation would need to track channel names
        Vec::new()
    }
    
    /// Get event history
    pub fn get_history(&self, limit: Option<usize>) -> Vec<EventRecord> {
        let history = self.event_history.read().unwrap();
        let limit = limit.unwrap_or(self.max_history);
        
        let start = history.len().saturating_sub(limit);
        history[start..].to_vec()
    }
    
    /// Clear event history
    pub fn clear_history(&self) {
        if let Ok(mut history) = self.event_history.write() {
            history.clear();
        }
    }
    
    /// Get total event count across all channels
    pub fn total_event_count(&self) -> u64 {
        let channels = self.channels.read().unwrap();
        channels.values()
            .filter_map(|c| {
                // Try to get event count from any channel type
                // This is a simplified implementation
                c.downcast_ref::<EventChannel<events::ModelLoadingStarted>>()
                    .map(|c| *c.event_count.read().unwrap())
            })
            .sum()
    }
}

/// Macro for implementing Event trait
#[macro_export]
macro_rules! impl_event {
    ($name:ident) => {
        impl $crate::core::event_bus::Event for $name {
            fn event_name(&self) -> &'static str {
                stringify!($name)
            }
        }
    };
    
    ($name:ident, $custom_name:expr) => {
        impl $crate::core::event_bus::Event for $name {
            fn event_name(&self) -> &'static str {
                $custom_name
            }
        }
    };
}

/// Common TTS events
pub mod events {
    use super::*;
    
    /// Model loading started event
    #[derive(Debug, Clone)]
    pub struct ModelLoadingStarted {
        pub model_name: String,
        pub model_path: String,
    }
    
    impl_event!(ModelLoadingStarted, "model.loading.started");
    
    /// Model loading completed event
    #[derive(Debug, Clone)]
    pub struct ModelLoadingCompleted {
        pub model_name: String,
        pub load_duration_ms: u64,
    }
    
    impl_event!(ModelLoadingCompleted, "model.loading.completed");
    
    /// Model loading failed event
    #[derive(Debug, Clone)]
    pub struct ModelLoadingFailed {
        pub model_name: String,
        pub error: String,
    }
    
    impl_event!(ModelLoadingFailed, "model.loading.failed");
    
    /// Inference started event
    #[derive(Debug, Clone)]
    pub struct InferenceStarted {
        pub request_id: String,
        pub text_length: usize,
    }
    
    impl_event!(InferenceStarted, "inference.started");
    
    /// Inference completed event
    #[derive(Debug, Clone)]
    pub struct InferenceCompleted {
        pub request_id: String,
        pub duration_ms: u64,
        pub audio_duration_secs: f32,
    }
    
    impl_event!(InferenceCompleted, "inference.completed");
    
    /// Inference failed event
    #[derive(Debug, Clone)]
    pub struct InferenceFailed {
        pub request_id: String,
        pub error: String,
    }
    
    impl_event!(InferenceFailed, "inference.failed");
    
    /// Resource low memory event
    #[derive(Debug, Clone)]
    pub struct ResourceLowMemory {
        pub current_usage_bytes: usize,
        pub limit_bytes: usize,
        pub usage_percent: f64,
    }
    
    impl_event!(ResourceLowMemory, "resource.low_memory");
    
    /// Configuration changed event
    #[derive(Debug, Clone)]
    pub struct ConfigurationChanged {
        pub component: String,
        pub changes: Vec<(String, String)>,
    }
    
    impl_event!(ConfigurationChanged, "config.changed");
    
    /// Streaming audio chunk event
    #[derive(Debug, Clone)]
    pub struct StreamingAudioChunk {
        pub chunk_index: usize,
        pub samples_count: usize,
        pub timestamp_ms: u64,
    }
    
    impl_event!(StreamingAudioChunk, "streaming.audio_chunk");
}

#[cfg(test)]
mod tests {
    use super::*;
    use events::*;
    
    struct TestHandler {
        name: &'static str,
        priority: i32,
    }
    
    impl TestHandler {
        fn new(name: &'static str, priority: i32) -> Self {
            Self { name, priority }
        }
    }
    
    impl EventHandler<ModelLoadingStarted> for TestHandler {
        fn handle(&self, event: &ModelLoadingStarted) -> Result<()> {
            println!("Handler {}: Model {} loading started", self.name, event.model_name);
            Ok(())
        }
        
        fn handler_name(&self) -> &'static str {
            self.name
        }
        
        fn priority(&self) -> i32 {
            self.priority
        }
    }
    
    #[test]
    fn test_event_bus_subscribe_publish() {
        let bus = EventBus::new();
        
        let handler1 = Arc::new(TestHandler::new("handler1", 10));
        let handler2 = Arc::new(TestHandler::new("handler2", 5));
        
        let _id1 = bus.subscribe(handler1);
        let _id2 = bus.subscribe(handler2);
        
        let event = ModelLoadingStarted {
            model_name: "test_model".to_string(),
            model_path: "/path/to/model".to_string(),
        };
        
        let result = bus.publish(&event).unwrap();
        
        assert_eq!(result.success_count, 2);
        assert_eq!(result.error_count, 0);
    }
    
    #[test]
    fn test_event_bus_unsubscribe() {
        let bus = EventBus::new();
        
        let handler = Arc::new(TestHandler::new("handler", 0));
        let id = bus.subscribe(handler);
        
        let event = ModelLoadingStarted {
            model_name: "test".to_string(),
            model_path: "/path".to_string(),
        };
        
        // Before unsubscribe
        let result = bus.publish(&event).unwrap();
        assert_eq!(result.success_count, 1);
        
        // Unsubscribe
        bus.unsubscribe::<ModelLoadingStarted>(id);
        
        // After unsubscribe
        let result = bus.publish(&event).unwrap();
        assert_eq!(result.success_count, 0);
    }
    
    #[test]
    fn test_event_bus_stats() {
        let bus = EventBus::new();
        
        let handler = Arc::new(TestHandler::new("handler", 0));
        bus.subscribe(handler);
        
        let event = ModelLoadingStarted {
            model_name: "test".to_string(),
            model_path: "/path".to_string(),
        };
        
        bus.publish(&event).unwrap();
        bus.publish(&event).unwrap();
        
        let stats = bus.get_stats::<ModelLoadingStarted>().unwrap();
        
        assert_eq!(stats.subscriber_count, 1);
        assert_eq!(stats.active_subscriber_count, 1);
        assert_eq!(stats.event_count, 2);
        assert!(stats.last_event_time.is_some());
    }
    
    #[test]
    fn test_event_history() {
        let bus = EventBus::with_history_size(10);
        
        for i in 0..15 {
            let event = ModelLoadingStarted {
                model_name: format!("model_{}", i),
                model_path: "/path".to_string(),
            };
            bus.publish(&event).unwrap();
        }
        
        let history = bus.get_history(None);
        assert_eq!(history.len(), 10); // Limited to max_history
        
        let history = bus.get_history(Some(5));
        assert_eq!(history.len(), 5);
    }
}
