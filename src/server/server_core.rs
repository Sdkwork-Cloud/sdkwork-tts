//! TTS Server Core
//!
//! Main server implementation with Axum web framework

use axum::{
    routing::{get, post, delete},
    Router,
};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tower_http::{
    cors::{Any, CorsLayer},
    trace::TraceLayer,
    request_id::PropagateRequestIdLayer,
};
use tracing::{info, warn};

use crate::server::config::ServerConfig;
use crate::server::types::*;
use crate::server::speaker_lib::SpeakerLibrary;
use crate::server::routes;

/// Server state shared across handlers
pub struct ServerState {
    /// Server configuration
    pub config: ServerConfig,
    /// Speaker library
    pub speaker_lib: SpeakerLibrary,
    /// Start time for uptime calculation
    pub start_time: Instant,
    /// Request counter
    pub request_count: Arc<tokio::sync::RwLock<u64>>,
    /// Local TTS engine (if enabled)
    pub local_engine: Option<Arc<LocalTtsEngine>>,
    /// Cloud channels (if enabled)
    pub cloud_channels: Vec<Arc<dyn Channel>>,
}

/// Local TTS engine wrapper
pub struct LocalTtsEngine {
    /// Engine name
    pub name: String,
    /// Engine instance (type-erased)
    pub engine: Box<dyn TtsEngineTrait + Send + Sync>,
}

/// Synthesis result for local engine
pub struct SynthesisResult {
    /// Audio data (PCM samples)
    pub audio: Vec<f32>,
    /// Sample rate
    pub sample_rate: u32,
    /// Duration in seconds
    pub duration: f32,
}

/// Trait for TTS engines (type erasure)
#[async_trait::async_trait]
pub trait TtsEngineTrait {
    async fn synthesize(&self, request: &SynthesisRequest) -> Result<SynthesisResult, String>;
    async fn list_speakers(&self) -> Result<Vec<SpeakerInfo>, String>;
}

/// Channel trait for cloud providers
#[async_trait::async_trait]
pub trait Channel: Send + Sync {
    fn name(&self) -> &str;
    fn channel_type(&self) -> ChannelType;
    
    async fn synthesize(&self, request: &SynthesisRequest) -> Result<SynthesisResponse, String>;
    async fn list_speakers(&self) -> Result<Vec<SpeakerInfo>, String>;
    async fn get_models(&self) -> Result<Vec<String>, String>;
}

/// Cloud channel types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChannelType {
    Aliyun,
    Openai,
    Volcano,
    Minimax,
    Azure,
    Google,
    AwsPolly,
}

impl ServerState {
    /// Create new server state
    pub fn new(config: ServerConfig) -> Self {
        let speaker_lib = SpeakerLibrary::new(
            &config.speaker_lib.local_path,
            config.speaker_lib.max_cache_size,
        );
        
        // Load speaker library
        if let Err(e) = speaker_lib.load() {
            warn!("Failed to load speaker library: {}", e);
        }
        
        Self {
            start_time: Instant::now(),
            request_count: Arc::new(tokio::sync::RwLock::new(0)),
            local_engine: None,
            cloud_channels: Vec::new(),
            config,
            speaker_lib,
        }
    }
    
    /// Initialize local engine
    pub async fn init_local_engine(&mut self) -> Result<(), String> {
        if !self.config.local.enabled {
            return Ok(());
        }
        
        info!("Initializing local TTS engine...");
        
        // TODO: Initialize actual TTS engine based on config
        // For now, create a placeholder
        
        info!("Local TTS engine initialized");
        Ok(())
    }
    
    /// Initialize cloud channels
    pub async fn init_cloud_channels(&mut self) -> Result<(), String> {
        if !self.config.cloud.enabled {
            return Ok(());
        }
        
        info!("Initializing cloud channels...");
        
        for channel_config in &self.config.cloud.channels {
            match channel_config.channel_type {
                crate::server::config::ChannelTypeConfig::Aliyun => {
                    // TODO: Create Aliyun channel
                    warn!("Aliyun channel not yet implemented");
                }
                crate::server::config::ChannelTypeConfig::Openai => {
                    // TODO: Create OpenAI channel
                    warn!("OpenAI channel not yet implemented");
                }
                crate::server::config::ChannelTypeConfig::Volcano => {
                    // TODO: Create Volcano channel
                    warn!("Volcano channel not yet implemented");
                }
                crate::server::config::ChannelTypeConfig::Minimax => {
                    // TODO: Create Minimax channel
                    warn!("Minimax channel not yet implemented");
                }
                _ => {
                    warn!("Unsupported channel type: {:?}", channel_config.channel_type);
                }
            }
        }
        
        info!("Cloud channels initialized");
        Ok(())
    }
    
    /// Increment request count
    pub async fn increment_request_count(&self) -> u64 {
        let mut count = self.request_count.write().await;
        *count += 1;
        *count
    }
    
    /// Get server uptime
    pub fn uptime(&self) -> Duration {
        self.start_time.elapsed()
    }
    
    /// Get server statistics
    pub async fn get_stats(&self) -> ServerStats {
        let count = *self.request_count.read().await;
        
        ServerStats {
            total_requests: count,
            successful_requests: count, // TODO: Track separately
            failed_requests: 0,
            avg_processing_time_ms: 0.0, // TODO: Track
            active_connections: 0, // TODO: Track
            queue_size: 0,
            memory_usage_mb: 0.0, // TODO: Get actual memory usage
            uptime: self.uptime().as_secs(),
        }
    }
}

/// Create the router with all routes
pub fn create_router(state: Arc<ServerState>) -> Router {
    // CORS layer
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);
    
    // Build router
    Router::new()
        // Health check
        .route("/health", get(routes::health::health_check))
        .route("/api/v1/health", get(routes::health::health_check))
        
        // Server stats
        .route("/api/v1/stats", get(routes::stats::get_stats))
        
        // Synthesis endpoints
        .route("/api/v1/synthesis", post(routes::synthesis::synthesize))
        .route("/api/v1/synthesis/stream", post(routes::synthesis::synthesize_stream))
        
        // Voice design
        .route("/api/v1/voice/design", post(routes::voice::voice_design))
        
        // Voice clone
        .route("/api/v1/voice/clone", post(routes::voice::voice_clone))
        
        // Speaker management
        .route("/api/v1/speakers", get(routes::speakers::list_speakers))
        .route("/api/v1/speakers/:id", get(routes::speakers::get_speaker))
        .route("/api/v1/speakers", post(routes::speakers::add_speaker))
        .route("/api/v1/speakers/:id", delete(routes::speakers::delete_speaker))
        
        // Channel management
        .route("/api/v1/channels", get(routes::channels::list_channels))
        .route("/api/v1/channels/:name/models", get(routes::channels::list_models))
        
        // State
        .with_state(state)
        // Middleware
        .layer(PropagateRequestIdLayer::new(axum::http::HeaderName::from_static("x-request-id")))
        .layer(TraceLayer::new_for_http())
        .layer(cors)
}

/// TTS Server
pub struct TtsServer {
    config: ServerConfig,
    state: Arc<ServerState>,
}

impl TtsServer {
    /// Create new TTS server
    pub fn new(config: ServerConfig) -> Self {
        let state = Arc::new(ServerState::new(config.clone()));
        
        Self {
            config,
            state,
        }
    }
    
    /// Run the server
    pub async fn run(self) -> Result<(), Box<dyn std::error::Error>> {
        let mut state = ServerState::new(self.config.clone());
        
        // Initialize local engine
        if self.config.local.enabled {
            state.init_local_engine().await?;
        }
        
        // Initialize cloud channels
        if self.config.cloud.enabled {
            state.init_cloud_channels().await?;
        }
        
        let state = Arc::new(state);
        let router = create_router(state);
        
        let addr = format!("{}:{}", self.config.host, self.config.port);
        info!("Starting TTS server on {}", addr);

        let listener = tokio::net::TcpListener::bind(&addr).await?;
        axum::serve(listener, router).await?;

        Ok(())
    }

    /// Get server state
    pub fn state(&self) -> Arc<ServerState> {
        self.state.clone()
    }
}
