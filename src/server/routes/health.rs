//! Health Check Routes

use axum::Json;
use crate::server::server_core::ServerState;
use crate::server::types::HealthResponse;
use axum::extract::State;

/// Health check endpoint
pub async fn health_check(
    State(state): State<std::sync::Arc<ServerState>>,
) -> Json<HealthResponse> {
    let mode = match state.config.mode {
        crate::server::config::ServerMode::Local => "local",
        crate::server::config::ServerMode::Cloud => "cloud",
        crate::server::config::ServerMode::Hybrid => "hybrid",
    };
    
    let mut channels = Vec::new();
    for channel in &state.cloud_channels {
        channels.push(channel.name().to_string());
    }
    
    Json(HealthResponse {
        status: "healthy".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        mode: mode.to_string(),
        uptime: state.uptime().as_secs(),
        channels,
        speaker_count: state.speaker_lib.count(),
    })
}
