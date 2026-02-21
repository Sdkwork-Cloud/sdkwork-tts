//! Channel Management Routes

use axum::{extract::Path, extract::State, http::StatusCode, Json};
use crate::server::server_core::ServerState;

/// Channel info
#[derive(Debug, serde::Serialize)]
pub struct ChannelInfo {
    pub name: String,
    #[serde(rename = "type")]
    pub channel_type: String,
    pub enabled: bool,
    pub models: Vec<String>,
}

/// List channels response
#[derive(Debug, serde::Serialize)]
pub struct ListChannelsResponse {
    pub channels: Vec<ChannelInfo>,
}

/// List all channels
pub async fn list_channels(
    State(state): State<std::sync::Arc<ServerState>>,
) -> Json<ListChannelsResponse> {
    let mut channels = Vec::new();
    
    // Add local channel if enabled
    if state.config.local.enabled {
        channels.push(ChannelInfo {
            name: "local".to_string(),
            channel_type: "local".to_string(),
            enabled: true,
            models: vec![state.config.local.default_engine.clone()],
        });
    }
    
    // Add cloud channels
    for channel in &state.cloud_channels {
        // TODO: Get actual models from channel
        channels.push(ChannelInfo {
            name: channel.name().to_string(),
            channel_type: format!("{:?}", channel.channel_type()),
            enabled: true,
            models: vec![], // TODO: Get from channel
        });
    }
    
    Json(ListChannelsResponse { channels })
}

/// List models for a channel
pub async fn list_models(
    State(state): State<std::sync::Arc<ServerState>>,
    Path(channel_name): Path<String>,
) -> Result<Json<Vec<String>>, StatusCode> {
    if channel_name == "local" {
        return Ok(Json(vec![state.config.local.default_engine.clone()]));
    }
    
    // Find channel
    for channel in &state.cloud_channels {
        if channel.name() == channel_name {
            match channel.get_models().await {
                Ok(models) => return Ok(Json(models)),
                Err(_) => return Err(StatusCode::INTERNAL_SERVER_ERROR),
            }
        }
    }
    
    Err(StatusCode::NOT_FOUND)
}
