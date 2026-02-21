//! Synthesis Routes

use axum::{extract::State, http::StatusCode, Json};
use crate::server::server_core::ServerState;
use crate::server::types::{SynthesisRequest, SynthesisResponse, SynthesisStatus};

/// Synthesize speech
pub async fn synthesize(
    State(state): State<std::sync::Arc<ServerState>>,
    Json(request): Json<SynthesisRequest>,
) -> Result<Json<SynthesisResponse>, StatusCode> {
    // Increment request count
    state.increment_request_count().await;
    
    // Route to appropriate handler based on channel
    let response = if let Some(channel_name) = &request.channel {
        if channel_name == "local" {
            // Use local engine
            synthesize_local(&state, &request).await
        } else {
            // Use cloud channel
            synthesize_cloud(&state, channel_name, &request).await
        }
    } else {
        // Use default channel
        synthesize_default(&state, &request).await
    };
    
    Ok(Json(response))
}

/// Stream synthesis (placeholder)
pub async fn synthesize_stream(
    State(state): State<std::sync::Arc<ServerState>>,
    Json(request): Json<SynthesisRequest>,
) -> Result<Json<SynthesisResponse>, StatusCode> {
    state.increment_request_count().await;
    
    // TODO: Implement streaming
    let mut response = synthesize_default(&state, &request).await;
    response.status = SynthesisStatus::Processing;
    
    Ok(Json(response))
}

/// Synthesize using local engine
async fn synthesize_local(
    _state: &ServerState,
    request: &SynthesisRequest,
) -> SynthesisResponse {
    // TODO: Implement local synthesis
    SynthesisResponse {
        request_id: uuid::Uuid::new_v4().to_string(),
        status: SynthesisStatus::Success,
        audio: None,
        audio_url: None,
        duration: None,
        sample_rate: None,
        format: Some(request.output_format),
        error: Some("Local synthesis not yet implemented".to_string()),
        processing_time_ms: None,
        channel: Some("local".to_string()),
        model: None,
    }
}

/// Synthesize using cloud channel
async fn synthesize_cloud(
    state: &ServerState,
    channel_name: &str,
    request: &SynthesisRequest,
) -> SynthesisResponse {
    // Find channel
    for channel in &state.cloud_channels {
        if channel.name() == channel_name {
            match channel.synthesize(request).await {
                Ok(response) => return response,
                Err(e) => {
                    return SynthesisResponse {
                        request_id: uuid::Uuid::new_v4().to_string(),
                        status: SynthesisStatus::Failed,
                        audio: None,
                        audio_url: None,
                        duration: None,
                        sample_rate: None,
                        format: Some(request.output_format),
                        error: Some(e),
                        processing_time_ms: None,
                        channel: Some(channel_name.to_string()),
                        model: request.model.clone(),
                    };
                }
            }
        }
    }
    
    SynthesisResponse {
        request_id: uuid::Uuid::new_v4().to_string(),
        status: SynthesisStatus::Failed,
        audio: None,
        audio_url: None,
        duration: None,
        sample_rate: None,
        format: Some(request.output_format),
        error: Some(format!("Channel {} not found", channel_name)),
        processing_time_ms: None,
        channel: None,
        model: None,
    }
}

/// Synthesize using default channel
async fn synthesize_default(
    state: &ServerState,
    request: &SynthesisRequest,
) -> SynthesisResponse {
    // Prefer local if available
    if state.config.local.enabled {
        synthesize_local(state, request).await
    } else if let Some(default_channel) = &state.config.cloud.default_channel {
        synthesize_cloud(state, default_channel, request).await
    } else if let Some(channel) = state.cloud_channels.first() {
        synthesize_cloud(state, channel.name(), request).await
    } else {
        SynthesisResponse {
            request_id: uuid::Uuid::new_v4().to_string(),
            status: SynthesisStatus::Failed,
            audio: None,
            audio_url: None,
            duration: None,
            sample_rate: None,
            format: Some(request.output_format),
            error: Some("No synthesis channel available".to_string()),
            processing_time_ms: None,
            channel: None,
            model: None,
        }
    }
}
