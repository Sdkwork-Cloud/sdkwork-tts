//! Voice Design and Clone Routes

use axum::{extract::State, http::StatusCode, Json};
use crate::server::server_core::ServerState;
use crate::server::types::{SynthesisRequest, SynthesisResponse, VoiceDesignOptions, VoiceCloneOptions};

/// Voice design request
#[derive(Debug, serde::Deserialize)]
pub struct VoiceDesignRequest {
    pub text: String,
    pub voice_design: VoiceDesignOptions,
    #[serde(default)]
    pub output_format: crate::server::types::AudioFormat,
}

/// Voice clone request
#[derive(Debug, serde::Deserialize)]
pub struct VoiceCloneRequest {
    pub text: String,
    pub voice_clone: VoiceCloneOptions,
    #[serde(default)]
    pub output_format: crate::server::types::AudioFormat,
}

/// Voice design endpoint
pub async fn voice_design(
    State(state): State<std::sync::Arc<ServerState>>,
    Json(request): Json<VoiceDesignRequest>,
) -> Result<Json<SynthesisResponse>, StatusCode> {
    state.increment_request_count().await;
    
    // Create synthesis request with voice design options
    let synthesis_request = SynthesisRequest {
        text: request.text,
        speaker: "voice_design".to_string(),
        channel: Some("local".to_string()),
        model: None,
        language: None,
        parameters: Default::default(),
        voice_design: Some(request.voice_design),
        voice_clone: None,
        output_format: request.output_format,
        streaming: false,
    };
    
    // Route to synthesis handler
    crate::server::routes::synthesis::synthesize(
        State(state),
        Json(synthesis_request),
    ).await
}

/// Voice clone endpoint
pub async fn voice_clone(
    State(state): State<std::sync::Arc<ServerState>>,
    Json(request): Json<VoiceCloneRequest>,
) -> Result<Json<SynthesisResponse>, StatusCode> {
    state.increment_request_count().await;
    
    // Create synthesis request with voice clone options
    let synthesis_request = SynthesisRequest {
        text: request.text,
        speaker: "voice_clone".to_string(),
        channel: Some("local".to_string()),
        model: None,
        language: None,
        parameters: Default::default(),
        voice_design: None,
        voice_clone: Some(request.voice_clone),
        output_format: request.output_format,
        streaming: false,
    };
    
    // Route to synthesis handler
    crate::server::routes::synthesis::synthesize(
        State(state),
        Json(synthesis_request),
    ).await
}
