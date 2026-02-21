//! Speaker Management Routes

use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    Json,
};
use serde::Deserialize;
use crate::server::server_core::ServerState;
use crate::server::types::{ListSpeakersResponse, SpeakerInfo, Pagination};

/// Query parameters for listing speakers
#[derive(Debug, Deserialize)]
pub struct ListSpeakersQuery {
    #[serde(default)]
    pub page: Option<usize>,
    #[serde(default = "default_page_size")]
    pub page_size: Option<usize>,
    pub gender: Option<String>,
    pub age: Option<String>,
    pub language: Option<String>,
    pub source: Option<String>,
    pub search: Option<String>,
}

fn default_page_size() -> Option<usize> {
    Some(20)
}

/// List all speakers
pub async fn list_speakers(
    State(state): State<std::sync::Arc<ServerState>>,
    Query(query): Query<ListSpeakersQuery>,
) -> Result<Json<ListSpeakersResponse>, StatusCode> {
    let page = query.page.unwrap_or(1);
    let page_size = query.page_size.unwrap_or(20);
    
    // Get speakers
    let all_speakers = if let Some(search) = &query.search {
        state.speaker_lib.search(search)
    } else {
        state.speaker_lib.list_speakers()
    };
    
    // Apply filters
    let filtered: Vec<SpeakerInfo> = all_speakers.into_iter()
        .filter(|s| {
            if let Some(gender) = &query.gender {
                if s.gender.as_ref().map(|g| format!("{:?}", g).to_lowercase()) != Some(gender.to_lowercase()) {
                    return false;
                }
            }
            if let Some(age) = &query.age {
                if s.age.as_ref().map(|a| format!("{:?}", a).to_lowercase()) != Some(age.to_lowercase()) {
                    return false;
                }
            }
            if let Some(lang) = &query.language {
                if !s.languages.iter().any(|l| l.to_lowercase() == lang.to_lowercase()) {
                    return false;
                }
            }
            true
        })
        .collect();
    
    // Pagination
    let total = filtered.len();
    let total_pages = total.div_ceil(page_size);
    let start = (page - 1) * page_size;
    let end = std::cmp::min(start + page_size, total);
    
    let speakers = if start < total {
        filtered[start..end].to_vec()
    } else {
        Vec::new()
    };
    
    Ok(Json(ListSpeakersResponse {
        total,
        speakers,
        pagination: Some(Pagination {
            page,
            page_size,
            total_pages,
        }),
    }))
}

/// Get speaker by ID
pub async fn get_speaker(
    State(state): State<std::sync::Arc<ServerState>>,
    Path(speaker_id): Path<String>,
) -> Result<Json<SpeakerInfo>, StatusCode> {
    if let Some(speaker) = state.speaker_lib.get_speaker(&speaker_id) {
        // Record usage
        state.speaker_lib.record_usage(&speaker_id);
        Ok(Json(speaker.info))
    } else {
        Err(StatusCode::NOT_FOUND)
    }
}

/// Add new speaker (placeholder)
pub async fn add_speaker(
    State(_state): State<std::sync::Arc<ServerState>>,
    Json(_info): Json<SpeakerInfo>,
) -> Result<Json<SpeakerInfo>, StatusCode> {
    // TODO: Implement speaker addition with audio upload
    Err(StatusCode::NOT_IMPLEMENTED)
}

/// Delete speaker
pub async fn delete_speaker(
    State(state): State<std::sync::Arc<ServerState>>,
    Path(speaker_id): Path<String>,
) -> Result<StatusCode, StatusCode> {
    state.speaker_lib.remove_speaker(&speaker_id)
        .map(|_| StatusCode::NO_CONTENT)
        .map_err(|_| StatusCode::NOT_FOUND)
}
