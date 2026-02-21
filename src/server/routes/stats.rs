//! Statistics Routes

use axum::{Json};
use crate::server::server_core::ServerState;
use crate::server::types::ServerStats;
use axum::extract::State;

/// Get server statistics
pub async fn get_stats(
    State(state): State<std::sync::Arc<ServerState>>,
) -> Json<ServerStats> {
    Json(state.get_stats().await)
}
