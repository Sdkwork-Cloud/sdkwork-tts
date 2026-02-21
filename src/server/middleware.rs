//! Performance Monitoring Middleware
//!
//! Provides request timing, metrics collection, and performance tracking

use axum::{
    body::Body,
    http::{Request, StatusCode},
    middleware::Next,
    response::Response,
};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::RwLock;
use tracing::{info, warn};

/// Server metrics
#[derive(Debug, Default, Clone)]
pub struct ServerMetrics {
    /// Total requests
    pub total_requests: u64,
    /// Successful requests
    pub successful_requests: u64,
    /// Failed requests
    pub failed_requests: u64,
    /// Total processing time (ms)
    pub total_processing_time_ms: f64,
    /// Requests by endpoint
    pub requests_by_endpoint: std::collections::HashMap<String, u64>,
    /// Processing time by endpoint (ms)
    pub time_by_endpoint: std::collections::HashMap<String, f64>,
}

impl ServerMetrics {
    /// Create new metrics
    pub fn new() -> Self {
        Self::default()
    }
    
    /// Record a request
    pub fn record_request(&mut self, endpoint: &str, status: StatusCode, duration_ms: f64) {
        self.total_requests += 1;
        
        if status.is_success() {
            self.successful_requests += 1;
        } else {
            self.failed_requests += 1;
        }
        
        self.total_processing_time_ms += duration_ms;
        
        *self.requests_by_endpoint.entry(endpoint.to_string()).or_insert(0) += 1;
        *self.time_by_endpoint.entry(endpoint.to_string()).or_insert(0.0) += duration_ms;
    }
    
    /// Get average processing time
    pub fn avg_processing_time(&self) -> f64 {
        if self.total_requests == 0 {
            return 0.0;
        }
        self.total_processing_time_ms / self.total_requests as f64
    }
    
    /// Get average processing time by endpoint
    pub fn avg_time_by_endpoint(&self, endpoint: &str) -> f64 {
        let count = self.requests_by_endpoint.get(endpoint).copied().unwrap_or(0);
        let time = self.time_by_endpoint.get(endpoint).copied().unwrap_or(0.0);
        
        if count == 0 {
            return 0.0;
        }
        time / count as f64
    }
    
    /// Get metrics summary
    pub fn summary(&self) -> MetricsSummary {
        MetricsSummary {
            total_requests: self.total_requests,
            successful_requests: self.successful_requests,
            failed_requests: self.failed_requests,
            avg_processing_time_ms: self.avg_processing_time(),
            success_rate: if self.total_requests > 0 {
                self.successful_requests as f64 / self.total_requests as f64 * 100.0
            } else {
                0.0
            },
        }
    }
}

/// Metrics summary
#[derive(Debug, Clone)]
pub struct MetricsSummary {
    pub total_requests: u64,
    pub successful_requests: u64,
    pub failed_requests: u64,
    pub avg_processing_time_ms: f64,
    pub success_rate: f64,
}

/// Performance metrics state
pub struct MetricsState {
    metrics: Arc<RwLock<ServerMetrics>>,
}

impl MetricsState {
    /// Create new metrics state
    pub fn new() -> Self {
        Self {
            metrics: Arc::new(RwLock::new(ServerMetrics::new())),
        }
    }
    
    /// Get metrics clone
    pub async fn get_metrics(&self) -> ServerMetrics {
        let metrics = self.metrics.read().await;
        metrics.clone()
    }
    
    /// Record request
    pub async fn record_request(&self, endpoint: &str, status: StatusCode, duration_ms: f64) {
        let mut metrics = self.metrics.write().await;
        metrics.record_request(endpoint, status, duration_ms);
    }
    
    /// Get summary
    pub async fn get_summary(&self) -> MetricsSummary {
        let metrics = self.metrics.read().await;
        metrics.summary()
    }
}

impl Default for MetricsState {
    fn default() -> Self {
        Self::new()
    }
}

/// Performance monitoring middleware
pub async fn performance_monitor(
    metrics: axum::extract::State<Arc<MetricsState>>,
    req: Request<Body>,
    next: Next,
) -> Result<Response, StatusCode> {
    let path = req.uri().path().to_string();
    let method = req.method().clone();
    let start = Instant::now();
    
    // Execute request
    let response = next.run(req).await;
    
    // Record metrics
    let duration_ms = start.elapsed().as_secs_f64() * 1000.0;
    let status = response.status();
    
    metrics.record_request(&path, status, duration_ms).await;
    
    // Log slow requests
    if duration_ms > 1000.0 {
        warn!(
            "Slow request: {} {} took {:.2}ms (status: {})",
            method, path, duration_ms, status
        );
    } else {
        info!(
            "Request: {} {} took {:.2}ms (status: {})",
            method, path, duration_ms, status
        );
    }
    
    Ok(response)
}

/// Create metrics router layer (placeholder)
#[allow(dead_code)]
pub fn create_metrics_layer(_metrics: Arc<MetricsState>) -> axum::Router {
    // This is a placeholder - in production, implement proper layer
    axum::Router::new()
}
