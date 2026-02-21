//! Batch Inference Support
//!
//! Provides batch processing capabilities for efficient multi-request inference.

use std::sync::Arc;
use std::time::Instant;

use crate::core::error::Result;
use crate::engine::traits::{SynthesisRequest, SynthesisResult};

/// Batch synthesis request
#[derive(Debug, Clone)]
pub struct BatchSynthesisRequest {
    /// Individual requests
    pub requests: Vec<SynthesisRequest>,
    /// Batch ID for tracking
    pub batch_id: Option<String>,
    /// Maximum parallelism
    pub max_parallelism: usize,
}

impl BatchSynthesisRequest {
    /// Create a new batch request
    pub fn new(requests: Vec<SynthesisRequest>) -> Self {
        Self {
            requests,
            batch_id: None,
            max_parallelism: 4,
        }
    }

    /// Set batch ID
    pub fn with_batch_id(mut self, id: impl Into<String>) -> Self {
        self.batch_id = Some(id.into());
        self
    }

    /// Set maximum parallelism
    pub fn with_parallelism(mut self, n: usize) -> Self {
        self.max_parallelism = n.max(1);
        self
    }

    /// Get number of requests
    pub fn len(&self) -> usize {
        self.requests.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.requests.is_empty()
    }
}

/// Batch synthesis result
#[derive(Debug, Clone)]
pub struct BatchSynthesisResult {
    /// Individual results
    pub results: Vec<SynthesisResult>,
    /// Batch ID
    pub batch_id: Option<String>,
    /// Total processing time in milliseconds
    pub total_time_ms: u64,
    /// Average time per request in milliseconds
    pub avg_time_ms: f64,
    /// Requests processed
    pub requests_processed: usize,
    /// Failed requests
    pub failed: Vec<(usize, String)>,
}

impl BatchSynthesisResult {
    /// Create a new batch result
    pub fn new(
        results: Vec<SynthesisResult>,
        batch_id: Option<String>,
        total_time_ms: u64,
        failed: Vec<(usize, String)>,
    ) -> Self {
        let requests_processed = results.len();
        let avg_time_ms = if requests_processed > 0 {
            total_time_ms as f64 / requests_processed as f64
        } else {
            0.0
        };

        Self {
            results,
            batch_id,
            total_time_ms,
            avg_time_ms,
            requests_processed,
            failed,
        }
    }

    /// Get success rate
    pub fn success_rate(&self) -> f64 {
        let total = self.requests_processed + self.failed.len();
        if total == 0 {
            1.0
        } else {
            self.requests_processed as f64 / total as f64
        }
    }

    /// Get total audio duration
    pub fn total_duration(&self) -> f32 {
        self.results.iter().map(|r| r.duration).sum()
    }
}

/// Batch processing trait for TTS engines
#[async_trait::async_trait]
pub trait BatchProcessor: Send + Sync {
    /// Process a batch of synthesis requests
    async fn process_batch(&self, batch: BatchSynthesisRequest) -> Result<BatchSynthesisResult>;

    /// Get maximum batch size
    fn max_batch_size(&self) -> usize {
        16
    }

    /// Get current batch size
    fn current_batch_size(&self) -> usize {
        self.max_batch_size()
    }
}

/// Simple batch processor implementation
pub struct SimpleBatchProcessor<E> {
    /// Engine reference
    engine: Arc<E>,
    /// Maximum parallelism
    max_parallelism: usize,
}

impl<E> SimpleBatchProcessor<E>
where
    E: crate::engine::traits::TtsEngine + 'static,
{
    /// Create a new batch processor
    pub fn new(engine: Arc<E>) -> Self {
        Self {
            engine,
            max_parallelism: 4,
        }
    }

    /// Set maximum parallelism
    pub fn with_parallelism(mut self, n: usize) -> Self {
        self.max_parallelism = n.max(1);
        self
    }
}

#[async_trait::async_trait]
impl<E> BatchProcessor for SimpleBatchProcessor<E>
where
    E: crate::engine::traits::TtsEngine + 'static,
{
    async fn process_batch(&self, batch: BatchSynthesisRequest) -> Result<BatchSynthesisResult> {
        let start = Instant::now();
        let max_parallelism = batch.max_parallelism.min(self.max_parallelism);

        // Process requests in parallel chunks
        let mut results = Vec::with_capacity(batch.requests.len());
        let mut failed = Vec::new();

        // Use tokio for parallel processing
        let chunks: Vec<Vec<_>> = batch
            .requests
            .chunks(max_parallelism)
            .map(|c| c.to_vec())
            .collect();

        for chunk in chunks {
            let futures: Vec<_> = chunk
                .into_iter()
                .enumerate()
                .map(|(i, req)| {
                    let engine = Arc::clone(&self.engine);
                    async move {
                        let result = engine.synthesize(&req).await;
                        (i, result)
                    }
                })
                .collect();

            let chunk_results = futures::future::join_all(futures).await;

            for (idx, result) in chunk_results {
                match result {
                    Ok(r) => results.push(r),
                    Err(e) => failed.push((idx, e.to_string())),
                }
            }
        }

        let total_time_ms = start.elapsed().as_millis() as u64;

        Ok(BatchSynthesisResult::new(
            results,
            batch.batch_id,
            total_time_ms,
            failed,
        ))
    }

    fn max_batch_size(&self) -> usize {
        16
    }
}

/// Batch processing statistics
#[derive(Debug, Clone, Default)]
pub struct BatchStats {
    /// Total batches processed
    pub total_batches: u64,
    /// Total requests processed
    pub total_requests: u64,
    /// Total failed requests
    pub total_failed: u64,
    /// Average batch processing time in milliseconds
    pub avg_batch_time_ms: f64,
    /// Average requests per batch
    pub avg_requests_per_batch: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batch_request_builder() {
        let requests = vec![];
        let batch = BatchSynthesisRequest::new(requests)
            .with_batch_id("test-batch")
            .with_parallelism(8);

        assert_eq!(batch.batch_id, Some("test-batch".to_string()));
        assert_eq!(batch.max_parallelism, 8);
        assert!(batch.is_empty());
    }

    #[test]
    fn test_batch_result_stats() {
        let results = vec![];
        let batch_result = BatchSynthesisResult::new(
            results,
            Some("test".to_string()),
            1000,
            vec![(0, "error".to_string())],
        );

        assert_eq!(batch_result.requests_processed, 0);
        assert_eq!(batch_result.failed.len(), 1);
        assert_eq!(batch_result.success_rate(), 0.0);
    }
}
