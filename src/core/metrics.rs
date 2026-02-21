//! Performance metrics and monitoring for IndexTTS2
//!
//! Provides comprehensive performance tracking, timing analysis,
//! and monitoring capabilities for the TTS pipeline.

use std::sync::{Arc, atomic::{AtomicU64, Ordering}};
use std::time::{Duration, Instant};
use dashmap::DashMap;

use super::error::Result;

/// Metrics collector for tracking performance
///
/// Uses lock-free data structures for high-performance metrics collection.
pub struct MetricsCollector {
    /// Named timers for different operations
    timers: Arc<DashMap<String, TimerStats>>,
    /// Counters for various events (lock-free)
    counters: Arc<DashMap<String, AtomicU64>>,
    /// Gauges for current values
    gauges: Arc<DashMap<String, f64>>,
    /// Histograms for value distributions
    histograms: Arc<DashMap<String, Arc<Mutex<Vec<f64>>>>>,
    /// Enable/disable collection
    enabled: bool,
    /// Maximum histogram samples
    max_histogram_samples: usize,
}

use std::sync::Mutex;

/// Timer statistics
#[derive(Debug, Clone)]
pub struct TimerStats {
    /// Total count of measurements
    pub count: u64,
    /// Total duration
    pub total_duration: Duration,
    /// Minimum duration
    pub min_duration: Duration,
    /// Maximum duration
    pub max_duration: Duration,
    /// Last duration
    pub last_duration: Duration,
}

impl Default for TimerStats {
    fn default() -> Self {
        Self {
            count: 0,
            total_duration: Duration::ZERO,
            min_duration: Duration::MAX,
            max_duration: Duration::ZERO,
            last_duration: Duration::ZERO,
        }
    }
}

impl TimerStats {
    /// Add a new duration measurement
    pub fn record(&mut self, duration: Duration) {
        self.count += 1;
        self.total_duration += duration;
        self.min_duration = self.min_duration.min(duration);
        self.max_duration = self.max_duration.max(duration);
        self.last_duration = duration;
    }

    /// Get average duration
    pub fn average(&self) -> Duration {
        if self.count > 0 {
            self.total_duration / self.count as u32
        } else {
            Duration::ZERO
        }
    }

    /// Get duration in milliseconds
    pub fn average_ms(&self) -> f64 {
        self.average().as_secs_f64() * 1000.0
    }
}

/// Performance metrics for a complete inference
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    /// Total inference time
    pub total_time: Duration,
    /// Text processing time
    pub text_processing_time: Duration,
    /// Speaker encoding time
    pub speaker_encoding_time: Duration,
    /// Emotion encoding time
    pub emotion_encoding_time: Duration,
    /// GPT generation time
    pub gpt_generation_time: Duration,
    /// Flow matching time
    pub flow_matching_time: Duration,
    /// Vocoding time
    pub vocoding_time: Duration,
    /// Number of generated mel frames
    pub mel_frames: usize,
    /// Number of generated tokens
    pub tokens_generated: usize,
    /// Real-time factor (audio_duration / processing_time)
    pub rtf: f64,
    /// Memory used (bytes)
    pub memory_used: usize,
    /// GPU memory used (bytes)
    pub gpu_memory_used: usize,
}

impl Default for PerformanceMetrics {
    fn default() -> Self {
        Self {
            total_time: Duration::ZERO,
            text_processing_time: Duration::ZERO,
            speaker_encoding_time: Duration::ZERO,
            emotion_encoding_time: Duration::ZERO,
            gpt_generation_time: Duration::ZERO,
            flow_matching_time: Duration::ZERO,
            vocoding_time: Duration::ZERO,
            mel_frames: 0,
            tokens_generated: 0,
            rtf: 0.0,
            memory_used: 0,
            gpu_memory_used: 0,
        }
    }
}

impl PerformanceMetrics {
    /// Get total processing time in milliseconds
    pub fn total_ms(&self) -> f64 {
        self.total_time.as_secs_f64() * 1000.0
    }

    /// Get RTF (Real-Time Factor)
    /// RTF < 1.0 means faster than real-time
    /// RTF > 1.0 means slower than real-time
    pub fn calculate_rtf(&self, audio_duration_secs: f64) -> f64 {
        if self.total_time.as_secs_f64() > 0.0 {
            audio_duration_secs / self.total_time.as_secs_f64()
        } else {
            0.0
        }
    }

    /// Format as human-readable string
    pub fn format_summary(&self) -> String {
        format!(
            "Inference completed in {:.2}ms (RTF: {:.2}x)\n\
             - Text processing: {:.2}ms\n\
             - Speaker encoding: {:.2}ms\n\
             - GPT generation: {:.2}ms\n\
             - Flow matching: {:.2}ms\n\
             - Vocoding: {:.2}ms\n\
             - Generated {} mel frames, {} tokens",
            self.total_ms(),
            self.rtf,
            self.text_processing_time.as_secs_f64() * 1000.0,
            self.speaker_encoding_time.as_secs_f64() * 1000.0,
            self.gpt_generation_time.as_secs_f64() * 1000.0,
            self.flow_matching_time.as_secs_f64() * 1000.0,
            self.vocoding_time.as_secs_f64() * 1000.0,
            self.mel_frames,
            self.tokens_generated
        )
    }
}

/// Timing information for a single operation
pub struct TimingInfo {
    name: String,
    start: Instant,
    collector: Option<Arc<MetricsCollector>>,
}

impl TimingInfo {
    /// Create a new timing info
    fn new(name: impl Into<String>, collector: Option<Arc<MetricsCollector>>) -> Self {
        Self {
            name: name.into(),
            start: Instant::now(),
            collector,
        }
    }

    /// Stop timing and record
    pub fn stop(self) -> Duration {
        let duration = self.start.elapsed();
        if let Some(ref collector) = self.collector {
            let _ = collector.record_timer(&self.name, duration);
        }
        duration
    }
}

impl Drop for TimingInfo {
    fn drop(&mut self) {
        // Auto-record on drop if not explicitly stopped
        let duration = self.start.elapsed();
        if let Some(ref collector) = self.collector {
            let _ = collector.record_timer(&self.name, duration);
        }
    }
}

impl Default for MetricsCollector {
    fn default() -> Self {
        Self::new()
    }
}

impl MetricsCollector {
    /// Create a new metrics collector
    pub fn new() -> Self {
        Self {
            timers: Arc::new(DashMap::new()),
            counters: Arc::new(DashMap::new()),
            gauges: Arc::new(DashMap::new()),
            histograms: Arc::new(DashMap::new()),
            enabled: true,
            max_histogram_samples: 1000,
        }
    }

    /// Create a disabled collector (no-op)
    pub fn disabled() -> Self {
        Self {
            timers: Arc::new(DashMap::new()),
            counters: Arc::new(DashMap::new()),
            gauges: Arc::new(DashMap::new()),
            histograms: Arc::new(DashMap::new()),
            enabled: false,
            max_histogram_samples: 1000,
        }
    }

    /// Enable collection
    pub fn enable(&mut self) {
        self.enabled = true;
    }

    /// Disable collection
    pub fn disable(&mut self) {
        self.enabled = false;
    }

    /// Check if enabled
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Start timing an operation
    pub fn start_timer(&self, name: impl Into<String>) -> TimingInfo {
        TimingInfo::new(name, Some(Arc::new(self.clone())))
    }

    /// Record a timer value (lock-free)
    pub fn record_timer(&self, name: &str, duration: Duration) -> Result<()> {
        if !self.enabled {
            return Ok(());
        }

        self.timers
            .entry(name.to_string())
            .or_insert_with(TimerStats::default)
            .record(duration);
        Ok(())
    }

    /// Increment a counter (lock-free)
    pub fn increment_counter(&self, name: &str, value: u64) -> Result<()> {
        if !self.enabled {
            return Ok(());
        }

        self.counters
            .entry(name.to_string())
            .or_insert_with(|| AtomicU64::new(0))
            .fetch_add(value, Ordering::Relaxed);
        Ok(())
    }

    /// Set a gauge value
    pub fn set_gauge(&self, name: &str, value: f64) -> Result<()> {
        if !self.enabled {
            return Ok(());
        }

        self.gauges.insert(name.to_string(), value);
        Ok(())
    }

    /// Record a histogram value
    pub fn record_histogram(&self, name: &str, value: f64) -> Result<()> {
        if !self.enabled {
            return Ok(());
        }

        let entry = self.histograms.entry(name.to_string()).or_insert_with(|| Arc::new(Mutex::new(Vec::new())));
        let mut samples = entry.lock().unwrap();
        samples.push(value);

        // Limit samples to prevent unbounded growth
        if samples.len() > self.max_histogram_samples {
            samples.remove(0);
        }
        Ok(())
    }

    /// Get timer statistics
    pub fn get_timer(&self, name: &str) -> Option<TimerStats> {
        self.timers.get(name).map(|r| r.value().clone())
    }

    /// Get counter value (lock-free)
    pub fn get_counter(&self, name: &str) -> u64 {
        self.counters
            .get(name)
            .map(|c| c.load(Ordering::Relaxed))
            .unwrap_or(0)
    }

    /// Get gauge value
    pub fn get_gauge(&self, name: &str) -> Option<f64> {
        self.gauges.get(name).map(|r| *r)
    }

    /// Get histogram statistics
    pub fn get_histogram_stats(&self, name: &str) -> Option<HistogramStats> {
        self.histograms.get(name).map(|r| {
            let samples = r.lock().unwrap();
            let mut sorted = samples.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

            HistogramStats {
                count: sorted.len(),
                min: sorted.first().copied().unwrap_or(0.0),
                max: sorted.last().copied().unwrap_or(0.0),
                mean: sorted.iter().sum::<f64>() / sorted.len() as f64,
                p50: percentile(&sorted, 0.5),
                p95: percentile(&sorted, 0.95),
                p99: percentile(&sorted, 0.99),
            }
        })
    }

    /// Get all timer names
    pub fn timer_names(&self) -> Vec<String> {
        self.timers.iter().map(|r| r.key().clone()).collect()
    }

    /// Reset all metrics
    pub fn reset(&self) {
        self.timers.clear();
        self.counters.clear();
        self.gauges.clear();
        self.histograms.clear();
    }

    /// Generate a summary report
    pub fn generate_report(&self) -> MetricsReport {
        let mut report = MetricsReport::default();

        for r in self.timers.iter() {
            let name = r.key();
            let stats = r.value();
            report.timers.push(TimerReport {
                name: name.clone(),
                count: stats.count,
                avg_ms: stats.average_ms(),
                min_ms: stats.min_duration.as_secs_f64() * 1000.0,
                max_ms: stats.max_duration.as_secs_f64() * 1000.0,
            });
        }

        for r in self.counters.iter() {
            let name = r.key();
            let value = r.value().load(Ordering::Relaxed);
            report.counters.push((name.clone(), value));
        }

        for r in self.gauges.iter() {
            report.gauges.push((r.key().clone(), *r.value()));
        }

        report
    }
}

impl Clone for MetricsCollector {
    fn clone(&self) -> Self {
        Self {
            timers: self.timers.clone(),
            counters: self.counters.clone(),
            gauges: self.gauges.clone(),
            histograms: self.histograms.clone(),
            enabled: self.enabled,
            max_histogram_samples: self.max_histogram_samples,
        }
    }
}

/// Histogram statistics
#[derive(Debug, Clone)]
pub struct HistogramStats {
    /// Number of samples
    pub count: usize,
    /// Minimum value
    pub min: f64,
    /// Maximum value
    pub max: f64,
    /// Mean value
    pub mean: f64,
    /// 50th percentile
    pub p50: f64,
    /// 95th percentile
    pub p95: f64,
    /// 99th percentile
    pub p99: f64,
}

/// Timer report entry
#[derive(Debug, Clone)]
pub struct TimerReport {
    /// Timer name
    pub name: String,
    /// Number of measurements
    pub count: u64,
    /// Average duration in milliseconds
    pub avg_ms: f64,
    /// Minimum duration in milliseconds
    pub min_ms: f64,
    /// Maximum duration in milliseconds
    pub max_ms: f64,
}

/// Complete metrics report
#[derive(Debug, Clone, Default)]
pub struct MetricsReport {
    /// Timer reports
    pub timers: Vec<TimerReport>,
    /// Counter values
    pub counters: Vec<(String, u64)>,
    /// Gauge values
    pub gauges: Vec<(String, f64)>,
}

impl MetricsReport {
    /// Format as human-readable string
    pub fn format(&self) -> String {
        let mut output = String::new();

        if !self.timers.is_empty() {
            output.push_str("=== Timers ===\n");
            for timer in &self.timers {
                output.push_str(&format!(
                    "{}: {} calls, avg={:.2}ms, min={:.2}ms, max={:.2}ms\n",
                    timer.name, timer.count, timer.avg_ms, timer.min_ms, timer.max_ms
                ));
            }
            output.push('\n');
        }

        if !self.counters.is_empty() {
            output.push_str("=== Counters ===\n");
            for (name, value) in &self.counters {
                output.push_str(&format!("{}: {}\n", name, value));
            }
            output.push('\n');
        }

        if !self.gauges.is_empty() {
            output.push_str("=== Gauges ===\n");
            for (name, value) in &self.gauges {
                output.push_str(&format!("{}: {:.2}\n", name, value));
            }
        }

        output
    }
}

/// Calculate percentile from sorted samples
fn percentile(sorted_samples: &[f64], p: f64) -> f64 {
    if sorted_samples.is_empty() {
        return 0.0;
    }

    let index = (p * (sorted_samples.len() - 1) as f64).round() as usize;
    sorted_samples[index.min(sorted_samples.len() - 1)]
}

/// Convenience macro for timing a block
#[macro_export]
macro_rules! time_it {
    ($collector:expr, $name:expr, $block:expr) => {{
        let _timer = $collector.start_timer($name);
        $block
    }};
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_timer_stats() {
        let mut stats = TimerStats::default();

        stats.record(Duration::from_millis(100));
        stats.record(Duration::from_millis(200));
        stats.record(Duration::from_millis(150));

        assert_eq!(stats.count, 3);
        assert_eq!(stats.min_duration, Duration::from_millis(100));
        assert_eq!(stats.max_duration, Duration::from_millis(200));
        assert_eq!(stats.average(), Duration::from_millis(150));
    }

    #[test]
    fn test_metrics_collector() {
        let collector = MetricsCollector::new();

        // Test timer
        collector.record_timer("test_op", Duration::from_millis(100)).unwrap();
        collector.record_timer("test_op", Duration::from_millis(200)).unwrap();

        let stats = collector.get_timer("test_op").unwrap();
        assert_eq!(stats.count, 2);

        // Test counter
        collector.increment_counter("requests", 1).unwrap();
        collector.increment_counter("requests", 2).unwrap();
        assert_eq!(collector.get_counter("requests"), 3);

        // Test gauge
        collector.set_gauge("memory", 1024.0).unwrap();
        assert_eq!(collector.get_gauge("memory"), Some(1024.0));
    }

    #[test]
    fn test_performance_metrics() {
        let metrics = PerformanceMetrics {
            total_time: Duration::from_millis(500),
            mel_frames: 100,
            tokens_generated: 50,
            ..Default::default()
        };

        let rtf = metrics.calculate_rtf(2.0); // 2 seconds of audio
        assert_eq!(rtf, 4.0); // 2.0 / 0.5 = 4.0

        let summary = metrics.format_summary();
        assert!(summary.contains("Inference completed"));
        assert!(summary.contains("RTF"));
    }

    #[test]
    fn test_metrics_report() {
        let collector = MetricsCollector::new();

        collector.record_timer("op1", Duration::from_millis(100)).unwrap();
        collector.record_timer("op2", Duration::from_millis(200)).unwrap();
        collector.increment_counter("count", 5).unwrap();
        collector.set_gauge("gauge", 42.0).unwrap();

        let report = collector.generate_report();
        assert_eq!(report.timers.len(), 2);
        assert_eq!(report.counters.len(), 1);
        assert_eq!(report.gauges.len(), 1);

        let formatted = report.format();
        assert!(formatted.contains("Timers"));
        assert!(formatted.contains("Counters"));
        assert!(formatted.contains("Gauges"));
    }
}
