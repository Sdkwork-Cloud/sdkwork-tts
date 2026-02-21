//! Qwen3-TTS Benchmark Utilities
//!
//! This module provides benchmarking utilities for measuring
//! Qwen3-TTS inference performance.

use std::time::Instant;

/// Benchmark result
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    /// Test name
    pub name: String,
    /// Number of iterations
    pub iterations: usize,
    /// Total time in milliseconds
    pub total_time_ms: f64,
    /// Average time per iteration in milliseconds
    pub avg_time_ms: f64,
    /// Minimum time in milliseconds
    pub min_time_ms: f64,
    /// Maximum time in milliseconds
    pub max_time_ms: f64,
    /// Standard deviation in milliseconds
    pub std_dev_ms: f64,
    /// Throughput (iterations per second)
    pub throughput_ips: f64,
}

impl BenchmarkResult {
    /// Create new benchmark result
    pub fn new(
        name: String,
        iterations: usize,
        times_ms: &[f64],
    ) -> Self {
        let total_time_ms = times_ms.iter().sum();
        let avg_time_ms = total_time_ms / iterations as f64;
        let min_time_ms = times_ms.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_time_ms = times_ms.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        
        // Calculate standard deviation
        let mut sum_sq = 0.0f64;
        for &t in times_ms {
            let diff = t - avg_time_ms;
            sum_sq += diff * diff;
        }
        let variance = sum_sq / iterations as f64;
        let std_dev_ms = variance.sqrt();
        
        let throughput_ips = iterations as f64 / (total_time_ms / 1000.0);
        
        Self {
            name,
            iterations,
            total_time_ms,
            avg_time_ms,
            min_time_ms,
            max_time_ms,
            std_dev_ms,
            throughput_ips,
        }
    }

    /// Print benchmark result
    pub fn print(&self) {
        println!("\n╔═══════════════════════════════════════════════════════════╗");
        println!("║  Benchmark: {:<48} ║", self.name);
        println!("╠═══════════════════════════════════════════════════════════╣");
        println!("║  Iterations:     {:>12} iterations                      ║", self.iterations);
        println!("║  Total Time:     {:>12.2} ms                           ║", self.total_time_ms);
        println!("║  Average Time:   {:>12.2} ± {:.2} ms                  ║", self.avg_time_ms, self.std_dev_ms);
        println!("║  Min Time:       {:>12.2} ms                           ║", self.min_time_ms);
        println!("║  Max Time:       {:>12.2} ms                           ║", self.max_time_ms);
        println!("║  Throughput:     {:>12.2} iter/sec                     ║", self.throughput_ips);
        println!("╚═══════════════════════════════════════════════════════════╝");
    }
}

/// Benchmark runner
pub struct BenchmarkRunner {
    name: String,
    warmup_iterations: usize,
    measurement_iterations: usize,
    times_ms: Vec<f64>,
}

impl BenchmarkRunner {
    /// Create new benchmark runner
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            warmup_iterations: 3,
            measurement_iterations: 10,
            times_ms: Vec::new(),
        }
    }

    /// Set warmup iterations
    pub fn warmup(mut self, iterations: usize) -> Self {
        self.warmup_iterations = iterations;
        self
    }

    /// Set measurement iterations
    pub fn iterations(mut self, iterations: usize) -> Self {
        self.measurement_iterations = iterations;
        self
    }

    /// Run benchmark with warmup
    pub fn run<F>(&mut self, mut benchmark_fn: F) -> BenchmarkResult
    where
        F: FnMut() -> anyhow::Result<()>,
    {
        // Warmup
        for _ in 0..self.warmup_iterations {
            let _ = benchmark_fn();
        }

        // Measurement
        self.times_ms.clear();
        for _ in 0..self.measurement_iterations {
            let start = Instant::now();
            let _ = benchmark_fn();
            let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
            self.times_ms.push(elapsed_ms);
        }

        BenchmarkResult::new(
            self.name.clone(),
            self.measurement_iterations,
            &self.times_ms,
        )
    }

    /// Run benchmark and print result
    pub fn run_and_print<F>(&mut self, benchmark_fn: F) -> BenchmarkResult
    where
        F: FnMut() -> anyhow::Result<()>,
    {
        let result = self.run(benchmark_fn);
        result.print();
        result
    }
}

/// Compare multiple benchmarks
pub struct BenchmarkComparator {
    results: Vec<BenchmarkResult>,
}

impl BenchmarkComparator {
    /// Create new comparator
    pub fn new() -> Self {
        Self {
            results: Vec::new(),
        }
    }

    /// Add benchmark result
    pub fn add(&mut self, result: BenchmarkResult) {
        self.results.push(result);
    }

    /// Run and compare multiple benchmarks
    pub fn compare<F>(&mut self, benchmarks: Vec<(&str, F)>)
    where
        F: FnMut() -> anyhow::Result<()>,
    {
        for (name, mut benchmark_fn) in benchmarks {
            let mut runner = BenchmarkRunner::new(name);
            let result = runner.run_and_print(&mut benchmark_fn);
            self.add(result);
        }
    }

    /// Print comparison table
    pub fn print_comparison(&self) {
        println!("\n╔════════════════════════════════════════════════════════════════════╗");
        println!("║                    Benchmark Comparison                            ║");
        println!("╠════════════════════════════════════════════════════════════════════╣");
        println!("║  Name                          │ Avg (ms) │ Min (ms) │ Max (ms)   ║");
        println!("╠════════════════════════════════════════════════════════════════════╣");
        
        for result in &self.results {
            println!("║  {:<30} │ {:>8.2} │ {:>8.2} │ {:>8.2} ║",
                result.name,
                result.avg_time_ms,
                result.min_time_ms,
                result.max_time_ms,
            );
        }
        
        println!("╚════════════════════════════════════════════════════════════════════╝");
    }
}

impl Default for BenchmarkComparator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_benchmark_result() {
        let times = vec![10.0, 11.0, 9.0, 10.5, 9.5];
        let result = BenchmarkResult::new("test".to_string(), 5, &times);
        
        assert_eq!(result.iterations, 5);
        assert!((result.avg_time_ms - 10.0).abs() < 0.1);
        assert!((result.min_time_ms - 9.0).abs() < 0.1);
        assert!((result.max_time_ms - 11.0).abs() < 0.1);
    }

    #[test]
    fn test_benchmark_runner() {
        let mut runner = BenchmarkRunner::new("test")
            .warmup(2)
            .iterations(5);
        
        let result = runner.run(|| Ok(()));
        
        assert_eq!(result.iterations, 5);
        assert!(result.avg_time_ms >= 0.0);
    }

    #[test]
    fn test_benchmark_comparator() {
        let mut comparator = BenchmarkComparator::new();
        
        let result1 = BenchmarkResult::new("test1".to_string(), 5, &[10.0, 11.0, 9.0, 10.5, 9.5]);
        let result2 = BenchmarkResult::new("test2".to_string(), 5, &[8.0, 9.0, 7.0, 8.5, 7.5]);
        
        comparator.add(result1);
        comparator.add(result2);
        
        assert_eq!(comparator.results.len(), 2);
    }
}
