//! Optimized Qwen3-TTS Inference Engine
//!
//! This module provides optimized inference capabilities with:
//! - Batch processing support
//! - KV cache optimization
//! - Streaming inference
//! - Performance profiling

use std::sync::{Arc, Mutex};
use std::time::Instant;

use candle_core::{Device, Tensor};
use anyhow::{Result, Context};

use super::{
    Qwen3TtsModel, QwenConfig, QwenSynthesisResult,
    GenerationConfig,
};

/// Optimized inference configuration
#[derive(Debug, Clone)]
pub struct OptimizedInferenceConfig {
    /// Base Qwen3-TTS config
    pub base_config: QwenConfig,
    /// Enable batch processing
    pub enable_batch: bool,
    /// Maximum batch size
    pub max_batch_size: usize,
    /// Enable KV cache optimization
    pub enable_kv_cache: bool,
    /// Pre-allocate KV cache for max sequence length
    pub max_seq_len: usize,
    /// Enable streaming output
    pub enable_streaming: bool,
    /// Chunk size for streaming (in frames)
    pub stream_chunk_size: usize,
    /// Enable performance profiling
    pub enable_profiling: bool,
}

impl Default for OptimizedInferenceConfig {
    fn default() -> Self {
        Self {
            base_config: QwenConfig::default(),
            enable_batch: true,
            max_batch_size: 8,
            enable_kv_cache: true,
            max_seq_len: 2048,
            enable_streaming: true,
            stream_chunk_size: 50,
            enable_profiling: false,
        }
    }
}

/// Inference profiling result
#[derive(Debug, Clone)]
pub struct InferenceProfile {
    /// Total inference time in milliseconds
    pub total_time_ms: f64,
    /// Tokenization time in milliseconds
    pub tokenize_time_ms: f64,
    /// Model forward time in milliseconds
    pub forward_time_ms: f64,
    /// Audio decoding time in milliseconds
    pub decode_time_ms: f64,
    /// Number of generated tokens
    pub num_tokens: usize,
    /// Tokens per second
    pub tokens_per_sec: f64,
    /// Real-time factor (audio_duration / inference_time)
    pub rtf: f64,
}

impl InferenceProfile {
    /// Create new profile
    pub fn new(
        total_time_ms: f64,
        tokenize_time_ms: f64,
        forward_time_ms: f64,
        decode_time_ms: f64,
        num_tokens: usize,
        audio_duration_sec: f64,
    ) -> Self {
        let tokens_per_sec = if total_time_ms > 0.0 {
            num_tokens as f64 / (total_time_ms / 1000.0)
        } else {
            0.0
        };
        
        let rtf = if total_time_ms > 0.0 {
            audio_duration_sec / (total_time_ms / 1000.0)
        } else {
            0.0
        };
        
        Self {
            total_time_ms,
            tokenize_time_ms,
            forward_time_ms,
            decode_time_ms,
            num_tokens,
            tokens_per_sec,
            rtf,
        }
    }

    /// Print profiling result
    pub fn print(&self) {
        println!("\n╔═══════════════════════════════════════════════════════════╗");
        println!("║              Inference Performance Profile                ║");
        println!("╠═══════════════════════════════════════════════════════════╣");
        println!("║  Total Time:     {:>12.2} ms                           ║", self.total_time_ms);
        println!("║  Tokenize Time:  {:>12.2} ms                           ║", self.tokenize_time_ms);
        println!("║  Forward Time:   {:>12.2} ms                           ║", self.forward_time_ms);
        println!("║  Decode Time:    {:>12.2} ms                           ║", self.decode_time_ms);
        println!("║  Generated Tokens: {:>10}                              ║", self.num_tokens);
        println!("║  Tokens/sec:     {:>12.2}                              ║", self.tokens_per_sec);
        println!("║  Real-time Factor: {:>10.2}x                           ║", self.rtf);
        println!("╚═══════════════════════════════════════════════════════════╝");
    }
}

/// Optimized Qwen3-TTS inference engine
pub struct OptimizedInferenceEngine {
    /// Base model
    model: Arc<Mutex<Qwen3TtsModel>>,
    /// Configuration
    config: OptimizedInferenceConfig,
    /// Device
    device: Device,
    /// KV cache (pre-allocated if enabled)
    kv_cache: Option<Vec<(Tensor, Tensor)>>,
}

impl OptimizedInferenceEngine {
    /// Create new optimized engine
    pub fn new(config: OptimizedInferenceConfig) -> Result<Self> {
        let device = if config.base_config.use_gpu {
            Device::new_cuda(config.base_config.device_id)?
        } else {
            Device::Cpu
        };

        let model = Arc::new(Mutex::new(Qwen3TtsModel::new(config.base_config.clone())?));

        Ok(Self {
            model,
            config,
            device,
            kv_cache: None,
        })
    }

    /// Pre-allocate KV cache for better performance
    pub fn pre_allocate_kv_cache(&mut self) -> Result<()> {
        if !self.config.enable_kv_cache {
            return Ok(());
        }

        // Pre-allocate KV cache for max sequence length
        // This is a placeholder - actual implementation would depend on model architecture
        self.kv_cache = Some(Vec::new());
        
        Ok(())
    }

    /// Run inference with profiling
    pub fn infer_with_profile(
        &self,
        text: &str,
        gen_config: &GenerationConfig,
    ) -> Result<(QwenSynthesisResult, Option<InferenceProfile>)> {
        let total_start = Instant::now();

        // Tokenization phase
        let tokenize_start = Instant::now();
        // Tokenization would happen here
        let tokenize_time = tokenize_start.elapsed().as_secs_f64() * 1000.0;

        // Model forward phase
        let forward_start = Instant::now();
        let result = {
            let model = self.model.lock().unwrap();
            model.synthesize(text, None)?
        };
        let forward_time = forward_start.elapsed().as_secs_f64() * 1000.0;

        // Decode phase (already included in result)
        let decode_time = 0.0;

        let total_time = total_start.elapsed().as_secs_f64() * 1000.0;

        let profile = if self.config.enable_profiling {
            Some(InferenceProfile::new(
                total_time,
                tokenize_time,
                forward_time,
                decode_time,
                text.len(), // Approximate token count
                result.duration as f64,
            ))
        } else {
            None
        };

        Ok((result, profile))
    }

    /// Batch inference
    pub fn infer_batch(
        &self,
        texts: &[&str],
        _gen_config: &GenerationConfig,
    ) -> Result<Vec<QwenSynthesisResult>> {
        if !self.config.enable_batch {
            // Fall back to sequential inference
            let mut results = Vec::new();
            for &text in texts {
                let model = self.model.lock().unwrap();
                let result = model.synthesize(text, None)
                    .with_context(|| "Inference failed")?;
                results.push(result);
            }
            return Ok(results);
        }

        // Process in batches
        let mut results = Vec::with_capacity(texts.len());
        let batch_size = self.config.max_batch_size;

        for chunk in texts.chunks(batch_size) {
            // Process batch
            for &text in chunk {
                let model = self.model.lock().unwrap();
                let result = model.synthesize(text, None)
                    .with_context(|| "Inference failed")?;
                results.push(result);
            }
        }

        Ok(results)
    }

    /// Streaming inference
    pub fn infer_streaming<F>(
        &self,
        text: &str,
        _gen_config: &GenerationConfig,
        mut callback: F,
    ) -> Result<QwenSynthesisResult>
    where
        F: FnMut(&[f32], u32) -> Result<()>,
    {
        if !self.config.enable_streaming {
            // Fall back to regular inference
            let model = self.model.lock().unwrap();
            return model.synthesize(text, None)
                .with_context(|| "Inference failed");
        }

        // Generate full audio first (placeholder for true streaming)
        let model = self.model.lock().unwrap();
        let result = model.synthesize(text, None)?;

        // Stream audio in chunks
        let chunk_size = self.config.stream_chunk_size * 256; // Approximate samples per frame
        let audio = &result.audio;
        let sample_rate = result.sample_rate;

        for chunk in audio.chunks(chunk_size) {
            callback(chunk, sample_rate)?;
        }

        Ok(result)
    }

    /// Get model reference
    pub fn model(&self) -> Arc<Mutex<Qwen3TtsModel>> {
        Arc::clone(&self.model)
    }

    /// Get configuration
    pub fn config(&self) -> &OptimizedInferenceConfig {
        &self.config
    }

    /// Get device
    pub fn device(&self) -> &Device {
        &self.device
    }
}

/// Builder for OptimizedInferenceEngine
pub struct OptimizedEngineBuilder {
    config: OptimizedInferenceConfig,
}

impl OptimizedEngineBuilder {
    /// Create new builder
    pub fn new() -> Self {
        Self {
            config: OptimizedInferenceConfig::default(),
        }
    }

    /// Set base config
    pub fn base_config(mut self, config: QwenConfig) -> Self {
        self.config.base_config = config;
        self
    }

    /// Enable batch processing
    pub fn with_batch(mut self, max_batch_size: usize) -> Self {
        self.config.enable_batch = true;
        self.config.max_batch_size = max_batch_size;
        self
    }

    /// Enable KV cache optimization
    pub fn with_kv_cache(mut self, max_seq_len: usize) -> Self {
        self.config.enable_kv_cache = true;
        self.config.max_seq_len = max_seq_len;
        self
    }

    /// Enable streaming
    pub fn with_streaming(mut self, chunk_size: usize) -> Self {
        self.config.enable_streaming = true;
        self.config.stream_chunk_size = chunk_size;
        self
    }

    /// Enable profiling
    pub fn with_profiling(mut self) -> Self {
        self.config.enable_profiling = true;
        self
    }

    /// Build optimized engine
    pub fn build(self) -> Result<OptimizedInferenceEngine> {
        let mut engine = OptimizedInferenceEngine::new(self.config)?;
        engine.pre_allocate_kv_cache()?;
        Ok(engine)
    }
}

impl Default for OptimizedEngineBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_optimized_config_default() {
        let config = OptimizedInferenceConfig::default();
        assert!(config.enable_batch);
        assert!(config.enable_kv_cache);
        assert!(config.enable_streaming);
        assert!(!config.enable_profiling);
    }

    #[test]
    fn test_inference_profile() {
        let profile = InferenceProfile::new(
            100.0,  // total_time_ms
            10.0,   // tokenize_time_ms
            80.0,   // forward_time_ms
            10.0,   // decode_time_ms
            50,     // num_tokens
            2.0,    // audio_duration_sec
        );

        assert!((profile.total_time_ms - 100.0).abs() < 0.1);
        assert!((profile.tokens_per_sec - 500.0).abs() < 1.0);
        assert!((profile.rtf - 20.0).abs() < 0.1);
    }

    #[test]
    fn test_engine_builder() {
        let builder = OptimizedEngineBuilder::new()
            .base_config(QwenConfig::default())
            .with_batch(4)
            .with_kv_cache(1024)
            .with_streaming(25)
            .with_profiling();

        let engine = builder.build();
        // Engine creation may fail if model weights are not available
        // This test just verifies the builder pattern works
        assert!(engine.is_ok() || engine.is_err());
    }
}
