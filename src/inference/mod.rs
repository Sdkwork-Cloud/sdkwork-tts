//! Inference module for text-to-speech synthesis
//!
//! This module provides the main entry point for TTS:
//! - IndexTTS2: Main inference pipeline
//! - Qwen3-TTS: Alibaba's 10-language TTS
//! - StreamingSynthesizer: Real-time audio streaming
//! - BatchProcessor: Batch inference support
//! - InferenceConfig: Runtime configuration
//! - ModelLoader: Load models from HuggingFace/ModelScope cache

mod model_loader;
mod pipeline;
mod streaming;
mod qwen3_tts;
mod batch;
pub mod optimized_index;

pub use model_loader::{
    get_hf_cache_dir, get_modelscope_cache_dir,
    find_model, find_in_hf_cache, find_in_modelscope_cache,
    get_model_or_error, get_default_indextts2_path,
    list_cached_models, CachedModel,
    resolve_model_id, DEFAULT_MODEL_ID, MODEL_ALIASES,
};
pub use pipeline::{IndexTTS2, InferenceConfig, InferenceResult, apply_high_pass};
pub use streaming::{StreamingSynthesizer, StreamingConfig, AudioChunk};
pub use qwen3_tts::{Qwen3Tts, QwenInferenceConfig, QwenInferenceResult, QwenModelVariant};
pub use batch::{BatchProcessor, BatchSynthesisRequest, BatchSynthesisResult, SimpleBatchProcessor, BatchStats};
pub use optimized_index::{
    OptimizedIndexEngine, OptimizedIndexConfig, OptimizedIndexBuilder,
    OptimizedKVCache, MemoryPool, IndexProfile,
};
