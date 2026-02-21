//! KV Cache for efficient autoregressive inference

use candle_core::{Device, Tensor};
use anyhow::Result;

/// KV Cache for a single transformer layer
pub struct KVCache {
    /// Key cache: [seq, num_kv_heads, head_dim]
    k_cache: Tensor,
    /// Value cache: [seq, num_kv_heads, head_dim]
    v_cache: Tensor,
    /// Current position in cache
    current_pos: usize,
}

impl KVCache {
    /// Create new KV cache with pre-allocated memory
    pub fn new(
        max_seq_len: usize,
        num_kv_heads: usize,
        head_dim: usize,
        device: &Device,
    ) -> Result<Self> {
        let k_cache = Tensor::zeros(
            (max_seq_len, num_kv_heads, head_dim),
            candle_core::DType::F32,
            device,
        )?;
        let v_cache = Tensor::zeros(
            (max_seq_len, num_kv_heads, head_dim),
            candle_core::DType::F32,
            device,
        )?;

        Ok(Self {
            k_cache,
            v_cache,
            current_pos: 0,
        })
    }

    /// Update cache with new key/value tensors (inplace operation)
    pub fn update(&mut self, k_val: &Tensor, v_val: &Tensor) -> Result<()> {
        let seq_len = k_val.dim(0)?;
        
        // Slice assign for inplace update
        self.k_cache = self.k_cache.slice_assign(
            &[self.current_pos..self.current_pos + seq_len, 0..self.k_cache.dim(1)?, 0..self.k_cache.dim(2)?],
            k_val,
        )?;
        self.v_cache = self.v_cache.slice_assign(
            &[self.current_pos..self.current_pos + seq_len, 0..self.v_cache.dim(1)?, 0..self.v_cache.dim(2)?],
            v_val,
        )?;
        
        self.current_pos += seq_len;
        Ok(())
    }

    /// Get all cached keys
    pub fn get_keys(&self) -> &Tensor {
        &self.k_cache
    }

    /// Get all cached values
    pub fn get_values(&self) -> &Tensor {
        &self.v_cache
    }

    /// Get current position
    pub fn position(&self) -> usize {
        self.current_pos
    }

    /// Reset cache position (for reuse)
    pub fn reset(&mut self) {
        self.current_pos = 0;
    }

    /// Get cached sequence length
    pub fn seq_len(&self) -> usize {
        self.current_pos
    }
}

/// Type alias for layer-specific KV cache
pub type LayerKVCache = KVCache;

/// Any KV cache type (for heterogeneous models)
pub enum AnyKVCache {
    /// Standard KV cache
    Standard(KVCache),
}

impl AnyKVCache {
    /// Create from standard KV cache
    pub fn from_standard(cache: KVCache) -> Self {
        Self::Standard(cache)
    }

    /// Get mutable reference to standard cache
    pub fn as_standard_mut(&mut self) -> Option<&mut KVCache> {
        match self {
            Self::Standard(cache) => Some(cache),
        }
    }
}

/// KV Cache manager for all transformer layers
pub struct KVCacheManager {
    /// Per-layer KV caches
    caches: Vec<KVCache>,
}

impl KVCacheManager {
    /// Create new cache manager
    pub fn new(
        num_layers: usize,
        max_seq_len: usize,
        num_kv_heads: usize,
        head_dim: usize,
        device: &Device,
    ) -> Result<Vec<KVCache>> {
        let mut caches = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            caches.push(KVCache::new(max_seq_len, num_kv_heads, head_dim, device)?);
        }
        Ok(caches)
    }

    /// Get mutable reference to layer cache
    pub fn get_layer(&mut self, layer_idx: usize) -> Option<&mut KVCache> {
        self.caches.get_mut(layer_idx)
    }

    /// Get all caches
    pub fn caches(&self) -> &[KVCache] {
        &self.caches
    }

    /// Reset all caches
    pub fn reset_all(&mut self) {
        for cache in &mut self.caches {
            cache.reset();
        }
    }
}
