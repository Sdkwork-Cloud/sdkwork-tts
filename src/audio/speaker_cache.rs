//! Speaker Embedding Cache
//!
//! Provides caching for speaker embeddings to improve performance
//! by avoiding redundant computation.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use candle_core::Tensor;
use dashmap::DashMap;

use crate::core::error::Result;

/// Cache entry for speaker embeddings
#[derive(Clone)]
pub struct SpeakerEmbeddingEntry {
    /// The embedding tensor
    pub embedding: Arc<Tensor>,
    /// When this entry was created
    pub created_at: Instant,
    /// Last access time
    pub last_accessed: Instant,
    /// Access count
    pub access_count: u64,
    /// Source audio path hash
    pub source_hash: u64,
}

impl SpeakerEmbeddingEntry {
    /// Create a new cache entry
    pub fn new(embedding: Tensor, source_hash: u64) -> Self {
        let now = Instant::now();
        Self {
            embedding: Arc::new(embedding),
            created_at: now,
            last_accessed: now,
            access_count: 0,
            source_hash,
        }
    }

    /// Record an access
    pub fn record_access(&mut self) {
        self.access_count += 1;
        self.last_accessed = Instant::now();
    }

    /// Get the age of this entry
    pub fn age(&self) -> Duration {
        self.created_at.elapsed()
    }

    /// Get time since last access
    pub fn idle_time(&self) -> Duration {
        self.last_accessed.elapsed()
    }
}

/// Speaker embedding cache configuration
#[derive(Debug, Clone)]
pub struct SpeakerCacheConfig {
    /// Maximum number of entries
    pub max_entries: usize,
    /// Time-to-live for entries
    pub ttl: Duration,
    /// Idle timeout before eviction
    pub idle_timeout: Duration,
    /// Enable/disable cache
    pub enabled: bool,
}

impl Default for SpeakerCacheConfig {
    fn default() -> Self {
        Self {
            max_entries: 100,
            ttl: Duration::from_secs(3600), // 1 hour
            idle_timeout: Duration::from_secs(300), // 5 minutes
            enabled: true,
        }
    }
}

/// Speaker embedding cache for performance optimization
///
/// Caches speaker embeddings to avoid redundant computation when
/// the same speaker audio is used multiple times.
///
/// # Example
///
/// ```rust,ignore
/// let cache = SpeakerEmbeddingCache::new(SpeakerCacheConfig::default());
///
/// // Get or compute embedding
/// let embedding = cache.get_or_compute(
///     &speaker_path,
///     || compute_embedding(&speaker_path),
/// )?;
/// ```
pub struct SpeakerEmbeddingCache {
    /// Cache storage
    entries: DashMap<PathBuf, SpeakerEmbeddingEntry>,
    /// Cache configuration
    config: SpeakerCacheConfig,
    /// Statistics
    hits: std::sync::atomic::AtomicU64,
    misses: std::sync::atomic::AtomicU64,
    evictions: std::sync::atomic::AtomicU64,
}

impl SpeakerEmbeddingCache {
    /// Create a new speaker embedding cache
    pub fn new(config: SpeakerCacheConfig) -> Self {
        Self {
            entries: DashMap::new(),
            config,
            hits: std::sync::atomic::AtomicU64::new(0),
            misses: std::sync::atomic::AtomicU64::new(0),
            evictions: std::sync::atomic::AtomicU64::new(0),
        }
    }

    /// Create a cache with default config
    pub fn with_defaults() -> Self {
        Self::new(SpeakerCacheConfig::default())
    }

    /// Create a disabled cache (no caching)
    pub fn disabled() -> Self {
        let config = SpeakerCacheConfig {
            enabled: false,
            ..Default::default()
        };
        Self::new(config)
    }

    /// Get the cache configuration
    pub fn config(&self) -> &SpeakerCacheConfig {
        &self.config
    }

    /// Enable/disable caching
    pub fn set_enabled(&mut self, enabled: bool) {
        self.config.enabled = enabled;
    }

    /// Check if an entry exists
    pub fn contains(&self, speaker_path: &Path) -> bool {
        if !self.config.enabled {
            return false;
        }
        self.entries.contains_key(speaker_path)
    }

    /// Get an embedding from cache
    pub fn get(&self, speaker_path: &Path) -> Option<Arc<Tensor>> {
        if !self.config.enabled {
            return None;
        }

        let mut entry = self.entries.get_mut(speaker_path)?;
        
        // Check TTL
        if entry.age() > self.config.ttl {
            drop(entry);
            self.entries.remove(speaker_path);
            self.evictions.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return None;
        }

        // Record access
        entry.record_access();
        self.hits.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        Some(entry.embedding.clone())
    }

    /// Insert an embedding into cache
    pub fn insert(&self, speaker_path: PathBuf, embedding: Tensor) -> Result<()> {
        if !self.config.enabled {
            return Ok(());
        }

        // Compute hash of source
        let source_hash = compute_path_hash(&speaker_path);

        // Check if we need to evict
        if self.entries.len() >= self.config.max_entries {
            self.evict_if_needed();
        }

        let entry = SpeakerEmbeddingEntry::new(embedding, source_hash);
        self.entries.insert(speaker_path, entry);
        self.misses.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        Ok(())
    }

    /// Get or compute an embedding
    pub fn get_or_compute<F, E>(
        &self,
        speaker_path: &Path,
        compute: F,
    ) -> std::result::Result<Arc<Tensor>, E>
    where
        F: FnOnce() -> std::result::Result<Tensor, E>,
    {
        // Try cache first
        if let Some(embedding) = self.get(speaker_path) {
            return Ok(embedding);
        }

        // Compute and cache
        let embedding = compute()?;
        let embedding_arc = Arc::new(embedding.clone());
        
        // Insert into cache (ignore errors)
        let _ = self.insert(speaker_path.to_path_buf(), embedding);

        Ok(embedding_arc)
    }

    /// Remove an entry
    pub fn remove(&self, speaker_path: &Path) -> Option<SpeakerEmbeddingEntry> {
        self.entries.remove(speaker_path).map(|(_, v)| v)
    }

    /// Clear all entries
    pub fn clear(&self) {
        self.entries.clear();
    }

    /// Evict old entries if needed
    fn evict_if_needed(&self) {
        let mut to_evict = Vec::new();

        // Find entries to evict
        for entry in self.entries.iter() {
            let key = entry.key();
            let value = entry.value();

            // Evict if TTL exceeded
            if value.age() > self.config.ttl {
                to_evict.push(key.clone());
                continue;
            }

            // Evict if idle too long
            if value.idle_time() > self.config.idle_timeout {
                to_evict.push(key.clone());
                continue;
            }
        }

        // If still over limit, evict LRU entries
        if to_evict.is_empty() && self.entries.len() >= self.config.max_entries {
            // Find LRU entry
            let mut lru: Option<(PathBuf, Duration)> = None;
            for entry in self.entries.iter() {
                let idle = entry.value().idle_time();
                match &lru {
                    None => lru = Some((entry.key().clone(), idle)),
                    Some((_, best_idle)) if idle > *best_idle => {
                        lru = Some((entry.key().clone(), idle));
                    }
                    _ => {}
                }
            }
            if let Some((key, _)) = lru {
                to_evict.push(key);
            }
        }

        // Evict
        for key in to_evict {
            self.entries.remove(&key);
            self.evictions.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
    }

    /// Get cache statistics
    pub fn stats(&self) -> SpeakerCacheStats {
        SpeakerCacheStats {
            entries: self.entries.len(),
            hits: self.hits.load(std::sync::atomic::Ordering::Relaxed),
            misses: self.misses.load(std::sync::atomic::Ordering::Relaxed),
            evictions: self.evictions.load(std::sync::atomic::Ordering::Relaxed),
            max_entries: self.config.max_entries,
            enabled: self.config.enabled,
        }
    }
}

impl Default for SpeakerEmbeddingCache {
    fn default() -> Self {
        Self::with_defaults()
    }
}

/// Cache statistics
#[derive(Debug, Clone)]
pub struct SpeakerCacheStats {
    /// Number of cached entries
    pub entries: usize,
    /// Cache hits
    pub hits: u64,
    /// Cache misses
    pub misses: u64,
    /// Evicted entries
    pub evictions: u64,
    /// Maximum entries
    pub max_entries: usize,
    /// Whether cache is enabled
    pub enabled: bool,
}

impl SpeakerCacheStats {
    /// Get hit rate
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }
}

/// Compute hash of a path
fn compute_path_hash(path: &Path) -> u64 {
    let mut hasher = DefaultHasher::new();
    path.hash(&mut hasher);
    hasher.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_basic() {
        let cache = SpeakerEmbeddingCache::with_defaults();
        
        // Create a dummy tensor
        let tensor = Tensor::zeros((1, 192), candle_core::DType::F32, &candle_core::Device::Cpu).unwrap();
        let path = PathBuf::from("test.wav");
        
        // Insert
        cache.insert(path.clone(), tensor.clone()).unwrap();
        
        // Get
        let retrieved = cache.get(&path);
        assert!(retrieved.is_some());
        
        // Stats
        let stats = cache.stats();
        assert_eq!(stats.entries, 1);
        assert_eq!(stats.hits, 1);
    }

    #[test]
    fn test_cache_disabled() {
        let cache = SpeakerEmbeddingCache::disabled();
        
        let tensor = Tensor::zeros((1, 192), candle_core::DType::F32, &candle_core::Device::Cpu).unwrap();
        let path = PathBuf::from("test.wav");
        
        // Insert should be no-op
        cache.insert(path.clone(), tensor).unwrap();
        
        // Get should return None
        assert!(cache.get(&path).is_none());
    }

    #[test]
    fn test_cache_stats() {
        let cache = SpeakerEmbeddingCache::with_defaults();
        
        let tensor = Tensor::zeros((1, 192), candle_core::DType::F32, &candle_core::Device::Cpu).unwrap();
        let path = PathBuf::from("test.wav");
        
        cache.insert(path.clone(), tensor.clone()).unwrap();
        cache.get(&path);
        cache.get(&path);
        
        let stats = cache.stats();
        assert_eq!(stats.hits, 2);
        assert!(stats.hit_rate() > 0.0);
    }
}
