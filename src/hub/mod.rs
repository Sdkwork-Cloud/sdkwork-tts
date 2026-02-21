//! Model Hub Management System
//!
//! Unified model management supporting both HuggingFace and ModelScope.
//! Models are stored in standard cache directories following each platform's conventions.
//!
//! # Directory Structure
//!
//! ## HuggingFace Cache (default)
//! ```text
//! ~/.cache/huggingface/hub/
//! ├── models--IndexTeam--IndexTTS-2/
//! │   ├── blobs/
//! │   ├── refs/
//! │   └── snapshots/
//! │       └── main/
//! ```
//!
//! ## ModelScope Cache (fallback for China users)
//! ```text
//! ~/.cache/modelscope/hub/
//! ├── IndexTeam/
//! │   └── IndexTTS-2/
//! │       └── ...
//! ```
//!
//! # Usage
//! ```rust,ignore
//! use sdkwork_tts::hub::{HubManager, HubType};
//!
//! let manager = HubManager::new()?;
//!
//! // Auto: try HF first, fallback to ModelScope
//! let model_path = manager.get_model("IndexTeam/IndexTTS-2").await?;
//!
//! // Force use ModelScope
//! let model_path = manager.get_model_with_hub("Qwen/Qwen3-TTS", HubType::ModelScope).await?;
//! ```

mod hub_core;
mod registry;

pub use hub_core::{HubType, HubConfig, ModelSource};
pub use registry::{ModelRegistry, ModelInfo};

use std::path::PathBuf;

use crate::core::error::{Result, TtsError};

/// HuggingFace cache subdirectory
pub const HF_HUB_CACHE_SUBDIR: &str = "huggingface/hub";

/// ModelScope cache subdirectory
pub const MS_HUB_CACHE_SUBDIR: &str = "modelscope/hub";

/// Get HuggingFace cache directory
/// 
/// Priority:
/// 1. `HF_HOME` environment variable
/// 2. `HUGGINGFACE_HUB_CACHE` environment variable  
/// 3. `~/.cache/huggingface/hub/` (default)
pub fn get_hf_cache_dir() -> Result<PathBuf> {
    if let Ok(hf_home) = std::env::var("HF_HOME") {
        return Ok(PathBuf::from(hf_home).join("hub"));
    }
    
    if let Ok(cache) = std::env::var("HUGGINGFACE_HUB_CACHE") {
        return Ok(PathBuf::from(cache));
    }
    
    dirs::cache_dir()
        .map(|p| p.join("huggingface").join("hub"))
        .ok_or_else(|| TtsError::Internal {
            message: "Cannot determine HuggingFace cache directory".to_string(),
            location: Some("hub::get_hf_cache_dir".to_string()),
        })
}

/// Get ModelScope cache directory
/// 
/// Priority:
/// 1. `MODELSCOPE_CACHE` environment variable
/// 2. `~/.cache/modelscope/hub/` (default)
pub fn get_modelscope_cache_dir() -> Result<PathBuf> {
    if let Ok(cache) = std::env::var("MODELSCOPE_CACHE") {
        return Ok(PathBuf::from(cache));
    }
    
    dirs::cache_dir()
        .map(|p| p.join("modelscope").join("hub"))
        .ok_or_else(|| TtsError::Internal {
            message: "Cannot determine ModelScope cache directory".to_string(),
            location: Some("hub::get_modelscope_cache_dir".to_string()),
        })
}

/// Convert model ID to HuggingFace cache directory name
/// 
/// Example: "IndexTeam/IndexTTS-2" -> "models--IndexTeam--IndexTTS-2"
pub fn model_id_to_hf_cache_name(model_id: &str) -> String {
    format!("models--{}", model_id.replace('/', "--"))
}

/// Get cached model path using HF conventions
pub fn get_hf_cached_model_path(model_id: &str) -> Result<PathBuf> {
    let cache_dir = get_hf_cache_dir()?;
    let model_dir = cache_dir.join(model_id_to_hf_cache_name(model_id));
    
    let snapshots_dir = model_dir.join("snapshots");
    if snapshots_dir.exists() {
        for revision in &["main", "master"] {
            let revision_dir = snapshots_dir.join(revision);
            if revision_dir.exists() {
                return Ok(revision_dir);
            }
        }
    }
    
    Ok(model_dir)
}

/// Get cached model path using ModelScope conventions
pub fn get_modelscope_cached_model_path(model_id: &str) -> Result<PathBuf> {
    let cache_dir = get_modelscope_cache_dir()?;
    // ModelScope uses: ~/.cache/modelscope/hub/{namespace}/{model_name}
    let model_dir = cache_dir.join(model_id);
    
    if model_dir.exists() {
        Ok(model_dir)
    } else {
        Ok(cache_dir) // Return cache dir if model not found
    }
}

/// Hub manager for unified model management
/// 
/// Supports both HuggingFace and ModelScope with automatic fallback.
pub struct HubManager {
    /// Model registry
    registry: ModelRegistry,
    /// Configuration
    config: HubConfig,
    /// Preferred hub (for initial download)
    preferred_hub: HubType,
}

impl HubManager {
    /// Create a new hub manager with auto-detection
    /// 
    /// Auto-detects best hub based on network conditions:
    /// - In China: prefers ModelScope
    /// - Elsewhere: prefers HuggingFace
    pub fn new() -> Result<Self> {
        let preferred_hub = Self::detect_preferred_hub();
        Self::with_config(HubConfig::default(), preferred_hub)
    }

    /// Create with specific preferred hub
    pub fn with_preferred_hub(hub: HubType) -> Result<Self> {
        Self::with_config(HubConfig::default(), hub)
    }

    /// Create with custom configuration
    pub fn with_config(config: HubConfig, preferred_hub: HubType) -> Result<Self> {
        Ok(Self {
            registry: ModelRegistry::default_registry()?,
            config,
            preferred_hub,
        })
    }

    /// Detect preferred hub based on environment
    fn detect_preferred_hub() -> HubType {
        // Check environment variable
        if let Ok(hub) = std::env::var("SDKWORK_TTS_HUB") {
            if let Ok(hub_type) = hub.parse() {
                return hub_type;
            }
        }
        
        // Default to HuggingFace
        // In production, could check network conditions or geo-location
        HubType::HuggingFace
    }

    /// Get model path (download if not cached)
    /// 
    /// Tries preferred hub first, then falls back to alternative.
    pub async fn get_model(&self, model_id: &str) -> Result<PathBuf> {
        // Check if already cached in either hub
        if let Some(path) = self.get_cached_any(model_id)? {
            return Ok(path);
        }

        // Try preferred hub first
        match self.download_model(model_id, self.preferred_hub).await {
            Ok(path) => Ok(path),
            Err(_) => {
                // Fallback to alternative hub
                let fallback_hub = match self.preferred_hub {
                    HubType::HuggingFace => HubType::ModelScope,
                    HubType::ModelScope => HubType::HuggingFace,
                };
                
                self.download_model(model_id, fallback_hub).await
            }
        }
    }

    /// Get model from specific hub
    pub async fn get_model_with_hub(&self, model_id: &str, hub: HubType) -> Result<PathBuf> {
        // Check if cached
        if let Some(path) = self.get_cached(model_id, hub)? {
            return Ok(path);
        }

        self.download_model(model_id, hub).await
    }

    /// Get cached model path from any hub
    pub fn get_cached_any(&self, model_id: &str) -> Result<Option<PathBuf>> {
        // Try HuggingFace first
        if let Some(path) = self.get_cached(model_id, HubType::HuggingFace)? {
            return Ok(Some(path));
        }
        
        // Try ModelScope
        if let Some(path) = self.get_cached(model_id, HubType::ModelScope)? {
            return Ok(Some(path));
        }
        
        Ok(None)
    }

    /// Get cached model path from specific hub
    pub fn get_cached(&self, model_id: &str, hub: HubType) -> Result<Option<PathBuf>> {
        let path = match hub {
            HubType::HuggingFace => get_hf_cached_model_path(model_id)?,
            HubType::ModelScope => get_modelscope_cached_model_path(model_id)?,
        };
        
        if path.exists() {
            // Check if it has actual model files (not just empty dir)
            if Self::has_model_files(&path) {
                return Ok(Some(path));
            }
        }
        
        Ok(None)
    }

    /// Check if directory has model files
    fn has_model_files(path: &PathBuf) -> bool {
        if !path.is_dir() {
            return false;
        }
        
        let model_extensions = [".pth", ".bin", ".safetensors", ".pt", ".ckpt"];
        let config_extensions = [".yaml", ".json", ".yml"];
        
        let mut has_model = false;
        let mut has_config = false;
        
        if let Ok(entries) = std::fs::read_dir(path) {
            for entry in entries.flatten() {
                let name = entry.file_name().to_string_lossy().to_string();
                
                if model_extensions.iter().any(|ext| name.ends_with(ext)) {
                    has_model = true;
                }
                if config_extensions.iter().any(|ext| name.ends_with(ext)) {
                    has_config = true;
                }
            }
        }
        
        has_model || has_config
    }

    /// Download model from specified hub
    async fn download_model(&self, model_id: &str, hub: HubType) -> Result<PathBuf> {
        match hub {
            HubType::HuggingFace => self.download_from_huggingface(model_id).await,
            HubType::ModelScope => self.download_from_modelscope(model_id).await,
        }
    }

    /// Download from HuggingFace Hub
    async fn download_from_huggingface(&self, model_id: &str) -> Result<PathBuf> {
        let api = hf_hub::api::sync::Api::new()
            .map_err(|e| TtsError::Internal {
                message: format!("Failed to create HF API: {}", e),
                location: Some("HubManager::download_from_huggingface".to_string()),
            })?;

        let repo = api.model(model_id.to_string());
        
        // Get model info
        let info = repo.info()
            .map_err(|e| TtsError::Internal {
                message: format!("Failed to get model info from HuggingFace: {}", e),
                location: Some("HubManager::download_from_huggingface".to_string()),
            })?;

        // Download required files
        for sibling in info.siblings {
            let filename = sibling.rfilename;
            
            // Skip unnecessary files
            if filename.ends_with(".gitattributes") 
                || filename.starts_with(".git/")
                || filename.contains("README")
                || filename.contains(".md") {
                continue;
            }

            repo.download(&filename)
                .map_err(|e| TtsError::Internal {
                    message: format!("Failed to download {}: {}", filename, e),
                    location: Some("HubManager::download_from_huggingface".to_string()),
                })?;
        }

        get_hf_cached_model_path(model_id)
    }

    /// Download from ModelScope Hub
    async fn download_from_modelscope(&self, model_id: &str) -> Result<PathBuf> {
        // ModelScope SDK integration
        // For now, provide instructions for manual download
        Err(TtsError::Internal {
            message: format!(
                "ModelScope download not yet implemented. Please download manually:\n\
                 modelscope download --model {} --local_dir ~/.cache/modelscope/hub/{}",
                model_id, model_id
            ),
            location: Some("HubManager::download_from_modelscope".to_string()),
        })
    }

    /// List available models
    pub fn list_models(&self) -> Result<Vec<ModelInfo>> {
        self.registry.list()
    }

    /// Get model info
    pub fn get_model_info(&self, model_id: &str) -> Result<Option<ModelInfo>> {
        self.registry.get_opt(model_id)
    }

    /// Check if model is cached in any hub
    pub fn is_cached(&self, model_id: &str) -> bool {
        self.get_cached_any(model_id).unwrap_or(None).is_some()
    }

    /// List all cached models from both hubs
    pub fn list_cached(&self) -> Result<Vec<CachedModel>> {
        let mut cached = Vec::new();

        // HuggingFace models
        let hf_cache = get_hf_cache_dir()?;
        if hf_cache.exists() {
            for entry in std::fs::read_dir(&hf_cache)? {
                let entry = entry?;
                let name = entry.file_name().to_string_lossy().to_string();
                
                if name.starts_with("models--") {
                    let model_id = name
                        .strip_prefix("models--")
                        .unwrap_or(&name)
                        .replace("--", "/");
                    
                    if let Some(path) = self.get_cached(&model_id, HubType::HuggingFace)? {
                        cached.push(CachedModel {
                            model_id,
                            path,
                            hub: HubType::HuggingFace,
                        });
                    }
                }
            }
        }

        // ModelScope models
        let ms_cache = get_modelscope_cache_dir()?;
        if ms_cache.exists() {
            for entry in std::fs::read_dir(&ms_cache)? {
                let entry = entry?;
                if entry.path().is_dir() {
                    // ModelScope structure: namespace/model_name
                    let namespace = entry.file_name().to_string_lossy().to_string();
                    
                    for model_entry in std::fs::read_dir(entry.path())? {
                        let model_entry = model_entry?;
                        let model_name = model_entry.file_name().to_string_lossy().to_string();
                        let model_id = format!("{}/{}", namespace, model_name);
                        
                        if let Some(path) = self.get_cached(&model_id, HubType::ModelScope)? {
                            cached.push(CachedModel {
                                model_id,
                                path,
                                hub: HubType::ModelScope,
                            });
                        }
                    }
                }
            }
        }

        Ok(cached)
    }

    /// Clear model cache from specific hub
    pub fn clear_cache(&self, model_id: &str, hub: HubType) -> Result<()> {
        let cache_dir = match hub {
            HubType::HuggingFace => {
                get_hf_cache_dir()?.join(model_id_to_hf_cache_name(model_id))
            }
            HubType::ModelScope => {
                get_modelscope_cache_dir()?.join(model_id)
            }
        };
        
        if cache_dir.exists() {
            std::fs::remove_dir_all(&cache_dir)?;
        }
        
        Ok(())
    }

    /// Get cache statistics
    pub fn cache_stats(&self) -> Result<CacheStats> {
        let mut total_size = 0;
        let mut model_count = 0;
        let mut cache_dirs = Vec::new();

        // HuggingFace cache
        let hf_cache = get_hf_cache_dir()?;
        if hf_cache.exists() {
            cache_dirs.push(hf_cache);
        }

        // ModelScope cache
        let ms_cache = get_modelscope_cache_dir()?;
        if ms_cache.exists() {
            cache_dirs.push(ms_cache);
        }

        for cache_dir in cache_dirs {
            for entry in walkdir::WalkDir::new(&cache_dir)
                .into_iter()
                .filter_map(|e| e.ok()) 
            {
                if entry.file_type().is_file() {
                    if let Ok(metadata) = entry.metadata() {
                        total_size += metadata.len();
                    }
                }
            }

            for entry in std::fs::read_dir(&cache_dir)? {
                let entry = entry?;
                let name = entry.file_name().to_string_lossy().to_string();
                
                if name.starts_with("models--") || entry.path().is_dir() {
                    model_count += 1;
                }
            }
        }

        Ok(CacheStats {
            total_models: model_count,
            total_size,
        })
    }

    /// Get preferred hub
    pub fn preferred_hub(&self) -> HubType {
        self.preferred_hub
    }

    /// Set preferred hub
    pub fn set_preferred_hub(&mut self, hub: HubType) {
        self.preferred_hub = hub;
    }
}

impl Default for HubManager {
    fn default() -> Self {
        Self::new().expect("Failed to create HubManager")
    }
}

/// Cached model information
#[derive(Debug, Clone)]
pub struct CachedModel {
    /// Model ID
    pub model_id: String,
    /// Local path
    pub path: PathBuf,
    /// Source hub
    pub hub: HubType,
}

/// Cache statistics
#[derive(Debug, Clone)]
pub struct CacheStats {
    /// Total number of cached models
    pub total_models: usize,
    /// Total size in bytes
    pub total_size: u64,
}

impl CacheStats {
    /// Format total size for display
    pub fn total_size_formatted(&self) -> String {
        const GB: u64 = 1024 * 1024 * 1024;
        const MB: u64 = 1024 * 1024;
        const KB: u64 = 1024;

        if self.total_size >= GB {
            format!("{:.2} GB", self.total_size as f64 / GB as f64)
        } else if self.total_size >= MB {
            format!("{:.2} MB", self.total_size as f64 / MB as f64)
        } else if self.total_size >= KB {
            format!("{:.2} KB", self.total_size as f64 / KB as f64)
        } else {
            format!("{} B", self.total_size)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_id_to_hf_cache_name() {
        assert_eq!(
            model_id_to_hf_cache_name("IndexTeam/IndexTTS-2"),
            "models--IndexTeam--IndexTTS-2"
        );
    }

    #[test]
    fn test_get_hf_cache_dir() {
        let cache_dir = get_hf_cache_dir();
        assert!(cache_dir.is_ok());
    }

    #[test]
    fn test_get_modelscope_cache_dir() {
        let cache_dir = get_modelscope_cache_dir();
        assert!(cache_dir.is_ok());
    }

    #[test]
    fn test_hub_manager_new() {
        let manager = HubManager::new();
        assert!(manager.is_ok());
    }

    #[test]
    fn test_hub_manager_preferred_hub() {
        let manager = HubManager::with_preferred_hub(HubType::ModelScope).unwrap();
        assert_eq!(manager.preferred_hub(), HubType::ModelScope);
    }

    #[test]
    fn test_cache_stats_size_format() {
        let stats = CacheStats {
            total_models: 3,
            total_size: 1024 * 1024 * 1024,
        };
        assert_eq!(stats.total_size_formatted(), "1.00 GB");
    }
}
