//! Model Loader
//!
//! Handles loading models from HuggingFace/ModelScope standard directories.
//! Models are stored in:
//! - ModelScope (default for China): ~/.cache/modelscope/hub/{org}/{model}/
//! - HuggingFace: ~/.cache/huggingface/hub/models--{org}--{model}/snapshots/main/
//!
//! # Default Behavior
//!
//! For users in China, ModelScope is preferred as it provides faster download speeds.
//! Set `SDKWORK_TTS_HUB=huggingface` to prefer HuggingFace instead.
//!
//! # Model Aliases
//!
//! Short names can be used instead of full model IDs:
//! - `indextts` or `indextts2` → `IndexTeam/IndexTTS-2`

use std::path::PathBuf;

use crate::core::error::{Result, TtsError};

/// Default model ID for IndexTTS2
pub const DEFAULT_MODEL_ID: &str = "IndexTeam/IndexTTS-2";

/// Model aliases for convenience
pub const MODEL_ALIASES: &[(&str, &str)] = &[
    ("indextts", "IndexTeam/IndexTTS-2"),
    ("indextts2", "IndexTeam/IndexTTS-2"),
    ("IndexTTS", "IndexTeam/IndexTTS-2"),
    ("IndexTTS-2", "IndexTeam/IndexTTS-2"),
];

/// Resolve model alias to full model ID
pub fn resolve_model_id(model_id: &str) -> &str {
    // Check if it's an alias
    for (alias, full_id) in MODEL_ALIASES {
        if model_id.eq_ignore_ascii_case(alias) {
            return full_id;
        }
    }
    // Return as-is if not an alias
    model_id
}

/// Get HuggingFace cache directory
/// 
/// Priority:
/// 1. `HF_HOME` environment variable
/// 2. `HUGGINGFACE_HUB_CACHE` environment variable  
/// 3. `~/.cache/huggingface/hub/` (default, same as Python huggingface_hub)
pub fn get_hf_cache_dir() -> Result<PathBuf> {
    if let Ok(hf_home) = std::env::var("HF_HOME") {
        return Ok(PathBuf::from(hf_home).join("hub"));
    }
    
    if let Ok(cache) = std::env::var("HUGGINGFACE_HUB_CACHE") {
        return Ok(PathBuf::from(cache));
    }
    
    // Use home directory + .cache (same as Python's os.path.expanduser('~/.cache'))
    dirs::home_dir()
        .map(|p| p.join(".cache").join("huggingface").join("hub"))
        .ok_or_else(|| TtsError::Internal {
            message: "Cannot determine HuggingFace cache directory".to_string(),
            location: Some("model_loader::get_hf_cache_dir".to_string()),
        })
}

/// Get ModelScope cache directory
/// 
/// Priority:
/// 1. `MODELSCOPE_CACHE` environment variable
/// 2. `~/.cache/modelscope/hub/` (default, same as Python modelscope)
pub fn get_modelscope_cache_dir() -> Result<PathBuf> {
    if let Ok(cache) = std::env::var("MODELSCOPE_CACHE") {
        return Ok(PathBuf::from(cache));
    }
    
    // Use home directory + .cache (same as Python's os.path.expanduser('~/.cache'))
    dirs::home_dir()
        .map(|p| p.join(".cache").join("modelscope").join("hub"))
        .ok_or_else(|| TtsError::Internal {
            message: "Cannot determine ModelScope cache directory".to_string(),
            location: Some("model_loader::get_modelscope_cache_dir".to_string()),
        })
}

/// Convert model ID to HuggingFace cache directory name
pub fn model_id_to_hf_cache_name(model_id: &str) -> String {
    format!("models--{}", model_id.replace('/', "--"))
}

/// Find model in HuggingFace cache
pub fn find_in_hf_cache(model_id: &str) -> Result<Option<PathBuf>> {
    let cache_dir = get_hf_cache_dir()?;
    let model_dir = cache_dir.join(model_id_to_hf_cache_name(model_id));
    
    if !model_dir.exists() {
        return Ok(None);
    }
    
    // Check for snapshots/main or snapshots/master
    let snapshots_dir = model_dir.join("snapshots");
    if snapshots_dir.exists() {
        for revision in &["main", "master"] {
            let revision_dir = snapshots_dir.join(revision);
            if revision_dir.exists() && has_model_files(&revision_dir) {
                return Ok(Some(revision_dir));
            }
        }
    }
    
    // Check for blobs directly (some setups)
    if has_model_files(&model_dir) {
        return Ok(Some(model_dir));
    }
    
    Ok(None)
}

/// Find model in ModelScope cache
pub fn find_in_modelscope_cache(model_id: &str) -> Result<Option<PathBuf>> {
    let cache_dir = get_modelscope_cache_dir()?;
    let model_dir = cache_dir.join(model_id);
    
    if model_dir.exists() && has_model_files(&model_dir) {
        return Ok(Some(model_dir));
    }
    
    Ok(None)
}

/// Check if directory has model files
fn has_model_files(path: &PathBuf) -> bool {
    if !path.is_dir() {
        return false;
    }
    
    let model_extensions = [".pth", ".bin", ".safetensors", ".pt"];
    
    if let Ok(entries) = std::fs::read_dir(path) {
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().to_string();
            if model_extensions.iter().any(|ext| name.ends_with(ext)) {
                return true;
            }
        }
    }
    
    false
}

/// Find model in any cache (ModelScope first for China users, then HuggingFace)
pub fn find_model(model_id: &str) -> Result<Option<PathBuf>> {
    let model_id = resolve_model_id(model_id);
    
    // Check environment variable for preferred hub
    let prefer_modelscope = std::env::var("SDKWORK_TTS_HUB")
        .map(|v| v.to_lowercase() != "huggingface")
        .unwrap_or(true); // Default to ModelScope for China users

    if prefer_modelscope {
        // Try ModelScope first
        if let Some(path) = find_in_modelscope_cache(model_id)? {
            return Ok(Some(path));
        }
        // Fallback to HuggingFace
        if let Some(path) = find_in_hf_cache(model_id)? {
            return Ok(Some(path));
        }
    } else {
        // Try HuggingFace first
        if let Some(path) = find_in_hf_cache(model_id)? {
            return Ok(Some(path));
        }
        // Fallback to ModelScope
        if let Some(path) = find_in_modelscope_cache(model_id)? {
            return Ok(Some(path));
        }
    }
    
    Ok(None)
}

/// Get model path or return error with download instructions
pub fn get_model_or_error(model_id: &str) -> Result<PathBuf> {
    let resolved_id = resolve_model_id(model_id);
    
    if let Some(path) = find_model(model_id)? {
        return Ok(path);
    }
    
    Err(TtsError::Internal {
        message: format!(
            "Model '{}' not found in standard directories.\n\n\
             Please download the model first:\n\n\
             # Using ModelScope CLI (recommended for China):\n\
             modelscope download --model {} --local_dir ~/.cache/modelscope/hub/{}\n\n\
             # Using HuggingFace CLI:\n\
             huggingface-cli download {}\n\n\
             # Environment variables:\n\
             $env:SDKWORK_TTS_HUB = 'modelscope'  # or 'huggingface'\n\
             $env:MODELSCOPE_CACHE = 'D:\\models\\modelscope'\n\
             $env:HF_HOME = 'D:\\models\\huggingface'",
            model_id, resolved_id, resolved_id, resolved_id
        ),
        location: Some("model_loader::get_model_or_error".to_string()),
    })
}

/// Get default IndexTTS2 model path
pub fn get_default_indextts2_path() -> Result<PathBuf> {
    get_model_or_error(DEFAULT_MODEL_ID)
}

/// List all cached models
pub fn list_cached_models() -> Result<Vec<CachedModel>> {
    let mut models = Vec::new();
    
    // HuggingFace models
    let hf_cache = get_hf_cache_dir()?;
    if hf_cache.exists() {
        if let Ok(entries) = std::fs::read_dir(&hf_cache) {
            for entry in entries.flatten() {
                let name = entry.file_name().to_string_lossy().to_string();
                if name.starts_with("models--") {
                    let model_id = name
                        .strip_prefix("models--")
                        .unwrap_or(&name)
                        .replace("--", "/");
                    
                    if let Some(path) = find_in_hf_cache(&model_id)? {
                        models.push(CachedModel {
                            model_id,
                            path,
                            source: "HuggingFace".to_string(),
                        });
                    }
                }
            }
        }
    }
    
    // ModelScope models
    let ms_cache = get_modelscope_cache_dir()?;
    if ms_cache.exists() {
        if let Ok(entries) = std::fs::read_dir(&ms_cache) {
            for entry in entries.flatten() {
                if entry.path().is_dir() {
                    let namespace = entry.file_name().to_string_lossy().to_string();
                    
                    if let Ok(model_entries) = std::fs::read_dir(entry.path()) {
                        for model_entry in model_entries.flatten() {
                            let model_name = model_entry.file_name().to_string_lossy().to_string();
                            let model_id = format!("{}/{}", namespace, model_name);
                            
                            if let Some(path) = find_in_modelscope_cache(&model_id)? {
                                models.push(CachedModel {
                                    model_id,
                                    path,
                                    source: "ModelScope".to_string(),
                                });
                            }
                        }
                    }
                }
            }
        }
    }
    
    Ok(models)
}

/// Cached model information
#[derive(Debug, Clone)]
pub struct CachedModel {
    /// Model ID (e.g., "IndexTeam/IndexTTS-2")
    pub model_id: String,
    /// Local path
    pub path: PathBuf,
    /// Source ("HuggingFace" or "ModelScope")
    pub source: String,
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
        println!("HF cache dir: {:?}", cache_dir.unwrap());
    }

    #[test]
    fn test_get_modelscope_cache_dir() {
        let cache_dir = get_modelscope_cache_dir();
        assert!(cache_dir.is_ok());
        let path = cache_dir.unwrap();
        println!("ModelScope cache dir: {:?}", path);
        
        // Check if model exists
        let model_dir = path.join("IndexTeam/IndexTTS-2");
        println!("Model dir exists: {}", model_dir.exists());
        if model_dir.exists() {
            println!("Model dir: {:?}", model_dir);
            if let Ok(entries) = std::fs::read_dir(&model_dir) {
                for entry in entries.flatten() {
                    println!("  - {:?}", entry.file_name());
                }
            }
        }
    }

    #[test]
    fn test_find_model_debug() {
        let model_id = "IndexTeam/IndexTTS-2";
        
        // Check ModelScope
        let ms_result = find_in_modelscope_cache(model_id);
        println!("ModelScope result: {:?}", ms_result);
        
        // Check HF
        let hf_result = find_in_hf_cache(model_id);
        println!("HF result: {:?}", hf_result);
        
        // Check combined
        let result = find_model(model_id);
        println!("Combined result: {:?}", result);
    }
}
