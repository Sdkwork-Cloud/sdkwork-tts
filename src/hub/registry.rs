//! Model Registry
//!
//! Central registry for all supported TTS models with their sources,
//! versions, and metadata.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::core::error::{Result, TtsError};
use super::HubType;

/// Model source information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelSource {
    /// Hub type
    pub hub: HubType,
    /// Model ID on the hub (e.g., "IndexTeam/IndexTTS-2")
    pub model_id: String,
    /// Revision/branch (default: "main")
    pub revision: Option<String>,
    /// Required files to download
    pub required_files: Vec<String>,
    /// Optional files
    pub optional_files: Vec<String>,
}

/// Model information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    /// Unique model identifier (e.g., "indextts2")
    pub id: String,
    /// Display name
    pub name: String,
    /// Description
    pub description: String,
    /// Engine type
    pub engine: String,
    /// Model version
    pub version: String,
    /// Model size (parameters)
    pub size: String,
    /// Supported languages
    pub languages: Vec<String>,
    /// Available sources
    pub sources: Vec<ModelSource>,
    /// License
    pub license: String,
    /// Author
    pub author: String,
    /// Homepage URL
    pub homepage: Option<String>,
    /// Paper URL
    pub paper: Option<String>,
    /// Minimum disk space required (bytes)
    pub min_disk_space: u64,
    /// Minimum RAM required (bytes)
    pub min_ram: u64,
    /// GPU memory required (bytes, 0 for CPU-only)
    pub min_gpu_memory: u64,
}

impl ModelInfo {
    /// Get primary source (first available)
    pub fn primary_source(&self) -> Option<&ModelSource> {
        self.sources.first()
    }

    /// Get source by hub type
    pub fn source_by_hub(&self, hub: HubType) -> Option<&ModelSource> {
        self.sources.iter().find(|s| s.hub == hub)
    }

    /// Get all required files from primary source
    pub fn all_required_files(&self) -> Vec<&String> {
        self.primary_source()
            .map(|s| s.required_files.iter().collect())
            .unwrap_or_default()
    }
}

/// Model registry
pub struct ModelRegistry {
    /// Registered models
    models: HashMap<String, ModelInfo>,
}

impl ModelRegistry {
    /// Create a new empty registry
    pub fn new() -> Self {
        Self {
            models: HashMap::new(),
        }
    }

    /// Create registry with default models
    pub fn default_registry() -> Result<Self> {
        let mut registry = Self::new();
        registry.register_defaults();
        Ok(registry)
    }

    /// Register default models
    fn register_defaults(&mut self) {
        // IndexTTS2
        self.register(ModelInfo {
            id: "indextts2".to_string(),
            name: "IndexTTS2".to_string(),
            description: "Bilibili's Industrial-Level Controllable and Efficient Zero-Shot Text-To-Speech System".to_string(),
            engine: "indextts2".to_string(),
            version: "2.0".to_string(),
            size: "1.2B".to_string(),
            languages: vec!["zh".to_string(), "en".to_string(), "ja".to_string()],
            sources: vec![
                ModelSource {
                    hub: HubType::HuggingFace,
                    model_id: "IndexTeam/IndexTTS-2".to_string(),
                    revision: Some("main".to_string()),
                    required_files: vec![
                        "config.yaml".to_string(),
                        "gpt.pth".to_string(),
                        "s2mel.pth".to_string(),
                        "vocoder.pth".to_string(),
                        "semantic_encoder.pth".to_string(),
                        "speaker_encoder.pth".to_string(),
                    ],
                    optional_files: vec![
                        "tokenizer.json".to_string(),
                        "tokenizer.model".to_string(),
                    ],
                },
                ModelSource {
                    hub: HubType::ModelScope,
                    model_id: "IndexTeam/IndexTTS-2".to_string(),
                    revision: Some("master".to_string()),
                    required_files: vec![
                        "config.yaml".to_string(),
                        "gpt.pth".to_string(),
                        "s2mel.pth".to_string(),
                        "vocoder.pth".to_string(),
                        "semantic_encoder.pth".to_string(),
                        "speaker_encoder.pth".to_string(),
                    ],
                    optional_files: vec![],
                },
            ],
            license: "Apache-2.0".to_string(),
            author: "Bilibili IndexTeam".to_string(),
            homepage: Some("https://github.com/index-tts/index-tts".to_string()),
            paper: Some("https://arxiv.org/abs/2501.07595".to_string()),
            min_disk_space: 5 * 1024 * 1024 * 1024, // 5GB
            min_ram: 8 * 1024 * 1024 * 1024, // 8GB
            min_gpu_memory: 4 * 1024 * 1024 * 1024, // 4GB
        });

        // Fish-Speech
        self.register(ModelInfo {
            id: "fish-speech".to_string(),
            name: "Fish-Speech".to_string(),
            description: "Open-source TTS framework with multi-language support".to_string(),
            engine: "fish-speech".to_string(),
            version: "1.5".to_string(),
            size: "1.2B".to_string(),
            languages: vec!["zh".to_string(), "en".to_string(), "ja".to_string(), "ko".to_string(), "de".to_string(), "fr".to_string()],
            sources: vec![
                ModelSource {
                    hub: HubType::HuggingFace,
                    model_id: "fishaudio/fish-speech-1.5".to_string(),
                    revision: Some("main".to_string()),
                    required_files: vec![
                        "config.json".to_string(),
                        "model.pth".to_string(),
                        "tokenizer.json".to_string(),
                    ],
                    optional_files: vec![],
                },
                ModelSource {
                    hub: HubType::ModelScope,
                    model_id: "fishaudio/fish-speech-1.5".to_string(),
                    revision: Some("master".to_string()),
                    required_files: vec![
                        "config.json".to_string(),
                        "model.pth".to_string(),
                    ],
                    optional_files: vec![],
                },
            ],
            license: "Apache-2.0".to_string(),
            author: "Fish Audio".to_string(),
            homepage: Some("https://github.com/fishaudio/fish-speech".to_string()),
            paper: None,
            min_disk_space: 4 * 1024 * 1024 * 1024, // 4GB
            min_ram: 8 * 1024 * 1024 * 1024, // 8GB
            min_gpu_memory: 4 * 1024 * 1024 * 1024, // 4GB
        });

        // Qwen3-TTS
        self.register(ModelInfo {
            id: "qwen3-tts".to_string(),
            name: "Qwen3-TTS".to_string(),
            description: "Alibaba's powerful speech generation model with multi-language support".to_string(),
            engine: "qwen3-tts".to_string(),
            version: "1.7B".to_string(),
            size: "1.7B".to_string(),
            languages: vec!["zh".to_string(), "en".to_string(), "ja".to_string(), "ko".to_string(), "de".to_string(), "fr".to_string(), "ru".to_string(), "pt".to_string(), "es".to_string(), "it".to_string()],
            sources: vec![
                ModelSource {
                    hub: HubType::HuggingFace,
                    model_id: "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice".to_string(),
                    revision: Some("main".to_string()),
                    required_files: vec![
                        "config.json".to_string(),
                        "model.safetensors".to_string(),
                        "tokenizer.json".to_string(),
                    ],
                    optional_files: vec![],
                },
                ModelSource {
                    hub: HubType::ModelScope,
                    model_id: "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice".to_string(),
                    revision: Some("master".to_string()),
                    required_files: vec![
                        "config.json".to_string(),
                        "model.safetensors".to_string(),
                    ],
                    optional_files: vec![],
                },
            ],
            license: "Apache-2.0".to_string(),
            author: "Alibaba Cloud Qwen Team".to_string(),
            homepage: Some("https://github.com/QwenLM/Qwen3-TTS".to_string()),
            paper: None,
            min_disk_space: 6 * 1024 * 1024 * 1024, // 6GB
            min_ram: 16 * 1024 * 1024 * 1024, // 16GB
            min_gpu_memory: 8 * 1024 * 1024 * 1024, // 8GB
        });

        // Qwen3-TTS 0.6B (smaller variant)
        self.register(ModelInfo {
            id: "qwen3-tts-small".to_string(),
            name: "Qwen3-TTS Small".to_string(),
            description: "Lightweight version of Qwen3-TTS".to_string(),
            engine: "qwen3-tts".to_string(),
            version: "0.6B".to_string(),
            size: "0.6B".to_string(),
            languages: vec!["zh".to_string(), "en".to_string(), "ja".to_string(), "ko".to_string()],
            sources: vec![
                ModelSource {
                    hub: HubType::HuggingFace,
                    model_id: "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice".to_string(),
                    revision: Some("main".to_string()),
                    required_files: vec![
                        "config.json".to_string(),
                        "model.safetensors".to_string(),
                        "tokenizer.json".to_string(),
                    ],
                    optional_files: vec![],
                },
            ],
            license: "Apache-2.0".to_string(),
            author: "Alibaba Cloud Qwen Team".to_string(),
            homepage: Some("https://github.com/QwenLM/Qwen3-TTS".to_string()),
            paper: None,
            min_disk_space: 2 * 1024 * 1024 * 1024, // 2GB
            min_ram: 8 * 1024 * 1024 * 1024, // 8GB
            min_gpu_memory: 4 * 1024 * 1024 * 1024, // 4GB
        });

        // GPT-SoVITS (planned)
        self.register(ModelInfo {
            id: "gpt-sovits".to_string(),
            name: "GPT-SoVITS".to_string(),
            description: "Zero-shot TTS with style transfer".to_string(),
            engine: "gpt-sovits".to_string(),
            version: "1.0".to_string(),
            size: "1.0B".to_string(),
            languages: vec!["zh".to_string(), "en".to_string(), "ja".to_string()],
            sources: vec![
                ModelSource {
                    hub: HubType::HuggingFace,
                    model_id: "lj1995/GPT-SoVITS".to_string(),
                    revision: Some("main".to_string()),
                    required_files: vec![],
                    optional_files: vec![],
                },
                ModelSource {
                    hub: HubType::ModelScope,
                    model_id: "iic/GPT-SoVITS".to_string(),
                    revision: Some("master".to_string()),
                    required_files: vec![],
                    optional_files: vec![],
                },
            ],
            license: "MIT".to_string(),
            author: "RVC-Boss".to_string(),
            homepage: Some("https://github.com/RVC-Boss/GPT-SoVITS".to_string()),
            paper: None,
            min_disk_space: 4 * 1024 * 1024 * 1024, // 4GB
            min_ram: 8 * 1024 * 1024 * 1024, // 8GB
            min_gpu_memory: 4 * 1024 * 1024 * 1024, // 4GB
        });

        // ChatTTS (planned)
        self.register(ModelInfo {
            id: "chattts".to_string(),
            name: "ChatTTS".to_string(),
            description: "Conversational TTS model".to_string(),
            engine: "chattts".to_string(),
            version: "1.0".to_string(),
            size: "2.0B".to_string(),
            languages: vec!["zh".to_string(), "en".to_string()],
            sources: vec![
                ModelSource {
                    hub: HubType::HuggingFace,
                    model_id: "2Noise/ChatTTS".to_string(),
                    revision: Some("main".to_string()),
                    required_files: vec![],
                    optional_files: vec![],
                },
            ],
            license: "Apache-2.0".to_string(),
            author: "2Noise".to_string(),
            homepage: Some("https://github.com/2noise/ChatTTS".to_string()),
            paper: None,
            min_disk_space: 4 * 1024 * 1024 * 1024, // 4GB
            min_ram: 8 * 1024 * 1024 * 1024, // 8GB
            min_gpu_memory: 4 * 1024 * 1024 * 1024, // 4GB
        });
    }

    /// Register a model
    pub fn register(&mut self, model: ModelInfo) {
        self.models.insert(model.id.clone(), model);
    }

    /// Get model info
    pub fn get(&self, id: &str) -> Result<ModelInfo> {
        self.models.get(id).cloned().ok_or_else(|| TtsError::Config {
            message: format!("Model '{}' not found in registry", id),
            path: None,
        })
    }

    /// Get model info (optional)
    pub fn get_opt(&self, id: &str) -> Result<Option<ModelInfo>> {
        Ok(self.models.get(id).cloned())
    }

    /// List all models
    pub fn list(&self) -> Result<Vec<ModelInfo>> {
        Ok(self.models.values().cloned().collect())
    }

    /// List models by engine
    pub fn list_by_engine(&self, engine: &str) -> Result<Vec<ModelInfo>> {
        Ok(self.models.values()
            .filter(|m| m.engine == engine)
            .cloned()
            .collect())
    }

    /// Check if model exists
    pub fn exists(&self, id: &str) -> bool {
        self.models.contains_key(id)
    }

    /// Get model count
    pub fn count(&self) -> usize {
        self.models.len()
    }
}

impl Default for ModelRegistry {
    fn default() -> Self {
        Self::default_registry().expect("Failed to create default registry")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_registry_new() {
        let registry = ModelRegistry::new();
        assert_eq!(registry.count(), 0);
    }

    #[test]
    fn test_registry_defaults() {
        let registry = ModelRegistry::default_registry().unwrap();
        assert!(registry.count() >= 3);
        assert!(registry.exists("indextts2"));
        assert!(registry.exists("fish-speech"));
        assert!(registry.exists("qwen3-tts"));
    }

    #[test]
    fn test_get_model() {
        let registry = ModelRegistry::default_registry().unwrap();
        let model = registry.get("indextts2").unwrap();
        assert_eq!(model.id, "indextts2");
        assert_eq!(model.engine, "indextts2");
    }

    #[test]
    fn test_list_by_engine() {
        let registry = ModelRegistry::default_registry().unwrap();
        let models = registry.list_by_engine("qwen3-tts").unwrap();
        assert!(models.len() >= 2); // qwen3-tts and qwen3-tts-small
    }

    #[test]
    fn test_model_sources() {
        let registry = ModelRegistry::default_registry().unwrap();
        let model = registry.get("indextts2").unwrap();
        
        assert!(!model.sources.is_empty());
        
        let hf_source = model.source_by_hub(HubType::HuggingFace);
        assert!(hf_source.is_some());
        
        let ms_source = model.source_by_hub(HubType::ModelScope);
        assert!(ms_source.is_some());
    }
}
