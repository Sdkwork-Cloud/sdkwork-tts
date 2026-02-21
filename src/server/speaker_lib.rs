//! Speaker Library Management
//!
//! Manages local and cloud speaker libraries

use crate::server::types::{SpeakerInfo, SpeakerSource, Gender, AgeRange};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};

/// Speaker library
pub struct SpeakerLibrary {
    /// Local speakers
    local_speakers: Arc<RwLock<HashMap<String, SpeakerEntry>>>,
    /// Cloud speakers (cached)
    cloud_speakers: Arc<RwLock<HashMap<String, SpeakerEntry>>>,
    /// Library path
    library_path: PathBuf,
    /// Max cache size
    max_cache_size: usize,
}

/// Speaker entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeakerEntry {
    /// Speaker info
    pub info: SpeakerInfo,
    
    /// Audio samples (paths or URLs)
    pub samples: Vec<SpeakerSample>,
    
    /// Embeddings (for local speakers)
    #[serde(skip)]
    pub embeddings: Option<SpeakerEmbedding>,
    
    /// Metadata
    pub metadata: SpeakerMetadata,
}

/// Speaker sample
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeakerSample {
    /// Sample ID
    pub id: String,
    
    /// Audio path or URL
    pub audio: String,
    
    /// Text content
    pub text: Option<String>,
    
    /// Duration (seconds)
    pub duration: Option<f32>,
    
    /// Quality score (0.0 - 1.0)
    pub quality: Option<f32>,
}

/// Speaker embedding
#[derive(Debug, Clone)]
pub struct SpeakerEmbedding {
    /// Embedding vector
    pub vector: Vec<f32>,

    /// Dimension
    pub dimension: usize,

    /// Model used
    pub model: String,
}

/// Speaker metadata
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SpeakerMetadata {
    /// Creation timestamp
    pub created_at: Option<String>,
    
    /// Last update timestamp
    pub updated_at: Option<String>,
    
    /// Usage count
    pub usage_count: u64,
    
    /// Last used timestamp
    pub last_used_at: Option<String>,
    
    /// Tags
    pub tags: Vec<String>,
    
    /// Custom attributes
    pub attributes: HashMap<String, String>,
}

impl SpeakerLibrary {
    /// Create new speaker library
    pub fn new<P: AsRef<Path>>(library_path: P, max_cache_size: usize) -> Self {
        Self {
            local_speakers: Arc::new(RwLock::new(HashMap::new())),
            cloud_speakers: Arc::new(RwLock::new(HashMap::new())),
            library_path: library_path.as_ref().to_path_buf(),
            max_cache_size,
        }
    }
    
    /// Load library from disk
    pub fn load(&self) -> Result<(), Box<dyn std::error::Error>> {
        // Create library directory if not exists
        if !self.library_path.exists() {
            std::fs::create_dir_all(&self.library_path)?;
        }
        
        // Load local speakers
        let speakers_dir = self.library_path.join("speakers");
        if speakers_dir.exists() {
            for entry in std::fs::read_dir(&speakers_dir)? {
                let entry = entry?;
                let path = entry.path();
                
                if path.extension().is_some_and(|ext| ext == "json") {
                    let content = std::fs::read_to_string(&path)?;
                    let speaker: SpeakerEntry = serde_json::from_str(&content)?;
                    
                    let mut local = self.local_speakers.write().unwrap();
                    local.insert(speaker.info.id.clone(), speaker);
                }
            }
        }
        
        Ok(())
    }
    
    /// Add local speaker
    pub fn add_speaker(&self, speaker: SpeakerEntry) -> Result<(), String> {
        let mut local = self.local_speakers.write().unwrap();
        
        if local.contains_key(&speaker.info.id) {
            return Err(format!("Speaker {} already exists", speaker.info.id));
        }
        
        // Check cache size
        if local.len() >= self.max_cache_size {
            // Remove least recently used
            // (simplified - in production, use LRU cache)
        }
        
        // Save to disk
        let speakers_dir = self.library_path.join("speakers");
        std::fs::create_dir_all(&speakers_dir)
            .map_err(|e| format!("Failed to create speakers directory: {}", e))?;
        
        let speaker_path = speakers_dir.join(format!("{}.json", speaker.info.id));
        let content = serde_json::to_string_pretty(&speaker)
            .map_err(|e| format!("Failed to serialize speaker: {}", e))?;
        
        std::fs::write(&speaker_path, content)
            .map_err(|e| format!("Failed to save speaker: {}", e))?;
        
        local.insert(speaker.info.id.clone(), speaker);
        
        Ok(())
    }
    
    /// Remove speaker
    pub fn remove_speaker(&self, speaker_id: &str) -> Result<(), String> {
        let mut local = self.local_speakers.write().unwrap();
        
        if !local.contains_key(speaker_id) {
            return Err(format!("Speaker {} not found", speaker_id));
        }
        
        // Remove from disk
        let speaker_path = self.library_path.join("speakers").join(format!("{}.json", speaker_id));
        if speaker_path.exists() {
            std::fs::remove_file(&speaker_path)
                .map_err(|e| format!("Failed to remove speaker file: {}", e))?;
        }
        
        local.remove(speaker_id);
        
        Ok(())
    }
    
    /// Get speaker by ID
    pub fn get_speaker(&self, speaker_id: &str) -> Option<SpeakerEntry> {
        // Check local first
        {
            let local = self.local_speakers.read().unwrap();
            if let Some(speaker) = local.get(speaker_id) {
                return Some(speaker.clone());
            }
        }
        
        // Check cloud cache
        {
            let cloud = self.cloud_speakers.read().unwrap();
            if let Some(speaker) = cloud.get(speaker_id) {
                return Some(speaker.clone());
            }
        }
        
        None
    }
    
    /// List all speakers
    pub fn list_speakers(&self) -> Vec<SpeakerInfo> {
        let mut speakers = Vec::new();
        
        // Local speakers
        {
            let local = self.local_speakers.read().unwrap();
            for entry in local.values() {
                speakers.push(entry.info.clone());
            }
        }
        
        // Cloud speakers
        {
            let cloud = self.cloud_speakers.read().unwrap();
            for entry in cloud.values() {
                speakers.push(entry.info.clone());
            }
        }
        
        speakers
    }
    
    /// List speakers with filters
    pub fn list_speakers_filtered(
        &self,
        gender: Option<Gender>,
        age: Option<AgeRange>,
        language: Option<&str>,
        source: Option<&SpeakerSource>,
    ) -> Vec<SpeakerInfo> {
        let all_speakers = self.list_speakers();
        
        all_speakers.into_iter().filter(|s| {
            // Filter by gender
            if let Some(g) = gender {
                if s.gender != Some(g) {
                    return false;
                }
            }
            
            // Filter by age
            if let Some(a) = age {
                if s.age != Some(a) {
                    return false;
                }
            }
            
            // Filter by language
            if let Some(lang) = language {
                if !s.languages.contains(&lang.to_string()) {
                    return false;
                }
            }
            
            // Filter by source
            if let Some(src) = source {
                if &s.source != src {
                    return false;
                }
            }
            
            true
        }).collect()
    }
    
    /// Add cloud speakers
    pub fn add_cloud_speakers(&self, speakers: Vec<SpeakerEntry>, channel: &str) {
        let mut cloud = self.cloud_speakers.write().unwrap();
        
        for mut speaker in speakers {
            speaker.info.source = SpeakerSource::Cloud {
                channel: channel.to_string(),
            };
            
            cloud.insert(speaker.info.id.clone(), speaker);
        }
    }
    
    /// Get speaker count
    pub fn count(&self) -> usize {
        let local = self.local_speakers.read().unwrap();
        let cloud = self.cloud_speakers.read().unwrap();
        local.len() + cloud.len()
    }
    
    /// Get local speaker count
    pub fn local_count(&self) -> usize {
        let local = self.local_speakers.read().unwrap();
        local.len()
    }
    
    /// Get cloud speaker count
    pub fn cloud_count(&self) -> usize {
        let cloud = self.cloud_speakers.read().unwrap();
        cloud.len()
    }
    
    /// Search speakers
    pub fn search(&self, query: &str) -> Vec<SpeakerInfo> {
        let query_lower = query.to_lowercase();
        let all_speakers = self.list_speakers();
        
        all_speakers.into_iter().filter(|s| {
            s.name.to_lowercase().contains(&query_lower) ||
            s.description.as_ref().is_some_and(|d| d.to_lowercase().contains(&query_lower)) ||
            s.tags.iter().any(|t| t.to_lowercase().contains(&query_lower))
        }).collect()
    }
    
    /// Update speaker usage
    pub fn record_usage(&self, speaker_id: &str) {
        let mut local = self.local_speakers.write().unwrap();
        
        if let Some(speaker) = local.get_mut(speaker_id) {
            speaker.metadata.usage_count += 1;
            speaker.metadata.last_used_at = Some(chrono::Utc::now().to_rfc3339());
        }
    }
    
    /// Clear cloud cache
    pub fn clear_cloud_cache(&self) {
        let mut cloud = self.cloud_speakers.write().unwrap();
        cloud.clear();
    }
}

impl Default for SpeakerLibrary {
    fn default() -> Self {
        Self::new("speaker_library", 1000)
    }
}

/// Speaker library builder
pub struct SpeakerLibraryBuilder {
    library_path: PathBuf,
    max_cache_size: usize,
    auto_load: bool,
}

impl SpeakerLibraryBuilder {
    pub fn new() -> Self {
        Self {
            library_path: PathBuf::from("speaker_library"),
            max_cache_size: 1000,
            auto_load: true,
        }
    }
    
    pub fn path<P: AsRef<Path>>(mut self, path: P) -> Self {
        self.library_path = path.as_ref().to_path_buf();
        self
    }
    
    pub fn max_cache_size(mut self, size: usize) -> Self {
        self.max_cache_size = size;
        self
    }
    
    pub fn auto_load(mut self, load: bool) -> Self {
        self.auto_load = load;
        self
    }
    
    pub fn build(self) -> SpeakerLibrary {
        let library = SpeakerLibrary::new(self.library_path, self.max_cache_size);
        
        if self.auto_load {
            let _ = library.load();
        }
        
        library
    }
}

impl Default for SpeakerLibraryBuilder {
    fn default() -> Self {
        Self::new()
    }
}
