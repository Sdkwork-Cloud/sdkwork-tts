//! Speaker management for TTS engines
//!
//! Provides unified speaker management across different TTS engines.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::RwLock;

use serde::{Deserialize, Serialize};

use crate::core::error::{Result, TtsError};

/// Speaker manager for managing speakers across engines
pub struct SpeakerManager {
    /// Registered speakers
    speakers: RwLock<HashMap<String, SpeakerInfo>>,
    /// Speaker embeddings cache
    embeddings: RwLock<HashMap<String, Vec<f32>>>,
    /// Reference audio cache
    audio_cache: RwLock<HashMap<String, Vec<f32>>>,
}

impl SpeakerManager {
    /// Create a new speaker manager
    pub fn new() -> Self {
        Self {
            speakers: RwLock::new(HashMap::new()),
            embeddings: RwLock::new(HashMap::new()),
            audio_cache: RwLock::new(HashMap::new()),
        }
    }

    /// Register a speaker
    pub fn register(&self, speaker: SpeakerInfo) -> Result<()> {
        let id = speaker.id.clone();
        self.speakers.write().map_err(|_| TtsError::Internal {
            message: "Failed to acquire write lock".to_string(),
            location: Some("SpeakerManager::register".to_string()),
        })?.insert(id, speaker);
        Ok(())
    }

    /// Unregister a speaker
    pub fn unregister(&self, id: &str) -> Result<()> {
        self.speakers.write().map_err(|_| TtsError::Internal {
            message: "Failed to acquire write lock".to_string(),
            location: Some("SpeakerManager::unregister".to_string()),
        })?.remove(id);
        self.embeddings.write().map_err(|_| TtsError::Internal {
            message: "Failed to acquire write lock".to_string(),
            location: Some("SpeakerManager::unregister".to_string()),
        })?.remove(id);
        self.audio_cache.write().map_err(|_| TtsError::Internal {
            message: "Failed to acquire write lock".to_string(),
            location: Some("SpeakerManager::unregister".to_string()),
        })?.remove(id);
        Ok(())
    }

    /// Get speaker info
    pub fn get(&self, id: &str) -> Result<Option<SpeakerInfo>> {
        let speakers = self.speakers.read().map_err(|_| TtsError::Internal {
            message: "Failed to acquire read lock".to_string(),
            location: Some("SpeakerManager::get".to_string()),
        })?;
        Ok(speakers.get(id).cloned())
    }

    /// List all speakers
    pub fn list(&self) -> Result<Vec<SpeakerInfo>> {
        let speakers = self.speakers.read().map_err(|_| TtsError::Internal {
            message: "Failed to acquire read lock".to_string(),
            location: Some("SpeakerManager::list".to_string()),
        })?;
        Ok(speakers.values().cloned().collect())
    }

    /// Load speakers from directory
    pub fn load_from_directory<P: AsRef<Path>>(&self, path: P) -> Result<Vec<SpeakerInfo>> {
        let path = path.as_ref();
        let mut loaded = Vec::new();

        if !path.exists() {
            return Ok(loaded);
        }

        for entry in std::fs::read_dir(path).map_err(|e| TtsError::Io {
            message: format!("Failed to read directory: {}", e),
            path: Some(path.to_path_buf()),
        })? {
            let entry = entry.map_err(|e| TtsError::Io {
                message: format!("Failed to read entry: {}", e),
                path: Some(path.to_path_buf()),
            })?;

            let speaker_path = entry.path();
            if speaker_path.extension().map(|e| e == "wav").unwrap_or(false) {
                let speaker_id = speaker_path
                    .file_stem()
                    .map(|s| s.to_string_lossy().to_string())
                    .unwrap_or_default();

                let speaker = SpeakerInfo {
                    id: speaker_id.clone(),
                    name: speaker_id.clone(),
                    language: "unknown".to_string(),
                    gender: None,
                    description: None,
                    preview_audio: Some(speaker_path.clone()),
                };

                self.register(speaker.clone())?;
                loaded.push(speaker);
            }
        }

        Ok(loaded)
    }

    /// Cache speaker embedding
    pub fn cache_embedding(&self, id: &str, embedding: Vec<f32>) -> Result<()> {
        self.embeddings.write().map_err(|_| TtsError::Internal {
            message: "Failed to acquire write lock".to_string(),
            location: Some("SpeakerManager::cache_embedding".to_string()),
        })?.insert(id.to_string(), embedding);
        Ok(())
    }

    /// Get cached embedding
    pub fn get_embedding(&self, id: &str) -> Result<Option<Vec<f32>>> {
        let embeddings = self.embeddings.read().map_err(|_| TtsError::Internal {
            message: "Failed to acquire read lock".to_string(),
            location: Some("SpeakerManager::get_embedding".to_string()),
        })?;
        Ok(embeddings.get(id).cloned())
    }

    /// Cache reference audio
    pub fn cache_audio(&self, id: &str, samples: Vec<f32>) -> Result<()> {
        self.audio_cache.write().map_err(|_| TtsError::Internal {
            message: "Failed to acquire write lock".to_string(),
            location: Some("SpeakerManager::cache_audio".to_string()),
        })?.insert(id.to_string(), samples);
        Ok(())
    }

    /// Get cached audio
    pub fn get_audio(&self, id: &str) -> Result<Option<Vec<f32>>> {
        let audio = self.audio_cache.read().map_err(|_| TtsError::Internal {
            message: "Failed to acquire read lock".to_string(),
            location: Some("SpeakerManager::get_audio".to_string()),
        })?;
        Ok(audio.get(id).cloned())
    }

    /// Clear all caches
    pub fn clear_cache(&self) -> Result<()> {
        self.embeddings.write().map_err(|_| TtsError::Internal {
            message: "Failed to acquire write lock".to_string(),
            location: Some("SpeakerManager::clear_cache".to_string()),
        })?.clear();
        self.audio_cache.write().map_err(|_| TtsError::Internal {
            message: "Failed to acquire write lock".to_string(),
            location: Some("SpeakerManager::clear_cache".to_string()),
        })?.clear();
        Ok(())
    }

    /// Get statistics
    pub fn stats(&self) -> SpeakerStats {
        let speakers_count = self.speakers.read()
            .map(|s| s.len())
            .unwrap_or(0);
        let embeddings_count = self.embeddings.read()
            .map(|e| e.len())
            .unwrap_or(0);
        let audio_count = self.audio_cache.read()
            .map(|a| a.len())
            .unwrap_or(0);

        SpeakerStats {
            total_speakers: speakers_count,
            cached_embeddings: embeddings_count,
            cached_audio: audio_count,
        }
    }
}

impl Default for SpeakerManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Speaker statistics
#[derive(Debug, Clone)]
pub struct SpeakerStats {
    pub total_speakers: usize,
    pub cached_embeddings: usize,
    pub cached_audio: usize,
}

/// Speaker information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeakerInfo {
    /// Speaker ID
    pub id: String,
    /// Speaker name
    pub name: String,
    /// Language
    pub language: String,
    /// Gender
    pub gender: Option<String>,
    /// Description
    pub description: Option<String>,
    /// Preview audio path
    pub preview_audio: Option<PathBuf>,
}

/// Speaker database for persistent storage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeakerDatabase {
    /// Database version
    pub version: String,
    /// Speakers
    pub speakers: Vec<SpeakerInfo>,
}

impl SpeakerDatabase {
    /// Create a new database
    pub fn new() -> Self {
        Self {
            version: "1.0".to_string(),
            speakers: Vec::new(),
        }
    }

    /// Load from file
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let content = std::fs::read_to_string(path.as_ref()).map_err(|e| TtsError::Io {
            message: format!("Failed to read speaker database: {}", e),
            path: Some(path.as_ref().to_path_buf()),
        })?;

        serde_json::from_str(&content).map_err(|e| TtsError::Internal {
            message: format!("Failed to parse speaker database: {}", e),
            location: None,
        })
    }

    /// Save to file
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let content = serde_json::to_string_pretty(self).map_err(|e| TtsError::Internal {
            message: format!("Failed to serialize speaker database: {}", e),
            location: None,
        })?;

        std::fs::write(path.as_ref(), content).map_err(|e| TtsError::Io {
            message: format!("Failed to write speaker database: {}", e),
            path: Some(path.as_ref().to_path_buf()),
        })?;

        Ok(())
    }

    /// Add speaker
    pub fn add(&mut self, speaker: SpeakerInfo) {
        self.speakers.push(speaker);
    }

    /// Find by ID
    pub fn find(&self, id: &str) -> Option<&SpeakerInfo> {
        self.speakers.iter().find(|s| s.id == id)
    }

    /// Find by name
    pub fn find_by_name(&self, name: &str) -> Option<&SpeakerInfo> {
        self.speakers.iter().find(|s| s.name == name)
    }

    /// Filter by language
    pub fn filter_by_language(&self, language: &str) -> Vec<&SpeakerInfo> {
        self.speakers.iter().filter(|s| s.language == language).collect()
    }
}

impl Default for SpeakerDatabase {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_speaker_manager_new() {
        let manager = SpeakerManager::new();
        let stats = manager.stats();
        assert_eq!(stats.total_speakers, 0);
    }

    #[test]
    fn test_speaker_manager_register() {
        let manager = SpeakerManager::new();
        let speaker = SpeakerInfo {
            id: "test".to_string(),
            name: "Test Speaker".to_string(),
            language: "en".to_string(),
            gender: Some("neutral".to_string()),
            description: None,
            preview_audio: None,
        };

        manager.register(speaker).unwrap();
        let stats = manager.stats();
        assert_eq!(stats.total_speakers, 1);
    }

    #[test]
    fn test_speaker_database() {
        let mut db = SpeakerDatabase::new();
        db.add(SpeakerInfo {
            id: "spk1".to_string(),
            name: "Speaker 1".to_string(),
            language: "en".to_string(),
            gender: None,
            description: None,
            preview_audio: None,
        });

        assert_eq!(db.speakers.len(), 1);
        assert!(db.find("spk1").is_some());
    }
}
