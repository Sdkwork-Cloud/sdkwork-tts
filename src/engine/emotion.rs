//! Emotion management for TTS engines
//!
//! Provides unified emotion/style control across different TTS engines.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::RwLock;

use serde::{Deserialize, Serialize};

use crate::core::error::{Result, TtsError};

/// Emotion manager for managing emotions across engines
pub struct EmotionManager {
    /// Registered emotions
    emotions: RwLock<HashMap<String, EmotionInfo>>,
    /// Emotion vectors cache
    vectors: RwLock<HashMap<String, Vec<f32>>>,
    /// Emotion categories
    categories: RwLock<HashMap<String, Vec<String>>>,
}

impl EmotionManager {
    /// Create a new emotion manager
    pub fn new() -> Self {
        let mut categories = HashMap::new();
        
        categories.insert("basic".to_string(), vec![
            "happy".to_string(),
            "sad".to_string(),
            "angry".to_string(),
            "fear".to_string(),
            "surprise".to_string(),
            "disgust".to_string(),
            "neutral".to_string(),
        ]);

        categories.insert("valence".to_string(), vec![
            "positive".to_string(),
            "negative".to_string(),
            "neutral".to_string(),
        ]);

        Self {
            emotions: RwLock::new(HashMap::new()),
            vectors: RwLock::new(HashMap::new()),
            categories: RwLock::new(categories),
        }
    }

    /// Register an emotion
    pub fn register(&self, emotion: EmotionInfo) -> Result<()> {
        let id = emotion.id.clone();
        self.emotions.write().map_err(|_| TtsError::Internal {
            message: "Failed to acquire write lock".to_string(),
            location: Some("EmotionManager::register".to_string()),
        })?.insert(id, emotion);
        Ok(())
    }

    /// Unregister an emotion
    pub fn unregister(&self, id: &str) -> Result<()> {
        self.emotions.write().map_err(|_| TtsError::Internal {
            message: "Failed to acquire write lock".to_string(),
            location: Some("EmotionManager::unregister".to_string()),
        })?.remove(id);
        self.vectors.write().map_err(|_| TtsError::Internal {
            message: "Failed to acquire write lock".to_string(),
            location: Some("EmotionManager::unregister".to_string()),
        })?.remove(id);
        Ok(())
    }

    /// Get emotion info
    pub fn get(&self, id: &str) -> Result<Option<EmotionInfo>> {
        let emotions = self.emotions.read().map_err(|_| TtsError::Internal {
            message: "Failed to acquire read lock".to_string(),
            location: Some("EmotionManager::get".to_string()),
        })?;
        Ok(emotions.get(id).cloned())
    }

    /// List all emotions
    pub fn list(&self) -> Result<Vec<EmotionInfo>> {
        let emotions = self.emotions.read().map_err(|_| TtsError::Internal {
            message: "Failed to acquire read lock".to_string(),
            location: Some("EmotionManager::list".to_string()),
        })?;
        Ok(emotions.values().cloned().collect())
    }

    /// Get emotions by category
    pub fn get_by_category(&self, category: &str) -> Result<Vec<EmotionInfo>> {
        let categories = self.categories.read().map_err(|_| TtsError::Internal {
            message: "Failed to acquire read lock".to_string(),
            location: Some("EmotionManager::get_by_category".to_string()),
        })?;

        let emotion_ids = categories.get(category).cloned().unwrap_or_default();
        let emotions = self.emotions.read().map_err(|_| TtsError::Internal {
            message: "Failed to acquire read lock".to_string(),
            location: Some("EmotionManager::get_by_category".to_string()),
        })?;

        Ok(emotion_ids
            .iter()
            .filter_map(|id| emotions.get(id).cloned())
            .collect())
    }

    /// List categories
    pub fn list_categories(&self) -> Result<Vec<String>> {
        let categories = self.categories.read().map_err(|_| TtsError::Internal {
            message: "Failed to acquire read lock".to_string(),
            location: Some("EmotionManager::list_categories".to_string()),
        })?;
        Ok(categories.keys().cloned().collect())
    }

    /// Cache emotion vector
    pub fn cache_vector(&self, id: &str, vector: Vec<f32>) -> Result<()> {
        self.vectors.write().map_err(|_| TtsError::Internal {
            message: "Failed to acquire write lock".to_string(),
            location: Some("EmotionManager::cache_vector".to_string()),
        })?.insert(id.to_string(), vector);
        Ok(())
    }

    /// Get cached vector
    pub fn get_vector(&self, id: &str) -> Result<Option<Vec<f32>>> {
        let vectors = self.vectors.read().map_err(|_| TtsError::Internal {
            message: "Failed to acquire read lock".to_string(),
            location: Some("EmotionManager::get_vector".to_string()),
        })?;
        Ok(vectors.get(id).cloned())
    }

    /// Blend multiple emotion vectors
    pub fn blend_vectors(&self, emotions: &[(String, f32)]) -> Result<Vec<f32>> {
        if emotions.is_empty() {
            return Ok(vec![0.0; 8]);
        }

        let vectors = self.vectors.read().map_err(|_| TtsError::Internal {
            message: "Failed to acquire read lock".to_string(),
            location: Some("EmotionManager::blend_vectors".to_string()),
        })?;

        let dim = vectors.values().next().map(|v| v.len()).unwrap_or(8);
        let mut result = vec![0.0f32; dim];
        let mut total_weight = 0.0f32;

        for (id, weight) in emotions {
            if let Some(vector) = vectors.get(id) {
                for (i, v) in vector.iter().enumerate() {
                    if i < dim {
                        result[i] += v * weight;
                    }
                }
                total_weight += weight;
            }
        }

        if total_weight > 0.0 {
            for v in &mut result {
                *v /= total_weight;
            }
        }

        Ok(result)
    }

    /// Initialize with default emotions
    pub fn initialize_defaults(&self) -> Result<()> {
        let defaults = vec![
            EmotionInfo {
                id: "neutral".to_string(),
                name: "Neutral".to_string(),
                description: Some("Neutral emotion".to_string()),
                intensity_range: (0.0, 1.0),
                preview_audio: None,
            },
            EmotionInfo {
                id: "happy".to_string(),
                name: "Happy".to_string(),
                description: Some("Happy, cheerful emotion".to_string()),
                intensity_range: (0.0, 1.0),
                preview_audio: None,
            },
            EmotionInfo {
                id: "sad".to_string(),
                name: "Sad".to_string(),
                description: Some("Sad, melancholic emotion".to_string()),
                intensity_range: (0.0, 1.0),
                preview_audio: None,
            },
            EmotionInfo {
                id: "angry".to_string(),
                name: "Angry".to_string(),
                description: Some("Angry, aggressive emotion".to_string()),
                intensity_range: (0.0, 1.0),
                preview_audio: None,
            },
            EmotionInfo {
                id: "fear".to_string(),
                name: "Fear".to_string(),
                description: Some("Fearful, anxious emotion".to_string()),
                intensity_range: (0.0, 1.0),
                preview_audio: None,
            },
            EmotionInfo {
                id: "surprise".to_string(),
                name: "Surprise".to_string(),
                description: Some("Surprised, amazed emotion".to_string()),
                intensity_range: (0.0, 1.0),
                preview_audio: None,
            },
            EmotionInfo {
                id: "disgust".to_string(),
                name: "Disgust".to_string(),
                description: Some("Disgusted emotion".to_string()),
                intensity_range: (0.0, 1.0),
                preview_audio: None,
            },
        ];

        for emotion in defaults {
            self.register(emotion)?;
        }

        Ok(())
    }
}

impl Default for EmotionManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Emotion information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmotionInfo {
    /// Emotion ID
    pub id: String,
    /// Emotion name
    pub name: String,
    /// Description
    pub description: Option<String>,
    /// Intensity range
    pub intensity_range: (f32, f32),
    /// Preview audio path
    pub preview_audio: Option<PathBuf>,
}

/// Emotion vector with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmotionVector {
    /// Vector values
    pub values: Vec<f32>,
    /// Dimension labels
    pub labels: Vec<String>,
    /// Source emotion
    pub source: Option<String>,
    /// Confidence
    pub confidence: f32,
}

impl EmotionVector {
    /// Create a new emotion vector
    pub fn new(values: Vec<f32>) -> Self {
        let dim = values.len();
        Self {
            values,
            labels: (0..dim).map(|i| format!("dim_{}", i)).collect(),
            source: None,
            confidence: 1.0,
        }
    }

    /// Create with labels
    pub fn with_labels(values: Vec<f32>, labels: Vec<String>) -> Self {
        Self {
            values,
            labels,
            source: None,
            confidence: 1.0,
        }
    }

    /// Get dimension count
    pub fn dim(&self) -> usize {
        self.values.len()
    }

    /// Normalize vector
    pub fn normalize(&mut self) {
        let sum: f32 = self.values.iter().map(|v| v.abs()).sum();
        if sum > 0.0 {
            for v in &mut self.values {
                *v /= sum;
            }
        }
    }

    /// Scale by intensity
    pub fn scale(&mut self, intensity: f32) {
        for v in &mut self.values {
            *v *= intensity;
        }
    }

    /// Blend with another vector
    pub fn blend(&mut self, other: &EmotionVector, weight: f32) {
        let min_len = self.values.len().min(other.values.len());
        for i in 0..min_len {
            self.values[i] = self.values[i] * (1.0 - weight) + other.values[i] * weight;
        }
    }
}

/// Emotion dimension labels (standard 8-dimensional model)
pub const EMOTION_DIMENSIONS: &[&str] = &[
    "happiness",
    "sadness",
    "anger",
    "fear",
    "surprise",
    "disgust",
    "neutral",
    "other",
];

/// Create default emotion vector
pub fn default_emotion_vector() -> Vec<f32> {
    vec![0.0; EMOTION_DIMENSIONS.len()]
}

/// Parse emotion vector from string
pub fn parse_emotion_vector(s: &str) -> Result<Vec<f32>> {
    let values: Vec<f32> = s
        .split(',')
        .map(|v| v.trim().parse::<f32>())
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(|e| TtsError::Validation {
            message: format!("Invalid emotion vector: {}", e),
            field: Some("emotion_vector".to_string()),
        })?;

    if values.len() != EMOTION_DIMENSIONS.len() {
        return Err(TtsError::Validation {
            message: format!(
                "Emotion vector must have {} dimensions, got {}",
                EMOTION_DIMENSIONS.len(),
                values.len()
            ),
            field: Some("emotion_vector".to_string()),
        });
    }

    Ok(values)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_emotion_manager_new() {
        let manager = EmotionManager::new();
        let emotions = manager.list().unwrap();
        assert!(emotions.is_empty());
    }

    #[test]
    fn test_emotion_manager_register() {
        let manager = EmotionManager::new();
        let emotion = EmotionInfo {
            id: "test".to_string(),
            name: "Test Emotion".to_string(),
            description: None,
            intensity_range: (0.0, 1.0),
            preview_audio: None,
        };

        manager.register(emotion).unwrap();
        let emotions = manager.list().unwrap();
        assert_eq!(emotions.len(), 1);
    }

    #[test]
    fn test_emotion_vector() {
        let mut vec = EmotionVector::new(vec![0.5, 0.3, 0.2]);
        assert_eq!(vec.dim(), 3);

        vec.normalize();
        let sum: f32 = vec.values.iter().map(|v| v.abs()).sum();
        assert!((sum - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_parse_emotion_vector() {
        let vec = parse_emotion_vector("0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8").unwrap();
        assert_eq!(vec.len(), 8);
        assert_eq!(vec[0], 0.1);
        assert_eq!(vec[7], 0.8);
    }

    #[test]
    fn test_initialize_defaults() {
        let manager = EmotionManager::new();
        manager.initialize_defaults().unwrap();
        let emotions = manager.list().unwrap();
        assert_eq!(emotions.len(), 7);
    }
}
