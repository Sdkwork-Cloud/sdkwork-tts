//! Voice Cloning Module
//!
//! Provides zero-shot voice cloning capabilities:
//! - Clone voice from 3-30 seconds of reference audio
//! - Support for audio file paths, URLs, and raw samples
//! - X-vector mode for audio-only cloning
//! - Reusable voice clone prompts

use std::path::PathBuf;
use crate::core::error::{Result, TtsError};
use crate::models::speaker::CAMPPlus;
use crate::audio::{AudioLoader, MelSpectrogram};
use candle_core::{Device, Tensor};

/// Voice cloning configuration
#[derive(Debug, Clone)]
pub struct VoiceCloneConfig {
    /// Reference audio path
    pub reference_audio: PathBuf,
    /// Reference audio transcription (optional, for better quality)
    pub reference_text: Option<String>,
    /// Use x-vector mode (audio only, no text required)
    pub x_vector_only: bool,
    /// Voice cloning strength (0.0 - 1.0)
    pub cloning_strength: f32,
    /// Blend with base speaker (0.0 = base, 1.0 = full clone)
    pub blend_alpha: f32,
}

impl Default for VoiceCloneConfig {
    fn default() -> Self {
        Self {
            reference_audio: PathBuf::new(),
            reference_text: None,
            x_vector_only: false,
            cloning_strength: 0.8,
            blend_alpha: 1.0,
        }
    }
}

impl VoiceCloneConfig {
    /// Create new config with reference audio
    pub fn new(reference_audio: impl Into<PathBuf>) -> Self {
        Self {
            reference_audio: reference_audio.into(),
            ..Default::default()
        }
    }

    /// Set reference text
    pub fn with_text(mut self, text: impl Into<String>) -> Self {
        self.reference_text = Some(text.into());
        self
    }

    /// Enable x-vector mode
    pub fn x_vector_mode(mut self, enable: bool) -> Self {
        self.x_vector_only = enable;
        self
    }

    /// Set cloning strength
    pub fn with_strength(mut self, strength: f32) -> Self {
        self.cloning_strength = strength.clamp(0.0, 1.0);
        self
    }

    /// Set blend alpha
    pub fn with_blend(mut self, alpha: f32) -> Self {
        self.blend_alpha = alpha.clamp(0.0, 1.0);
        self
    }
}

/// Voice clone prompt (reusable)
#[derive(Debug, Clone)]
pub struct VoiceClonePrompt {
    /// Prompt ID
    pub id: String,
    /// Speaker embedding
    pub speaker_embedding: Tensor,
    /// Reference mel spectrogram
    pub reference_mel: Tensor,
    /// Reference audio duration in seconds
    pub duration_secs: f32,
    /// Created timestamp
    pub created_at: std::time::Instant,
}

impl VoiceClonePrompt {
    /// Create new voice clone prompt
    pub fn new(
        id: impl Into<String>,
        speaker_embedding: Tensor,
        reference_mel: Tensor,
        duration_secs: f32,
    ) -> Self {
        Self {
            id: id.into(),
            speaker_embedding,
            reference_mel,
            duration_secs,
            created_at: std::time::Instant::now(),
        }
    }

    /// Get prompt age in seconds
    pub fn age(&self) -> f32 {
        self.created_at.elapsed().as_secs_f32()
    }
}

/// Voice cloning manager
pub struct VoiceCloningManager {
    /// Speaker encoder (CAMPPlus)
    speaker_encoder: CAMPPlus,
    /// Mel spectrogram extractor
    mel_extractor: MelSpectrogram,
    /// Cached voice prompts
    cached_prompts: std::collections::HashMap<String, VoiceClonePrompt>,
    /// Device
    device: Device,
}

impl VoiceCloningManager {
    /// Create new voice cloning manager
    pub fn new(
        speaker_encoder: CAMPPlus,
        mel_extractor: MelSpectrogram,
        device: Device,
    ) -> Self {
        Self {
            speaker_encoder,
            mel_extractor,
            cached_prompts: std::collections::HashMap::new(),
            device,
        }
    }

    /// Create voice clone from reference audio
    pub fn create_clone(&self, config: &VoiceCloneConfig) -> Result<VoiceClonePrompt> {
        // Load reference audio
        let (samples, sample_rate) = AudioLoader::load(&config.reference_audio, 16000)?;
        
        // Validate audio duration
        let duration_secs = samples.len() as f32 / sample_rate as f32;
        if duration_secs < 3.0 {
            return Err(TtsError::Validation {
                message: format!(
                    "Reference audio too short ({}s). Minimum 3 seconds required.",
                    duration_secs
                ),
                field: Some("reference_audio".to_string()),
            });
        }
        
        if duration_secs > 30.0 {
            return Err(TtsError::Validation {
                message: format!(
                    "Reference audio too long ({}s). Maximum 30 seconds supported.",
                    duration_secs
                ),
                field: Some("reference_audio".to_string()),
            });
        }

        // Compute mel spectrogram
        let mel_vec = self.mel_extractor.compute(&samples)?;
        
        // Flatten mel spectrogram and convert to tensor
        let mel_flat: Vec<f32> = mel_vec.iter().flatten().cloned().collect();
        let mel_rows = mel_vec.len();
        let mel_cols = if mel_rows > 0 { mel_vec[0].len() } else { 0 };
        let mel = Tensor::from_vec(mel_flat, (mel_rows, mel_cols), &self.device)?;

        // Extract speaker embedding - first convert samples to tensor
        let samples_tensor = Tensor::from_vec(samples.clone(), samples.len(), &self.device)?;
        let speaker_embedding = self.speaker_encoder.encode(&samples_tensor)?;

        // Create prompt ID
        let prompt_id = format!(
            "clone_{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis()
        );

        Ok(VoiceClonePrompt::new(
            prompt_id,
            speaker_embedding,
            mel,
            duration_secs,
        ))
    }

    /// Create and cache voice clone prompt
    pub fn create_and_cache_clone(
        &mut self,
        config: &VoiceCloneConfig,
        prompt_id: Option<String>,
    ) -> Result<String> {
        let prompt = self.create_clone(config)?;
        
        let id = prompt_id.unwrap_or_else(|| prompt.id.clone());
        self.cached_prompts.insert(id.clone(), prompt);
        
        Ok(id)
    }

    /// Get cached voice prompt
    pub fn get_cached_prompt(&self, prompt_id: &str) -> Option<&VoiceClonePrompt> {
        self.cached_prompts.get(prompt_id)
    }

    /// Remove cached voice prompt
    pub fn remove_cached_prompt(&mut self, prompt_id: &str) -> bool {
        self.cached_prompts.remove(prompt_id).is_some()
    }

    /// Clear all cached prompts
    pub fn clear_cached_prompts(&mut self) {
        self.cached_prompts.clear();
    }

    /// Get number of cached prompts
    pub fn cached_prompt_count(&self) -> usize {
        self.cached_prompts.len()
    }

    /// Blend voice clone with base speaker
    pub fn blend_embeddings(
        &self,
        base_embedding: &Tensor,
        clone_embedding: &Tensor,
        alpha: f32,
    ) -> Result<Tensor> {
        // Linear interpolation: result = (1 - alpha) * base + alpha * clone
        let alpha_tensor = Tensor::new(alpha, &self.device)?;
        let one_minus_alpha = Tensor::new(1.0 - alpha, &self.device)?;
        
        let base_scaled = base_embedding.mul(&one_minus_alpha)?;
        let clone_scaled = clone_embedding.mul(&alpha_tensor)?;
        Ok(base_scaled.add(&clone_scaled)?)
    }

    /// Get speaker embedding from prompt
    pub fn get_speaker_embedding<'a>(&self, prompt: &'a VoiceClonePrompt) -> &'a Tensor {
        &prompt.speaker_embedding
    }

    /// Get reference mel from prompt
    pub fn get_reference_mel<'a>(&self, prompt: &'a VoiceClonePrompt) -> &'a Tensor {
        &prompt.reference_mel
    }
}

/// Voice clone builder for fluent API
pub struct VoiceCloneBuilder {
    reference_audio: Option<PathBuf>,
    reference_text: Option<String>,
    x_vector_only: bool,
    strength: f32,
    blend_alpha: f32,
    prompt_id: Option<String>,
}

impl VoiceCloneBuilder {
    /// Create new builder
    pub fn new() -> Self {
        Self {
            reference_audio: None,
            reference_text: None,
            x_vector_only: false,
            strength: 0.8,
            blend_alpha: 1.0,
            prompt_id: None,
        }
    }

    /// Set reference audio path
    pub fn audio(mut self, path: impl Into<PathBuf>) -> Self {
        self.reference_audio = Some(path.into());
        self
    }

    /// Set reference text
    pub fn text(mut self, text: impl Into<String>) -> Self {
        self.reference_text = Some(text.into());
        self
    }

    /// Enable x-vector mode
    pub fn x_vector(mut self, enable: bool) -> Self {
        self.x_vector_only = enable;
        self
    }

    /// Set cloning strength
    pub fn strength(mut self, strength: f32) -> Self {
        self.strength = strength.clamp(0.0, 1.0);
        self
    }

    /// Set blend alpha
    pub fn blend(mut self, alpha: f32) -> Self {
        self.blend_alpha = alpha.clamp(0.0, 1.0);
        self
    }

    /// Set prompt ID for caching
    pub fn prompt_id(mut self, id: impl Into<String>) -> Self {
        self.prompt_id = Some(id.into());
        self
    }

    /// Build configuration
    pub fn build(self) -> Result<VoiceCloneConfig> {
        let reference_audio = self.reference_audio.ok_or_else(|| TtsError::Validation {
            message: "Reference audio path is required".to_string(),
            field: Some("reference_audio".to_string()),
        })?;

        Ok(VoiceCloneConfig {
            reference_audio,
            reference_text: self.reference_text,
            x_vector_only: self.x_vector_only,
            cloning_strength: self.strength,
            blend_alpha: self.blend_alpha,
        })
    }
}

impl Default for VoiceCloneBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_voice_clone_config_builder() {
        let config = VoiceCloneBuilder::new()
            .audio("speaker.wav")
            .text("Hello world")
            .strength(0.9)
            .blend(0.8)
            .build();

        assert!(config.is_ok());
        let config = config.unwrap();
        assert_eq!(config.reference_audio, PathBuf::from("speaker.wav"));
        assert_eq!(config.reference_text, Some("Hello world".to_string()));
        assert_eq!(config.cloning_strength, 0.9);
        assert_eq!(config.blend_alpha, 0.8);
    }

    #[test]
    fn test_voice_clone_config_default() {
        let config = VoiceCloneConfig::default();
        assert_eq!(config.cloning_strength, 0.8);
        assert_eq!(config.blend_alpha, 1.0);
        assert!(!config.x_vector_only);
    }
}
