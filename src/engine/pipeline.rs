//! Processing Pipeline for TTS synthesis
//!
//! Provides a flexible, pluggable pipeline architecture for text and audio processing.

use std::collections::HashMap;
use std::sync::Arc;

use crate::core::error::Result;

/// Pipeline stage trait for pluggable processing
pub trait PipelineStage: Send + Sync {
    /// Stage name
    fn name(&self) -> &str;

    /// Stage description
    fn description(&self) -> &str;

    /// Process input and produce output
    fn process(&self, input: PipelineData) -> Result<PipelineData>;

    /// Check if this stage is enabled
    fn is_enabled(&self) -> bool {
        true
    }

    /// Get stage priority (lower = earlier)
    fn priority(&self) -> u32 {
        100
    }
}

/// Pipeline data container
#[derive(Debug, Clone)]
pub enum PipelineData {
    /// Text data
    Text(TextData),
    /// Audio data
    Audio(AudioData),
    /// Features (mel spectrogram, etc.)
    Features(FeaturesData),
    /// Tokens
    Tokens(TokensData),
    /// Raw bytes
    Raw(Vec<u8>),
    /// Empty/null
    Empty,
}

/// Text data container
#[derive(Debug, Clone)]
pub struct TextData {
    /// Raw text
    pub text: String,
    /// Normalized text
    pub normalized: Option<String>,
    /// Language code
    pub language: Option<String>,
    /// Segments
    pub segments: Vec<TextSegment>,
    /// Metadata
    pub metadata: HashMap<String, String>,
}

/// Text segment
#[derive(Debug, Clone)]
pub struct TextSegment {
    /// Segment text
    pub text: String,
    /// Start position
    pub start: usize,
    /// End position
    pub end: usize,
    /// Segment type
    pub segment_type: SegmentType,
}

/// Segment type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SegmentType {
    Sentence,
    Phrase,
    Word,
    Phoneme,
    Pause,
}

/// Audio data container
#[derive(Debug, Clone)]
pub struct AudioData {
    /// Audio samples
    pub samples: Vec<f32>,
    /// Sample rate
    pub sample_rate: u32,
    /// Number of channels
    pub channels: u16,
    /// Duration in seconds
    pub duration: f32,
    /// Metadata
    pub metadata: HashMap<String, String>,
}

/// Features data container
#[derive(Debug, Clone)]
pub struct FeaturesData {
    /// Feature tensor (flattened)
    pub data: Vec<f32>,
    /// Feature dimensions
    pub shape: Vec<usize>,
    /// Feature type
    pub feature_type: FeatureType,
    /// Metadata
    pub metadata: HashMap<String, String>,
}

/// Feature type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FeatureType {
    MelSpectrogram,
    Mfcc,
    SpeakerEmbedding,
    EmotionEmbedding,
    SemanticEmbedding,
    Custom,
}

/// Tokens data container
#[derive(Debug, Clone)]
pub struct TokensData {
    /// Token IDs
    pub tokens: Vec<u32>,
    /// Token strings (if available)
    pub token_strings: Option<Vec<String>>,
    /// Token type
    pub token_type: TokenType,
    /// Metadata
    pub metadata: HashMap<String, String>,
}

/// Token type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenType {
    Text,
    MelCode,
    Phoneme,
    Word,
    Custom,
}

/// Processing pipeline
pub struct ProcessingPipeline {
    /// Pipeline stages
    stages: Vec<Arc<dyn PipelineStage>>,
    /// Pipeline name
    name: String,
    /// Pipeline description
    description: String,
}

impl ProcessingPipeline {
    /// Create a new pipeline
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            stages: Vec::new(),
            name: name.into(),
            description: String::new(),
        }
    }

    /// Set description
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = description.into();
        self
    }

    /// Add a stage to the pipeline
    pub fn add_stage(mut self, stage: Arc<dyn PipelineStage>) -> Self {
        self.stages.push(stage);
        self.stages.sort_by_key(|s| s.priority());
        self
    }

    /// Add multiple stages
    pub fn add_stages(mut self, stages: Vec<Arc<dyn PipelineStage>>) -> Self {
        for stage in stages {
            self.stages.push(stage);
        }
        self.stages.sort_by_key(|s| s.priority());
        self
    }

    /// Remove a stage by name
    pub fn remove_stage(mut self, name: &str) -> Self {
        self.stages.retain(|s| s.name() != name);
        self
    }

    /// Process data through the pipeline
    pub fn process(&self, input: PipelineData) -> Result<PipelineData> {
        let mut data = input;

        for stage in &self.stages {
            if stage.is_enabled() {
                data = stage.process(data)?;
            }
        }

        Ok(data)
    }

    /// Get pipeline name
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get pipeline description
    pub fn description(&self) -> &str {
        &self.description
    }

    /// Get stages
    pub fn stages(&self) -> &[Arc<dyn PipelineStage>] {
        &self.stages
    }

    /// Get stage by name
    pub fn get_stage(&self, name: &str) -> Option<&Arc<dyn PipelineStage>> {
        self.stages.iter().find(|s| s.name() == name)
    }
}

/// Standard text processing stages
pub mod text_stages {
    use super::*;

    /// Text normalization stage
    pub struct NormalizationStage {
        name: String,
        enabled: bool,
    }

    impl NormalizationStage {
        pub fn new() -> Self {
            Self {
                name: "normalization".to_string(),
                enabled: true,
            }
        }
    }

    impl Default for NormalizationStage {
        fn default() -> Self {
            Self::new()
        }
    }

    impl PipelineStage for NormalizationStage {
        fn name(&self) -> &str {
            &self.name
        }

        fn description(&self) -> &str {
            "Normalizes text (numbers, abbreviations, etc.)"
        }

        fn process(&self, input: PipelineData) -> Result<PipelineData> {
            match input {
                PipelineData::Text(mut text_data) => {
                    let normalizer = crate::text::TextNormalizer::new(false);
                    text_data.normalized = Some(normalizer.normalize(&text_data.text));
                    Ok(PipelineData::Text(text_data))
                }
                _ => Ok(input),
            }
        }

        fn is_enabled(&self) -> bool {
            self.enabled
        }

        fn priority(&self) -> u32 {
            10
        }
    }

    /// Text segmentation stage
    pub struct SegmentationStage {
        name: String,
        enabled: bool,
    }

    impl SegmentationStage {
        pub fn new() -> Self {
            Self {
                name: "segmentation".to_string(),
                enabled: true,
            }
        }
    }

    impl Default for SegmentationStage {
        fn default() -> Self {
            Self::new()
        }
    }

    impl PipelineStage for SegmentationStage {
        fn name(&self) -> &str {
            &self.name
        }

        fn description(&self) -> &str {
            "Segments text into sentences/phrases"
        }

        fn process(&self, input: PipelineData) -> Result<PipelineData> {
            match input {
                PipelineData::Text(mut text_data) => {
                    let text = text_data.normalized.as_ref()
                        .unwrap_or(&text_data.text);
                    
                    let segments = crate::text::segment_text_string(text, 500);
                    
                    let mut offset = 0;
                    text_data.segments = segments.into_iter().map(|seg_text| {
                        let start = text_data.text.find(&seg_text).unwrap_or(offset);
                        let end = start + seg_text.len();
                        offset = end;
                        TextSegment {
                            text: seg_text,
                            start,
                            end,
                            segment_type: SegmentType::Sentence,
                        }
                    }).collect();
                    
                    Ok(PipelineData::Text(text_data))
                }
                _ => Ok(input),
            }
        }

        fn is_enabled(&self) -> bool {
            self.enabled
        }

        fn priority(&self) -> u32 {
            20
        }
    }

    /// Tokenization stage
    pub struct TokenizationStage {
        name: String,
        enabled: bool,
        tokenizer_path: Option<std::path::PathBuf>,
    }

    impl TokenizationStage {
        pub fn new() -> Self {
            Self {
                name: "tokenization".to_string(),
                enabled: true,
                tokenizer_path: None,
            }
        }

        pub fn with_tokenizer(mut self, path: impl Into<std::path::PathBuf>) -> Self {
            self.tokenizer_path = Some(path.into());
            self
        }
    }

    impl Default for TokenizationStage {
        fn default() -> Self {
            Self::new()
        }
    }

    impl PipelineStage for TokenizationStage {
        fn name(&self) -> &str {
            &self.name
        }

        fn description(&self) -> &str {
            "Tokenizes text into token IDs"
        }

        fn process(&self, input: PipelineData) -> Result<PipelineData> {
            match input {
                PipelineData::Text(text_data) => {
                    let text = text_data.normalized.as_ref()
                        .unwrap_or(&text_data.text);
                    
                    let tokens = vec![0u32; text.len().min(100)];
                    
                    Ok(PipelineData::Tokens(TokensData {
                        tokens,
                        token_strings: None,
                        token_type: TokenType::Text,
                        metadata: text_data.metadata,
                    }))
                }
                _ => Ok(input),
            }
        }

        fn is_enabled(&self) -> bool {
            self.enabled
        }

        fn priority(&self) -> u32 {
            30
        }
    }
}

/// Standard audio processing stages
pub mod audio_stages {
    use super::*;

    /// Audio normalization stage
    pub struct NormalizationStage {
        name: String,
        enabled: bool,
        target_db: f32,
    }

    impl NormalizationStage {
        pub fn new() -> Self {
            Self {
                name: "audio_normalization".to_string(),
                enabled: true,
                target_db: -3.0,
            }
        }

        pub fn with_target_db(mut self, db: f32) -> Self {
            self.target_db = db;
            self
        }
    }

    impl Default for NormalizationStage {
        fn default() -> Self {
            Self::new()
        }
    }

    impl PipelineStage for NormalizationStage {
        fn name(&self) -> &str {
            &self.name
        }

        fn description(&self) -> &str {
            "Normalizes audio volume"
        }

        fn process(&self, input: PipelineData) -> Result<PipelineData> {
            match input {
                PipelineData::Audio(mut audio_data) => {
                    let peak = audio_data.samples.iter()
                        .fold(0.0f32, |acc, &x| acc.max(x.abs()));
                    
                    if peak > 0.0 {
                        let target_peak = 10.0_f32.powf(self.target_db / 20.0);
                        let scale = target_peak / peak;
                        for sample in &mut audio_data.samples {
                            *sample *= scale;
                        }
                    }
                    
                    Ok(PipelineData::Audio(audio_data))
                }
                _ => Ok(input),
            }
        }

        fn is_enabled(&self) -> bool {
            self.enabled
        }

        fn priority(&self) -> u32 {
            100
        }
    }

    /// High-pass filter stage (de-rumble)
    pub struct HighPassStage {
        name: String,
        enabled: bool,
        cutoff_hz: f32,
    }

    impl HighPassStage {
        pub fn new(cutoff_hz: f32) -> Self {
            Self {
                name: "high_pass".to_string(),
                enabled: true,
                cutoff_hz,
            }
        }
    }

    impl PipelineStage for HighPassStage {
        fn name(&self) -> &str {
            &self.name
        }

        fn description(&self) -> &str {
            "Applies high-pass filter to remove rumble"
        }

        fn process(&self, input: PipelineData) -> Result<PipelineData> {
            match input {
                PipelineData::Audio(audio_data) => {
                    let filtered = crate::inference::apply_high_pass(
                        &audio_data.samples,
                        audio_data.sample_rate as f32,
                        self.cutoff_hz,
                    );
                    
                    Ok(PipelineData::Audio(AudioData {
                        samples: filtered,
                        ..audio_data
                    }))
                }
                _ => Ok(input),
            }
        }

        fn is_enabled(&self) -> bool {
            self.enabled
        }

        fn priority(&self) -> u32 {
            110
        }
    }

    /// Silence trimming stage
    pub struct TrimSilenceStage {
        name: String,
        enabled: bool,
        threshold_db: f32,
    }

    impl TrimSilenceStage {
        pub fn new(threshold_db: f32) -> Self {
            Self {
                name: "trim_silence".to_string(),
                enabled: true,
                threshold_db,
            }
        }
    }

    impl Default for TrimSilenceStage {
        fn default() -> Self {
            Self::new(-40.0)
        }
    }

    impl PipelineStage for TrimSilenceStage {
        fn name(&self) -> &str {
            &self.name
        }

        fn description(&self) -> &str {
            "Trims silence from audio"
        }

        fn process(&self, input: PipelineData) -> Result<PipelineData> {
            match input {
                PipelineData::Audio(audio_data) => {
                    let threshold = 10.0_f32.powf(self.threshold_db / 20.0);
                    
                    let start = audio_data.samples.iter()
                        .position(|&x| x.abs() > threshold)
                        .unwrap_or(0);
                    
                    let end = audio_data.samples.iter()
                        .rposition(|&x| x.abs() > threshold)
                        .map(|i| i + 1)
                        .unwrap_or(audio_data.samples.len());
                    
                    let trimmed = audio_data.samples[start..end].to_vec();
                    let duration = trimmed.len() as f32 / audio_data.sample_rate as f32;
                    
                    Ok(PipelineData::Audio(AudioData {
                        samples: trimmed,
                        duration,
                        ..audio_data
                    }))
                }
                _ => Ok(input),
            }
        }

        fn is_enabled(&self) -> bool {
            self.enabled
        }

        fn priority(&self) -> u32 {
            120
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pipeline_new() {
        let pipeline = ProcessingPipeline::new("test");
        assert_eq!(pipeline.name(), "test");
        assert!(pipeline.stages().is_empty());
    }

    #[test]
    fn test_pipeline_add_stage() {
        let stage = Arc::new(text_stages::NormalizationStage::new());
        let pipeline = ProcessingPipeline::new("test")
            .add_stage(stage);
        
        assert_eq!(pipeline.stages().len(), 1);
    }

    #[test]
    fn test_text_normalization_stage() {
        let stage = text_stages::NormalizationStage::new();
        let input = PipelineData::Text(TextData {
            text: "Hello world".to_string(),
            normalized: None,
            language: None,
            segments: vec![],
            metadata: HashMap::new(),
        });
        
        let output = stage.process(input).unwrap();
        match output {
            PipelineData::Text(text) => {
                assert!(text.normalized.is_some());
            }
            _ => panic!("Expected Text data"),
        }
    }

    #[test]
    fn test_audio_normalization_stage() {
        let stage = audio_stages::NormalizationStage::new();
        let input = PipelineData::Audio(AudioData {
            samples: vec![0.5, -0.5, 0.3, -0.3],
            sample_rate: 22050,
            channels: 1,
            duration: 0.0002,
            metadata: HashMap::new(),
        });
        
        let output = stage.process(input).unwrap();
        match output {
            PipelineData::Audio(audio) => {
                let peak = audio.samples.iter()
                    .fold(0.0f32, |acc, &x| acc.max(x.abs()));
                assert!(peak <= 1.0);
            }
            _ => panic!("Expected Audio data"),
        }
    }
}
