//! Enhanced Emotion Control Module
//!
//! Provides fine-grained emotion control inspired by Qwen3-TTS:
//! - 16+ emotion types with intensity control
//! - Multi-dimensional emotion vectors
//! - Natural language emotion instructions
//! - Prosody control (pitch, speed, energy)
//! - Text-based emotion extraction


/// Emotion configuration
#[derive(Debug, Clone)]
pub struct EmotionConfig {
    /// Primary emotion
    pub primary_emotion: EmotionType,
    /// Emotion intensity (0.0 - 1.0)
    pub intensity: f32,
    /// Secondary emotion (for blending)
    pub secondary_emotion: Option<EmotionType>,
    /// Secondary emotion intensity
    pub secondary_intensity: f32,
    /// Emotion vector (8-dimensional)
    pub emotion_vector: Option<Vec<f32>>,
    /// Natural language instruction
    pub instruction: Option<String>,
    /// Prosody modifications
    pub prosody: ProsodyConfig,
}

impl Default for EmotionConfig {
    fn default() -> Self {
        Self {
            primary_emotion: EmotionType::Neutral,
            intensity: 0.5,
            secondary_emotion: None,
            secondary_intensity: 0.0,
            emotion_vector: None,
            instruction: None,
            prosody: ProsodyConfig::default(),
        }
    }
}

impl EmotionConfig {
    /// Create new emotion config with primary emotion
    pub fn new(emotion: EmotionType) -> Self {
        Self {
            primary_emotion: emotion,
            ..Default::default()
        }
    }

    /// Set intensity
    pub fn with_intensity(mut self, intensity: f32) -> Self {
        self.intensity = intensity.clamp(0.0, 1.0);
        self
    }

    /// Set secondary emotion for blending
    pub fn with_secondary(mut self, emotion: EmotionType, intensity: f32) -> Self {
        self.secondary_emotion = Some(emotion);
        self.secondary_intensity = intensity.clamp(0.0, 1.0);
        self
    }

    /// Set natural language instruction
    pub fn with_instruction(mut self, instruction: impl Into<String>) -> Self {
        self.instruction = Some(instruction.into());
        self
    }

    /// Set prosody modifications
    pub fn with_prosody(mut self, prosody: ProsodyConfig) -> Self {
        self.prosody = prosody;
        self
    }

    /// Build emotion vector from configuration
    pub fn build_vector(&self) -> Vec<f32> {
        if let Some(ref vec) = self.emotion_vector {
            return vec.clone();
        }

        // Generate vector from emotion type
        let mut vector = vec![0.0f32; 8];
        let primary_idx = self.primary_emotion.vector_index();
        vector[primary_idx] = self.intensity;

        if let Some(secondary) = self.secondary_emotion {
            let sec_idx = secondary.vector_index();
            vector[sec_idx] = self.secondary_intensity;
        }

        vector
    }
}

/// Emotion types (16 emotions inspired by Qwen3-TTS)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmotionType {
    /// Neutral, calm
    Neutral,
    /// Happy, joyful
    Happy,
    /// Sad, melancholic
    Sad,
    /// Angry, furious
    Angry,
    /// Fearful, scared
    Fearful,
    /// Disgusted, repulsed
    Disgusted,
    /// Surprised, amazed
    Surprised,
    /// Excited, enthusiastic
    Excited,
    /// Calm, peaceful
    Calm,
    /// Anxious, worried
    Anxious,
    /// Confident, assured
    Confident,
    /// Nervous, uneasy
    Nervous,
    /// Friendly, warm
    Friendly,
    /// Hostile, aggressive
    Hostile,
    /// Sarcastic, ironic
    Sarcastic,
    /// Sincere, genuine
    Sincere,
}

impl EmotionType {
    /// Get emotion name
    pub fn name(&self) -> &'static str {
        match self {
            EmotionType::Neutral => "neutral",
            EmotionType::Happy => "happy",
            EmotionType::Sad => "sad",
            EmotionType::Angry => "angry",
            EmotionType::Fearful => "fearful",
            EmotionType::Disgusted => "disgusted",
            EmotionType::Surprised => "surprised",
            EmotionType::Excited => "excited",
            EmotionType::Calm => "calm",
            EmotionType::Anxious => "anxious",
            EmotionType::Confident => "confident",
            EmotionType::Nervous => "nervous",
            EmotionType::Friendly => "friendly",
            EmotionType::Hostile => "hostile",
            EmotionType::Sarcastic => "sarcastic",
            EmotionType::Sincere => "sincere",
        }
    }

    /// Get vector index (0-7 for 8-dimensional vector)
    pub fn vector_index(&self) -> usize {
        match self {
            EmotionType::Neutral => 0,
            EmotionType::Happy => 1,
            EmotionType::Sad => 2,
            EmotionType::Angry => 3,
            EmotionType::Fearful => 4,
            EmotionType::Disgusted => 5,
            EmotionType::Surprised => 6,
            EmotionType::Excited => 7,
            EmotionType::Calm => 0,
            EmotionType::Anxious => 4,
            EmotionType::Confident => 1,
            EmotionType::Nervous => 4,
            EmotionType::Friendly => 1,
            EmotionType::Hostile => 3,
            EmotionType::Sarcastic => 6,
            EmotionType::Sincere => 0,
        }
    }

    /// Parse from string
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Option<Self> {
        let s = s.to_lowercase();
        match s.as_str() {
            "neutral" | "calm" => Some(EmotionType::Neutral),
            "happy" | "joyful" | "cheerful" => Some(EmotionType::Happy),
            "sad" | "melancholic" | "depressed" => Some(EmotionType::Sad),
            "angry" | "furious" | "mad" => Some(EmotionType::Angry),
            "fearful" | "scared" | "afraid" => Some(EmotionType::Fearful),
            "disgusted" | "repulsed" => Some(EmotionType::Disgusted),
            "surprised" | "amazed" | "astonished" => Some(EmotionType::Surprised),
            "excited" | "enthusiastic" => Some(EmotionType::Excited),
            "peaceful" | "serene" => Some(EmotionType::Calm),
            "anxious" | "worried" | "nervous" => Some(EmotionType::Anxious),
            "confident" | "assured" => Some(EmotionType::Confident),
            "uneasy" => Some(EmotionType::Nervous),
            "friendly" | "warm" | "kind" => Some(EmotionType::Friendly),
            "hostile" | "aggressive" => Some(EmotionType::Hostile),
            "sarcastic" | "ironic" => Some(EmotionType::Sarcastic),
            "sincere" | "genuine" => Some(EmotionType::Sincere),
            _ => None,
        }
    }

    /// Get all emotion types
    pub fn all() -> &'static [EmotionType] {
        &[
            EmotionType::Neutral,
            EmotionType::Happy,
            EmotionType::Sad,
            EmotionType::Angry,
            EmotionType::Fearful,
            EmotionType::Disgusted,
            EmotionType::Surprised,
            EmotionType::Excited,
            EmotionType::Calm,
            EmotionType::Anxious,
            EmotionType::Confident,
            EmotionType::Nervous,
            EmotionType::Friendly,
            EmotionType::Hostile,
            EmotionType::Sarcastic,
            EmotionType::Sincere,
        ]
    }
}

impl std::fmt::Display for EmotionType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// Prosody configuration for fine-grained control
#[derive(Debug, Clone)]
pub struct ProsodyConfig {
    /// Pitch shift in semitones (-12 to +12)
    pub pitch_shift: f32,
    /// Speaking rate multiplier (0.5 - 2.0)
    pub speaking_rate: f32,
    /// Energy/volume multiplier (0.0 - 2.0)
    pub energy: f32,
    /// Pause duration multiplier (0.5 - 2.0)
    pub pause_duration: f32,
    /// Intonation variation (0.0 - 1.0)
    pub intonation_variation: f32,
}

impl Default for ProsodyConfig {
    fn default() -> Self {
        Self {
            pitch_shift: 0.0,
            speaking_rate: 1.0,
            energy: 1.0,
            pause_duration: 1.0,
            intonation_variation: 0.5,
        }
    }
}

impl ProsodyConfig {
    /// Create prosody config for happy emotion
    pub fn happy() -> Self {
        Self {
            pitch_shift: 2.0,
            speaking_rate: 1.1,
            energy: 1.2,
            pause_duration: 0.9,
            intonation_variation: 0.8,
        }
    }

    /// Create prosody config for sad emotion
    pub fn sad() -> Self {
        Self {
            pitch_shift: -2.0,
            speaking_rate: 0.85,
            energy: 0.7,
            pause_duration: 1.2,
            intonation_variation: 0.3,
        }
    }

    /// Create prosody config for angry emotion
    pub fn angry() -> Self {
        Self {
            pitch_shift: 1.0,
            speaking_rate: 1.2,
            energy: 1.5,
            pause_duration: 0.8,
            intonation_variation: 0.7,
        }
    }

    /// Create prosody config for calm emotion
    pub fn calm() -> Self {
        Self {
            pitch_shift: 0.0,
            speaking_rate: 0.9,
            energy: 0.8,
            pause_duration: 1.1,
            intonation_variation: 0.4,
        }
    }
}

/// Emotion instruction parser for natural language control
pub struct EmotionInstructionParser;

impl EmotionInstructionParser {
    /// Parse natural language instruction into emotion config
    pub fn parse(instruction: &str) -> EmotionConfig {
        let lower = instruction.to_lowercase();
        let mut config = EmotionConfig::default();

        // Detect primary emotion
        for &emotion in EmotionType::all() {
            if lower.contains(emotion.name()) {
                config.primary_emotion = emotion;
                config.intensity = 0.7;
                break;
            }
        }

        // Detect intensity modifiers
        if lower.contains("very") || lower.contains("extremely") {
            config.intensity = 0.9;
        } else if lower.contains("slightly") || lower.contains("a bit") {
            config.intensity = 0.4;
        } else if lower.contains("moderately") {
            config.intensity = 0.6;
        }

        // Detect prosody modifiers
        if lower.contains("high pitch") || lower.contains("high-pitched") {
            config.prosody.pitch_shift = 3.0;
        } else if lower.contains("low pitch") || lower.contains("low-pitched") || lower.contains("deep") {
            config.prosody.pitch_shift = -3.0;
        }

        if lower.contains("fast") || lower.contains("quickly") {
            config.prosody.speaking_rate = 1.3;
        } else if lower.contains("slow") || lower.contains("slowly") {
            config.prosody.speaking_rate = 0.7;
        }

        if lower.contains("loud") || lower.contains("shout") {
            config.prosody.energy = 1.5;
        } else if lower.contains("quiet") || lower.contains("whisper") {
            config.prosody.energy = 0.5;
        }

        // Set instruction
        config.instruction = Some(instruction.to_string());

        config
    }

    /// Parse Chinese emotion instruction
    pub fn parse_chinese(instruction: &str) -> EmotionConfig {
        let mut config = EmotionConfig::default();

        // Chinese emotion keywords
        if instruction.contains("开心") || instruction.contains("高兴") || instruction.contains("快乐") {
            config.primary_emotion = EmotionType::Happy;
            config.intensity = 0.7;
        } else if instruction.contains("伤心") || instruction.contains("难过") || instruction.contains("悲伤") {
            config.primary_emotion = EmotionType::Sad;
            config.intensity = 0.7;
        } else if instruction.contains("生气") || instruction.contains("愤怒") {
            config.primary_emotion = EmotionType::Angry;
            config.intensity = 0.7;
        } else if instruction.contains("害怕") || instruction.contains("恐惧") {
            config.primary_emotion = EmotionType::Fearful;
            config.intensity = 0.7;
        } else if instruction.contains("惊讶") || instruction.contains("吃惊") {
            config.primary_emotion = EmotionType::Surprised;
            config.intensity = 0.7;
        } else if instruction.contains("平静") || instruction.contains("冷静") {
            config.primary_emotion = EmotionType::Calm;
            config.intensity = 0.7;
        } else if instruction.contains("兴奋") || instruction.contains("激动") {
            config.primary_emotion = EmotionType::Excited;
            config.intensity = 0.7;
        } else if instruction.contains("友好") || instruction.contains("温暖") {
            config.primary_emotion = EmotionType::Friendly;
            config.intensity = 0.7;
        }

        // Intensity modifiers
        if instruction.contains("特别") || instruction.contains("非常") || instruction.contains("极其") {
            config.intensity = 0.9;
        } else if instruction.contains("有点") || instruction.contains("稍微") {
            config.intensity = 0.4;
        }

        // Prosody modifiers
        if instruction.contains("高音") || instruction.contains("音调高") {
            config.prosody.pitch_shift = 3.0;
        } else if instruction.contains("低音") || instruction.contains("音调低") || instruction.contains("深沉") {
            config.prosody.pitch_shift = -3.0;
        }

        if instruction.contains("快") || instruction.contains("快速") {
            config.prosody.speaking_rate = 1.3;
        } else if instruction.contains("慢") || instruction.contains("缓慢") {
            config.prosody.speaking_rate = 0.7;
        }

        config.instruction = Some(instruction.to_string());
        config
    }
}

/// Emotion preset configurations
pub struct EmotionPresets;

impl EmotionPresets {
    /// Get preset emotion config by name
    pub fn get_preset(name: &str) -> Option<EmotionConfig> {
        match name {
            "cheerful" => Some(EmotionConfig::new(EmotionType::Happy)
                .with_intensity(0.8)
                .with_prosody(ProsodyConfig::happy())),
            
            "melancholic" => Some(EmotionConfig::new(EmotionType::Sad)
                .with_intensity(0.7)
                .with_prosody(ProsodyConfig::sad())),
            
            "furious" => Some(EmotionConfig::new(EmotionType::Angry)
                .with_intensity(0.9)
                .with_prosody(ProsodyConfig::angry())),
            
            "serene" => Some(EmotionConfig::new(EmotionType::Calm)
                .with_intensity(0.8)
                .with_prosody(ProsodyConfig::calm())),
            
            "enthusiastic" => Some(EmotionConfig::new(EmotionType::Excited)
                .with_intensity(0.85)
                .with_prosody(ProsodyConfig {
                    pitch_shift: 2.5,
                    speaking_rate: 1.2,
                    energy: 1.3,
                    pause_duration: 0.85,
                    intonation_variation: 0.85,
                })),
            
            "worried" => Some(EmotionConfig::new(EmotionType::Anxious)
                .with_intensity(0.7)),
            
            "confident" => Some(EmotionConfig::new(EmotionType::Confident)
                .with_intensity(0.8)
                .with_prosody(ProsodyConfig {
                    pitch_shift: -1.0,
                    speaking_rate: 0.95,
                    energy: 1.1,
                    pause_duration: 1.0,
                    intonation_variation: 0.5,
                })),
            
            "warm" => Some(EmotionConfig::new(EmotionType::Friendly)
                .with_intensity(0.75)
                .with_prosody(ProsodyConfig {
                    pitch_shift: 1.0,
                    speaking_rate: 0.95,
                    energy: 0.9,
                    pause_duration: 1.05,
                    intonation_variation: 0.6,
                })),
            
            _ => None,
        }
    }

    /// List available preset names
    pub fn available_presets() -> &'static [&'static str] {
        &[
            "cheerful",
            "melancholic",
            "furious",
            "serene",
            "enthusiastic",
            "worried",
            "confident",
            "warm",
        ]
    }
}

/// 8-dimensional emotion vector builder
pub struct EmotionVectorBuilder {
    vector: [f32; 8],
}

impl EmotionVectorBuilder {
    /// Create new builder with zeros
    pub fn new() -> Self {
        Self { vector: [0.0; 8] }
    }

    /// Set neutral
    pub fn neutral(mut self, value: f32) -> Self {
        self.vector[0] = value.clamp(0.0, 1.0);
        self
    }

    /// Set happy
    pub fn happy(mut self, value: f32) -> Self {
        self.vector[1] = value.clamp(0.0, 1.0);
        self
    }

    /// Set sad
    pub fn sad(mut self, value: f32) -> Self {
        self.vector[2] = value.clamp(0.0, 1.0);
        self
    }

    /// Set angry
    pub fn angry(mut self, value: f32) -> Self {
        self.vector[3] = value.clamp(0.0, 1.0);
        self
    }

    /// Set fearful
    pub fn fearful(mut self, value: f32) -> Self {
        self.vector[4] = value.clamp(0.0, 1.0);
        self
    }

    /// Set disgusted
    pub fn disgusted(mut self, value: f32) -> Self {
        self.vector[5] = value.clamp(0.0, 1.0);
        self
    }

    /// Set surprised
    pub fn surprised(mut self, value: f32) -> Self {
        self.vector[6] = value.clamp(0.0, 1.0);
        self
    }

    /// Set excited
    pub fn excited(mut self, value: f32) -> Self {
        self.vector[7] = value.clamp(0.0, 1.0);
        self
    }

    /// Build the vector
    pub fn build(self) -> Vec<f32> {
        self.vector.to_vec()
    }

    /// Normalize the vector (sum to 1.0)
    pub fn normalize(mut self) -> Self {
        let sum: f32 = self.vector.iter().sum();
        if sum > 0.0 {
            for v in &mut self.vector {
                *v /= sum;
            }
        }
        self
    }
}

impl Default for EmotionVectorBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_emotion_type_from_str() {
        assert_eq!(EmotionType::from_str("happy"), Some(EmotionType::Happy));
        assert_eq!(EmotionType::from_str("ANGRY"), Some(EmotionType::Angry));
        assert_eq!(EmotionType::from_str("unknown"), None);
    }

    #[test]
    fn test_emotion_config_builder() {
        let config = EmotionConfig::new(EmotionType::Happy)
            .with_intensity(0.8)
            .with_secondary(EmotionType::Excited, 0.3);

        assert_eq!(config.primary_emotion, EmotionType::Happy);
        assert_eq!(config.intensity, 0.8);
        assert_eq!(config.secondary_emotion, Some(EmotionType::Excited));
    }

    #[test]
    fn test_emotion_vector_builder() {
        let vector = EmotionVectorBuilder::new()
            .happy(0.8)
            .excited(0.5)
            .normalize()
            .build();

        assert_eq!(vector.len(), 8);
        let sum: f32 = vector.iter().sum();
        assert!((sum - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_instruction_parser() {
        let config = EmotionInstructionParser::parse("Speak very happily and quickly");
        assert_eq!(config.primary_emotion, EmotionType::Happy);
        assert!(config.intensity > 0.8);
        assert!(config.prosody.speaking_rate > 1.2);
    }

    #[test]
    fn test_chinese_instruction_parser() {
        let config = EmotionInstructionParser::parse_chinese("用特别开心的语气说");
        assert_eq!(config.primary_emotion, EmotionType::Happy);
        assert!(config.intensity > 0.8);
    }

    #[test]
    fn test_emotion_presets() {
        let presets = EmotionPresets::available_presets();
        assert!(!presets.is_empty());

        for &name in presets {
            let preset = EmotionPresets::get_preset(name);
            assert!(preset.is_some());
        }
    }
}
