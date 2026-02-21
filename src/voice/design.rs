//! Voice Design Module
//!
//! Provides voice design capabilities through natural language descriptions:
//! - Design voices using text descriptions
//! - Control gender, age, timbre, pitch, emotion, accent
//! - Generate reference audio from voice design
//! - Combine with voice cloning for consistent character voices

use std::collections::HashMap;

/// Voice design configuration
#[derive(Debug, Clone, Default)]
pub struct VoiceDesignConfig {
    /// Natural language description of the voice
    pub description: String,
    /// Gender (male/female/neutral)
    pub gender: Option<Gender>,
    /// Age range
    pub age_range: Option<AgeRange>,
    /// Voice timbre
    pub timbre: Option<Timbre>,
    /// Pitch level
    pub pitch: Option<PitchLevel>,
    /// Emotion style
    pub emotion: Option<EmotionStyle>,
    /// Accent or dialect
    pub accent: Option<String>,
    /// Speaking rate
    pub speaking_rate: Option<f32>,
    /// Additional characteristics
    pub characteristics: Vec<String>,
}

impl VoiceDesignConfig {
    /// Create new config with description
    pub fn new(description: impl Into<String>) -> Self {
        Self {
            description: description.into(),
            ..Default::default()
        }
    }

    /// Set gender
    pub fn with_gender(mut self, gender: Gender) -> Self {
        self.gender = Some(gender);
        self
    }

    /// Set age range
    pub fn with_age(mut self, age: AgeRange) -> Self {
        self.age_range = Some(age);
        self
    }

    /// Set timbre
    pub fn with_timbre(mut self, timbre: Timbre) -> Self {
        self.timbre = Some(timbre);
        self
    }

    /// Set pitch
    pub fn with_pitch(mut self, pitch: PitchLevel) -> Self {
        self.pitch = Some(pitch);
        self
    }

    /// Set emotion
    pub fn with_emotion(mut self, emotion: EmotionStyle) -> Self {
        self.emotion = Some(emotion);
        self
    }

    /// Set accent
    pub fn with_accent(mut self, accent: impl Into<String>) -> Self {
        self.accent = Some(accent.into());
        self
    }

    /// Set speaking rate
    pub fn with_speaking_rate(mut self, rate: f32) -> Self {
        self.speaking_rate = Some(rate.clamp(0.5, 2.0));
        self
    }

    /// Add characteristic
    pub fn add_characteristic(mut self, characteristic: impl Into<String>) -> Self {
        self.characteristics.push(characteristic.into());
        self
    }

    /// Build description from structured fields
    pub fn build_description(&self) -> String {
        let mut parts = Vec::new();

        if let Some(gender) = &self.gender {
            parts.push(format!("{}", gender));
        }

        if let Some(age) = &self.age_range {
            parts.push(format!("{}", age));
        }

        if let Some(timbre) = &self.timbre {
            parts.push(format!("{} timbre", timbre));
        }

        if let Some(pitch) = &self.pitch {
            parts.push(format!("{} pitch", pitch));
        }

        if let Some(emotion) = &self.emotion {
            parts.push(format!("{} emotion", emotion));
        }

        if let Some(accent) = &self.accent {
            parts.push(format!("{} accent", accent));
        }

        if let Some(rate) = self.speaking_rate {
            if rate < 1.0 {
                parts.push(format!("slow speaking rate ({:.1}x)", rate));
            } else if rate > 1.0 {
                parts.push(format!("fast speaking rate ({:.1}x)", rate));
            }
        }

        if !self.characteristics.is_empty() {
            parts.push(format!("characteristics: {}", self.characteristics.join(", ")));
        }

        let mut description = if parts.is_empty() {
            self.description.clone()
        } else if self.description.is_empty() {
            parts.join(", ")
        } else {
            format!("{}, {}", self.description, parts.join(", "))
        };

        // Capitalize first letter
        if let Some(first) = description.chars().next() {
            description = first.to_uppercase().collect::<String>() + &description[1..];
        }

        description
    }
}

/// Gender enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Gender {
    Male,
    Female,
    Neutral,
}

impl std::fmt::Display for Gender {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Gender::Male => write!(f, "Male"),
            Gender::Female => write!(f, "Female"),
            Gender::Neutral => write!(f, "Neutral"),
        }
    }
}

/// Age range enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AgeRange {
    Child,
    Teenager,
    YoungAdult,
    Adult,
    MiddleAged,
    Senior,
}

impl std::fmt::Display for AgeRange {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AgeRange::Child => write!(f, "Child"),
            AgeRange::Teenager => write!(f, "Teenager"),
            AgeRange::YoungAdult => write!(f, "Young adult"),
            AgeRange::Adult => write!(f, "Adult"),
            AgeRange::MiddleAged => write!(f, "Middle-aged"),
            AgeRange::Senior => write!(f, "Senior"),
        }
    }
}

/// Voice timbre enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Timbre {
    Warm,
    Cold,
    Bright,
    Dark,
    Rich,
    Thin,
    Smooth,
    Rough,
    Clear,
    Muffled,
}

impl std::fmt::Display for Timbre {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Timbre::Warm => write!(f, "Warm"),
            Timbre::Cold => write!(f, "Cold"),
            Timbre::Bright => write!(f, "Bright"),
            Timbre::Dark => write!(f, "Dark"),
            Timbre::Rich => write!(f, "Rich"),
            Timbre::Thin => write!(f, "Thin"),
            Timbre::Smooth => write!(f, "Smooth"),
            Timbre::Rough => write!(f, "Rough"),
            Timbre::Clear => write!(f, "Clear"),
            Timbre::Muffled => write!(f, "Muffled"),
        }
    }
}

/// Pitch level enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PitchLevel {
    VeryLow,
    Low,
    MediumLow,
    Medium,
    MediumHigh,
    High,
    VeryHigh,
}

impl std::fmt::Display for PitchLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PitchLevel::VeryLow => write!(f, "Very low"),
            PitchLevel::Low => write!(f, "Low"),
            PitchLevel::MediumLow => write!(f, "Medium-low"),
            PitchLevel::Medium => write!(f, "Medium"),
            PitchLevel::MediumHigh => write!(f, "Medium-high"),
            PitchLevel::High => write!(f, "High"),
            PitchLevel::VeryHigh => write!(f, "Very high"),
        }
    }
}

/// Emotion style enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmotionStyle {
    Neutral,
    Happy,
    Sad,
    Angry,
    Fearful,
    Disgusted,
    Surprised,
    Excited,
    Calm,
    Anxious,
    Confident,
    Nervous,
    Friendly,
    Hostile,
    Sarcastic,
    Sincere,
}

impl std::fmt::Display for EmotionStyle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EmotionStyle::Neutral => write!(f, "Neutral"),
            EmotionStyle::Happy => write!(f, "Happy"),
            EmotionStyle::Sad => write!(f, "Sad"),
            EmotionStyle::Angry => write!(f, "Angry"),
            EmotionStyle::Fearful => write!(f, "Fearful"),
            EmotionStyle::Disgusted => write!(f, "Disgusted"),
            EmotionStyle::Surprised => write!(f, "Surprised"),
            EmotionStyle::Excited => write!(f, "Excited"),
            EmotionStyle::Calm => write!(f, "Calm"),
            EmotionStyle::Anxious => write!(f, "Anxious"),
            EmotionStyle::Confident => write!(f, "Confident"),
            EmotionStyle::Nervous => write!(f, "Nervous"),
            EmotionStyle::Friendly => write!(f, "Friendly"),
            EmotionStyle::Hostile => write!(f, "Hostile"),
            EmotionStyle::Sarcastic => write!(f, "Sarcastic"),
            EmotionStyle::Sincere => write!(f, "Sincere"),
        }
    }
}

/// Preset voice designs
pub struct VoicePresets;

impl VoicePresets {
    /// Get available preset names
    pub fn available_presets() -> Vec<&'static str> {
        vec![
            "friendly_female_adult",
            "professional_male",
            "cheerful_child",
            "calm_narrator",
            "energetic_teenager",
            "wise_senior",
            "warm_customer_service",
            "authoritative_news_anchor",
            "gentle_storyteller",
            "excited_sports_commentator",
        ]
    }

    /// Get preset configuration
    pub fn get_preset(name: &str) -> Option<VoiceDesignConfig> {
        match name {
            "friendly_female_adult" => Some(VoiceDesignConfig {
                description: "Friendly and approachable".to_string(),
                gender: Some(Gender::Female),
                age_range: Some(AgeRange::Adult),
                timbre: Some(Timbre::Warm),
                pitch: Some(PitchLevel::MediumHigh),
                emotion: Some(EmotionStyle::Friendly),
                accent: None,
                speaking_rate: Some(1.0),
                characteristics: vec!["approachable".to_string(), "warm tone".to_string()],
            }),

            "professional_male" => Some(VoiceDesignConfig {
                description: "Professional and confident".to_string(),
                gender: Some(Gender::Male),
                age_range: Some(AgeRange::Adult),
                timbre: Some(Timbre::Clear),
                pitch: Some(PitchLevel::Medium),
                emotion: Some(EmotionStyle::Confident),
                accent: None,
                speaking_rate: Some(0.95),
                characteristics: vec!["authoritative".to_string(), "clear articulation".to_string()],
            }),

            "cheerful_child" => Some(VoiceDesignConfig {
                description: "Cheerful and energetic".to_string(),
                gender: Some(Gender::Neutral),
                age_range: Some(AgeRange::Child),
                timbre: Some(Timbre::Bright),
                pitch: Some(PitchLevel::High),
                emotion: Some(EmotionStyle::Happy),
                accent: None,
                speaking_rate: Some(1.1),
                characteristics: vec!["playful".to_string(), "innocent".to_string()],
            }),

            "calm_narrator" => Some(VoiceDesignConfig {
                description: "Calm and soothing narrator".to_string(),
                gender: Some(Gender::Neutral),
                age_range: Some(AgeRange::MiddleAged),
                timbre: Some(Timbre::Smooth),
                pitch: Some(PitchLevel::MediumLow),
                emotion: Some(EmotionStyle::Calm),
                accent: None,
                speaking_rate: Some(0.85),
                characteristics: vec!["soothing".to_string(), "steady pace".to_string()],
            }),

            "energetic_teenager" => Some(VoiceDesignConfig {
                description: "Energetic and enthusiastic".to_string(),
                gender: Some(Gender::Neutral),
                age_range: Some(AgeRange::Teenager),
                timbre: Some(Timbre::Bright),
                pitch: Some(PitchLevel::MediumHigh),
                emotion: Some(EmotionStyle::Excited),
                accent: None,
                speaking_rate: Some(1.15),
                characteristics: vec!["enthusiastic".to_string(), "youthful".to_string()],
            }),

            "wise_senior" => Some(VoiceDesignConfig {
                description: "Wise and experienced".to_string(),
                gender: Some(Gender::Male),
                age_range: Some(AgeRange::Senior),
                timbre: Some(Timbre::Warm),
                pitch: Some(PitchLevel::Low),
                emotion: Some(EmotionStyle::Calm),
                accent: None,
                speaking_rate: Some(0.9),
                characteristics: vec!["authoritative".to_string(), "experienced".to_string()],
            }),

            "warm_customer_service" => Some(VoiceDesignConfig {
                description: "Warm and helpful customer service".to_string(),
                gender: Some(Gender::Female),
                age_range: Some(AgeRange::YoungAdult),
                timbre: Some(Timbre::Warm),
                pitch: Some(PitchLevel::Medium),
                emotion: Some(EmotionStyle::Friendly),
                accent: None,
                speaking_rate: Some(1.0),
                characteristics: vec!["helpful".to_string(), "patient".to_string()],
            }),

            "authoritative_news_anchor" => Some(VoiceDesignConfig {
                description: "Authoritative news anchor".to_string(),
                gender: Some(Gender::Male),
                age_range: Some(AgeRange::MiddleAged),
                timbre: Some(Timbre::Clear),
                pitch: Some(PitchLevel::Medium),
                emotion: Some(EmotionStyle::Confident),
                accent: None,
                speaking_rate: Some(1.05),
                characteristics: vec!["authoritative".to_string(), "clear".to_string()],
            }),

            "gentle_storyteller" => Some(VoiceDesignConfig {
                description: "Gentle storyteller".to_string(),
                gender: Some(Gender::Female),
                age_range: Some(AgeRange::MiddleAged),
                timbre: Some(Timbre::Warm),
                pitch: Some(PitchLevel::Medium),
                emotion: Some(EmotionStyle::Calm),
                accent: None,
                speaking_rate: Some(0.9),
                characteristics: vec!["gentle".to_string(), "expressive".to_string()],
            }),

            "excited_sports_commentator" => Some(VoiceDesignConfig {
                description: "Excited sports commentator".to_string(),
                gender: Some(Gender::Male),
                age_range: Some(AgeRange::Adult),
                timbre: Some(Timbre::Bright),
                pitch: Some(PitchLevel::High),
                emotion: Some(EmotionStyle::Excited),
                accent: None,
                speaking_rate: Some(1.3),
                characteristics: vec!["energetic".to_string(), "passionate".to_string()],
            }),

            _ => None,
        }
    }

    /// Create config from natural language description
    pub fn from_description(description: &str) -> VoiceDesignConfig {
        let desc_lower = description.to_lowercase();
        let mut config = VoiceDesignConfig::new(description);

        // Parse gender
        if desc_lower.contains("female") || desc_lower.contains("woman") || desc_lower.contains("girl") {
            config = config.with_gender(Gender::Female);
        } else if desc_lower.contains("male") || desc_lower.contains("man") || desc_lower.contains("boy") {
            config = config.with_gender(Gender::Male);
        }

        // Parse age
        if desc_lower.contains("child") || desc_lower.contains("kid") || desc_lower.contains("young") {
            config = config.with_age(AgeRange::Child);
        } else if desc_lower.contains("teen") || desc_lower.contains("teenager") {
            config = config.with_age(AgeRange::Teenager);
        } else if desc_lower.contains("young adult") {
            config = config.with_age(AgeRange::YoungAdult);
        } else if desc_lower.contains("senior") || desc_lower.contains("elderly") || desc_lower.contains("old") {
            config = config.with_age(AgeRange::Senior);
        } else if desc_lower.contains("middle") {
            config = config.with_age(AgeRange::MiddleAged);
        } else {
            config = config.with_age(AgeRange::Adult);
        }

        // Parse emotion
        if desc_lower.contains("happy") || desc_lower.contains("cheerful") || desc_lower.contains("joyful") {
            config = config.with_emotion(EmotionStyle::Happy);
        } else if desc_lower.contains("sad") || desc_lower.contains("melancholy") {
            config = config.with_emotion(EmotionStyle::Sad);
        } else if desc_lower.contains("angry") || desc_lower.contains("furious") {
            config = config.with_emotion(EmotionStyle::Angry);
        } else if desc_lower.contains("calm") || desc_lower.contains("peaceful") || desc_lower.contains("soothing") {
            config = config.with_emotion(EmotionStyle::Calm);
        } else if desc_lower.contains("excited") || desc_lower.contains("enthusiastic") {
            config = config.with_emotion(EmotionStyle::Excited);
        } else if desc_lower.contains("friendly") || desc_lower.contains("warm") {
            config = config.with_emotion(EmotionStyle::Friendly);
        } else if desc_lower.contains("confident") || desc_lower.contains("authoritative") {
            config = config.with_emotion(EmotionStyle::Confident);
        }

        // Parse pitch
        if desc_lower.contains("high pitch") || desc_lower.contains("high-pitched") {
            config = config.with_pitch(PitchLevel::High);
        } else if desc_lower.contains("low pitch") || desc_lower.contains("low-pitched") || desc_lower.contains("deep") {
            config = config.with_pitch(PitchLevel::Low);
        }

        config
    }
}

/// Voice design manager
pub struct VoiceDesignManager {
    /// Cached voice designs
    cached_designs: HashMap<String, VoiceDesignConfig>,
}

impl VoiceDesignManager {
    /// Create new voice design manager
    pub fn new() -> Self {
        Self {
            cached_designs: HashMap::new(),
        }
    }

    /// Create voice design from description
    pub fn design_voice(&self, description: &str) -> VoiceDesignConfig {
        VoicePresets::from_description(description)
    }

    /// Get preset by name
    pub fn get_preset(&self, name: &str) -> Option<VoiceDesignConfig> {
        VoicePresets::get_preset(name)
    }

    /// List available presets
    pub fn list_presets(&self) -> Vec<&'static str> {
        VoicePresets::available_presets()
    }

    /// Cache a voice design
    pub fn cache_design(&mut self, name: impl Into<String>, config: VoiceDesignConfig) {
        self.cached_designs.insert(name.into(), config);
    }

    /// Get cached design
    pub fn get_cached_design(&self, name: &str) -> Option<&VoiceDesignConfig> {
        self.cached_designs.get(name)
    }

    /// Remove cached design
    pub fn remove_cached_design(&mut self, name: &str) -> bool {
        self.cached_designs.remove(name).is_some()
    }
}

impl Default for VoiceDesignManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_voice_design_config_builder() {
        let config = VoiceDesignConfig::new("Test voice")
            .with_gender(Gender::Female)
            .with_age(AgeRange::Adult)
            .with_emotion(EmotionStyle::Happy)
            .with_pitch(PitchLevel::MediumHigh)
            .build_description();

        assert!(config.contains("Test voice"));
        assert!(config.contains("Female"));
        assert!(config.contains("Adult"));
        assert!(config.contains("Happy"));
    }

    #[test]
    fn test_voice_presets() {
        let presets = VoicePresets::available_presets();
        assert!(!presets.is_empty());
        assert!(presets.contains(&"friendly_female_adult"));
        assert!(presets.contains(&"professional_male"));
    }

    #[test]
    fn test_preset_from_description() {
        let config = VoicePresets::from_description("A happy young female voice");
        assert_eq!(config.gender, Some(Gender::Female));
        assert_eq!(config.emotion, Some(EmotionStyle::Happy));
    }

    #[test]
    fn test_voice_design_manager() {
        let mut manager = VoiceDesignManager::new();
        
        // Test presets
        let presets = manager.list_presets();
        assert!(!presets.is_empty());

        // Test getting preset
        let preset = manager.get_preset("friendly_female_adult");
        assert!(preset.is_some());

        // Test caching
        let config = VoiceDesignConfig::new("Custom voice");
        manager.cache_design("my_voice", config);
        assert!(manager.get_cached_design("my_voice").is_some());
    }
}
