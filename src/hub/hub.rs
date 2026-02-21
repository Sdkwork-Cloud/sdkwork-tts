//! Hub Types and Configuration
//!
//! Defines hub types (HuggingFace, ModelScope) and configuration options.

use serde::{Deserialize, Serialize};

/// Hub type enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HubType {
    /// HuggingFace Hub (https://huggingface.co)
    HuggingFace,
    /// ModelScope Hub (https://modelscope.cn)
    ModelScope,
}

impl HubType {
    /// Get hub display name
    pub fn display_name(&self) -> &'static str {
        match self {
            Self::HuggingFace => "HuggingFace",
            Self::ModelScope => "ModelScope",
        }
    }

    /// Get hub base URL
    pub fn base_url(&self) -> &'static str {
        match self {
            Self::HuggingFace => "https://huggingface.co",
            Self::ModelScope => "https://modelscope.cn",
        }
    }
}

impl std::fmt::Display for HubType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.display_name())
    }
}

impl std::str::FromStr for HubType {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "huggingface" | "hf" | "hugging-face" => Ok(Self::HuggingFace),
            "modelscope" | "ms" | "model-scope" => Ok(Self::ModelScope),
            _ => Err(format!("Unknown hub type: {}", s)),
        }
    }
}

/// Hub configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HubConfig {
    /// HuggingFace token (optional, for private models)
    pub hf_token: Option<String>,
    /// ModelScope token (optional, for private models)
    pub ms_token: Option<String>,
    /// Enable progress bar
    pub show_progress: bool,
    /// Mirror URL for HuggingFace (for users in China)
    pub hf_mirror: Option<String>,
}

impl Default for HubConfig {
    fn default() -> Self {
        Self {
            hf_token: std::env::var("HF_TOKEN").ok(),
            ms_token: std::env::var("MODELSCOPE_TOKEN").ok(),
            show_progress: true,
            hf_mirror: std::env::var("HF_ENDPOINT").ok(),
        }
    }
}

impl HubConfig {
    /// Create builder for HubConfig
    pub fn builder() -> HubConfigBuilder {
        HubConfigBuilder::default()
    }
}

/// Builder for HubConfig
#[derive(Debug, Default)]
pub struct HubConfigBuilder {
    config: HubConfig,
}

impl HubConfigBuilder {
    /// Set HuggingFace token
    pub fn hf_token(mut self, token: impl Into<String>) -> Self {
        self.config.hf_token = Some(token.into());
        self
    }

    /// Set ModelScope token
    pub fn ms_token(mut self, token: impl Into<String>) -> Self {
        self.config.ms_token = Some(token.into());
        self
    }

    /// Set HuggingFace mirror
    pub fn hf_mirror(mut self, url: impl Into<String>) -> Self {
        self.config.hf_mirror = Some(url.into());
        self
    }

    /// Build the configuration
    pub fn build(self) -> HubConfig {
        self.config
    }
}

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

#[cfg(test)]
mod tests {
    use super::*;
    use std::str::FromStr;

    #[test]
    fn test_hub_type_from_str() {
        assert_eq!(HubType::from_str("hf").unwrap(), HubType::HuggingFace);
        assert_eq!(HubType::from_str("modelscope").unwrap(), HubType::ModelScope);
    }

    #[test]
    fn test_hub_config_default() {
        let config = HubConfig::default();
        assert!(config.show_progress);
    }

    #[test]
    fn test_hub_config_builder() {
        let config = HubConfig::builder()
            .hf_token("test_token")
            .build();

        assert_eq!(config.hf_token, Some("test_token".to_string()));
    }
}
