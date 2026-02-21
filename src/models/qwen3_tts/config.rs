//! Qwen3-TTS Model Configuration
//!
//! Configuration structures for Qwen3-TTS model components.

use serde::{Deserialize, Serialize};

/// Model type detected from config.json
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "PascalCase")]
pub enum ModelType {
    /// Base model for voice cloning
    Base,
    /// CustomVoice model with preset speakers
    CustomVoice,
    /// VoiceDesign model with text descriptions
    VoiceDesign,
}

/// Parsed model configuration from config.json
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParsedModelConfig {
    /// Model type
    #[serde(rename = "type")]
    pub model_type: ModelType,
    /// Model label/name
    pub label: Option<String>,
    /// Talker configuration
    pub talker: TalkerConfig,
    /// Code predictor configuration
    pub code_predictor: Option<CodePredictorConfig>,
    /// Speaker encoder configuration
    pub speaker_encoder_config: Option<SpeakerEncoderConfig>,
}

impl ParsedModelConfig {
    /// Load from config.json file
    pub fn from_file(path: &std::path::Path) -> anyhow::Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let config: Self = serde_json::from_str(&content)?;
        Ok(config)
    }

    /// Get model label
    pub fn label(&self) -> &str {
        self.label.as_deref().unwrap_or("unknown")
    }
}

/// Talker model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TalkerConfig {
    /// Hidden dimension
    pub hidden_size: usize,
    /// Number of attention heads
    pub num_attention_heads: usize,
    /// Number of KV heads (for GQA)
    pub num_key_value_heads: Option<usize>,
    /// Number of layers
    pub num_hidden_layers: usize,
    /// Intermediate dimension (FFN)
    pub intermediate_size: usize,
    /// Vocabulary size
    pub vocab_size: usize,
    /// Max sequence length
    pub max_position_embeddings: usize,
    /// RMSNorm epsilon
    pub rms_norm_eps: f64,
    /// RoPE theta
    pub rope_theta: f64,
    /// Whether model uses bias in linear layers
    pub bias: Option<bool>,
}

impl Default for TalkerConfig {
    fn default() -> Self {
        Self {
            hidden_size: 2048,
            num_attention_heads: 16,
            num_key_value_heads: Some(8),
            num_hidden_layers: 28,
            intermediate_size: 5632,
            vocab_size: 151936,
            max_position_embeddings: 4096,
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
            bias: Some(false),
        }
    }
}

impl TalkerConfig {
    /// Create config for CustomVoice model
    pub fn custom_voice() -> Self {
        Self {
            hidden_size: 2048,
            num_attention_heads: 16,
            num_key_value_heads: Some(8),
            num_hidden_layers: 28,
            intermediate_size: 5632,
            ..Default::default()
        }
    }

    /// Create config for Base model
    pub fn base() -> Self {
        Self {
            hidden_size: 1024,
            num_attention_heads: 8,
            num_key_value_heads: Some(4),
            num_hidden_layers: 24,
            intermediate_size: 2816,
            ..Default::default()
        }
    }

    /// Create from parsed config
    pub fn from_parsed(config: &ParsedModelConfig) -> Self {
        config.talker.clone()
    }

    /// Get head dimension
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    /// Get number of KV heads
    pub fn num_kv_heads(&self) -> usize {
        self.num_key_value_heads.unwrap_or(self.num_attention_heads)
    }

    /// Check if uses GQA (Grouped Query Attention)
    pub fn uses_gqa(&self) -> bool {
        self.num_kv_heads() < self.num_attention_heads
    }
}

/// Code Predictor configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodePredictorConfig {
    /// Hidden dimension
    pub hidden_size: usize,
    /// Number of attention heads
    pub num_attention_heads: usize,
    /// Number of decoder layers
    pub num_decoder_layers: usize,
    /// Intermediate dimension
    pub intermediate_size: Option<usize>,
    /// Codec embedding dimension
    pub codec_embed_dim: Option<usize>,
    /// Number of codebooks (Qwen3-TTS uses 16 codebooks: 1 semantic + 15 acoustic RVQ)
    pub num_codebooks: usize,
    /// Codebook size (Qwen3-TTS uses 2048 per codebook)
    pub codebook_size: usize,
    /// RMSNorm epsilon
    pub rms_norm_eps: f64,
}

impl Default for CodePredictorConfig {
    fn default() -> Self {
        Self {
            hidden_size: 1024,
            num_attention_heads: 8,
            num_decoder_layers: 5,
            intermediate_size: None,
            codec_embed_dim: None,
            num_codebooks: 16,  // Qwen3-TTS: 16 codebooks
            codebook_size: 2048, // Qwen3-TTS: 2048 per codebook
            rms_norm_eps: 1e-6,
        }
    }
}

impl CodePredictorConfig {
    /// Create from parsed config
    pub fn from_parsed(config: &ParsedModelConfig) -> Self {
        config.code_predictor.clone().unwrap_or_default()
    }
}

/// Speaker Encoder configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeakerEncoderConfig {
    /// Embedding dimension
    pub embed_dim: usize,
    /// Number of mel bands
    pub n_mels: usize,
    /// Sample rate
    pub sample_rate: u32,
}

impl Default for SpeakerEncoderConfig {
    fn default() -> Self {
        Self {
            embed_dim: 1024,
            n_mels: 80,
            sample_rate: 24000,
        }
    }
}

/// Decoder (ConvNeXt) configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecoderConfig {
    /// Input channels (codebooks × codebook_dim)
    pub in_channels: usize,
    /// Hidden channels
    pub hidden_channels: usize,
    /// Output channels
    pub out_channels: usize,
    /// Upsample strides (12.5 Hz → 24000 Hz = 1920× total)
    /// Using 16×16×8 = 2048× (closest to 1920×)
    pub upsample_strides: Vec<usize>,
    /// Number of codebooks
    pub num_codebooks: usize,
    /// Codebook size (Qwen3-TTS uses 2048)
    pub codebook_size: usize,
    /// Number of ConvNeXt blocks
    pub num_convnext_blocks: usize,
}

impl Default for DecoderConfig {
    fn default() -> Self {
        Self {
            in_channels: 16,
            hidden_channels: 512,
            out_channels: 1,
            // 12.5 Hz → 24000 Hz requires 1920× upsampling
            // Using 16×16×8 = 2048× (closest power-of-2 factorization)
            upsample_strides: vec![16, 16, 8],
            num_codebooks: 16,
            codebook_size: 2048,  // Qwen3-TTS: 2048 per codebook
            num_convnext_blocks: 12,
        }
    }
}
