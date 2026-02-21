//! Qwen3-TTS Model Implementation
//!
//! Pure Rust implementation of Qwen3-TTS using Candle ML framework.
//! Based on the architecture from https://github.com/TrevorS/qwen3-tts-rs
//!
//! ## Status
//!
//! - ✅ Config system
//! - ✅ KV Cache
//! - ✅ Core components (RMSNorm, RoPE, Attention, SwiGLU)
//! - 🚧 TalkerModel (in progress)
//! - 🚧 CodePredictor (TODO)
//! - 🚧 Decoder12Hz (TODO)
//! - 🚧 Generation loop (TODO)

pub mod config;
pub mod kv_cache;
pub mod components;
pub mod talker;
pub mod code_predictor;
pub mod decoder12hz;
pub mod generation;
pub mod speaker_encoder;
pub mod rvq;
pub mod audio_features;
pub mod wavlm;
pub mod stft;
pub mod benchmark;
pub mod inference_engine;
#[cfg(test)]
mod tests;

// Re-export key types
pub use config::{
    ModelType, ParsedModelConfig, TalkerConfig, CodePredictorConfig,
    SpeakerEncoderConfig, DecoderConfig,
};
pub use kv_cache::KVCache;
pub use talker::TalkerModel;
pub use code_predictor::CodePredictor;
pub use decoder12hz::Decoder12Hz;
pub use generation::{Generator, GenerationOutput};
pub use speaker_encoder::SpeakerEncoder;
pub use components::{RMSNorm, RotaryEmbedding, CausalSelfAttention, SwiGLU};

use candle_core::{Device, DType, Tensor};
use std::path::Path;
use crate::core::error::{Result, TtsError};

/// Qwen3-TTS model variants
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum QwenModelVariant {
    Base06B,
    CustomVoice06B,
    #[default]
    Base17B,
    CustomVoice17B,
    VoiceDesign17B,
}

impl QwenModelVariant {
    pub fn name(&self) -> &'static str {
        match self {
            Self::Base06B => "Qwen3-TTS-12Hz-0.6B-Base",
            Self::CustomVoice06B => "Qwen3-TTS-12Hz-0.6B-CustomVoice",
            Self::Base17B => "Qwen3-TTS-12Hz-1.7B-Base",
            Self::CustomVoice17B => "Qwen3-TTS-12Hz-1.7B-CustomVoice",
            Self::VoiceDesign17B => "Qwen3-TTS-12Hz-1.7B-VoiceDesign",
        }
    }

    pub fn model_id(&self) -> &'static str {
        match self {
            Self::Base06B => "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
            Self::CustomVoice06B => "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
            Self::Base17B => "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
            Self::CustomVoice17B => "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
            Self::VoiceDesign17B => "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
        }
    }

    pub fn supports_voice_cloning(&self) -> bool {
        matches!(self, Self::Base06B | Self::Base17B)
    }

    pub fn supports_custom_voice(&self) -> bool {
        matches!(self, Self::CustomVoice06B | Self::CustomVoice17B)
    }

    pub fn supports_voice_design(&self) -> bool {
        matches!(self, Self::VoiceDesign17B)
    }
}

/// Qwen3-TTS configuration
#[derive(Debug, Clone)]
pub struct QwenConfig {
    pub variant: QwenModelVariant,
    pub use_gpu: bool,
    pub use_bf16: bool,
    pub use_flash_attn: bool,
    pub device_id: usize,
    pub verbose: bool,
}

impl Default for QwenConfig {
    fn default() -> Self {
        Self {
            variant: QwenModelVariant::Base17B,
            use_gpu: true,
            use_bf16: true,
            use_flash_attn: false,
            device_id: 0,
            verbose: false,
        }
    }
}

/// Voice clone prompt
#[derive(Debug, Clone)]
pub struct VoiceClonePrompt {
    pub speaker_embedding: Tensor,
    pub ref_codes: Option<Tensor>,
    pub ref_text_ids: Option<Vec<u32>>,
}

impl VoiceClonePrompt {
    pub fn new(speaker_embedding: Tensor) -> Self {
        Self {
            speaker_embedding,
            ref_codes: None,
            ref_text_ids: None,
        }
    }

    pub fn with_icl(
        speaker_embedding: Tensor,
        ref_codes: Tensor,
        ref_text_ids: Vec<u32>,
    ) -> Self {
        Self {
            speaker_embedding,
            ref_codes: Some(ref_codes),
            ref_text_ids: Some(ref_text_ids),
        }
    }

    pub fn is_icl(&self) -> bool {
        self.ref_codes.is_some() && self.ref_text_ids.is_some()
    }
}

/// Synthesis result
#[derive(Debug, Clone)]
pub struct QwenSynthesisResult {
    pub audio: Vec<f32>,
    pub sample_rate: u32,
    pub duration: f32,
    pub processing_time_ms: u64,
    pub rtf: f32,
    pub tokens: Option<Vec<u32>>,
    pub timing: Option<SynthesisTiming>,
}

#[derive(Debug, Clone)]
pub struct SynthesisTiming {
    pub prefill_ms: f64,
    pub generation_ms: f64,
    pub generation_frames: usize,
    pub decode_ms: f64,
}

impl QwenSynthesisResult {
    pub fn new(audio: Vec<f32>, sample_rate: u32, processing_time_ms: u64) -> Self {
        let duration = audio.len() as f32 / sample_rate as f32;
        let rtf = if processing_time_ms > 0 {
            (duration * 1000.0 / processing_time_ms as f32).max(0.0)
        } else {
            0.0
        };

        Self {
            audio,
            sample_rate,
            duration,
            processing_time_ms,
            rtf,
            tokens: None,
            timing: None,
        }
    }

    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        use hound::{WavSpec, WavWriter};

        let spec = WavSpec {
            channels: 1,
            sample_rate: self.sample_rate,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };

        let mut writer = WavWriter::create(path.as_ref(), spec)
            .map_err(|e| TtsError::Io {
                message: e.to_string(),
                path: Some(path.as_ref().to_path_buf()),
            })?;

        for sample in &self.audio {
            let sample_i16 = (sample * 32767.0).clamp(-32768.0, 32767.0) as i16;
            writer.write_sample(sample_i16)
                .map_err(|e| TtsError::Io {
                    message: e.to_string(),
                    path: Some(path.as_ref().to_path_buf()),
                })?;
        }

        writer.finalize()
            .map_err(|e| TtsError::Io {
                message: e.to_string(),
                path: Some(path.as_ref().to_path_buf()),
            })?;

        Ok(())
    }
}

// CodePredictor and Decoder12Hz are already defined in their respective modules

/// TextTokenizer wrapper
pub use tokenizer_wrapper::TextTokenizer;

mod tokenizer_wrapper {
    use super::*;
    
    /// TextTokenizer - placeholder for HF tokenizers integration
    pub struct TextTokenizer;

    impl TextTokenizer {
        pub fn from_pretrained<P: AsRef<Path>>(_path: P) -> Result<Self> {
            Ok(Self)
        }

        pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
            Ok(text.bytes().map(|b| b as u32).collect())
        }
    }
}

/// Generation config - re-export from generation module
pub use generation::GenerationConfig;

/// Sampling context - re-export from generation module  
pub use generation::SamplingContext;

/// Synthesis options
#[derive(Debug, Clone, Default)]
pub struct SynthesisOptions {
    pub seed: u64,
    pub temperature: f64,
    pub top_k: usize,
    pub top_p: f64,
    pub repetition_penalty: f64,
}

impl SynthesisOptions {
    pub fn to_gen_config(&self) -> GenerationConfig {
        GenerationConfig {
            temperature: self.temperature,
            top_k: Some(self.top_k),
            top_p: Some(self.top_p),
            repetition_penalty: self.repetition_penalty,
            seed: self.seed,
            ..Default::default()
        }
    }
}

/// Language
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Language {
    Chinese,
    English,
    Japanese,
    Korean,
    German,
    French,
    Russian,
    Portuguese,
    Spanish,
    Italian,
    #[default]
    Auto,
}

/// Speaker
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Speaker {
    Serena,
    #[default]
    Vivian,
    UncleFu,
    Dylan,
    Eric,
    Ryan,
    Aiden,
    OnoAnna,
    Sohee,
}

/// Main Qwen3-TTS model
pub struct Qwen3TtsModel {
    talker: TalkerModel,
    code_predictor: CodePredictor,
    decoder: Decoder12Hz,
    text_tokenizer: TextTokenizer,
    speaker_encoder: Option<SpeakerEncoder>,
    model_type: Option<ModelType>,
    device: Device,
    compute_dtype: DType,
}

impl Qwen3TtsModel {
    pub fn new(config: QwenConfig) -> Result<Self> {
        let device = if config.use_gpu {
            Device::new_cuda(config.device_id).unwrap_or(Device::Cpu)
        } else {
            Device::Cpu
        };

        Ok(Self {
            talker: TalkerModel::from_weights(&std::collections::HashMap::new(), None, &device, DType::F32)?,
            code_predictor: CodePredictor::from_weights(&std::collections::HashMap::new(), None, &device, DType::F32)?,
            decoder: Decoder12Hz::from_weights(&std::collections::HashMap::new(), &DecoderConfig::default())?,
            text_tokenizer: TextTokenizer::from_pretrained("dummy")?,
            speaker_encoder: None,
            model_type: None,
            device,
            compute_dtype: DType::F32,
        })
    }

    pub fn supports_voice_cloning(&self) -> bool {
        self.speaker_encoder.is_some()
    }

    pub fn synthesize(&self, text: &str, _options: Option<SynthesisOptions>) -> Result<QwenSynthesisResult> {
        self.synthesize_with_voice(text, Speaker::Vivian, Language::Auto, _options)
    }

    pub fn synthesize_with_voice(
        &self,
        _text: &str,
        _speaker: Speaker,
        _language: Language,
        _options: Option<SynthesisOptions>,
    ) -> Result<QwenSynthesisResult> {
        let audio = vec![0.0f32; 24000];
        Ok(QwenSynthesisResult::new(audio, 24000, 0))
    }

    pub fn synthesize_voice_clone(
        &self,
        text: &str,
        _prompt: &VoiceClonePrompt,
        _language: Language,
        _options: Option<SynthesisOptions>,
    ) -> Result<QwenSynthesisResult> {
        self.synthesize(text, _options)
    }

    pub fn synthesize_voice_design(
        &self,
        text: &str,
        _description: &str,
        _language: Language,
        _options: Option<SynthesisOptions>,
    ) -> Result<QwenSynthesisResult> {
        self.synthesize(text, _options)
    }

    pub fn create_voice_clone_prompt(
        &self,
        _ref_audio: &[f32],
        _ref_text: Option<&str>,
    ) -> Result<VoiceClonePrompt> {
        let speaker_emb = Tensor::zeros((1, 512), DType::F32, &self.device)?;
        Ok(VoiceClonePrompt::new(speaker_emb))
    }
}
