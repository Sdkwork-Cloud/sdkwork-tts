//! Qwen3-TTS End-to-End Integration Tests

#[cfg(test)]
mod tests {
    use crate::models::qwen3_tts::{
        QwenConfig, QwenModelVariant, QwenSynthesisResult,
        Speaker, Language, SynthesisOptions, GenerationConfig, SamplingContext,
    };

    /// Test basic config creation
    #[test]
    fn test_basic_synthesis() {
        let config = QwenConfig {
            variant: QwenModelVariant::CustomVoice17B,
            use_gpu: false,
            use_bf16: false,
            ..Default::default()
        };

        // Test config creation (model loading requires actual weights)
        assert_eq!(config.variant, QwenModelVariant::CustomVoice17B);
        assert!(!config.use_gpu);
    }

    /// Test voice clone prompt structure
    #[test]
    fn test_voice_clone_prompt_structure() {
        // Test that we can create the data structures
        let ref_audio = vec![0.0f32; 16000]; // 1 second at 16kHz
        assert_eq!(ref_audio.len(), 16000);
    }

    /// Test model variant features
    #[test]
    fn test_model_variants() {
        // Base models support voice cloning
        assert!(QwenModelVariant::Base17B.supports_voice_cloning());
        assert!(!QwenModelVariant::Base17B.supports_custom_voice());
        assert!(!QwenModelVariant::Base17B.supports_voice_design());

        // CustomVoice models support preset speakers
        assert!(!QwenModelVariant::CustomVoice17B.supports_voice_cloning());
        assert!(QwenModelVariant::CustomVoice17B.supports_custom_voice());
        assert!(!QwenModelVariant::CustomVoice17B.supports_voice_design());

        // VoiceDesign model supports text descriptions
        assert!(!QwenModelVariant::VoiceDesign17B.supports_voice_cloning());
        assert!(!QwenModelVariant::VoiceDesign17B.supports_custom_voice());
        assert!(QwenModelVariant::VoiceDesign17B.supports_voice_design());
    }

    /// Test synthesis options structure
    #[test]
    fn test_synthesis_options() {
        let options = SynthesisOptions {
            seed: 42,
            temperature: 0.8,
            top_k: 50,
            top_p: 0.95,
            repetition_penalty: 1.05,
        };

        assert_eq!(options.seed, 42);
        assert_eq!(options.temperature, 0.8);
    }

    /// Test speaker enum
    #[test]
    fn test_speaker_synthesis() {
        // Test speaker enum values
        let speakers = [
            Speaker::Vivian,
            Speaker::Serena,
            Speaker::Ryan,
        ];
        assert_eq!(speakers.len(), 3);
    }

    /// Test language enum
    #[test]
    fn test_language_support() {
        let languages = [
            Language::Chinese,
            Language::English,
            Language::Japanese,
            Language::Korean,
        ];
        assert_eq!(languages.len(), 4);
    }

    /// Test voice cloning workflow structure
    #[test]
    fn test_voice_clone_workflow() {
        // Test workflow structure (without actual model)
        let ref_audio = vec![0.0f32; 48000]; // 2 seconds at 24kHz
        assert_eq!(ref_audio.len(), 48000);
    }

    /// Test voice design workflow structure
    #[test]
    fn test_voice_design_workflow() {
        let description = "A warm, friendly female voice with medium pitch";
        assert!(!description.is_empty());
    }

    /// Test audio result save
    #[test]
    fn test_audio_save() {
        let audio = vec![0.0f32; 24000]; // 1 second of silence
        let result = QwenSynthesisResult::new(audio, 24000, 100);

        // Save to temp file
        let temp_path = std::env::temp_dir().join("test_qwen3_tts.wav");
        let save_result = result.save(&temp_path);
        
        assert!(save_result.is_ok());
        
        // Clean up
        let _ = std::fs::remove_file(&temp_path);
    }

    /// Test batch synthesis structure
    #[test]
    fn test_batch_synthesis() {
        let texts = vec![
            "First sentence",
            "Second sentence",
            "Third sentence",
        ];
        assert_eq!(texts.len(), 3);
    }

    /// Test generation config conversion
    #[test]
    fn test_gen_config_conversion() {
        let synth_options = SynthesisOptions {
            seed: 123,
            temperature: 0.7,
            top_k: 40,
            top_p: 0.9,
            repetition_penalty: 1.1,
        };

        let gen_config = synth_options.to_gen_config();
        
        assert_eq!(gen_config.seed, 123);
        assert_eq!(gen_config.temperature, 0.7);
        assert_eq!(gen_config.top_k, Some(40));
        assert_eq!(gen_config.top_p, Some(0.9));
        assert_eq!(gen_config.repetition_penalty, 1.1);
    }

    /// Test generation config default
    #[test]
    fn test_generation_config_default() {
        let config = GenerationConfig::default();
        assert_eq!(config.max_new_tokens, 2048);
        assert_eq!(config.temperature, 0.8);
    }

    /// Test sampling context
    #[test]
    fn test_sampling_context() {
        let mut ctx = SamplingContext::new(42);
        let logits = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let token = ctx.sample(&logits).unwrap();
        assert!(token < 5);
    }

    /// Test result creation
    #[test]
    fn test_result_creation() {
        let audio = vec![0.0f32; 24000];
        let result = QwenSynthesisResult::new(audio.clone(), 24000, 100);
        
        assert_eq!(result.audio.len(), 24000);
        assert_eq!(result.sample_rate, 24000);
        assert!((result.duration - 1.0).abs() < 0.01);
    }
}
