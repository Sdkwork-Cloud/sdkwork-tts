//! Qwen3-TTS Example - Basic Inference
//!
//! This example demonstrates basic Qwen3-TTS usage.

use sdkwork_tts::models::qwen3_tts::{
    QwenConfig, QwenModelVariant, QwenSynthesisResult,
    Speaker, Language, SynthesisOptions,
};

fn main() -> anyhow::Result<()> {
    println!("╔═══════════════════════════════════════════════════════════╗");
    println!("║         Qwen3-TTS Rust Implementation Example             ║");
    println!("╚═══════════════════════════════════════════════════════════╝\n");

    // Example 1: Configuration
    println!("1. Creating Qwen3-TTS configuration...");
    let config = QwenConfig {
        variant: QwenModelVariant::CustomVoice17B,
        use_gpu: false, // Set to true for GPU inference
        use_bf16: false,
        ..Default::default()
    };
    println!("   ✓ Config created: {:?}", config.variant.name());
    println!("   ✓ GPU: {}, BF16: {}\n", config.use_gpu, config.use_bf16);

    // Example 2: Model variants
    println!("2. Available model variants:");
    let variants = [
        QwenModelVariant::Base06B,
        QwenModelVariant::Base17B,
        QwenModelVariant::CustomVoice06B,
        QwenModelVariant::CustomVoice17B,
        QwenModelVariant::VoiceDesign17B,
    ];
    
    for variant in &variants {
        println!("   • {:25} - Cloning: {:5}, Custom: {:5}, Design: {:5}",
            variant.name(),
            variant.supports_voice_cloning(),
            variant.supports_custom_voice(),
            variant.supports_voice_design(),
        );
    }
    println!();

    // Example 3: Preset speakers
    println!("3. Available preset speakers:");
    let speakers = [
        (Speaker::Vivian, "Chinese", "Bright, slightly husky young female"),
        (Speaker::Serena, "Chinese", "Warm, gentle young female"),
        (Speaker::UncleFu, "Chinese", "Deep, mellow mature male"),
        (Speaker::Dylan, "Chinese", "Youthful, clear Beijing male"),
        (Speaker::Eric, "Chinese", "Lively, slightly husky Chengdu male"),
        (Speaker::Ryan, "English", "Dynamic, rhythmic male"),
        (Speaker::Aiden, "English", "Sunny, mid-frequency clear American male"),
        (Speaker::OnoAnna, "Japanese", "Playful Japanese female"),
        (Speaker::Sohee, "Korean", "Warm, emotional Korean female"),
    ];
    
    for (speaker, lang, desc) in &speakers {
        println!("   • {:12} ({:8}) - {}", speaker.id(), lang, desc);
    }
    println!();

    // Example 4: Supported languages
    println!("4. Supported languages:");
    let languages = [
        (Language::Chinese, "中文", "Chinese"),
        (Language::English, "English", "English"),
        (Language::Japanese, "日本語", "Japanese"),
        (Language::Korean, "한국어", "Korean"),
        (Language::German, "Deutsch", "German"),
        (Language::French, "Français", "French"),
        (Language::Russian, "Русский", "Russian"),
        (Language::Portuguese, "Português", "Portuguese"),
        (Language::Spanish, "Español", "Spanish"),
        (Language::Italian, "Italiano", "Italian"),
    ];
    
    for (lang, native, name) in &languages {
        println!("   • {:12} - {:12} ({})", lang.code(), native, name);
    }
    println!();

    // Example 5: Synthesis options
    println!("5. Synthesis options:");
    let options = SynthesisOptions {
        seed: 42,
        temperature: 0.8,
        top_k: 50,
        top_p: 0.95,
        repetition_penalty: 1.05,
    };
    println!("   • Seed: {}", options.seed);
    println!("   • Temperature: {}", options.temperature);
    println!("   • Top-k: {}", options.top_k);
    println!("   • Top-p: {}", options.top_p);
    println!("   • Repetition Penalty: {}", options.repetition_penalty);
    println!();

    // Example 6: Create synthesis result (placeholder)
    println!("6. Creating synthesis result (placeholder)...");
    let audio = vec![0.0f32; 24000]; // 1 second of silence
    let result = QwenSynthesisResult::new(audio, 24000, 100);
    println!("   • Sample rate: {} Hz", result.sample_rate);
    println!("   • Duration: {:.2} seconds", result.duration);
    println!("   • RTF: {:.2}", result.rtf);
    
    // Save to temp file
    let temp_path = std::env::temp_dir().join("qwen3_tts_example.wav");
    result.save(&temp_path)?;
    println!("   ✓ Saved to: {:?}", temp_path);
    
    // Clean up
    let _ = std::fs::remove_file(&temp_path);
    println!();

    // Example 7: Generation config
    println!("7. Generation config:");
    let gen_config = options.to_gen_config();
    println!("   • Max new tokens: {}", gen_config.max_new_tokens);
    println!("   • Temperature: {}", gen_config.temperature);
    println!("   • Top-k: {:?}", gen_config.top_k);
    println!("   • Top-p: {:?}", gen_config.top_p);
    println!("   • Repetition penalty: {}", gen_config.repetition_penalty);
    println!();

    println!("╔═══════════════════════════════════════════════════════════╗");
    println!("║                    Example Complete                       ║");
    println!("╚═══════════════════════════════════════════════════════════╝");
    
    Ok(())
}

// Helper trait for speaker ID
trait SpeakerId {
    fn id(&self) -> &'static str;
}

impl SpeakerId for Speaker {
    fn id(&self) -> &'static str {
        match self {
            Speaker::Serena => "serena",
            Speaker::Vivian => "vivian",
            Speaker::UncleFu => "uncle_fu",
            Speaker::Dylan => "dylan",
            Speaker::Eric => "eric",
            Speaker::Ryan => "ryan",
            Speaker::Aiden => "aiden",
            Speaker::OnoAnna => "ono_anna",
            Speaker::Sohee => "sohee",
        }
    }
}

// Helper trait for language code
trait LanguageCode {
    fn code(&self) -> &'static str;
}

impl LanguageCode for Language {
    fn code(&self) -> &'static str {
        match self {
            Language::Chinese => "zh",
            Language::English => "en",
            Language::Japanese => "ja",
            Language::Korean => "ko",
            Language::German => "de",
            Language::French => "fr",
            Language::Russian => "ru",
            Language::Portuguese => "pt",
            Language::Spanish => "es",
            Language::Italian => "it",
            Language::Auto => "auto",
        }
    }
}
