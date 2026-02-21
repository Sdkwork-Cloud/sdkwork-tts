//! Qwen3-TTS CLI - Main Implementation

use anyhow::{Context, Result};
use clap::Parser;
use std::time::Instant;
use tracing::{info, Level};
use tracing_subscriber::FmtSubscriber;

use sdkwork_tts::models::qwen3_tts::{
    QwenConfig, QwenModelVariant, Qwen3TtsModel, QwenSynthesisResult,
    Speaker, Language, SynthesisOptions,
};

// CLI argument structures
mod args {
    use clap::{Parser, Subcommand, ValueEnum};
    use std::path::PathBuf;

    /// Qwen3-TTS model variants
    #[derive(Debug, Clone, Copy, ValueEnum)]
    pub enum ModelVariant {
        Base06B,
        CustomVoice06B,
        Base17B,
        CustomVoice17B,
        VoiceDesign17B,
    }

    /// Supported languages
    #[derive(Debug, Clone, Copy, ValueEnum)]
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
        Auto,
    }

    /// Preset speakers
    #[derive(Debug, Clone, Copy, ValueEnum)]
    pub enum Speaker {
        Vivian,
        Serena,
        UncleFu,
        Dylan,
        Eric,
        Ryan,
        Aiden,
        OnoAnna,
        Sohee,
    }

    /// Qwen3-TTS CLI
    #[derive(Parser)]
    #[command(name = "qwen3-tts")]
    pub struct Cli {
        #[command(subcommand)]
        pub command: Commands,

        #[arg(short, long, global = true)]
        pub verbose: bool,

        #[arg(long, global = true)]
        pub cpu: bool,

        #[arg(long, global = true)]
        pub bf16: bool,
    }

    #[derive(Subcommand)]
    pub enum Commands {
        Synthesize {
            #[arg(short, long)]
            text: String,

            #[arg(short, long, default_value = "Vivian")]
            speaker: Speaker,

            #[arg(short, long, default_value = "Auto")]
            language: Language,

            #[arg(short, long, default_value = "output.wav")]
            output: PathBuf,

            #[arg(long, default_value = "CustomVoice17B")]
            model: ModelVariant,

            #[arg(long, default_value = "checkpoints/qwen3-tts")]
            model_dir: PathBuf,

            #[arg(long, default_value = "0.8")]
            temperature: f64,

            #[arg(long, default_value = "50")]
            top_k: usize,

            #[arg(long, default_value = "0.95")]
            top_p: f64,

            #[arg(long, default_value = "1.05")]
            repetition_penalty: f64,

            #[arg(long, default_value = "42")]
            seed: u64,
        },

        Clone {
            #[arg(short, long)]
            text: String,

            #[arg(short, long)]
            reference: PathBuf,

            #[arg(long)]
            reference_text: Option<String>,

            #[arg(short, long, default_value = "cloned.wav")]
            output: PathBuf,

            #[arg(long, default_value = "Base17B")]
            model: ModelVariant,

            #[arg(long, default_value = "checkpoints/qwen3-tts")]
            model_dir: PathBuf,

            #[arg(short, long, default_value = "Auto")]
            language: Language,
        },

        Design {
            #[arg(short, long)]
            text: String,

            #[arg(short, long)]
            description: String,

            #[arg(short, long, default_value = "designed.wav")]
            output: PathBuf,

            #[arg(long, default_value = "VoiceDesign17B")]
            model: ModelVariant,

            #[arg(long, default_value = "checkpoints/qwen3-tts")]
            model_dir: PathBuf,

            #[arg(short, long, default_value = "Auto")]
            language: Language,
        },

        Download {
            #[arg(long, default_value = "CustomVoice17B")]
            model: ModelVariant,

            #[arg(short, long, default_value = "checkpoints")]
            output: PathBuf,
        },

        List,

        Info {
            #[arg(short, long, default_value = "checkpoints/qwen3-tts")]
            model_dir: PathBuf,
        },
    }

    impl From<ModelVariant> for sdkwork_tts::models::qwen3_tts::QwenModelVariant {
        fn from(variant: ModelVariant) -> Self {
            match variant {
                ModelVariant::Base06B => Self::Base06B,
                ModelVariant::CustomVoice06B => Self::CustomVoice06B,
                ModelVariant::Base17B => Self::Base17B,
                ModelVariant::CustomVoice17B => Self::CustomVoice17B,
                ModelVariant::VoiceDesign17B => Self::VoiceDesign17B,
            }
        }
    }

    impl From<Language> for sdkwork_tts::models::qwen3_tts::Language {
        fn from(lang: Language) -> Self {
            match lang {
                Language::Chinese => Self::Chinese,
                Language::English => Self::English,
                Language::Japanese => Self::Japanese,
                Language::Korean => Self::Korean,
                Language::German => Self::German,
                Language::French => Self::French,
                Language::Russian => Self::Russian,
                Language::Portuguese => Self::Portuguese,
                Language::Spanish => Self::Spanish,
                Language::Italian => Self::Italian,
                Language::Auto => Self::Auto,
            }
        }
    }

    impl From<Speaker> for sdkwork_tts::models::qwen3_tts::Speaker {
        fn from(speaker: Speaker) -> Self {
            match speaker {
                Speaker::Vivian => Self::Vivian,
                Speaker::Serena => Self::Serena,
                Speaker::UncleFu => Self::UncleFu,
                Speaker::Dylan => Self::Dylan,
                Speaker::Eric => Self::Eric,
                Speaker::Ryan => Self::Ryan,
                Speaker::Aiden => Self::Aiden,
                Speaker::OnoAnna => Self::OnoAnna,
                Speaker::Sohee => Self::Sohee,
            }
        }
    }
}

use args::{Cli, Commands};

fn setup_logging(verbose: bool) {
    let level = if verbose { Level::DEBUG } else { Level::INFO };
    let subscriber = FmtSubscriber::builder()
        .with_max_level(level)
        .with_target(false)
        .with_thread_ids(false)
        .compact()
        .finish();
    tracing::subscriber::set_global_default(subscriber)
        .expect("Failed to set tracing subscriber");
}

fn run_synthesize(args: &cli::SynthesizeArgs) -> Result<()> {
    info!("Synthesizing speech...");
    info!("  Text: {}", args.text);
    info!("  Speaker: {:?}", args.speaker);
    info!("  Language: {:?}", args.language);
    info!("  Model: {:?}", args.model);

    let start = Instant::now();

    // Create config
    let config = QwenConfig {
        variant: args.model.into(),
        use_gpu: !args.cpu,
        use_bf16: args.bf16,
        ..Default::default()
    };

    // Create model (placeholder - requires actual weights for full functionality)
    let model = Qwen3TtsModel::new(config)
        .context("Failed to create model")?;

    // Create synthesis options
    let options = SynthesisOptions {
        seed: args.seed,
        temperature: args.temperature,
        top_k: args.top_k,
        top_p: args.top_p,
        repetition_penalty: args.repetition_penalty,
    };

    // Synthesize
    let result = model.synthesize_with_voice(
        &args.text,
        args.speaker.into(),
        args.language.into(),
        Some(options),
    )
    .context("Synthesis failed")?;

    // Save audio
    result.save(&args.output)
        .context("Failed to save audio")?;

    let duration = start.elapsed();
    info!("Synthesis completed in {:.2}s", duration.as_secs_f32());
    info!("Output saved to: {:?}", args.output);
    info!("Audio duration: {:.2}s", result.duration);
    info!("Sample rate: {} Hz", result.sample_rate);

    Ok(())
}

fn run_clone(args: &args::Commands) -> Result<()> {
    if let Commands::Clone { model, .. } = args {
        info!("Cloning voice...");
        info!("  Model: {:?}", model);

        // Validate model variant
        let qwen_variant: QwenModelVariant = (*model).into();
        if !qwen_variant.supports_voice_cloning() {
            anyhow::bail!("Voice cloning requires Base model variant (Base06B or Base17B)");
        }

        info!("Voice cloning is a placeholder - full implementation pending");
    }
    Ok(())
}

fn run_design(args: &args::Commands) -> Result<()> {
    if let Commands::Design { model, .. } = args {
        info!("Designing voice...");
        info!("  Model: {:?}", model);

        // Validate model variant
        let qwen_variant: QwenModelVariant = (*model).into();
        if !qwen_variant.supports_voice_design() {
            anyhow::bail!("Voice design requires VoiceDesign17B model");
        }

        info!("Voice design is a placeholder - full implementation pending");
    }
    Ok(())
}

fn run_download(args: &cli::DownloadArgs) -> Result<()> {
    use hf_hub::{api::sync::Api, Repo, RepoType};

    info!("Downloading model...");
    info!("  Model: {:?}", args.model);
    info!("  Output: {:?}", args.output);

    let model_id = match args.model {
        ModelVariant::Base06B => "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        ModelVariant::CustomVoice06B => "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
        ModelVariant::Base17B => "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        ModelVariant::CustomVoice17B => "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        ModelVariant::VoiceDesign17B => "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    };

    info!("Downloading from HuggingFace: {}", model_id);

    let api = Api::new()
        .context("Failed to create HuggingFace API")?;
    
    let repo = Repo::with_revision(
        model_id.to_string(),
        RepoType::Model,
        "main".to_string(),
    );

    let model_dir = api.repo(repo)
        .get(".")
        .context("Failed to download model")?;

    info!("Model downloaded to: {:?}", model_dir);

    Ok(())
}

fn run_list() -> Result<()> {
    println!("\n╔═══════════════════════════════════════════════════════════╗");
    println!("║              Qwen3-TTS Available Models                   ║");
    println!("╚═══════════════════════════════════════════════════════════╝\n");

    println!("Model Variants:");
    println!("  • Base06B         - 0.6B parameters, voice cloning");
    println!("  • Base17B         - 1.7B parameters, voice cloning");
    println!("  • CustomVoice06B  - 0.6B parameters, 9 preset speakers");
    println!("  • CustomVoice17B  - 1.7B parameters, 9 preset speakers");
    println!("  • VoiceDesign17B  - 1.7B parameters, text voice design");
    println!();

    println!("Preset Speakers:");
    println!("  • Vivian       - Chinese, bright young female");
    println!("  • Serena       - Chinese, warm gentle female");
    println!("  • UncleFu      - Chinese, deep mature male");
    println!("  • Dylan        - Chinese, youthful Beijing male");
    println!("  • Eric         - Chinese, lively Chengdu male");
    println!("  • Ryan         - English, dynamic rhythmic male");
    println!("  • Aiden        - English, sunny American male");
    println!("  • OnoAnna      - Japanese, playful female");
    println!("  • Sohee        - Korean, warm emotional female");
    println!();

    println!("Supported Languages:");
    println!("  • Chinese (zh), English (en), Japanese (ja)");
    println!("  • Korean (ko), German (de), French (fr)");
    println!("  • Russian (ru), Portuguese (pt)");
    println!("  • Spanish (es), Italian (it)");
    println!();

    Ok(())
}

fn run_info(args: &cli::InfoArgs) -> Result<()> {
    info!("Model information:");
    info!("  Directory: {:?}", args.model_dir);

    // Check if model directory exists
    if !args.model_dir.exists() {
        println!("\n⚠️  Model directory does not exist: {:?}", args.model_dir);
        println!("   Run 'qwen3-tts download' to download the model.\n");
        return Ok(());
    }

    // List model files
    println!("\nModel files:");
    for entry in std::fs::read_dir(&args.model_dir)? {
        let entry = entry?;
        let path = entry.path();
        let size = path.metadata()?.len();
        println!("  • {:30} ({:.2} MB)", 
            path.file_name().unwrap().to_string_lossy(),
            size as f64 / 1024.0 / 1024.0
        );
    }
    println!();

    Ok(())
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    setup_logging(cli.verbose);

    info!("Qwen3-TTS CLI");

    match &cli.command {
        Commands::Synthesize(args) => run_synthesize(args)?,
        Commands::Clone(args) => run_clone(args)?,
        Commands::Design(args) => run_design(args)?,
        Commands::Download(args) => run_download(args)?,
        Commands::List => run_list()?,
        Commands::Info(args) => run_info(args)?,
    }

    Ok(())
}
