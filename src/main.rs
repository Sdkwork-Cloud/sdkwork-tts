//! SDKWork-TTS CLI - Command-line interface for text-to-speech synthesis
//!
//! A unified CLI supporting multiple TTS engines including:
//! - Qwen3-TTS (default)
//! - IndexTTS2
//! - Fish-Speech

use anyhow::{Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use indicatif::{ProgressBar, ProgressStyle};
use std::path::PathBuf;
use std::time::Instant;
use tracing::{info, warn, Level};
use tracing_subscriber::FmtSubscriber;

use sdkwork_tts::{IndexTTS2, ModelConfig, VERSION};
use sdkwork_tts::inference::{InferenceConfig, Qwen3Tts, QwenInferenceConfig, QwenModelVariant};
use sdkwork_tts::engine::{global_registry, init_engines};

/// Available TTS engines
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum EngineType {
    /// Qwen3-TTS - Alibaba's 10-language TTS (default)
    Qwen3Tts,
    /// IndexTTS2 - Bilibili's zero-shot TTS
    Indextts2,
    /// Fish-Speech - Multi-language TTS
    FishSpeech,
}

impl EngineType {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Indextts2 => "indextts2",
            Self::FishSpeech => "fish-speech",
            Self::Qwen3Tts => "qwen3-tts",
        }
    }
}

impl Default for EngineType {
    fn default() -> Self {
        Self::Qwen3Tts
    }
}

/// SDKWork-TTS - Unified TTS framework supporting multiple engines
#[derive(Parser, Debug)]
#[command(name = "sdkwork-tts")]
#[command(author, version, about, long_about = None)]
#[command(about = "A unified TTS framework supporting multiple engines")]
#[command(long_about = "
SDKWork-TTS is a unified, extensible TTS framework supporting multiple engines.

Supported Engines:
  - Qwen3-TTS: 10-language TTS with 97ms streaming latency (default)
  - IndexTTS2: Zero-shot voice cloning with emotion control
  - Fish-Speech: Multi-language TTS with streaming support

Examples:
  # Basic synthesis with Qwen3-TTS (default)
  sdkwork-tts infer --speaker voice.wav --text \"Hello world\" --output out.wav

  # Use IndexTTS2 engine
  sdkwork-tts infer --engine indextts2 --speaker voice.wav --text \"Hello world\" --output out.wav

  # List available engines
  sdkwork-tts engines
")]
struct Cli {
    /// Enable verbose logging
    #[arg(short, long, global = true)]
    verbose: bool,

    /// Use CPU instead of GPU
    #[arg(long, global = true)]
    cpu: bool,

    /// Use FP16 precision (faster, slightly lower quality)
    #[arg(long, global = true)]
    fp16: bool,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Synthesize speech from text
    Infer {
        /// TTS engine to use
        #[arg(long, value_enum, default_value = "qwen3-tts")]
        engine: EngineType,

        /// Text to synthesize
        #[arg(short, long)]
        text: String,

        /// Path to speaker reference audio
        #[arg(short, long)]
        speaker: PathBuf,

        /// Output audio file path
        #[arg(short, long, default_value = "output.wav")]
        output: PathBuf,

        /// Speaker ID for multi-speaker engines (Fish-Speech, Qwen3-TTS)
        #[arg(long)]
        speaker_id: Option<String>,

        /// Language code for multi-language engines (e.g., zh, en, ja)
        #[arg(long)]
        language: Option<String>,

        /// Path to emotion reference audio (optional, IndexTTS2 only)
        #[arg(long)]
        emotion_audio: Option<PathBuf>,

        /// Emotion blending alpha (0.0 - 1.0)
        #[arg(long, default_value = "1.0")]
        emotion_alpha: f32,

        /// Emotion vector as comma-separated values
        /// Order: happy,angry,sad,afraid,disgusted,melancholic,surprised,calm
        #[arg(long)]
        emotion_vector: Option<String>,

        /// Derive emotion vector from text via Qwen
        #[arg(long)]
        use_emo_text: bool,

        /// Text to analyze for emotion (defaults to main text)
        #[arg(long)]
        emo_text: Option<String>,

        /// Path to model config file
        #[arg(short, long, default_value = "checkpoints/config.yaml")]
        config: PathBuf,

        /// Path to model weights directory
        #[arg(long)]
        model_dir: Option<PathBuf>,

        /// Maximum text tokens per segment
        #[arg(long, default_value = "120")]
        max_tokens: usize,

        /// Generation temperature (0.0-1.0)
        #[arg(long, default_value = "0.8")]
        temperature: f32,

        /// Top-k sampling (0 = disabled)
        #[arg(long, default_value = "50")]
        top_k: usize,

        /// Top-p (nucleus) sampling (1.0 = disabled)
        #[arg(long, default_value = "0.95")]
        top_p: f32,

        /// Repetition penalty (higher discourages loops)
        #[arg(long, default_value = "1.1")]
        repetition_penalty: f32,

        /// Maximum mel tokens generated before fallback/stop
        #[arg(long, default_value = "1815")]
        max_mel_tokens: usize,

        /// Number of flow-matching denoising steps
        #[arg(long, default_value = "25")]
        flow_steps: usize,

        /// Flow-matching classifier-free guidance rate
        #[arg(long = "flow-cfg-rate", default_value = "0.7")]
        flow_cfg_rate: f32,

        /// Apply a post-vocoder high-pass filter to reduce low-frequency rumble
        #[arg(long)]
        de_rumble: bool,

        /// High-pass cutoff in Hz used when --de-rumble is enabled
        #[arg(long, default_value = "140.0")]
        de_rumble_cutoff_hz: f32,

        /// Enable streaming output (play as generated)
        #[arg(long)]
        stream: bool,
    },

    /// List available TTS engines and their capabilities
    Engines {
        /// Show detailed information
        #[arg(short, long)]
        detailed: bool,
    },

    /// Start streaming TTS server
    Serve {
        /// TTS engine to use
        #[arg(long, value_enum, default_value = "qwen3-tts")]
        engine: EngineType,

        /// Port to listen on
        #[arg(short, long, default_value = "8080")]
        port: u16,

        /// Path to model config file
        #[arg(short, long, default_value = "checkpoints/config.yaml")]
        config: PathBuf,
    },

    /// Download model weights from HuggingFace
    Download {
        /// Engine to download models for
        #[arg(long, value_enum, default_value = "qwen3-tts")]
        engine: EngineType,

        /// Model version to download (for IndexTTS2: 1.5 or 2)
        #[arg(long, default_value = "2")]
        version: String,

        /// Output directory for checkpoints
        #[arg(short, long, default_value = "checkpoints")]
        output: PathBuf,
    },

    /// Show model information
    Info {
        /// TTS engine to show info for
        #[arg(long, value_enum, default_value = "qwen3-tts")]
        engine: EngineType,

        /// Path to model config file
        #[arg(short, long, default_value = "checkpoints/config.yaml")]
        config: PathBuf,
    },
}

fn parse_emotion_vector(raw: &str) -> Result<Vec<f32>> {
    let parts: Vec<&str> = raw.split(',').collect();
    if parts.len() != 8 {
        anyhow::bail!("Emotion vector must have 8 comma-separated values");
    }
    let mut vec = Vec::with_capacity(8);
    for p in parts {
        let v: f32 = p.trim().parse()
            .with_context(|| format!("Invalid emotion value: '{}'", p))?;
        vec.push(v);
    }
    Ok(vec)
}

fn run_qwen3tts_inference(
    cli: &Cli,
    text: &str,
    speaker: &PathBuf,
    output: &PathBuf,
    speaker_id: &Option<String>,
    language: &Option<String>,
    temperature: f32,
    top_k: usize,
    top_p: f32,
    repetition_penalty: f32,
) -> Result<()> {
    if !speaker.exists() {
        anyhow::bail!("Speaker audio file not found: {:?}", speaker);
    }

    let pb = create_progress_bar("Loading Qwen3-TTS model...");
    let start = Instant::now();

    // Determine model variant
    let model_variant = QwenModelVariant::VoiceDesign17B;

    let inference_config = QwenInferenceConfig {
        model_variant,
        use_gpu: !cli.cpu,
        use_fp16: cli.fp16,
        temperature,
        top_k,
        top_p,
        repetition_penalty,
        speaker_id: speaker_id.clone(),
        language: language.clone(),
        verbose_weights: cli.verbose,
        ..Default::default()
    };

    let mut tts = Qwen3Tts::with_config(inference_config)?;

    // Try to load model from cache
    pb.set_message("Searching for Qwen3-TTS model...");
    let model_path = sdkwork_tts::find_model("Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign")?;

    if let Some(path) = &model_path {
        pb.set_message("Loading model weights from cache...");
        tts.load_weights(path)?;
    } else {
        warn!("Qwen3-TTS model not found in cache directories.");
        warn!("Please download the model first:");
        warn!("  hf download Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign --local-dir=checkpoints/qwen3-tts");
        warn!("Falling back to IndexTTS2 for inference...");

        // Fallback to IndexTTS2
        drop(tts);
        pb.finish_with_message("Fallback to IndexTTS2");

        return run_indextts2_inference(
            cli,
            text,
            speaker,
            output,
            &PathBuf::from("checkpoints/config.yaml"),
            &None,
            1.0,
            &None,
            false,
            &None,
            temperature,
            top_k,
            top_p,
            repetition_penalty,
            1815,
            25,
            0.7,
            false,
            140.0,
        );
    }

    pb.finish_with_message(format!("Model loaded in {:.1}s", start.elapsed().as_secs_f32()));

    info!("Engine: Qwen3-TTS");
    let text_preview: String = text.chars().take(50).collect();
    info!("Text: {} ({} chars)", text_preview, text.len());
    info!("Speaker: {:?}", speaker);
    if let Some(id) = speaker_id {
        info!("Speaker ID: {}", id);
    }
    if let Some(lang) = language {
        info!("Language: {}", lang);
    }
    info!("Output: {:?}", output);

    let pb = create_progress_bar("Generating speech...");
    let start = Instant::now();

    let result = if let Some(lang) = language {
        tts.infer_with_language(text, speaker, lang)
    } else if let Some(id) = speaker_id {
        tts.infer_with_speaker_id(text, id)
    } else {
        tts.infer(text, speaker)
    }
    .context("Qwen3-TTS inference failed")?;

    let duration = result.duration;
    pb.finish_with_message(format!(
        "Generated {:.1}s of audio in {:.1}s (RTF: {:.2}x)",
        duration,
        start.elapsed().as_secs_f32(),
        duration / start.elapsed().as_secs_f32()
    ));

    result.save(output).context("Failed to save audio")?;

    info!("Saved to {:?}", output);
    info!("Sample rate: {} Hz", result.sample_rate);

    Ok(())
}

fn setup_logging(verbose: bool) {
    let level = if verbose { Level::DEBUG } else { Level::INFO };
    let subscriber = FmtSubscriber::builder()
        .with_max_level(level)
        .with_target(false)
        .with_thread_ids(false)
        .compact()
        .finish();
    tracing::subscriber::set_global_default(subscriber).expect("Failed to set tracing subscriber");
}

fn create_progress_bar(msg: &str) -> ProgressBar {
    let pb = ProgressBar::new_spinner();
    pb.set_style(
        ProgressStyle::default_spinner()
            .template("{spinner:.green} {msg}")
            .unwrap(),
    );
    pb.set_message(msg.to_string());
    pb
}

fn print_engine_info(engine_id: &str, detailed: bool) {
    if let Ok(Some(info)) = global_registry().get_info(engine_id) {
        println!("┌─────────────────────────────────────────────────────────────┐");
        println!("│ {:^59} │", format!("{} v{}", info.name, info.version));
        println!("├─────────────────────────────────────────────────────────────┤");
        println!("│ ID: {:55} │", info.id);
        println!("│ Author: {:51} │", info.author);
        println!("│ License: {:50} │", info.license);
        println!("│ Type: {:52} │", format!("{:?}", info.engine_type));
        
        if detailed {
            println!("├─────────────────────────────────────────────────────────────┤");
            println!("│ Description: {:46} │", 
                if info.description.len() > 46 { 
                    format!("{}...", &info.description[..43]) 
                } else { 
                    info.description.clone() 
                }
            );
            
            println!("│ Features: {:48} │", "");
            for feature in &info.features {
                println!("│   - {:53} │", format!("{:?}", feature));
            }
            
            if let Some(repo) = &info.repository {
                println!("│ Repository: {:47} │", 
                    if repo.len() > 47 { format!("{}...", &repo[..44]) } else { repo.clone() }
                );
            }
        }
        println!("└─────────────────────────────────────────────────────────────┘");
    }
}

fn run_indextts2_inference(
    cli: &Cli,
    text: &str,
    speaker: &PathBuf,
    output: &PathBuf,
    config: &PathBuf,
    emotion_audio: &Option<PathBuf>,
    emotion_alpha: f32,
    emotion_vector: &Option<String>,
    use_emo_text: bool,
    emo_text: &Option<String>,
    temperature: f32,
    top_k: usize,
    top_p: f32,
    repetition_penalty: f32,
    max_mel_tokens: usize,
    flow_steps: usize,
    flow_cfg_rate: f32,
    de_rumble: bool,
    de_rumble_cutoff_hz: f32,
) -> Result<()> {
    if !speaker.exists() {
        anyhow::bail!("Speaker audio file not found: {:?}", speaker);
    }

    let pb = create_progress_bar("Loading IndexTTS2 model...");
    let start = Instant::now();

    let emotion_vector = match emotion_vector.as_deref() {
        Some(raw) => Some(parse_emotion_vector(raw)?),
        None => None,
    };

    let inference_config = InferenceConfig {
        temperature,
        top_k,
        top_p,
        repetition_penalty,
        max_mel_tokens,
        use_gpu: !cli.cpu,
        verbose_weights: cli.verbose,
        flow_steps,
        cfg_rate: flow_cfg_rate,
        de_rumble,
        de_rumble_cutoff_hz,
        emotion_alpha,
        emotion_vector,
        use_emo_text,
        emotion_text: emo_text.clone(),
        ..Default::default()
    };

    // Try to load from ModelScope/HuggingFace standard directories first
    let mut tts = if config.to_string_lossy().contains("checkpoints") {
        // Default config path, try to load from standard model directories
        match IndexTTS2::from_cache_with_config(sdkwork_tts::DEFAULT_MODEL_ID, inference_config.clone()) {
            Ok(tts) => tts,
            Err(e) => {
                // Fallback to config file if model not in cache
                warn!("Model not found in standard directories: {}", e);
                warn!("Falling back to config file: {:?}", config);
                IndexTTS2::with_config(config, inference_config)?
            }
        }
    } else {
        // Custom config path specified
        IndexTTS2::with_config(config, inference_config)?
    };

    // Load weights if not already loaded
    if !config.parent().map(|d| d.join("gpt.pth").exists()).unwrap_or(false) {
        // Try to find model in standard directories
        if let Some(model_path) = sdkwork_tts::find_model(sdkwork_tts::DEFAULT_MODEL_ID)? {
            pb.set_message("Loading model weights from cache...");
            tts.load_weights(&model_path)?;
        } else {
            warn!("No model weights found. Using random initialization.");
            warn!("Download model with: modelscope download --model IndexTeam/IndexTTS-2");
        }
    } else if let Some(model_dir) = config.parent() {
        if model_dir.join("gpt.pth").exists() || model_dir.join("s2mel.pth").exists() {
            pb.set_message("Loading model weights...");
            tts.load_weights(model_dir)?;
        }
    }

    pb.finish_with_message(format!("Model loaded in {:.1}s", start.elapsed().as_secs_f32()));

    info!("Engine: IndexTTS2");
    let text_preview: String = text.chars().take(50).collect();
    info!("Text: {} ({} chars)", text_preview, text.len());
    info!("Speaker: {:?}", speaker);
    info!("Output: {:?}", output);
    info!("Flow: steps={}, cfg_rate={:.3}", flow_steps, flow_cfg_rate);

    let pb = create_progress_bar("Generating speech...");
    let start = Instant::now();

    let result = if emotion_audio.is_some() {
        tts.infer_with_emotion(text, speaker, emotion_audio.as_ref())
    } else {
        tts.infer(text, speaker)
    }
    .context("Inference failed")?;

    let duration = result.duration();
    pb.finish_with_message(format!(
        "Generated {:.1}s of audio in {:.1}s (RTF: {:.2}x)",
        duration,
        start.elapsed().as_secs_f32(),
        duration / start.elapsed().as_secs_f32()
    ));

    result.save(output).context("Failed to save audio")?;

    info!("Saved to {:?}", output);
    info!("Generated {} mel codes", result.mel_codes.len());

    Ok(())
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    setup_logging(cli.verbose);

    info!("SDKWork-TTS v{}", VERSION);

    match &cli.command {
        Commands::Infer {
            engine,
            text,
            speaker,
            output,
            speaker_id,
            language,
            emotion_audio,
            emotion_alpha,
            emotion_vector,
            use_emo_text,
            emo_text,
            config,
            model_dir: _,
            max_tokens: _,
            temperature,
            top_k,
            top_p,
            repetition_penalty,
            max_mel_tokens,
            flow_steps,
            flow_cfg_rate,
            de_rumble,
            de_rumble_cutoff_hz,
            stream: _,
        } => {
            match engine {
                EngineType::Qwen3Tts => {
                    // Qwen3-TTS is the default engine
                    run_qwen3tts_inference(
                        &cli,
                        text,
                        speaker,
                        output,
                        speaker_id,
                        language,
                        *temperature,
                        *top_k,
                        *top_p,
                        *repetition_penalty,
                    )
                }
                EngineType::Indextts2 => {
                    run_indextts2_inference(
                        &cli,
                        text,
                        speaker,
                        output,
                        config,
                        emotion_audio,
                        *emotion_alpha,
                        emotion_vector,
                        *use_emo_text,
                        emo_text,
                        *temperature,
                        *top_k,
                        *top_p,
                        *repetition_penalty,
                        *max_mel_tokens,
                        *flow_steps,
                        *flow_cfg_rate,
                        *de_rumble,
                        *de_rumble_cutoff_hz,
                    )
                }
                EngineType::FishSpeech => {
                    info!("Engine: Fish-Speech");
                    info!("Text: {} ({} chars)", &text[..text.len().min(50)], text.len());
                    warn!("Fish-Speech engine adapter is ready, but model integration is not yet complete.");
                    warn!("Using IndexTTS2 as fallback...");

                    run_indextts2_inference(
                        &cli,
                        text,
                        speaker,
                        output,
                        config,
                        emotion_audio,
                        *emotion_alpha,
                        emotion_vector,
                        *use_emo_text,
                        emo_text,
                        *temperature,
                        *top_k,
                        *top_p,
                        *repetition_penalty,
                        *max_mel_tokens,
                        *flow_steps,
                        *flow_cfg_rate,
                        *de_rumble,
                        *de_rumble_cutoff_hz,
                    )
                }
            }
        }

        Commands::Engines { detailed } => {
            init_engines()?;
            
            println!("\n╔═══════════════════════════════════════════════════════════════╗");
            println!("║              SDKWork-TTS - Available Engines                  ║");
            println!("╚═══════════════════════════════════════════════════════════════╝\n");

            let engines = global_registry().list_engines()?;
            
            for engine in &engines {
                print_engine_info(&engine.id, *detailed);
                println!();
            }

            println!("Total: {} engine(s) registered", engines.len());
            println!("\nUsage: sdkwork-tts infer --engine <engine-id> --speaker voice.wav --text \"Hello\" --output out.wav");
            
            Ok(())
        }

        Commands::Serve { engine, port, config: _ } => {
            info!("Starting TTS server with {} engine on port {}", engine.as_str(), port);
            eprintln!("Server mode not yet implemented.");
            Ok(())
        }

        Commands::Download { engine, version, output } => {
            info!("Downloading {} models (version {}) to {:?}", engine.as_str(), version, output);
            eprintln!("Model download not yet implemented.");
            eprintln!("For IndexTTS2, use: hf download IndexTeam/IndexTTS-2 --local-dir=checkpoints");
            Ok(())
        }

        Commands::Info { engine, config } => {
            init_engines()?;
            
            println!("\nEngine: {}", engine.as_str());
            print_engine_info(engine.as_str(), true);
            
            if engine == &EngineType::Indextts2 && config.exists() {
                println!("\nModel Config:");
                let cfg = ModelConfig::load(config)
                    .context("Failed to load config")?;
                println!("{:#?}", cfg);
            } else if !config.exists() {
                eprintln!("\nConfig file not found: {:?}", config);
                eprintln!("Download models first with: sdkwork-tts download --engine {}", engine.as_str());
            }
            
            Ok(())
        }
    }
}
