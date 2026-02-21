//! SDKWork-TTS Library Integration Examples
//!
//! This file demonstrates how to use SDKWork-TTS as a library in your Rust projects

use sdkwork_tts::server::{
    TtsServer, ServerConfig, ServerMode,
    LocalConfig, CloudConfig, ChannelConfig,
    SpeakerLibrary,
};
use sdkwork_tts::inference::{InferenceConfig, IndexTTS2};
use anyhow::Result;

/// Example 1: Basic server setup
///
/// This example shows how to create and run a basic TTS server
pub async fn example_basic_server() -> Result<()> {
    // Create default configuration
    let config = ServerConfig::default();
    
    // Create and run server
    let server = TtsServer::new(config);
    server.run().await?;
    
    Ok(())
}

/// Example 2: Local mode configuration
///
/// Configure server for local inference with GPU acceleration
pub fn example_local_config() -> ServerConfig {
    ServerConfig {
        mode: ServerMode::Local,
        host: "0.0.0.0".to_string(),
        port: 8080,
        local: LocalConfig {
            enabled: true,
            checkpoints_dir: "checkpoints".into(),
            default_engine: "indextts2".to_string(),
            use_gpu: true,
            use_fp16: true,
            batch_size: 4,
            max_concurrent: 10,
        },
        ..Default::default()
    }
}

/// Example 3: Cloud mode configuration
///
/// Configure server to use cloud TTS providers
pub fn example_cloud_config() -> ServerConfig {
    ServerConfig {
        mode: ServerMode::Cloud,
        cloud: CloudConfig {
            enabled: true,
            channels: vec![
                ChannelConfig {
                    name: "openai".to_string(),
                    channel_type: sdkwork_tts::server::ChannelTypeConfig::Openai,
                    api_key: std::env::var("OPENAI_API_KEY").unwrap_or_default(),
                    models: vec!["tts-1".to_string(), "tts-1-hd".to_string()],
                    default_model: Some("tts-1".to_string()),
                    timeout: 30,
                    retries: 3,
                    ..Default::default()
                },
                ChannelConfig {
                    name: "aliyun".to_string(),
                    channel_type: sdkwork_tts::server::ChannelTypeConfig::Aliyun,
                    api_key: std::env::var("ALIYUN_API_KEY").unwrap_or_default(),
                    api_secret: std::env::var("ALIYUN_API_SECRET").ok(),
                    models: vec!["tts-v1".to_string()],
                    timeout: 30,
                    retries: 3,
                    ..Default::default()
                },
            ],
            ..Default::default()
        },
        ..Default::default()
    }
}

/// Example 4: Hybrid mode configuration
///
/// Configure server to use local inference with cloud fallback
pub fn example_hybrid_config() -> ServerConfig {
    ServerConfig {
        mode: ServerMode::Hybrid,
        local: LocalConfig {
            enabled: true,
            use_gpu: true,
            ..Default::default()
        },
        cloud: CloudConfig {
            enabled: true,
            default_channel: Some("openai".to_string()),
            ..Default::default()
        },
        ..Default::default()
    }
}

/// Example 5: Speaker library management
///
/// This example shows how to manage speakers programmatically
pub fn example_speaker_library() -> Result<()> {
    // Create speaker library
    let speaker_lib = SpeakerLibrary::new("speaker_library", 1000);
    
    // Load existing speakers
    speaker_lib.load()?;
    
    // List all speakers
    let speakers = speaker_lib.list_speakers();
    println!("Total speakers: {}", speakers.len());
    
    // Search for speakers
    let results = speaker_lib.search("vivian");
    println!("Found {} speakers matching 'vivian'", results.len());
    
    // Get speaker count
    println!("Local speakers: {}", speaker_lib.local_count());
    println!("Cloud speakers: {}", speaker_lib.cloud_count());
    
    Ok(())
}

/// Example 6: Direct TTS inference (without server)
///
/// Use TTS engine directly in your application
pub async fn example_direct_inference() -> Result<()> {
    // Create IndexTTS2 instance
    let config = InferenceConfig::default();
    
    // Note: Actual inference requires model weights
    // let tts = IndexTTS2::new("checkpoints/config.yaml")?;
    // let result = tts.infer("Hello world", "speaker.wav")?;
    // result.save("output.wav")?;
    
    println!("Inference configuration: {:?}", config);
    
    Ok(())
}

/// Example 7: Custom server with middleware
///
/// Create server with custom middleware and routes
pub async fn example_custom_server() -> Result<()> {
    use axum::{Router, routing::get};
    use std::sync::Arc;
    use sdkwork_tts::server::{ServerState, MetricsState};
    
    // Create configuration
    let config = ServerConfig::default();
    
    // Create server state
    let state = Arc::new(ServerState::new(config.clone()));
    
    // Create metrics state
    let metrics = Arc::new(MetricsState::new());
    
    // Create custom router
    let app = Router::new()
        .route("/custom", get(|| async { "Custom endpoint" }))
        .with_state(state);
    
    // Run server
    let listener = tokio::net::TcpListener::bind("0.0.0.0:8080").await?;
    axum::serve(listener, app).await?;
    
    Ok(())
}

/// Example 8: Batch synthesis
///
/// Process multiple synthesis requests in batch
pub async fn example_batch_synthesis() -> Result<()> {
    let texts = vec![
        "First sentence to synthesize",
        "Second sentence to synthesize",
        "Third sentence to synthesize",
    ];
    
    // Note: Actual implementation would use the server's batch processing
    for (i, text) in texts.iter().enumerate() {
        println!("Processing text {}: {}", i + 1, text);
        // In production, send requests to server or use engine directly
    }
    
    Ok(())
}

/// Example 9: Streaming synthesis
///
/// Stream audio as it's generated
pub async fn example_streaming_synthesis() -> Result<()> {
    // Note: Streaming implementation depends on server configuration
    println!("Streaming synthesis example");
    
    // In production:
    // 1. Create streaming request
    // 2. Process audio chunks as they arrive
    // 3. Play or save chunks in real-time
    
    Ok(())
}

/// Example 10: Error handling
///
/// Proper error handling for TTS operations
pub fn example_error_handling() {
    use sdkwork_tts::server::ServerConfig;
    
    let result = ServerConfig::load("nonexistent.yaml");
    
    match result {
        Ok(config) => {
            println!("Config loaded: {:?}", config);
        }
        Err(e) => {
            eprintln!("Failed to load config: {}", e);
            // Use default config instead
            let _config = ServerConfig::default();
        }
    }
}

// Main function to run examples
#[tokio::main]
async fn main() -> Result<()> {
    println!("SDKWork-TTS Library Integration Examples\n");
    
    // Example 1: Basic server
    println!("Example 1: Basic Server Setup");
    // example_basic_server().await?;
    
    // Example 2: Local config
    println!("\nExample 2: Local Mode Configuration");
    let _local_config = example_local_config();
    
    // Example 3: Cloud config
    println!("\nExample 3: Cloud Mode Configuration");
    let _cloud_config = example_cloud_config();
    
    // Example 4: Hybrid config
    println!("\nExample 4: Hybrid Mode Configuration");
    let _hybrid_config = example_hybrid_config();
    
    // Example 5: Speaker library
    println!("\nExample 5: Speaker Library Management");
    example_speaker_library()?;
    
    // Example 6: Direct inference
    println!("\nExample 6: Direct Inference");
    example_direct_inference().await?;
    
    // Example 7: Custom server
    println!("\nExample 7: Custom Server");
    // example_custom_server().await?;
    
    // Example 8: Batch synthesis
    println!("\nExample 8: Batch Synthesis");
    example_batch_synthesis().await?;
    
    // Example 9: Streaming
    println!("\nExample 9: Streaming Synthesis");
    example_streaming_synthesis().await?;
    
    // Example 10: Error handling
    println!("\nExample 10: Error Handling");
    example_error_handling();
    
    println!("\n✓ All examples completed!");
    
    Ok(())
}
