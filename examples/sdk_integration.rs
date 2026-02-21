//! SDK Integration Examples
//!
//! This file demonstrates how to integrate SDKWork-TTS into third-party applications.

#[cfg(test)]
mod tests {
    use sdkwork_tts::sdk::*;

    /// Example 1: Basic SDK initialization and synthesis
    #[test]
    #[ignore] // Requires model files
    fn example_basic_synthesis() -> Result<(), Box<dyn std::error::Error>> {
        // Initialize SDK with defaults
        let sdk = SdkBuilder::new()
            .cpu() // Use CPU for this example
            .with_default_engines()
            .build()?;

        // Simple synthesis
        let audio = sdk.synthesize("Hello world", "speaker.wav")?;

        // Save to file
        sdk.save_audio(&audio, "output.wav")?;

        Ok(())
    }

    /// Example 2: GPU-accelerated synthesis with custom config
    #[test]
    #[ignore]
    fn example_gpu_synthesis() -> Result<(), Box<dyn std::error::Error>> {
        // Configure SDK for GPU usage
        let config = SdkConfig::builder()
            .gpu(true)
            .default_engine("indextts2")
            .memory_limit(4 * 1024 * 1024 * 1024) // 4GB limit
            .metrics(true)
            .build();

        // Build SDK
        let sdk = SdkBuilder::from_config(config)
            .with_default_engines()
            .build()?;

        // Use fluent builder API
        sdk.synthesis()
            .text("Hello from GPU!")
            .speaker("speaker.wav")
            .temperature(0.8)
            .save("gpu_output.wav")?;

        Ok(())
    }

    /// Example 3: Synthesis with emotion control
    #[test]
    #[ignore]
    fn example_emotion_synthesis() -> Result<(), Box<dyn std::error::Error>> {
        let sdk = SdkBuilder::new()
            .gpu()
            .with_default_engines()
            .build()?;

        // Synthesis with named emotion
        let audio = sdk.synthesis()
            .text("I am so happy to speak with you!")
            .speaker("speaker.wav")
            .emotion("happy", 0.8)
            .temperature(0.7)
            .build()?;

        sdk.save_audio(&audio, "happy_output.wav")?;

        // Synthesis with emotion vector
        let emotion_vector = vec![0.8, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1]; // Happy vector
        let audio2 = sdk.synthesis()
            .text("Custom emotion vector test")
            .speaker("speaker.wav")
            .emotion(emotion_vector, 1.0)
            .build()?;

        sdk.save_audio(&audio2, "vector_output.wav")?;

        Ok(())
    }

    /// Example 4: Multi-engine usage
    #[test]
    #[ignore]
    fn example_multi_engine() -> Result<(), Box<dyn std::error::Error>> {
        let sdk = SdkBuilder::new()
            .gpu()
            .with_all_engines()
            .build()?;

        // List available engines
        let engines = sdk.list_engines()?;
        println!("Available engines:");
        for engine in &engines {
            println!("  - {} (v{})", engine.name, engine.version);
        }

        // Get statistics
        let stats = sdk.stats();
        println!("SDK Stats: {} engines loaded", stats.loaded_engines);

        Ok(())
    }

    /// Example 5: Metrics and monitoring
    #[test]
    #[ignore]
    fn example_metrics() -> Result<(), Box<dyn std::error::Error>> {
        use sdkwork_tts::core::{PrometheusExporter, JsonExporter, MetricsExporter};

        let sdk = SdkBuilder::new()
            .gpu()
            .with_metrics(true)
            .with_default_engines()
            .build()?;

        // Do some synthesis
        let _ = sdk.synthesize("Test 1", "speaker.wav");
        let _ = sdk.synthesize("Test 2", "speaker.wav");

        // Get metrics
        if let Some(metrics) = sdk.metrics() {
            let report = metrics.generate_report();

            // Export as Prometheus format
            let prom = PrometheusExporter::new().export(&report)?;
            println!("Prometheus metrics:\n{}", prom);

            // Export as JSON
            let json = JsonExporter::new().pretty(true).export(&report)?;
            println!("JSON metrics:\n{}", json);
        }

        // Get SDK statistics
        let stats = sdk.stats();
        println!("Total synthesis: {}", stats.total_synthesis);
        println!("Success rate: {:.2}%", stats.success_rate() * 100.0);

        Ok(())
    }

    /// Example 6: Event handling
    #[test]
    #[ignore]
    fn example_events() -> Result<(), Box<dyn std::error::Error>> {
        use std::sync::Arc;
        use sdkwork_tts::core::{EventBus, Event, EventHandler, events::*};

        // Create event handler
        struct LogHandler;
        impl EventHandler<ModelLoadingStarted> for LogHandler {
            fn handle(&self, event: &ModelLoadingStarted) -> sdkwork_tts::core::error::Result<()> {
                println!("Loading model: {} from {}", event.model_name, event.model_path);
                Ok(())
            }
            fn handler_name(&self) -> &'static str { "log_handler" }
        }

        let sdk = SdkBuilder::new()
            .with_event_logging(true)
            .with_default_engines()
            .build()?;

        // Subscribe to events
        if let Some(event_bus) = sdk.event_bus() {
            event_bus.subscribe::<ModelLoadingStarted>(Arc::new(LogHandler));
        }

        Ok(())
    }

    /// Example 7: Error handling
    #[test]
    fn example_error_handling() {
        use sdkwork_tts::sdk::SdkError;

        // Try to initialize without model files - should handle gracefully
        let result = SdkBuilder::new()
            .cpu()
            .build();

        match result {
            Ok(_sdk) => {
                // SDK initialized successfully
                println!("SDK initialized");
            }
            Err(e) => {
                // Handle error
                println!("SDK error: {}", e);
            }
        }

        // Test error type matching
        let sdk_error = SdkError::InvalidConfig {
            field: "model_path".to_string(),
            message: "Path not found".to_string(),
        };

        match sdk_error {
            SdkError::InvalidConfig { field, message } => {
                println!("Config error in {}: {}", field, message);
            }
            _ => {}
        }
    }

    /// Example 8: Audio data processing
    #[test]
    #[ignore]
    fn example_audio_processing() -> Result<(), Box<dyn std::error::Error>> {
        let sdk = SdkBuilder::new()
            .cpu()
            .with_default_engines()
            .build()?;

        let audio = sdk.synthesize("Test audio processing", "speaker.wav")?;

        // Analyze audio
        println!("Duration: {:.2}s", audio.duration_secs);
        println!("Sample rate: {} Hz", audio.sample_rate);
        println!("Samples: {}", audio.samples.len());
        println!("RMS: {:.4}", audio.rms());
        println!("Peak: {:.4}", audio.peak());

        // Convert to i16 for WAV output
        let i16_samples = audio.to_i16();
        println!("i16 samples: {}", i16_samples.len());

        Ok(())
    }
}

/// Example: Web API integration (Axum)
#[cfg(feature = "web")]
pub mod web_api {
    use axum::{
        extract::State,
        Json,
        routing::post,
        Router,
    };
    use serde::{Deserialize, Serialize};
    use sdkwork_tts::sdk::*;

    #[derive(Deserialize)]
    pub struct SynthesisRequest {
        text: String,
        speaker: String,
        temperature: Option<f32>,
        emotion: Option<String>,
    }

    #[derive(Serialize)]
    pub struct SynthesisResponse {
        success: bool,
        duration_secs: f32,
        message: String,
    }

    pub struct AppState {
        sdk: Sdk,
    }

    pub fn create_router(sdk: Sdk) -> Router {
        let state = AppState { sdk };

        Router::new()
            .route("/synthesize", post(synthesize_handler))
            .with_state(state)
    }

    async fn synthesize_handler(
        State(state): State<AppState>,
        Json(req): Json<SynthesisRequest>,
    ) -> Json<SynthesisResponse> {
        let mut builder = state.sdk.synthesis()
            .text(req.text)
            .speaker(req.speaker);

        if let Some(temp) = req.temperature {
            builder = builder.temperature(temp);
        }

        if let Some(emotion) = req.emotion {
            builder = builder.emotion(emotion, 0.8);
        }

        match builder.build() {
            Ok(audio) => Json(SynthesisResponse {
                success: true,
                duration_secs: audio.duration_secs,
                message: format!("Generated {:.2}s of audio", audio.duration_secs),
            }),
            Err(e) => Json(SynthesisResponse {
                success: false,
                duration_secs: 0.0,
                message: format!("Error: {}", e),
            }),
        }
    }
}

/// Example: Async batch processing
#[cfg(test)]
mod async_examples {
    use sdkwork_tts::sdk::*;
    use futures::future::join_all;

    #[test]
    #[ignore]
    fn example_batch_synthesis() -> Result<(), Box<dyn std::error::Error>> {
        use tokio::runtime::Runtime;

        let rt = Runtime::new()?;
        rt.block_on(async {
            let sdk = SdkBuilder::new()
                .gpu()
                .with_default_engines()
                .build()?;

            let texts = vec![
                "First text to synthesize",
                "Second text to synthesize",
                "Third text to synthesize",
            ];

            // Process in parallel
            let futures: Vec<_> = texts.iter()
                .map(|text| async {
                    sdk.synthesis()
                        .text(*text)
                        .speaker("speaker.wav")
                        .build()
                })
                .collect();

            let results = join_all(futures).await;

            for (i, result) in results.iter().enumerate() {
                match result {
                    Ok(audio) => println!("Text {}: Generated {:.2}s", i, audio.duration_secs),
                    Err(e) => println!("Text {}: Error {}", e),
                }
            }

            Ok::<_, SdkError>(())
        })?;

        Ok(())
    }
}
