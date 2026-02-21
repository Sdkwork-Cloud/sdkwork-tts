# IndexTTS2 Architecture Guide

## Overview

IndexTTS2 is a production-ready, framework-level text-to-speech system built with Rust. This document describes the architecture, design patterns, and best practices for extending and maintaining the system.

## Core Architecture

### Layer Structure

```
┌─────────────────────────────────────────────────────────────┐
│                    Application Layer                        │
│         (CLI, API, Streaming, Batch Processing)             │
├─────────────────────────────────────────────────────────────┤
│                    Core Framework                           │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐          │
│  │  Error  │ │  Traits │ │Resource │ │ Metrics │          │
│  │Handling │ │         │ │ Manager │ │         │          │
│  └─────────┘ └─────────┘ └─────────┘ └─────────┘          │
│  ┌─────────┐ ┌─────────┐                                   │
│  │ Builder │ │Validation│                                  │
│  │ Pattern │ │         │                                   │
│  └─────────┘ └─────────┘                                   │
├─────────────────────────────────────────────────────────────┤
│                    Inference Engine                         │
│         (Pipeline, Streaming, Audio I/O)                    │
├─────────────────────────────────────────────────────────────┤
│                    Model Components                         │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐          │
│  │   GPT   │ │ S2Mel   │ │ Vocoder │ │ Speaker │          │
│  │  Model  │ │  Model  │ │         │ │ Encoder │          │
│  └─────────┘ └─────────┘ └─────────┘ └─────────┘          │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐                       │
│  │ Semantic│ │ Emotion │ │  Codec  │                       │
│  │ Encoder │ │ Control │ │         │                       │
│  └─────────┘ └─────────┘ └─────────┘                       │
├─────────────────────────────────────────────────────────────┤
│                    Infrastructure                           │
│    (Text Processing, Audio Processing, Utilities)           │
└─────────────────────────────────────────────────────────────┘
```

## Core Framework (`src/core/`)

### 1. Error Handling (`error.rs`)

Structured error types for the entire TTS pipeline:

```rust
pub enum TtsError {
    Config { message, path },
    ModelLoad { message, component, path },
    Inference { stage, message, recoverable },
    Audio { message, operation },
    Text { message, operation },
    Resource { message, resource_type },
    Device { message, device_type },
    Validation { message, field },
    Io { message, path },
    Timeout { message, duration_ms },
    Internal { message, location },
}
```

**Key Features:**
- Rich context with operation, component, and suggestions
- Error chaining with `ResultExt` trait
- Conversion from `anyhow`, `std::io::Error`, `candle_core::Error`

### 2. Traits (`traits.rs`)

Core abstractions for all components:

```rust
// Base trait for all model components
pub trait ModelComponent: Send + Sync {
    fn name(&self) -> &str;
    fn is_ready(&self) -> bool;
    fn memory_usage(&self) -> Option<usize>;
}

// For configurable components
pub trait Configurable {
    type Config;
    fn with_config(config: &Self::Config, device: &Device) -> Result<Self>;
    fn update_config(&mut self, config: &Self::Config) -> Result<()>;
}

// For loadable components (weights, vocabularies)
pub trait Loadable: ModelComponent {
    fn load<P: AsRef<Path>>(path: P, device: &Device) -> Result<Self>;
    fn is_loaded(&self) -> bool;
    fn unload(&mut self) -> Result<()>;
}

// For encoder components
pub trait Encoder: ModelComponent {
    type Input;
    type Output;
    fn encode(&self, input: &Self::Input) -> Result<Self::Output>;
    fn output_dim(&self) -> usize;
}

// For decoder components
pub trait Decoder: ModelComponent {
    type Input;
    type Output;
    fn decode(&self, input: &Self::Input) -> Result<Self::Output>;
}

// For synthesizer components
pub trait Synthesizer: ModelComponent {
    type Condition;
    type Output;
    type Options;
    fn synthesize(&self, condition: &Self::Condition, options: &Self::Options) -> Result<Self::Output>;
}
```

### 3. Resource Management (`resource.rs`)

Centralized resource lifecycle management:

```rust
pub struct ResourceManager {
    device: Device,
    resources: HashMap<ResourceId, ResourceEntry>,
    memory_limit: usize,
    gpu_memory_limit: Option<usize>,
    idle_timeout: Duration,
}

pub struct ResourceHandle {
    id: ResourceId,
    resource_type: ResourceType,
    name: String,
    ref_count: Arc<Mutex<usize>>,
}
```

**Features:**
- Reference counting for shared resources
- Automatic idle resource cleanup
- Memory usage tracking (CPU and GPU)
- Resource statistics and reporting

### 4. Metrics (`metrics.rs`)

Comprehensive performance monitoring:

```rust
pub struct MetricsCollector {
    timers: HashMap<String, TimerStats>,
    counters: HashMap<String, u64>,
    gauges: HashMap<String, f64>,
    histograms: HashMap<String, Vec<f64>>,
}

pub struct PerformanceMetrics {
    total_time: Duration,
    text_processing_time: Duration,
    speaker_encoding_time: Duration,
    gpt_generation_time: Duration,
    flow_matching_time: Duration,
    vocoding_time: Duration,
    mel_frames: usize,
    tokens_generated: usize,
    rtf: f64,
}
```

**Features:**
- Timer statistics (avg, min, max)
- Counter increments
- Gauge values
- Histogram percentiles (p50, p95, p99)

### 5. Builder Pattern (`builder.rs`)

Ergonomic APIs for complex object construction:

```rust
// TTS Builder
let tts = TtsBuilder::new("checkpoints/config.yaml")
    .with_gpu(true)
    .with_temperature(0.8)
    .with_flow_steps(25)
    .with_memory_limit(4 * 1024 * 1024 * 1024)
    .build()?;

// Config Builder
let config = TtsConfigBuilder::new()
    .temperature(0.8)
    .flow_steps(25)
    .gpu(true)
    .build();

// Presets
let config = presets::high_quality();
let config = presets::fast();
let config = presets::streaming();
```

## Design Patterns

### 1. Trait-Based Architecture

All major components implement standardized traits:

```rust
impl ModelComponent for MyEncoder {
    fn name(&self) -> &str { "my_encoder" }
    fn is_ready(&self) -> bool { self.weights_loaded }
}

impl Encoder for MyEncoder {
    type Input = Vec<f32>;
    type Output = Vec<f32>;
    
    fn encode(&self, input: &Self::Input) -> Result<Self::Output> {
        // Implementation
    }
    
    fn output_dim(&self) -> usize { 128 }
}
```

### 2. Error Context

Rich error reporting with context:

```rust
let result = operation
    .with_context(|| ErrorContext::new()
        .with_operation("loading model")
        .with_component("GPT")
        .with_suggestion("Check model path"))?;
```

### 3. Resource Management

Automatic resource lifecycle:

```rust
let handle = resource_manager.register(
    ResourceType::Model,
    "gpt_model",
    Some(PathBuf::from("model.safetensors")),
    1024 * 1024 * 100, // 100 MB
)?;

// Use resource
handle.acquire();
// ... use resource ...
handle.release();

// Auto-cleanup when idle
resource_manager.cleanup_idle()?;
```

### 4. Metrics Collection

Performance tracking:

```rust
let collector = MetricsCollector::new();

// Time an operation
let timer = collector.start_timer("inference");
// ... do work ...
timer.stop();

// Or use RAII
{
    let _timer = collector.start_timer("encode");
    encode(input)?;
} // Automatically recorded on drop

// Generate report
let report = collector.generate_report();
println!("{}", report.format());
```

## Extension Guide

### Adding a New Model Component

1. **Implement Core Traits:**

```rust
use indextts2::core::{
    ModelComponent, Configurable, Loadable,
    error::{Result, TtsError},
};

pub struct MyModel {
    name: String,
    device: Device,
    weights_loaded: bool,
}

impl ModelComponent for MyModel {
    fn name(&self) -> &str { &self.name }
    fn is_ready(&self) -> bool { self.weights_loaded }
}

impl Configurable for MyModel {
    type Config = MyModelConfig;
    
    fn with_config(config: &Self::Config, device: &Device) -> Result<Self> {
        Ok(Self {
            name: config.name.clone(),
            device: device.clone(),
            weights_loaded: false,
        })
    }
}

impl Loadable for MyModel {
    fn load<P: AsRef<Path>>(path: P, device: &Device) -> Result<Self> {
        // Load weights
        Ok(Self {
            name: "my_model".to_string(),
            device: device.clone(),
            weights_loaded: true,
        })
    }
    
    fn is_loaded(&self) -> bool { self.weights_loaded }
}
```

2. **Add to Pipeline:**

```rust
impl IndexTTS2 {
    pub fn with_my_model(mut self, model: MyModel) -> Self {
        self.my_model = Some(model);
        self
    }
}
```

3. **Add Builder Support:**

```rust
impl TtsBuilder {
    pub fn with_my_model(mut self, model: MyModel) -> Self {
        self.my_model = Some(model);
        self
    }
}
```

### Adding a New Inference Stage

1. **Add to Error Types:**

```rust
pub enum InferenceStage {
    // ... existing stages
    MyNewStage,
}
```

2. **Implement Stage:**

```rust
impl IndexTTS2 {
    fn my_new_stage(&self, input: &Input) -> Result<Output> {
        let timer = self.metrics.start_timer("my_new_stage");
        
        let result = self.my_model
            .as_ref()
            .ok_or_else(|| TtsError::Inference {
                stage: InferenceStage::MyNewStage,
                message: "Model not loaded".to_string(),
                recoverable: false,
            })?
            .process(input)
            .map_err(|e| TtsError::Inference {
                stage: InferenceStage::MyNewStage,
                message: e.to_string(),
                recoverable: false,
            })?;
        
        timer.stop();
        Ok(result)
    }
}
```

## Best Practices

### 1. Error Handling

- Use structured `TtsError` types
- Add context with `ErrorContext`
- Mark recoverable vs non-recoverable errors
- Provide actionable suggestions

### 2. Resource Management

- Always use `ResourceManager` for model weights
- Properly acquire/release handles
- Set appropriate memory limits
- Enable idle cleanup for long-running services

### 3. Performance

- Use `MetricsCollector` for all operations
- Profile before optimizing
- Consider batching for throughput
- Use streaming for real-time applications

### 4. Testing

- Unit tests for all components
- Integration tests for full pipeline
- Benchmarks for performance regression
- Property-based tests for invariants

### 5. Documentation

- Document all public APIs
- Include examples in doc comments
- Keep architecture docs updated
- Document breaking changes

## Configuration Presets

### High Quality

```rust
let config = presets::high_quality();
// temperature: 0.7, flow_steps: 50, cfg_rate: 0.7
```

### Fast

```rust
let config = presets::fast();
// temperature: 0.9, flow_steps: 10, cfg_rate: 0.0
```

### Streaming

```rust
let config = presets::streaming();
// temperature: 0.8, flow_steps: 15, max_mel_tokens: 500
```

### Low Latency

```rust
let config = presets::low_latency();
// temperature: 0.9, flow_steps: 5, max_mel_tokens: 200
```

## Migration Guide

### From Old API to New Builder API

**Before:**
```rust
let mut tts = IndexTTS2::new("config.yaml")?;
tts.inference_config.temperature = 0.8;
tts.inference_config.flow_steps = 25;
tts.load_weights("checkpoints/")?;
```

**After:**
```rust
let tts = TtsBuilder::new("checkpoints/config.yaml")
    .with_temperature(0.8)
    .with_flow_steps(25)
    .build()?;
```

### Error Handling Migration

**Before:**
```rust
let result = operation.map_err(|e| anyhow!("Failed: {}", e))?;
```

**After:**
```rust
let result = operation.map_err(|e| TtsError::Inference {
    stage: InferenceStage::MyStage,
    message: e.to_string(),
    recoverable: false,
})?;
```

## Future Enhancements

### Planned Features

1. **Async Support**: Full async/await for I/O operations
2. **Distributed Inference**: Multi-GPU and multi-node support
3. **Model Serving**: gRPC/HTTP API for production deployment
4. **Quantization**: INT8/INT4 for faster inference
5. **ONNX Export**: Cross-platform model deployment

### Extension Points

- Custom encoders/decoders via traits
- Pluggable emotion models
- Alternative vocoders
- Custom text processors
- Streaming adapters

---

For more information, see:
- [API Documentation](https://docs.rs/indextts2)
- [Examples](../examples/)
- [Contributing Guide](CONTRIBUTING.md)
