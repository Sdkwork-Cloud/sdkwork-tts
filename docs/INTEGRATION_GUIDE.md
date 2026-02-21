# SDKWork-TTS 第三方集成指南

## 概述

SDKWork-TTS 提供了多层次的 API 供第三方应用集成：

1. **SDK API** - 简化的 Facade 模式，适合大多数应用场景
2. **Engine API** - 引擎级别的抽象，适合需要多引擎切换的场景
3. **Core API** - 核心框架组件，适合深度定制场景

## 快速开始

### 1. 添加依赖

```toml
[dependencies]
sdkwork-tts = "0.2"
```

### 2. 基本使用

```rust
use sdkwork_tts::sdk::{Sdk, SdkBuilder};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 初始化 SDK
    let sdk = SdkBuilder::new()
        .gpu()  // 使用 GPU
        .with_default_engines()
        .build()?;

    // 简单合成
    let audio = sdk.synthesize("你好世界", "speaker.wav")?;
    
    // 保存音频
    sdk.save_audio(&audio, "output.wav")?;
    
    Ok(())
}
```

### 3. 使用 Fluent Builder

```rust
use sdkwork_tts::sdk::SdkBuilder;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let sdk = SdkBuilder::new().gpu().build()?;

    // 流式构建 API
    sdk.synthesis()
        .text("你好，这是 SDKWork-TTS")
        .speaker("speaker.wav")
        .emotion("happy", 0.8)
        .temperature(0.8)
        .save("output.wav")?;
    
    Ok(())
}
```

## SDK API 详解

### SdkBuilder

`SdkBuilder` 提供流畅的 SDK 初始化接口：

```rust
use sdkwork_tts::sdk::{Sdk, SdkBuilder, SdkConfig};

// 方式 1: 使用 Builder
let sdk = SdkBuilder::new()
    .gpu(true)                    // 启用 GPU
    .default_engine("indextts2")  // 设置默认引擎
    .memory_limit(4 * 1024 * 1024 * 1024)  // 4GB 内存限制
    .log_level("info")            // 日志级别
    .with_metrics(true)           // 启用指标收集
    .with_event_logging(true)     // 启用事件日志
    .with_default_engines()       // 初始化默认引擎
    .build()?;

// 方式 2: 使用配置对象
let config = SdkConfig::builder()
    .gpu(true)
    .default_engine("indextts2")
    .metrics(true)
    .build();

let sdk = SdkBuilder::from_config(config)
    .with_default_engines()
    .build()?;

// 方式 3: 预设配置
let sdk = Sdk::cpu()?;     // CPU 优化配置
let sdk = Sdk::gpu()?;     // GPU 优化配置
let sdk = Sdk::new_default()?;  // 默认配置
```

### 合成选项

```rust
use sdkwork_tts::sdk::{Sdk, SynthesisOptions, SpeakerRef, EmotionRef};

let sdk = Sdk::new_default()?;

// 方式 1: 简单 API
let audio = sdk.synthesize("文本内容", "speaker.wav")?;

// 方式 2: 使用选项
let options = SynthesisOptions::new("文本", "speaker.wav")
    .with_emotion("happy")
    .with_language("zh")
    .with_temperature(0.8)
    .with_output("output.wav");

let audio = sdk.synthesize_with_options(&options)?;

// 方式 3: 使用 Builder
let audio = sdk.synthesis()
    .text("文本内容")
    .speaker("speaker.wav")
    .emotion(EmotionRef::Name("happy".to_string()), 0.8)
    .language("zh")
    .temperature(0.8)
    .top_k(50)
    .top_p(0.95)
    .build()?;
```

### Speaker 和 Emotion 引用

```rust
use sdkwork_tts::sdk::{SpeakerRef, EmotionRef};

// Speaker 引用
let speaker1 = SpeakerRef::AudioPath(PathBuf::from("voice.wav"));
let speaker2 = SpeakerRef::SpeakerId("speaker_001".to_string());
let speaker3 = SpeakerRef::Default;

// Emotion 引用
let emotion1 = EmotionRef::Name("happy".to_string());
let emotion2 = EmotionRef::Vector(vec![0.8, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1]);
let emotion3 = EmotionRef::AudioPath(PathBuf::from("emotion.wav"));
let emotion4 = EmotionRef::Text("我感到非常开心".to_string());
```

## 多引擎支持

### 引擎管理

```rust
use sdkwork_tts::sdk::SdkBuilder;

let sdk = SdkBuilder::new()
    .with_all_engines()  // 初始化所有可用引擎
    .build()?;

// 列出可用引擎
let engines = sdk.list_engines()?;
for engine in &engines {
    println!("{} v{} - {:?}", engine.id, engine.version, engine.features);
}

// 获取默认引擎
println!("Default engine: {}", sdk.default_engine());
```

### 引擎类型

| 引擎 | ID | 说明 |
|------|-----|------|
| IndexTTS2 | `indextts2` | 零样本声音克隆，情感控制 |
| Fish-Speech | `fish-speech` | 多语言支持，流式合成 |
| Qwen3-TTS | `qwen3-tts` | 10 种语言，97ms 低延迟 |

## 指标和监控

### 性能指标

```rust
use sdkwork_tts::sdk::SdkBuilder;
use sdkwork_tts::core::{PrometheusExporter, JsonExporter, MetricsExporter};

let sdk = SdkBuilder::new()
    .with_metrics(true)
    .build()?;

// 执行一些合成
let _ = sdk.synthesize("Test 1", "speaker.wav");
let _ = sdk.synthesize("Test 2", "speaker.wav");

// 获取指标
if let Some(metrics) = sdk.metrics() {
    let report = metrics.generate_report();
    
    // Prometheus 格式
    let prom = PrometheusExporter::new().export(&report)?;
    
    // JSON 格式
    let json = JsonExporter::new().pretty(true).export(&report)?;
    
    println!("{}", json);
}
```

### SDK 统计

```rust
let stats = sdk.stats();

println!("总合成次数：{}", stats.total_synthesis);
println!("成功次数：{}", stats.successful_synthesis);
println!("失败次数：{}", stats.failed_synthesis);
println!("成功率：{:.2}%", stats.success_rate() * 100.0);
println!("总音频时长：{:.2}s", stats.total_audio_secs);
println!("平均处理时间：{:.2}ms", stats.avg_synthesis_time_ms());
println!("平均 RTF: {:.2}x", stats.avg_rtf);
```

## 事件系统

### 订阅事件

```rust
use std::sync::Arc;
use sdkwork_tts::sdk::SdkBuilder;
use sdkwork_tts::core::{Event, EventHandler, events::*};

// 创建事件处理器
struct LogHandler;

impl EventHandler<ModelLoadingStarted> for LogHandler {
    fn handle(&self, event: &ModelLoadingStarted) -> Result<(), sdkwork_tts::core::error::TtsError> {
        println!("开始加载模型：{} from {}", event.model_name, event.model_path);
        Ok(())
    }
    
    fn handler_name(&self) -> &'static str { "log_handler" }
    
    fn priority(&self) -> i32 { 0 }  // 优先级，越低越先执行
}

impl EventHandler<InferenceCompleted> for LogHandler {
    fn handle(&self, event: &InferenceCompleted) -> Result<(), sdkwork_tts::core::error::TtsError> {
        println!(
            "推理完成：{} 耗时 {}ms 生成 {:.2}s 音频",
            event.request_id, event.duration_ms, event.audio_duration_secs
        );
        Ok(())
    }
}

// 注册事件处理器
let sdk = SdkBuilder::new()
    .with_event_logging(true)
    .build()?;

if let Some(event_bus) = sdk.event_bus() {
    event_bus.subscribe::<ModelLoadingStarted>(Arc::new(LogHandler));
    event_bus.subscribe::<InferenceCompleted>(Arc::new(LogHandler));
}
```

### 内置事件类型

| 事件 | 说明 |
|------|------|
| `ModelLoadingStarted` | 模型加载开始 |
| `ModelLoadingCompleted` | 模型加载完成 |
| `ModelLoadingFailed` | 模型加载失败 |
| `InferenceStarted` | 推理开始 |
| `InferenceCompleted` | 推理完成 |
| `InferenceFailed` | 推理失败 |
| `ResourceLowMemory` | 低内存警告 |
| `ConfigurationChanged` | 配置变更 |
| `StreamingAudioChunk` | 流式音频块 |

## 错误处理

```rust
use sdkwork_tts::sdk::{SdkError, SdkBuilder};

match SdkBuilder::new().build() {
    Ok(sdk) => {
        // SDK 初始化成功
    }
    Err(SdkError::NotInitialized { component }) => {
        eprintln!("组件未初始化：{}", component);
    }
    Err(SdkError::InvalidConfig { field, message }) => {
        eprintln!("配置错误 - {}: {}", field, message);
    }
    Err(SdkError::Engine { engine_id, error }) => {
        eprintln!("引擎 {} 错误：{}", engine_id, error);
    }
    Err(SdkError::Synthesis { message, details }) => {
        eprintln!("合成错误：{}", message);
        if let Some(d) = details {
            eprintln!("详情：{}", d);
        }
    }
    Err(SdkError::Audio { operation, message }) => {
        eprintln!("音频{}失败：{}", operation, message);
    }
    Err(SdkError::Resource { resource_type, message }) => {
        eprintln!("资源{}错误：{}", resource_type, message);
    }
    Err(SdkError::Internal { message }) => {
        eprintln!("内部错误：{}", message);
    }
}
```

## 高级用法

### Web API 集成 (Axum)

```rust
use axum::{Router, Json, extract::State, routing::post};
use sdkwork_tts::sdk::{Sdk, SdkBuilder};
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
struct TtsRequest {
    text: String,
    speaker: String,
    emotion: Option<String>,
}

#[derive(Serialize)]
struct TtsResponse {
    success: bool,
    duration: f32,
}

struct AppState {
    sdk: Sdk,
}

async fn synthesize(
    State(state): State<AppState>,
    Json(req): Json<TtsRequest>,
) -> Json<TtsResponse> {
    let mut builder = state.sdk.synthesis()
        .text(req.text)
        .speaker(req.speaker);
    
    if let Some(emotion) = req.emotion {
        builder = builder.emotion(emotion, 0.8);
    }
    
    match builder.build() {
        Ok(audio) => Json(TtsResponse {
            success: true,
            duration: audio.duration_secs,
        }),
        Err(e) => Json(TtsResponse {
            success: false,
            duration: 0.0,
        }),
    }
}

#[tokio::main]
async fn main() {
    let sdk = SdkBuilder::new().gpu().build().unwrap();
    let state = AppState { sdk };
    
    let app = Router::new()
        .route("/tts", post(synthesize))
        .with_state(state);
    
    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
```

### 批量处理

```rust
use sdkwork_tts::sdk::SdkBuilder;
use futures::future::join_all;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let sdk = SdkBuilder::new().gpu().build()?;
    
    let texts = vec![
        "第一段文本",
        "第二段文本",
        "第三段文本",
    ];
    
    // 并行处理
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
            Ok(audio) => println!("文本 {}: 生成 {:.2}s 音频", i, audio.duration_secs),
            Err(e) => println!("文本 {}: 错误 {}", i, e),
        }
    }
    
    Ok(())
}
```

## 最佳实践

### 1. SDK 生命周期管理

```rust
// 推荐：在应用启动时初始化 SDK，全局复用
lazy_static::lazy_static! {
    static ref SDK: Sdk = SdkBuilder::new()
        .gpu()
        .with_default_engines()
        .build()
        .expect("Failed to initialize SDK");
}

// 避免：每次请求都重新初始化
fn handle_request() {
    let sdk = SdkBuilder::new().build().unwrap();  // 不推荐
    // ...
}
```

### 2. 内存管理

```rust
// 设置内存限制
let sdk = SdkBuilder::new()
    .memory_limit(4 * 1024 * 1024 * 1024)  // 4GB
    .build()?;

// 监控内存使用
let stats = sdk.stats();
println!("已加载引擎：{}", stats.loaded_engines);
```

### 3. 错误恢复

```rust
fn synthesize_with_retry(sdk: &Sdk, text: &str, speaker: &str) -> Result<AudioData, SdkError> {
    let mut attempts = 0;
    let max_attempts = 3;
    
    loop {
        match sdk.synthesize(text, speaker) {
            Ok(audio) => return Ok(audio),
            Err(e) => {
                attempts += 1;
                if attempts >= max_attempts {
                    return Err(e);
                }
                std::thread::sleep(std::time::Duration::from_millis(100 * attempts as u64));
            }
        }
    }
}
```

### 4. 日志配置

```rust
use tracing_subscriber::{fmt, EnvFilter};

// 配置日志
tracing_subscriber::fmt()
    .with_env_filter(EnvFilter::from_default_env())
    .init();

// 设置环境变量控制日志级别
// RUST_LOG=info cargo run
// RUST_LOG=sdkwork_tts=debug cargo run
```

## 故障排除

### 常见问题

1. **GPU 不可用**
   - 确保安装了正确的 CUDA 驱动
   - 使用 `Sdk::cpu()` 回退到 CPU 模式

2. **模型加载失败**
   - 检查模型文件路径是否正确
   - 确保模型文件已下载完整

3. **内存不足**
   - 设置 `memory_limit` 限制内存使用
   - 使用 CPU 模式减少 GPU 内存压力

## 更多资源

- API 文档：`cargo doc --open`
- 示例代码：`examples/sdk_integration.rs`
- 项目仓库：https://github.com/sdkwork/sdkwork-tts
