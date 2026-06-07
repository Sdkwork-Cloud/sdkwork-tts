# SDKWork-TTS 项目上下文

## 项目概述

**SDKWork-TTS** 是一个统一的、可扩展的文本转语音 (TTS) 框架，支持多种引擎（IndexTTS2、Fish-Speech、Qwen3-TTS 等）。项目使用 Rust 编写，基于 HuggingFace Candle 机器学习框架构建。

### 当前状态 (2026 年 2 月)

- **生产就绪** - 框架功能完整，具有模块化架构
- ✅ 187 个单元测试全部通过
- ✅ 端到端 GPU 推理生成清晰语音
- ✅ 支持零样本声音克隆和情感控制
- ✅ 多引擎架构（IndexTTS2 稳定，Fish-Speech/Qwen3-TTS 适配器就绪）
- ✅ **第三方 SDK 集成 API** - 简化的 Facade 模式，易于集成

### 技术栈

| 类别 | 技术 |
|------|------|
| 语言 | Rust 1.75+ |
| ML 框架 | Candle (candle-core, candle-nn, candle-transformers 0.8) |
| 音频处理 | cpal, rodio, rubato, rustfft, hound, symphonia |
| 异步运行时 | Tokio 1.42 |
| CLI | Clap 4.5 |
| 配置 | Serde + YAML/JSON/TOML |
| 日志 | Tracing |
| Web 集成 | Axum (可选) |

## 构建与运行

### 环境要求

- Rust 1.75+
- CUDA 兼容 GPU（推荐 RTX 5090，已测试）
- 模型权重文件位于 `checkpoints/` 目录

### 构建命令

```powershell
# CPU 构建
cargo build --release

# CUDA 构建
$env:CUDA_COMPUTE_CAP='90'
cargo build --release --features cuda

# Metal 构建 (macOS)
cargo build --release --features metal
```

### 运行测试

```powershell
# 运行所有测试
cargo test

# 运行基准测试
cargo bench
```

### CLI 使用

```powershell
# 列出可用引擎
./target/release/sdkwork-tts.exe engines

# 基本合成 (IndexTTS2)
./target/release/sdkwork-tts.exe infer `
  --speaker checkpoints/speaker_16k.wav `
  --text "Hello from SDKWork-TTS" `
  --output output.wav `
  --de-rumble --de-rumble-cutoff-hz 180

# 使用 Fish-Speech 引擎
./target/release/sdkwork-tts.exe infer `
  --engine fish-speech `
  --speaker checkpoints/speaker_16k.wav `
  --text "你好世界" `
  --language zh `
  --output output.wav

# 情感控制 (IndexTTS2)
./target/release/sdkwork-tts.exe infer `
  --speaker checkpoints/speaker_16k.wav `
  --emotion-audio emotion.wav --emotion-alpha 0.35 `
  --text "这应该听起来平静自然" `
  --output emotion_output.wav
```

## 项目结构

```
sdkwork-tts/
├── src/
│   ├── main.rs                  # CLI 入口点
│   ├── lib.rs                   # 库导出
│   ├── core/                    # 核心框架
│   │   ├── error.rs            # 结构化错误处理
│   │   ├── error_ext.rs        # 增强错误处理（错误码、恢复策略）
│   │   ├── traits.rs           # 组件特征
│   │   ├── resource.rs         # 资源管理
│   │   ├── metrics.rs          # 性能监控
│   │   ├── metrics_export.rs   # 多格式指标导出
│   │   ├── event_bus.rs        # 发布 - 订阅事件系统
│   │   ├── plugin.rs           # 插件系统
│   │   └── builder.rs          # 构建器模式
│   ├── engine/                  # 引擎抽象层
│   │   ├── traits.rs           # TtsEngine 特征
│   │   ├── registry.rs         # 引擎注册表
│   │   ├── pipeline.rs         # 处理管道
│   │   ├── config.rs           # 引擎配置
│   │   ├── speaker.rs          # 说话人管理
│   │   ├── emotion.rs          # 情感管理
│   │   ├── indextts2_adapter.rs
│   │   ├── fish_speech_adapter.rs
│   │   └── qwen3_tts_adapter.rs
│   ├── models/                  # 神经网络模型
│   │   ├── semantic/           # Wav2Vec-BERT, codec
│   │   ├── speaker/            # CAMPPlus
│   │   ├── gpt/                # UnifiedVoice, Conformer
│   │   ├── s2mel/              # DiT, Flow Matching
│   │   └── vocoder/            # BigVGAN
│   ├── inference/              # 推理管道
│   ├── audio/                  # 音频 I/O
│   ├── text/                   # 文本处理
│   └── utils/                  # 工具函数
├── checkpoints/                 # 模型权重
├── docs/                        # 文档
├── examples/                    # 示例代码
├── benches/                     # 基准测试
├── tests/                       # 集成测试
├── scripts/                     # Python 脚本工具
└── debug/                       # 调试输出
```

## 支持的引擎

| 引擎 | 状态 | 语言 | 声音克隆 | 流式 | 情感控制 |
|------|------|------|----------|------|----------|
| **IndexTTS2** | ✅ 稳定 | zh, en, ja | ✅ 3-30s | ✅ | ✅ |
| **Fish-Speech** | 🚧 适配器 | zh, en, ja, ko, de, fr | ✅ | ✅ | 📋 |
| **Qwen3-TTS** | 🚧 适配器 | 10 种语言 | ✅ 3s | ✅ 97ms | ✅ |

## 架构概览

```
┌─────────────────────────────────────────────────────────────┐
│                    SDKWork-TTS Framework                    │
├─────────────────────────────────────────────────────────────┤
│                    Unified TTS API                          │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              TtsEngine Trait                        │   │
│  │  - synthesize()    - synthesize_streaming()         │   │
│  │  - get_speakers()  - get_emotions()                 │   │
│  │  - load_model()    - unload_model()                 │   │
│  └─────────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────────┤
│                    Engine Registry                          │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐     │
│  │ IndexTTS │ │   Fish   │ │  Qwen3   │ │  Future  │     │
│  │    2     │ │  Speech  │ │   TTS    │ │  Engines │     │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘     │
├─────────────────────────────────────────────────────────────┤
│                    IndexTTS2 Pipeline:                      │
│  1. TEXT PROCESSING    → Tokenizer → Token IDs             │
│  2. SPEAKER ENCODING   → Wav2Vec-BERT + CAMPPlus           │
│  3. GPT GENERATION     → UnifiedVoice (1280d, 24 layers)   │
│  4. S2MEL (DiT)        → Flow Matching (25 steps)          │
│  5. VOCODER (BigVGAN)  → 22050 Hz Waveform                 │
└─────────────────────────────────────────────────────────────┘
```

## 核心模块说明

### Core 框架 (`src/core/`)

- **Error Handling** (`error.rs`): 结构化错误类型 (`TtsError`)，支持错误链和上下文
- **Extended Error** (`error_ext.rs`): 增强错误处理，支持错误码、严重性级别、恢复策略
- **Traits** (`traits.rs`): 统一组件接口 (`ModelComponent`, `Encoder`, `Decoder`, `Synthesizer`)
- **Resource Management** (`resource.rs`): 集中式资源生命周期管理，支持引用计数和内存跟踪
- **Metrics** (`metrics.rs`): 性能监控，支持计时器、计数器、直方图统计
- **Metrics Export** (`metrics_export.rs`): 多格式指标导出 (Prometheus、JSON、CSV、Console)
- **Event Bus** (`event_bus.rs`): 发布 - 订阅事件系统，支持类型安全的事件通道
- **Plugin System** (`plugin.rs`): 插件架构，支持动态插件注册和依赖管理
- **Builder** (`builder.rs`): 构建器模式，提供流畅 API 和预设配置

### Engine 层 (`src/engine/`)

- **TtsEngine Trait**: 统一引擎接口
- **EngineRegistry**: 引擎注册和发现
- **ProcessingPipeline**: 标准化处理流程
- **Speaker/Emotion Manager**: 说话人和情感管理

### Models (`src/models/`)

| 模块 | 组件 | 说明 |
|------|------|------|
| semantic/ | Wav2Vec-BERT 2.0, Codec | 语义编码器 |
| speaker/ | CAMPPlus | 说话人风格向量 (192-dim) |
| gpt/ | UnifiedVoice, Conformer, Perceiver | 1280 维，24 层，20 头 |
| s2mel/ | DiT, Flow Matching, LengthRegulator | 13 层 DiT，25 步流匹配 |
| vocoder/ | BigVGAN v2 | 22050 Hz 波形生成 |

## 开发约定

### 代码风格

- 使用 `rustfmt` 格式化代码
- 公共 API 必须有文档注释 (`///`)
- 错误处理使用 `Result<T, TtsError>` 而非 `anyhow`
- 使用构建器模式构造复杂对象

### 测试实践

- 单元测试放在各模块的 `#[cfg(test)]` 中
- 集成测试位于 `tests/` 目录
- 基准测试使用 Criterion，位于 `benches/`
- 所有测试必须通过才能提交

### 调试工具

项目包含多个调试工具：

```powershell
# 权重诊断
./target/release/diagnose_weights

# 长度调节器测试
./target/release/test_length_regulator

# Python 脚本工具
python scripts/analyze_weights.py
python scripts/audio_metrics.py
```

## 扩展新引擎

```rust
use sdkwork_tts::engine::{TtsEngine, TtsEngineInfo, SynthesisRequest, SynthesisResult};
use async_trait::async_trait;

pub struct MyTtsEngine {
    info: TtsEngineInfo,
}

#[async_trait]
impl TtsEngine for MyTtsEngine {
    fn info(&self) -> &TtsEngineInfo { &self.info }

    async fn initialize(&mut self, config: &EngineConfig) -> Result<()> {
        // 加载模型
    }

    async fn synthesize(&self, request: &SynthesisRequest) -> Result<SynthesisResult> {
        // 实现合成逻辑
    }
}

// 注册引擎
sdkwork_tts::engine::global_registry().register_lazy(
    "my-engine",
    info,
    || Ok(Box::new(MyTtsEngine::new()))
)?;
```

## 配置预设

```rust
use sdkwork_tts::core::builder::presets;

// 高质量
let config = presets::high_quality();  // temp=0.7, steps=50, cfg=0.7

// 快速
let config = presets::fast();          // temp=0.9, steps=10, cfg=0.0

// 流式
let config = presets::streaming();     // temp=0.8, steps=15, max_mel=500

// 低延迟
let config = presets::low_latency();   // temp=0.9, steps=5, max_mel=200
```

## 已知问题与调试

### 当前技术差距

- GPT step0 logits 接近参考实现，但 step1+ 缓存解码存在漂移
- 下一步工作：对比 step1 内部张量（pre/post LN, q/k/v, 注意力分数）

### 推荐配置

从质量扫描测试中得出的推荐预设：

- `top-k 0`, `top-p 1.0`, `temperature 0.8`
- `repetition-penalty 1.05`
- `flow-steps 25`, `flow-cfg-rate 0.7`
- `de-rumble cutoff 180 Hz`

### 调试资源

- `DEBUGGING.md` - 详细调试日志
- `docs/HANDOFF_2026-02-11.md` - 交接文档
- `docs/STATUS_2026-02-11.md` - 状态更新
- `scripts/parity_compare.py` - Python/Rust 对比工具

## 关键文件

| 文件 | 用途 |
|------|------|
| `README.md` | 项目概述和快速开始 |
| `Cargo.toml` | 依赖和构建配置 |
| `docs/ARCHITECTURE.md` | 详细架构文档 |
| `src/lib.rs` | 库入口和导出 |
| `src/inference/pipeline.rs` | 核心推理管道 |
| `CLAUDE.md` / `CURRENT_STATUS.md` | 当前状态摘要 |

## 相关文档

- 架构文档：`docs/ARCHITECTURE.md`
- API 参考：`docs/API.md` (如存在)
- 状态报告：`docs/STATUS_2026-02-11.md`
- 调试日志：`DEBUGGING.md`

---

## 新增框架级组件 (2026 年 2 月迭代)

本次迭代新增了以下框架级通用组件，提升了框架的可扩展性和专业性：

### 1. 增强错误处理 (`error_ext.rs`)

```rust
use sdkwork_tts::core::{ErrorCode, ErrorSeverity, RecoveryStrategy, ExtendedErrorInfo, to_extended_error};

// 获取错误码和严重性
let err = TtsError::ModelLoad { ... };
let ext = to_extended_error(&err);
assert_eq!(ext.code, ErrorCode::ModelNotFound);
assert_eq!(ext.severity, ErrorSeverity::Error);

// 使用恢复策略
let recovery = RecoveryStrategy::Retry {
    max_attempts: 3,
    delay_ms: 100,
    backoff_percent: 200,
};
```

### 2. 事件总线系统 (`event_bus.rs`)

```rust
use sdkwork_tts::core::{EventBus, Event, EventHandler, events::*};

// 创建事件总线
let bus = EventBus::new();

// 定义事件处理器
struct LogHandler;
impl EventHandler<ModelLoadingStarted> for LogHandler {
    fn handle(&self, event: &ModelLoadingStarted) -> Result<()> {
        println!("Loading model: {}", event.model_name);
        Ok(())
    }
    fn priority(&self) -> i32 { 0 }
}

// 订阅事件
bus.subscribe(Arc::new(LogHandler));

// 发布事件
bus.publish(&ModelLoadingStarted {
    model_name: "test".to_string(),
    model_path: "/path".to_string(),
})?;
```

### 3. 多格式指标导出 (`metrics_export.rs`)

```rust
use sdkwork_tts::core::{
    MetricsCollector, PrometheusExporter, JsonExporter,
    ConsoleExporter, MultiExporter, MetricsExporter
};

let collector = MetricsCollector::new();
// ... 记录指标 ...
let report = collector.generate_report();

// Prometheus 格式
let prom = PrometheusExporter::new().export(&report)?;

// JSON 格式
let json = JsonExporter::new().pretty(true).export(&report)?;

// 控制台格式
let console = ConsoleExporter::new().colored(true).export(&report)?;

// 多格式同时导出
let mut multi = MultiExporter::new();
multi.add_exporter(PrometheusExporter::new());
multi.add_exporter(JsonExporter::new());
let all = multi.export_all(&report)?;
```

### 4. 插件系统 (`plugin.rs`)

```rust
use sdkwork_tts::core::{Plugin, PluginRegistry, PluginContext, PluginBuilder};

// 定义插件
struct MyPlugin;
impl Plugin for MyPlugin {
    fn plugin_id(&self) -> &'static str { "my-plugin" }
    fn name(&self) -> &'static str { "My Plugin" }
    fn version(&self) -> &'static str { "1.0.0" }
    fn description(&self) -> &'static str { "A demo plugin" }
    
    fn initialize(&mut self, ctx: &PluginContext) -> Result<()> {
        // 初始化逻辑
        Ok(())
    }
}

// 注册插件
let registry = PluginRegistry::new("/plugins")?;
registry.register(MyPlugin)?;

// 管理插件
registry.enable_plugin("my-plugin")?;
registry.disable_plugin("my-plugin")?;
registry.unregister("my-plugin")?;
```

### 5. 内置事件类型

框架预定义了以下事件类型：

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

---

## SDK 集成 API (第三方应用)

### 快速开始

```rust
use sdkwork_tts::sdk::{Sdk, SdkBuilder};

// 初始化 SDK
let sdk = SdkBuilder::new()
    .gpu()
    .with_default_engines()
    .build()?;

// 简单合成
let audio = sdk.synthesize("Hello world", "speaker.wav")?;
sdk.save_audio(&audio, "output.wav")?;

// 或使用 Fluent Builder
sdk.synthesis()
    .text("Hello world")
    .speaker("speaker.wav")
    .temperature(0.8)
    .save("output.wav")?;
```

### SDK 类型

| 类型 | 说明 |
|------|------|
| `Sdk` | 主 SDK Facade |
| `SdkBuilder` | SDK 构建器 |
| `SdkConfig` | SDK 配置 |
| `SynthesisOptions` | 合成选项 |
| `SpeakerRef` | 说话人引用 |
| `EmotionRef` | 情感引用 |
| `AudioData` | 音频数据 |
| `SdkError` | SDK 错误类型 |
| `SdkStats` | SDK 统计信息 |

### 配置选项

```rust
use sdkwork_tts::sdk::SdkConfig;

// Builder 模式
let config = SdkConfig::builder()
    .gpu(true)
    .default_engine("indextts2")
    .memory_limit(4 * 1024 * 1024 * 1024)
    .metrics(true)
    .event_logging(true)
    .build();

// 预设
let config = SdkConfig::cpu();     // CPU 优化
let config = SdkConfig::gpu();     // GPU 优化
let config = SdkConfig::high_quality();  // 高质量
let config = SdkConfig::fast();    // 快速合成
```

### 错误处理

```rust
use sdkwork_tts::sdk::SdkError;

match sdk.synthesize("text", "speaker.wav") {
    Ok(audio) => { /* 成功 */ }
    Err(SdkError::NotInitialized { component }) => { /* 未初始化 */ }
    Err(SdkError::InvalidConfig { field, message }) => { /* 配置错误 */ }
    Err(SdkError::Engine { engine_id, error }) => { /* 引擎错误 */ }
    Err(SdkError::Synthesis { message, details }) => { /* 合成错误 */ }
    Err(SdkError::Audio { operation, message }) => { /* 音频错误 */ }
    Err(SdkError::Resource { resource_type, message }) => { /* 资源错误 */ }
    Err(SdkError::Internal { message }) => { /* 内部错误 */ }
}
```

### 文档

- 集成指南：`docs/INTEGRATION_GUIDE.md`
- 示例代码：`examples/sdk_integration.rs`
