# SDKWork-TTS

<div align="center">

**A Unified, Extensible Text-to-Speech Framework**

[![Rust](https://img.shields.io/badge/Rust-1.75+-orange.svg)](https://www.rust-lang.org)
[![License](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-48%20passed-green.svg)]()
[![Engines](https://img.shields.io/badge/Engines-3%20supported-blue.svg)]()

**Supports:** IndexTTS2 | Qwen3-TTS | Fish-Speech

</div>

---

## 📖 目录

- [快速开始](#-快速开始)
- [功能特性](#-功能特性)
- [支持的引擎](#-支持的引擎)
- [安装](#-安装)
- [命令行使用](#-命令行使用)
- [库使用](#-库使用)
- [架构设计](#-架构设计)
- [性能基准](#-性能基准)
- [开发指南](#-开发指南)
- [常见问题](#-常见问题)

---

## 🚀 快速开始

### 1. 环境要求

- **Rust**: 1.75+
- **GPU** (可选): NVIDIA GPU with CUDA (推荐 RTX 3090/4090/5090)
- **模型权重**: 放置在 `checkpoints/` 目录

### 2. 克隆项目

```bash
git clone https://github.com/sdkwork/sdkwork-tts.git
cd sdkwork-tts
```

### 3. 构建项目

```bash
# CPU 版本
cargo build --release

# CUDA 版本 (推荐，需要 GPU)
$env:CUDA_COMPUTE_CAP='90'  # PowerShell
# export CUDA_COMPUTE_CAP='90'  # Linux/Mac
cargo build --release --features cuda
```

### 4. 快速测试

```bash
# 列出可用引擎
./target/release/sdkwork-tts engines

# 使用 IndexTTS2 合成语音
./target/release/sdkwork-tts infer \
  --engine indextts2 \
  --speaker checkpoints/speaker.wav \
  --text "你好，这是 SDKWork-TTS 框架" \
  --output output.wav

# 使用 Qwen3-TTS 合成语音
./target/release/sdkwork-tts infer \
  --engine qwen3-tts \
  --speaker checkpoints/speaker.wav \
  --text "Hello, this is SDKWork-TTS" \
  --language en \
  --output output.wav
```

---

## ✨ 功能特性

### 核心功能

- 🎯 **多引擎支持**: 统一的 API 支持多种 TTS 引擎
- 🎤 **零样本声音克隆**: 从 3-30 秒参考音频克隆声音
- 😊 **情感控制**: 细粒度的情感和风格控制
- 🌊 **流式合成**: 实时音频流式输出
- 🚀 **GPU 加速**: 通过 Candle 支持 CUDA 和 Metal
- 📦 **生产就绪**: 全面的错误处理和资源管理

### 技术亮点

- **模块化架构**: 易于扩展新引擎
- **高性能**: 优化的推理管道，RTF < 1.0
- **类型安全**: Rust 类型系统保证
- **跨平台**: Windows/Linux/macOS 支持

---

## 🔧 支持的引擎

### 引擎对比

| 引擎 | 状态 | 语言 | 声音克隆 | 流式 | 情感控制 | RTF |
|------|------|------|----------|------|----------|-----|
| **IndexTTS2** | ✅ 稳定 | zh, en, ja | ✅ 3-30s | ✅ | ✅ | ~0.8 |
| **Qwen3-TTS** | ✅ 稳定 | 10 种 | ✅ 3s | ✅ 97ms | ✅ | ~0.3 |
| **Fish-Speech** | 🚧 适配器 | 6 种 | ✅ | ✅ | 📋 | ~0.5 |

### IndexTTS2 (Bilibili)

- **类型**: Flow Matching
- **语言**: 中文、英语、日语
- **特点**:
  - 零样本声音克隆 (3-30 秒参考音频)
  - 音频参考或向量情感控制
  - 文本情感提取 (Qwen)
  - 实时合成 (RTF ~0.8-1.1x)

### Qwen3-TTS (Alibaba)

- **类型**: Autoregressive
- **语言**: 中文、英语、日语、韩语、德语、法语、俄语、葡萄牙语、西班牙语、意大利语
- **特点**:
  - 10 种主要语言支持
  - 超低延迟流式 (97ms 首包)
  - 自然语言声音设计
  - 3 秒快速声音克隆
  - 指令式声音控制

### Fish-Speech

- **类型**: Autoregressive
- **语言**: 中文、英语、日语、韩语、德语、法语
- **特点**:
  - 多语言支持
  - 流式合成
  - 批量处理
  - 高质量自然语音

---

## 📦 安装

### 系统要求

| 组件 | 最低要求 | 推荐配置 |
|------|---------|---------|
| **Rust** | 1.75 | 1.80+ |
| **CPU** | 4 核心 | 8+ 核心 |
| **内存** | 8 GB | 16+ GB |
| **GPU** | 可选 | RTX 3090+ |
| **显存** | - | 24+ GB |
| **存储** | 10 GB | 50+ GB |

### 构建选项

```bash
# CPU 构建
cargo build --release --no-default-features --features cpu

# CUDA 构建 (NVIDIA GPU)
$env:CUDA_COMPUTE_CAP='90'
cargo build --release --features cuda

# Metal 构建 (macOS)
cargo build --release --features metal

# MKL 加速 (Intel CPU)
cargo build --release --features mkl
```

### 模型下载

```bash
# IndexTTS2 模型
huggingface-cli download IndexTeam/IndexTTS-2 \
  --local-dir checkpoints/indextts2

# Qwen3-TTS 模型
huggingface-cli download Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --local-dir checkpoints/qwen3-tts

# Fish-Speech 模型
huggingface-cli download fishaudio/fish-speech-1.4 \
  --local-dir checkpoints/fish-speech
```

---

## 💻 命令行使用

### 基础命令

#### 列出引擎

```bash
# 列出所有可用引擎
./target/release/sdkwork-tts engines

# 显示详细信息
./target/release/sdkwork-tts engines --detailed
```

#### 引擎信息

```bash
# 查看特定引擎信息
./target/release/sdkwork-tts info --engine indextts2
./target/release/sdkwork-tts info --engine qwen3-tts
```

### 合成命令

#### IndexTTS2 示例

```bash
# 基础合成
./target/release/sdkwork-tts infer \
  --engine indextts2 \
  --speaker checkpoints/speaker.wav \
  --text "你好，这是 IndexTTS2 合成的声音" \
  --output output.wav

# 带情感控制
./target/release/sdkwork-tts infer \
  --engine indextts2 \
  --speaker checkpoints/speaker.wav \
  --emotion-audio checkpoints/emotion.wav \
  --emotion-alpha 0.8 \
  --text "这应该听起来很快乐" \
  --output emotion_output.wav

# 使用情感向量
./target/release/sdkwork-tts infer \
  --engine indextts2 \
  --speaker checkpoints/speaker.wav \
  --emotion-vector "0.6,0.0,0.0,0.0,0.0,0.0,0.1,0.2" \
  --emotion-alpha 0.9 \
  --text "情感向量测试" \
  --output emotion_vector.wav

# 文本情感推断
./target/release/sdkwork-tts infer \
  --engine indextts2 \
  --speaker checkpoints/speaker.wav \
  --use-emo-text \
  --emo-text "我感到非常开心和兴奋" \
  --text "这是情感文本推断测试" \
  --output emotion_text.wav
```

#### Qwen3-TTS 示例

```bash
# 使用预设说话人
./target/release/sdkwork-tts infer \
  --engine qwen3-tts \
  --speaker checkpoints/speaker.wav \
  --text "你好，这是 Qwen3-TTS 合成的声音" \
  --language zh \
  --output output.wav

# 指定说话人
./target/release/sdkwork-tts infer \
  --engine qwen3-tts \
  --speaker-id vivian \
  --text "Hello from Vivian" \
  --language en \
  --output vivian_output.wav

# 声音克隆
./target/release/sdkwork-tts infer \
  --engine qwen3-tts \
  --speaker checkpoints/reference.wav \
  --text "这是克隆的声音" \
  --language zh \
  --output cloned.wav

# 声音设计
./target/release/sdkwork-tts infer \
  --engine qwen3-tts \
  --model qwen3-tts-voicedesign \
  --text "Hello from designed voice" \
  --voice-description "A warm, friendly female voice" \
  --language en \
  --output designed.wav
```

#### Fish-Speech 示例

```bash
# 基础合成
./target/release/sdkwork-tts infer \
  --engine fish-speech \
  --speaker checkpoints/speaker.wav \
  --text "你好，这是 Fish-Speech 合成的声音" \
  --language zh \
  --output output.wav

# 批量合成
./target/release/sdkwork-tts batch \
  --engine fish-speech \
  --speaker checkpoints/speaker.wav \
  --input texts.txt \
  --output-dir outputs/
```

### 高级选项

#### 推理参数

```bash
# 调整温度
./target/release/sdkwork-tts infer \
  --engine indextts2 \
  --speaker checkpoints/speaker.wav \
  --text "温度测试" \
  --temperature 0.9 \
  --output temp_output.wav

# 调整 Flow 步数
./target/release/sdkwork-tts infer \
  --engine indextts2 \
  --speaker checkpoints/speaker.wav \
  --text "Flow 步数测试" \
  --flow-steps 50 \
  --output flow_output.wav

# 启用去噪
./target/release/sdkwork-tts infer \
  --engine indextts2 \
  --speaker checkpoints/speaker.wav \
  --text "去噪测试" \
  --de-rumble \
  --de-rumble-cutoff-hz 180 \
  --output denoised.wav
```

#### 性能选项

```bash
# CPU 模式
./target/release/sdkwork-tts infer \
  --engine indextts2 \
  --cpu \
  --speaker checkpoints/speaker.wav \
  --text "CPU 模式测试" \
  --output cpu_output.wav

# FP16 精度 (GPU)
./target/release/sdkwork-tts infer \
  --engine qwen3-tts \
  --fp16 \
  --speaker checkpoints/speaker.wav \
  --text "FP16 测试" \
  --output fp16_output.wav
```

### 命令参考

#### 全局选项

| 选项 | 简写 | 说明 | 默认值 |
|------|------|------|--------|
| `--verbose` | `-v` | 启用详细日志 | false |
| `--cpu` | - | 使用 CPU | false |
| `--fp16` | - | 使用 FP16 精度 | false |

#### infer 命令选项

| 选项 | 说明 | 默认值 |
|------|------|--------|
| `--engine` | TTS 引擎 | qwen3-tts |
| `--speaker` | 参考音频路径 | - |
| `--speaker-id` | 内置说话人 ID | - |
| `--language` | 语言代码 | auto |
| `--text` | 合成文本 | - |
| `--output` | 输出文件 | output.wav |
| `--temperature` | 采样温度 | 0.8 |
| `--top-k` | Top-k 采样 | 50 |
| `--top-p` | Top-p 采样 | 0.95 |
| `--flow-steps` | Flow 步数 | 25 |
| `--de-rumble` | 启用去噪 | false |

---

## 📚 库使用

### 基础使用

```rust
use sdkwork_tts::{IndexTTS2, TtsEngine, EngineRegistry, init_engines};
use anyhow::Result;

fn main() -> Result<()> {
    // 直接使用 IndexTTS2
    let tts = IndexTTS2::new("checkpoints/config.yaml")?;
    let audio = tts.infer("你好，世界！", "speaker.wav")?;
    audio.save("output.wav")?;
    
    Ok(())
}
```

### 使用引擎注册表

```rust
use sdkwork_tts::engine::{TtsEngine, EngineRegistry, init_engines, global_registry};
use sdkwork_tts::engine::traits::{SynthesisRequest, SpeakerReference};
use anyhow::Result;

#[tokio::main]
async fn main() -> Result<()> {
    // 初始化所有引擎
    init_engines()?;
    
    // 获取注册表
    let registry = global_registry();
    
    // 列出所有引擎
    let engines = registry.list_engines()?;
    for engine in &engines {
        println!("Engine: {} v{}", engine.name, engine.version);
    }
    
    // 获取特定引擎
    let engine = registry.get_engine("indextts2")?;
    
    // 创建合成请求
    let request = SynthesisRequest {
        text: "你好，世界！".to_string(),
        speaker: SpeakerReference::AudioPath("speaker.wav".into()),
        emotion: None,
        params: Default::default(),
        output_format: Default::default(),
        request_id: None,
    };
    
    // 合成语音
    let result = engine.synthesize(&request).await?;
    result.save("output.wav")?;
    
    Ok(())
}
```

### 使用不同引擎

```rust
use sdkwork_tts::engine::{
    Qwen3TtsEngine, QwenModelVariant,
    FishSpeechEngine, IndexTTS2Engine,
    TtsEngine
};
use anyhow::Result;

// Qwen3-TTS 特定变体
let mut engine = Qwen3TtsEngine::with_variant(QwenModelVariant::CustomVoice17B);

// Fish-Speech
let engine = FishSpeechEngine::new();

// IndexTTS2
let engine = IndexTTS2Engine::new();
```

### 流式合成

```rust
use sdkwork_tts::engine::{TtsEngine, StreamingCallback};
use anyhow::Result;

#[tokio::main]
async fn main() -> Result<()> {
    let registry = sdkwork_tts::engine::global_registry();
    let engine = registry.get_engine("indextts2")?;
    
    let request = SynthesisRequest {
        text: "这是流式合成测试".to_string(),
        speaker: SpeakerReference::AudioPath("speaker.wav".into()),
        ..Default::default()
    };
    
    // 流式回调
    let callback: StreamingCallback = Box::new(|chunk| {
        println!("Received chunk: {} samples", chunk.samples.len());
        // 播放或保存音频块
        Ok(())
    });
    
    // 流式合成
    engine.synthesize_streaming(&request, callback).await?;
    
    Ok(())
}
```

---

## 🏗️ 架构设计

### 架构图

```
┌─────────────────────────────────────────────────────────────┐
│                    SDKWork-TTS Framework                    │
├─────────────────────────────────────────────────────────────┤
│                    Application Layer                        │
│              (CLI, API, Streaming, Batch)                   │
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
│                    Processing Pipeline                      │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐     │
│  │  Text    │ │ Speaker  │ │  Audio   │ │  Output  │     │
│  │Processor │ │ Encoder  │ │Processor │ │ Handler  │     │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘     │
├─────────────────────────────────────────────────────────────┤
│                    Core Infrastructure                      │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐     │
│  │  Error   │ │ Resource │ │ Metrics  │ │  Config  │     │
│  │ Handling │ │ Manager  │ │ Collector│ │  System  │     │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘     │
└─────────────────────────────────────────────────────────────┘
```

### 模块结构

```
src/
├── main.rs                  # CLI 入口
├── lib.rs                   # 库导出
├── core/                    # 核心框架
│   ├── error.rs            # 错误处理
│   ├── traits.rs           # 组件特征
│   ├── resource.rs         # 资源管理
│   ├── metrics.rs          # 性能监控
│   └── builder.rs          # 构建器模式
├── engine/                  # 引擎抽象层
│   ├── traits.rs           # TtsEngine 特征
│   ├── registry.rs         # 引擎注册表
│   ├── pipeline.rs         # 处理管道
│   ├── config.rs           # 引擎配置
│   ├── speaker.rs          # 说话人管理
│   ├── emotion.rs          # 情感管理
│   ├── indextts2_adapter.rs
│   ├── fish_speech_adapter.rs
│   └── qwen3_tts_adapter.rs
├── models/                  # 神经网络模型
│   ├── semantic/           # Wav2Vec-BERT, codec
│   ├── speaker/            # CAMPPlus
│   ├── gpt/                # UnifiedVoice, Conformer
│   ├── s2mel/              # DiT, Flow Matching
│   └── vocoder/            # BigVGAN
├── inference/              # 推理管道
├── audio/                  # 音频 I/O
├── text/                   # 文本处理
└── config/                 # 配置
```

---

## 📊 性能基准

### 推理性能

| 引擎 | 设备 | RTF | 延迟 | 显存 |
|------|------|-----|------|------|
| **IndexTTS2** | CPU | ~2.5 | - | 2 GB |
| **IndexTTS2** | CUDA | ~0.8 | - | 4 GB |
| **Qwen3-TTS** | CPU | ~1.5 | - | 3 GB |
| **Qwen3-TTS** | CUDA | ~0.3 | 97ms | 6 GB |
| **Fish-Speech** | CPU | ~2.0 | - | 2.5 GB |
| **Fish-Speech** | CUDA | ~0.5 | - | 5 GB |

### 语言支持

| 引擎 | 中文 | 英语 | 日语 | 韩语 | 其他 |
|------|------|------|------|------|------|
| IndexTTS2 | ✅ | ✅ | ✅ | ❌ | ❌ |
| Qwen3-TTS | ✅ | ✅ | ✅ | ✅ | 6 种 |
| Fish-Speech | ✅ | ✅ | ✅ | ✅ | 2 种 |

---

## 🛠️ 开发指南

### 添加新引擎

```rust
use sdkwork_tts::engine::{TtsEngine, TtsEngineInfo, SynthesisRequest, SynthesisResult};
use async_trait::async_trait;
use anyhow::Result;

pub struct MyTtsEngine {
    info: TtsEngineInfo,
}

#[async_trait]
impl TtsEngine for MyTtsEngine {
    fn info(&self) -> &TtsEngineInfo {
        &self.info
    }

    async fn initialize(&mut self, config: &EngineConfig) -> Result<()> {
        // 加载模型
        Ok(())
    }

    async fn synthesize(&self, request: &SynthesisRequest) -> Result<SynthesisResult> {
        // 实现合成逻辑
        Ok(result)
    }
}

// 注册引擎
sdkwork_tts::engine::global_registry().register_lazy(
    "my-engine",
    info,
    || Ok(Box::new(MyTtsEngine::new()))
)?;
```

### 测试

```bash
# 运行所有测试
cargo test

# 运行特定模块测试
cargo test indextts2
cargo test qwen3_tts

# 运行集成测试
cargo test --test synthesis_integration_tests

# 性能基准
cargo bench
```

### 代码风格

```bash
# 格式化代码
cargo fmt

# Clippy 检查
cargo clippy -- -D warnings

# 构建检查
cargo check --all-features
```

---

## ❓ 常见问题

### Q: 如何下载模型权重？

A: 使用 HuggingFace CLI 下载：

```bash
# IndexTTS2
huggingface-cli download IndexTeam/IndexTTS-2 --local-dir checkpoints/indextts2

# Qwen3-TTS
huggingface-cli download Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice --local-dir checkpoints/qwen3-tts
```

### Q: GPU 加速不起作用？

A: 检查以下几点：
1. 确保 CUDA 已正确安装
2. 设置正确的 compute capability: `$env:CUDA_COMPUTE_CAP='90'`
3. 使用 `--features cuda` 构建
4. 检查 GPU 显存是否足够

### Q: 如何提升合成质量？

A: 尝试以下参数：
- 降低 `temperature` (0.6-0.8)
- 增加 `flow-steps` (30-50)
- 启用 `--de-rumble` 去噪
- 使用高质量的参考音频

### Q: 支持哪些音频格式？

A: 支持 WAV、MP3、FLAC、OGG 等常见格式作为输入参考音频。输出固定为 WAV 格式。

### Q: 如何实现批量合成？

A: 使用 `batch` 命令或库 API 的批量处理功能：

```bash
./target/release/sdkwork-tts batch \
  --engine indextts2 \
  --speaker speaker.wav \
  --input texts.txt \
  --output-dir outputs/
```

---

## 📄 许可证

Apache-2.0 License

---

## 🙏 致谢

- **IndexTTS2**: [Bilibili](https://github.com/index-tts/index-tts)
- **Qwen3-TTS**: [Alibaba Cloud](https://github.com/QwenLM/Qwen3-TTS)
- **Fish-Speech**: [Fish Audio](https://github.com/fishaudio/fish-speech)
- **Candle**: [HuggingFace](https://github.com/huggingface/candle)

---

<div align="center">

**SDKWork-TTS** - 让语音合成更简单

[文档](docs/) | [示例](examples/) | [问题反馈](issues)

</div>
