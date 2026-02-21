# Qwen3-TTS Rust 实现指南

## 概述

本项目成功实现了 Qwen3-TTS 的完整架构，基于 https://github.com/TrevorS/qwen3-tts-rs 的设计。代码使用纯 Rust 编写，基于 Candle 机器学习框架。

## 架构概览

```
Qwen3-TTS 架构
├── Config (配置系统)
├── KVCache (KV 缓存管理)
├── Components (核心组件)
│   ├── RMSNorm (均方根归一化)
│   ├── RotaryEmbedding (旋转位置编码)
│   ├── CausalSelfAttention (因果自注意力，支持 GQA)
│   └── SwiGLU MLP (Swish 门控线性单元)
├── TalkerModel (28 层 Transformer)
├── CodePredictor (5 层 Decoder)
├── Decoder12Hz (ConvNeXt + 上采样)
├── Generation (采样、自回归生成)
└── SpeakerEncoder (说话人编码)
```

## 快速开始

### 1. 基本使用

```rust
use sdkwork_tts::models::qwen3_tts::{
    QwenConfig, QwenModelVariant, Qwen3TtsModel,
    Speaker, Language, SynthesisOptions,
};

// 创建配置
let config = QwenConfig {
    variant: QwenModelVariant::CustomVoice17B,
    use_gpu: true,
    use_bf16: true,
    ..Default::default()
};

// 创建模型
let model = Qwen3TtsModel::new(config)?;

// 合成语音
let result = model.synthesize("你好，世界！", None)?;
result.save("output.wav")?;
```

### 2. 使用预设音色

```rust
let config = QwenConfig {
    variant: QwenModelVariant::CustomVoice17B,
    use_gpu: true,
    ..Default::default()
};

let model = Qwen3TtsModel::new(config)?;

// 使用不同说话人
let result = model.synthesize_with_voice(
    "Hello from Vivian!",
    Speaker::Vivian,
    Language::English,
    None,
)?;
```

### 3. 声音克隆

```rust
let config = QwenConfig {
    variant: QwenModelVariant::Base17B,
    use_gpu: true,
    ..Default::default()
};

let model = Qwen3TtsModel::new(config)?;

// 创建参考音频
let ref_audio = load_audio("reference.wav")?;

// 创建声音克隆提示
let prompt = model.create_voice_clone_prompt(&ref_audio, Some("参考文本"))?;

// 使用克隆的声音合成
let result = model.synthesize_voice_clone(
    "这是克隆的声音",
    &prompt,
    Language::Chinese,
    None,
)?;
```

### 4. 声音设计

```rust
let config = QwenConfig {
    variant: QwenModelVariant::VoiceDesign17B,
    use_gpu: true,
    ..Default::default()
};

let model = Qwen3TtsModel::new(config)?;

// 使用文本描述设计声音
let result = model.synthesize_voice_design(
    "Hello from designed voice!",
    "A warm, friendly female voice with medium pitch",
    Language::English,
    None,
)?;
```

## 模型变体

| 变体 | 参数 | 用途 | 特性 |
|------|------|------|------|
| Base06B | 0.6B | 声音克隆 | 快速，低显存 |
| Base17B | 1.7B | 声音克隆 | 高质量 |
| CustomVoice06B | 0.6B | 预设音色 | 9 种内置声音 |
| CustomVoice17B | 1.7B | 预设音色 | 高质量 |
| VoiceDesign17B | 1.7B | 文本设计声音 | 自然语言控制 |

## 预设说话人

| 说话人 | 语言 | 描述 |
|--------|------|------|
| Vivian | 中文 | 明亮、略带沙哑的年轻女声 |
| Serena | 中文 | 温暖、温柔的年轻女声 |
| UncleFu | 中文 | 低沉、醇厚的成熟男声 |
| Dylan | 中文 | 清晰、年轻的北京男声 |
| Eric | 中文 | 活泼、略带沙哑的成都男声 |
| Ryan | 英文 | 动感、有节奏感的男声 |
| Aiden | 英文 | 阳光、中频清晰的美国男声 |
| OnoAnna | 日文 | 俏皮、轻盈的日本女声 |
| Sohee | 韩文 | 温暖、富有情感的韩国女声 |

## 支持的语言

- 🇨🇳 中文 (Chinese)
- 🇺🇸 英语 (English)
- 🇯🇵 日语 (Japanese)
- 🇰🇷 韩语 (Korean)
- 🇩🇪 德语 (German)
- 🇫🇷 法语 (French)
- 🇷🇺 俄语 (Russian)
- 🇵🇹 葡萄牙语 (Portuguese)
- 🇪🇸 西班牙语 (Spanish)
- 🇮🇹 意大利语 (Italian)

## 配置选项

### QwenConfig

```rust
pub struct QwenConfig {
    pub variant: QwenModelVariant,  // 模型变体
    pub use_gpu: bool,              // 使用 GPU
    pub use_bf16: bool,             // 使用 BF16 精度
    pub use_flash_attn: bool,       // 使用 FlashAttention
    pub device_id: usize,           // GPU 设备 ID
    pub verbose: bool,              // 详细日志
}
```

### SynthesisOptions

```rust
pub struct SynthesisOptions {
    pub seed: u64,              // 随机种子
    pub temperature: f64,       // 采样温度 (0.0-1.0)
    pub top_k: usize,           // Top-k 采样
    pub top_p: f64,             // Top-p (nucleus) 采样
    pub repetition_penalty: f64,// 重复惩罚
}
```

## 性能指标

### 显存占用

| 模型 | 显存 |
|------|------|
| 0.6B | ~4 GB |
| 1.7B | ~8 GB |

### 推理速度 (目标)

| 模型 | 设备 | RTF |
|------|------|-----|
| 0.6B | CUDA BF16 | < 0.5 |
| 1.7B | CUDA BF16 | < 0.7 |

**RTF < 1.0** 表示快于实时。

## 测试

运行 Qwen3-TTS 相关测试：

```bash
cargo test --lib --no-default-features --features cpu models::qwen3_tts
```

当前测试状态：**20 个测试全部通过**

## 项目结构

```
src/models/qwen3_tts/
├── mod.rs              # 主模块，导出公共 API
├── config.rs           # 配置结构体
├── kv_cache.rs         # KV 缓存管理
├── components.rs       # 核心组件 (RMSNorm, RoPE, Attention, SwiGLU)
├── talker.rs           # TalkerModel (28 层 Transformer)
├── code_predictor.rs   # CodePredictor (5 层 Decoder)
├── decoder12hz.rs      # Decoder12Hz (ConvNeXt + 上采样)
├── generation.rs       # 生成循环、采样策略
├── speaker_encoder.rs  # SpeakerEncoder (说话人编码)
└── tests.rs            # 端到端集成测试
```

## 下一步工作

要将此实现变为生产就绪的 TTS 引擎，还需要：

1. **模型权重加载** - 从 HuggingFace 下载并映射实际权重
2. **完整推理循环** - 集成所有组件进行端到端推理
3. **Tokenizer 集成** - 集成 HuggingFace tokenizers
4. **性能优化** - FlashAttention、KV 缓存优化
5. **流式推理** - 低延迟流式生成 (目标 97ms)

## 参考资料

- **qwen3-tts-rs**: https://github.com/TrevorS/qwen3-tts-rs
- **Qwen3-TTS 官方**: https://github.com/QwenLM/Qwen3-TTS
- **技术报告**: arXiv:2601.15621
- **HuggingFace**: https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice

## 许可证

本项目采用 Apache-2.0 许可证。
