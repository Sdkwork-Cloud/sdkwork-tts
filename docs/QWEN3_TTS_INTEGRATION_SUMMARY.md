# Qwen3-TTS 实现总结

## 概述

已成功将 Qwen3-TTS 基础架构整合到 SDKWork-TTS 框架中，基于 https://github.com/TrevorS/qwen3-tts-rs 的架构设计。

## 完成的工作

### 1. 依赖更新 ✅

```toml
# 升级到 Candle 0.9.x
candle-core = "0.9"
candle-nn = "0.9"
candle-transformers = "0.9"
candle-flash-attn = { version = "0.9", optional = true }

# Tokenizers 0.22
tokenizers = "0.22"

# Safetensors 0.7
safetensors = "0.7"

# 其他更新
hf-hub = "0.4"
indicatif = "0.18"
rodio = "0.20"
ndarray = "0.17"
```

### 2. 特性标志 ✅

```toml
[features]
default = ["cuda"]
cpu = []
cuda = ["candle-core/cuda", "candle-nn/cuda", "candle-transformers/cuda"]
metal = ["candle-core/metal", "candle-nn/metal", "candle-transformers/metal"]
mkl = ["candle-core/mkl", "candle-nn/mkl", "candle-transformers/mkl"]
accelerate = ["candle-core/accelerate", "candle-nn/accelerate", "candle-transformers/accelerate"]
flash-attn = ["cuda", "dep:candle-flash-attn"]
profiling = ["dep:tracing-chrome"]
```

### 3. 模块结构 ✅

```
src/models/qwen3_tts/
├── mod.rs              # 主模块，模型接口
├── config.rs           # 配置系统
├── kv_cache.rs         # KV 缓存管理
├── talker.rs           # (占位符) 28 层 Transformer
├── code_predictor.rs   # (占位符) 5 层 Decoder
├── decoder.rs          # (占位符) ConvNeXt 音频合成
├── tokenizer.rs        # (占位符) 文本分词
├── speaker_encoder.rs  # (占位符) 说话人编码
└── generation.rs       # (占位符) 生成循环
```

### 4. 核心类型 ✅

| 类型 | 状态 | 说明 |
|------|------|------|
| `QwenModelVariant` | ✅ | 5 种模型变体 |
| `QwenConfig` | ✅ | 模型配置 |
| `Qwen3TtsModel` | 🚧 | 主模型类（占位符实现） |
| `VoiceClonePrompt` | ✅ | 声音克隆提示 |
| `QwenSynthesisResult` | ✅ | 合成结果 |
| `SynthesisTiming` | ✅ | 性能计时 |
| `TalkerConfig` | ✅ | Talker 配置 |
| `CodePredictorConfig` | ✅ | CodePredictor 配置 |
| `KVCache` | ✅ | KV 缓存 |
| `Language` | ✅ | 10 种语言 |
| `Speaker` | ✅ | 9 种预设说话人 |

### 5. API 接口 ✅

```rust
// 创建模型
let model = Qwen3TtsModel::new(QwenConfig::default())?;

// 合成语音
let result = model.synthesize("你好，世界！", None)?;
result.save("output.wav")?;

// 声音克隆
let prompt = model.create_voice_clone_prompt(&ref_audio, None)?;
let result = model.synthesize_voice_clone("新内容", &prompt, Language::Chinese, None)?;

// 声音设计
let result = model.synthesize_voice_design("文本", "声音描述", Language::Chinese, None)?;
```

### 6. 引擎适配器 ✅

- `Qwen3TtsEngine` 实现 `TtsEngine` trait
- 支持引擎注册和发现
- 支持流式合成（分块输出）
- 支持多模型变体

### 7. 测试验证 ✅

```
running 3 tests
test models::qwen3_tts::tests::test_model_variant ... ok
test models::qwen3_tts::tests::test_model_new ... ok
test models::qwen3_tts::tests::test_config_default ... ok

test result: ok. 3 passed; 0 failed
```

## 待完成工作

### 高优先级（核心推理）

以下模块需要完整实现（参考 qwen3-tts-rs）：

| 模块 | 预估行数 | 说明 |
|------|---------|------|
| `talker.rs` | ~800 | 28 层 Transformer，MRoPE 位置编码，GQA 注意力 |
| `code_predictor.rs` | ~300 | 5 层 Decoder，因果注意力，声学码本预测 |
| `decoder.rs` | ~400 | ConvNeXt 块，上采样，GRN 归一化 |
| `generation.rs` | ~350 | 自回归生成循环，GPU 端采样，KV 缓存管理 |

### 中优先级（功能增强）

| 模块 | 预估行数 | 说明 |
|------|---------|------|
| `speaker_encoder.rs` | ~250 | ECAPA-TDNN，统计池化，说话人嵌入 |
| `tokenizer.rs` | ~200 | HuggingFace tokenizers 集成 |

### 低优先级（性能优化）

- FlashAttention 2 集成
- 预分配 KV 缓存
- 零拷贝更新
- 融合 CUDA 内核

## 性能目标

基于 qwen3-tts-rs 0.4.0 的基准测试：

| 模型 | 设备 | RTF (短) | RTF (长) | Tok/s | 显存 |
|------|------|---------|---------|-------|------|
| 0.6B Base | CUDA BF16 | 0.48 | 0.50 | 25.9 | 767 MB |
| 1.7B Base | CUDA BF16 | 0.65 | 0.65 | 19.4 | 767 MB |
| 1.7B CustomVoice | CUDA BF16 | 0.64 | 0.67 | 19.2 | 772 MB |

**RTF < 1.0** 表示快于实时。

## 编译验证

```bash
# CPU 模式
cargo check --no-default-features --features cpu

# CUDA 模式（需要 Visual Studio）
cargo build --release --features cuda

# 测试
cargo test --lib --no-default-features --features cpu models::qwen3_tts
```

## 参考资料

1. **qwen3-tts-rs**: https://github.com/TrevorS/qwen3-tts-rs
2. **Qwen3-TTS 官方**: https://github.com/QwenLM/Qwen3-TTS
3. **技术报告**: arXiv:2601.15621
4. **HuggingFace**: https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice

## 下一步

1. **完成 TalkerModel 实现** (~800 行)
   - 28 层 Transformer
   - MRoPE 位置编码
   - GQA 注意力机制
   - KV 缓存支持

2. **完成 CodePredictor 实现** (~300 行)
   - 5 层 Decoder
   - 因果注意力掩码
   - 声学码本预测

3. **完成 Decoder12Hz 实现** (~400 行)
   - ConvNeXt 块
   - 上采样（12Hz → 24kHz）
   - GRN 归一化

4. **完成 Generation Loop 实现** (~350 行)
   - 自回归生成
   - GPU 端采样
   - 重复惩罚

预计总工作量：约 1850 行核心代码。

## 总结

当前已完成 Qwen3-TTS 的基础架构搭建，包括：
- ✅ 依赖升级到 Candle 0.9
- ✅ 配置系统
- ✅ KV 缓存系统
- ✅ 模型接口定义
- ✅ 引擎适配器
- ✅ 编译测试通过

核心推理模块（TalkerModel、CodePredictor、Decoder12Hz）需要约 1850 行代码实现完整功能。建议按优先级顺序完成，预计 2-3 天开发时间可实现完整推理能力。
