> Migrated from `docs/QWEN3_TTS_STATUS.md` on 2026-06-24.
> Owner: SDKWork maintainers

# Qwen3-TTS 实现状态

## 项目概述

基于 https://github.com/TrevorS/qwen3-tts-rs 的架构，实现高性能 Qwen3-TTS 推理引擎。

## 当前状态 (2026 年 2 月 21 日)

### ✅ 已完成

1. **架构设计**
   - 完整的模块结构设计
   - 配置文件定义 (`config.rs`)
   - KV 缓存系统 (`kv_cache.rs`)
   - 主模型接口 (`mod.rs`)

2. **依赖更新**
   - Candle 0.9.x
   - Tokenizers 0.22
   - Safetensors 0.7
   - 支持 CUDA/Metal/FlashAttention

3. **API 设计**
   - `Qwen3TtsModel` - 主模型类
   - `VoiceClonePrompt` - 声音克隆提示
   - `QwenSynthesisResult` - 合成结果
   - `SynthesisTiming` - 性能计时

### 🚧 待完成模块

以下模块需要完整实现（参考 qwen3-tts-rs）：

| 模块 | 行数 | 说明 |
|------|------|------|
| `talker.rs` | ~800 | 28 层 Transformer，MRoPE 位置编码 |
| `code_predictor.rs` | ~300 | 5 层 Decoder，声学码本预测 |
| `decoder.rs` | ~400 | ConvNeXt + 上采样，12Hz→24kHz |
| `tokenizer.rs` | ~200 | 文本 tokenizer 封装 |
| `speaker_encoder.rs` | ~250 | ECAPA-TDNN 说话人编码 |
| `generation.rs` | ~350 | 采样、KV 缓存管理、生成循环 |

### 📋 实现优先级

#### 高优先级（核心推理）

1. **TalkerModel** - 文本→语义 token
   - 28 层 Transformer（CustomVoice 1.7B）
   - MRoPE 位置编码
   - KV 缓存支持
   - FlashAttention 2 集成

2. **CodePredictor** - 语义→声学 token
   - 5 层 Decoder
   - 每帧生成 15 个声学码本
   - 因果注意力掩码

3. **Decoder12Hz** - 声学码本→音频
   - ConvNeXt 块（12 层）
   - 上采样（12Hz → 24kHz）
   - GRN 归一化

#### 中优先级（功能增强）

4. **SpeakerEncoder** - 参考音频→说话人嵌入
   - ECAPA-TDNN 架构
   - 统计池化
   - 192 维输出

5. **Generation Loop** - 自回归生成
   - GPU 端采样
   - 重复惩罚
   - 流式分块

#### 低优先级（优化）

6. **性能优化**
   - 预分配 KV 缓存
   - 零拷贝更新
   - 融合内核

## 架构参考

### TalkerModel (28 层 Transformer)

```
Input: Token IDs [batch, seq_len]
  ↓
Text Embedding [batch, seq_len, 2048]
  ↓
┌─────────────────────────────┐
│ Transformer Block × 28      │
│  - Self-Attention (GQA)     │
│  - MRoPE 位置编码            │
│  - RMSNorm                  │
│  - SwiGLU MLP               │
└─────────────────────────────┘
  ↓
Output: Semantic Tokens [batch, seq_len, 4096]
```

### CodePredictor (5 层 Decoder)

```
Input: Semantic Token [batch, 1, 2048]
  ↓
Input Projection [batch, 1, 1024]
  ↓
┌─────────────────────────────┐
│ Decoder Block × 5           │
│  - Causal Self-Attention    │
│  - RMSNorm                  │
│  - MLP                      │
└─────────────────────────────┘
  ↓
Output Head: [batch, 1, 15 × 1024]
  ↓
Reshape: [batch, 1, 15, 1024]
```

### Decoder12Hz (ConvNeXt)

```
Input: 16 Codebooks [batch, 16, seq_len]
  ↓
Codebook Embedding [batch, 512, seq_len]
  ↓
┌─────────────────────────────┐
│ ConvNeXt Block × 12         │
│  - Depthwise Conv (7×7)     │
│  - Pointwise Conv (1×1)     │
│  - GRN                      │
└─────────────────────────────┘
  ↓
Upsample × 3 (8×, 8×, 4×)
  ↓
Output: Audio [batch, 24000 × duration]
```

## 性能目标

基于 qwen3-tts-rs 0.4.0 的基准测试：

| 模型 | 设备 | RTF (短) | RTF (长) | Tok/s | 显存 |
|------|------|---------|---------|-------|------|
| 0.6B Base | CUDA BF16 | 0.48 | 0.50 | 25.9 | 767 MB |
| 1.7B Base | CUDA BF16 | 0.65 | 0.65 | 19.4 | 767 MB |
| 1.7B CustomVoice | CUDA BF16 | 0.64 | 0.67 | 19.2 | 772 MB |
| 1.7B VoiceDesign | CUDA BF16 | 0.64 | 0.66 | 19.3 | 770 MB |

**RTF < 1.0** 表示快于实时。

## 下一步工作

### 1. 完成核心模块实现

```bash
# 待创建的文件
src/models/qwen3_tts/talker.rs
src/models/qwen3_tts/code_predictor.rs
src/models/qwen3_tts/decoder.rs
src/models/qwen3_tts/tokenizer.rs
src/models/qwen3_tts/speaker_encoder.rs
src/models/qwen3_tts/generation.rs
```

### 2. 权重加载

从 HuggingFace 下载模型：
```bash
huggingface-cli download Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --local-dir checkpoints/qwen3-tts-customvoice
```

### 3. 测试验证

```rust
#[test]
fn test_end_to_end() {
    let model = Qwen3TtsModel::from_pretrained(
        "checkpoints/qwen3-tts",
        Device::new_cuda(0)?
    )?;
    
    let result = model.synthesize_with_voice(
        "你好，世界！",
        Speaker::Vivian,
        Language::Chinese,
        None,
    )?;
    
    result.save("output.wav")?;
}
```

## 参考资料

- **qwen3-tts-rs**: https://github.com/TrevorS/qwen3-tts-rs
- **Qwen3-TTS 官方**: https://github.com/QwenLM/Qwen3-TTS
- **技术报告**: arXiv:2601.15621
- **HuggingFace**: https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice

## 总结

当前已完成 Qwen3-TTS 的基础架构设计，包括配置系统、KV 缓存、主模型接口。核心推理模块（TalkerModel、CodePredictor、Decoder12Hz）需要约 2000 行代码实现。

建议按以下顺序完成：
1. TalkerModel（800 行）- 文本编码
2. CodePredictor（300 行）- 声学码本预测
3. Decoder12Hz（400 行）- 音频合成
4. Generation Loop（350 行）- 自回归生成
5. SpeakerEncoder（250 行）- 声音克隆
6. Tokenizer（200 行）- 文本处理

预计总工作量：约 3000 行代码，2-3 天开发时间。

