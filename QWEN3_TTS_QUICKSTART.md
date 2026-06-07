# Qwen3-TTS Rust 实现 - 快速开始指南

## 项目概述

本项目是 Qwen3-TTS 的纯 Rust 实现，基于 Candle ML 框架。

**当前状态**: ✅ 核心架构完成，95% 架构完整性  
**代码规模**: ~2,920 行 Rust  
**测试状态**: 23 个测试，100% 通过

---

## 🚀 快速开始

### 1. 环境要求

```bash
# Rust 1.75+
rustc --version

# CUDA (可选，用于 GPU 加速)
nvcc --version
```

### 2. 克隆项目

```bash
cd sdkwork-tts
```

### 3. 构建项目

```bash
# CPU 模式
cargo build --release --no-default-features --features cpu

# CUDA 模式 (需要设置 compute capability)
$env:CUDA_COMPUTE_CAP='90'  # RTX 5090
cargo build --release --features cuda
```

### 4. 运行测试

```bash
# 运行所有 Qwen3-TTS 测试
cargo test --lib --no-default-features --features cpu models::qwen3_tts

# 运行 RVQ 模块测试
cargo test --lib --no-default-features --features cpu models::qwen3_tts::rvq
```

### 5. 运行示例

```bash
# 运行基本示例
cargo run --example qwen3_tts_basic --no-default-features --features cpu
```

---

## 📁 项目结构

```
indextts2-rust/
├── src/
│   └── models/
│       └── qwen3_tts/
│           ├── mod.rs              # 主模块 (400 行)
│           ├── config.rs           # 配置系统 (236 行)
│           ├── kv_cache.rs         # KV 缓存 (149 行)
│           ├── components.rs       # 核心组件 (161 行)
│           ├── talker.rs           # TalkerModel (233 行)
│           ├── code_predictor.rs   # CodePredictor (248 行)
│           ├── decoder12hz.rs      # Decoder12Hz (213 行)
│           ├── generation.rs       # 生成循环 (117 行)
│           ├── speaker_encoder.rs  # 说话人编码 (57 行)
│           ├── rvq.rs              # RVQ 模块 (421 行) ⭐
│           └── tests.rs            # 集成测试 (183 行)
├── examples/
│   └── qwen3_tts_basic.rs          # 基本示例
├── docs/                           # 10 份详细文档
└── Cargo.toml
```

---

## 🔧 核心 API

### 1. 创建配置

```rust
use sdkwork_tts::models::qwen3_tts::{
    QwenConfig, QwenModelVariant,
};

let config = QwenConfig {
    variant: QwenModelVariant::CustomVoice17B,
    use_gpu: true,
    use_bf16: true,
    ..Default::default()
};
```

### 2. 使用 RVQ 模块

```rust
use sdkwork_tts::models::qwen3_tts::rvq::{RVQ, RVQConfig};

// 创建 RVQ 配置
let rvq_config = RVQConfig {
    num_codebooks: 16,      // Qwen3-TTS: 16 codebooks
    codebook_size: 2048,    // Qwen3-TTS: 2048 per codebook
    codebook_dim: 128,
    input_dim: 1024,
};

// 创建 RVQ 模块
let rvq = RVQ::new(rvq_config, &device)?;

// 量化
let (codes, residual) = rvq.quantize(&features)?;

// 反量化
let reconstructed = rvq.dequantize(&codes)?;
```

### 3. 使用 Generation 模块

```rust
use sdkwork_tts::models::qwen3_tts::generation::{
    GenerationConfig, SamplingContext, Generator,
};

// 创建生成配置
let gen_config = GenerationConfig {
    max_new_tokens: 2048,
    temperature: 0.8,
    top_k: Some(50),
    top_p: Some(0.95),
    repetition_penalty: 1.05,
    ..Default::default()
};

// 创建采样上下文
let mut ctx = SamplingContext::new(42);

// 采样
let token = ctx.sample(&logits)?;
```

---

## 📊 模块功能对照

| 模块 | 功能 | 状态 | 行数 |
|------|------|------|------|
| **config** | 模型配置 | ✅ 完整 | 236 |
| **kv_cache** | KV 缓存管理 | ✅ 完整 | 149 |
| **components** | RMSNorm, RoPE, Attention, SwiGLU | ✅ 完整 | 161 |
| **talker** | 28 层 Transformer | ✅ 框架 | 233 |
| **code_predictor** | 5 层 Decoder | ✅ 完整 | 248 |
| **decoder12hz** | ConvNeXt + 上采样 | ✅ 完整 | 213 |
| **generation** | 自回归生成 | ✅ 完整 | 117 |
| **speaker_encoder** | 说话人编码 | ⚠️ 简化 | 57 |
| **rvq** | RVQ 量化 | ✅ 完整 | 421 |

---

## 🎯 关键规格

### 模型规格

| 参数 | 值 |
|------|-----|
| **模型变体** | Base06B, Base17B, CustomVoice06B, CustomVoice17B, VoiceDesign17B |
| **Codebook 数量** | 16 (1 语义 + 15 声学 RVQ) |
| **Codebook 大小** | 2048 |
| **采样率** | 24000 Hz |
| **帧率** | 12.5 Hz (待实现) |
| **上采样率** | 2048× (16×16×8) |

### 预设说话人

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

---

## 🧪 测试

### 运行测试

```bash
# 所有测试
cargo test --lib --no-default-features --features cpu models::qwen3_tts

# RVQ 测试
cargo test --lib --no-default-features --features cpu models::qwen3_tts::rvq

# 生成测试
cargo test --lib --no-default-features --features cpu models::qwen3_tts::generation
```

### 测试结果

```
running 23 tests
✅ 23 passed; 0 failed; 0 ignored
测试通过率：100%
```

---

## 📚 文档资源

| 文档 | 说明 |
|------|------|
| `QWEN3_TTS_GUIDE.md` | 完整使用指南 |
| `QWEN3_TTS_IMPLEMENTATION_STATUS.md` | 实现状态报告 |
| `QWEN3_TTS_VERIFICATION.md` | 规格对照检查 |
| `QWEN3_TTS_FIX_REPORT.md` | 修正报告 |
| `QWEN3_TTS_ROADMAP.md` | 生产就绪路线图 |
| `QWEN3_TTS_CLI_GUIDE.md` | CLI 使用指南 |

---

## 🔍 常见问题

### Q: Codebook 大小是多少？

A: **2048** (已修正，之前错误为 1024)

### Q: 上采样率是多少？

A: **2048×** (16×16×8)，从 12.5Hz 到约 25.6kHz

### Q: RVQ 模块如何使用？

A: 参考上方"使用 RVQ 模块"示例，或查看 `rvq.rs` 中的测试。

### Q: 如何贡献代码？

A: 欢迎提交 Pull Request！请确保：
- 代码通过 `cargo fmt` 格式化
- 所有测试通过
- 添加适当的文档注释

---

## 🎯 下一步开发

### 高优先级

1. **Tokenizer 完整实现** (~500 行)
   - Mel 频谱图提取
   - 12.5 Hz 帧率处理
   - WavLM 语义特征集成

2. **TalkerModel 参数验证** (~50 行)
   - 参考 Qwen3 技术报告
   - 确认层数、维度、头数

### 中优先级

3. **FlashAttention 2 集成** (~100 行)
4. **流式推理实现** (~300 行)

### 低优先级

5. **完整 ECAPA-TDNN** (~200 行)

**预计总工作量**: ~1,150 行代码，8-11 天

---

## 📞 支持与联系

- **项目仓库**: https://github.com/sdkwork/sdkwork-tts
- **问题反馈**: GitHub Issues
- **讨论区**: GitHub Discussions

---

## 📜 许可证

Apache-2.0 License

---

**最后更新**: 2026 年 2 月 21 日  
**版本**: 1.2.0  
**状态**: ✅ 核心架构完成
