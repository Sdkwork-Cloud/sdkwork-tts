# Qwen3-TTS CLI 使用指南

## 概述

Qwen3-TTS CLI 是一个命令行工具，用于使用 Qwen3-TTS 模型进行语音合成。

## 安装

```bash
# 构建 CLI 工具
cargo build --release --features cuda

# 或者 CPU 模式
cargo build --release --no-default-features --features cpu
```

## 使用方法

### 1. 列出可用模型和说话人

```bash
cargo run --bin qwen3-tts -- list
```

输出示例：
```
╔═══════════════════════════════════════════════════════════╗
║              Qwen3-TTS Available Models                   ║
╚═══════════════════════════════════════════════════════════╝

Model Variants:
  • Base06B         - 0.6B parameters, voice cloning
  • Base17B         - 1.7B parameters, voice cloning
  • CustomVoice06B  - 0.6B parameters, 9 preset speakers
  • CustomVoice17B  - 1.7B parameters, 9 preset speakers
  • VoiceDesign17B  - 1.7B parameters, text voice design

Preset Speakers:
  • Vivian       - Chinese, bright young female
  • Serena       - Chinese, warm gentle female
  ...
```

### 2. 使用预设音色合成

```bash
cargo run --bin qwen3-tts -- synthesize \
  --text "你好，世界！" \
  --speaker Vivian \
  --language Chinese \
  --output output.wav
```

参数说明：
- `--text`: 要合成的文本
- `--speaker`: 说话人 (默认：Vivian)
- `--language`: 语言 (默认：Auto)
- `--output`: 输出文件路径 (默认：output.wav)
- `--model`: 模型变体 (默认：CustomVoice17B)
- `--temperature`: 采样温度 (默认：0.8)
- `--top-k`: Top-k 采样 (默认：50)
- `--top-p`: Top-p 采样 (默认：0.95)
- `--seed`: 随机种子 (默认：42)

### 3. 声音克隆

```bash
cargo run --bin qwen3-tts -- clone \
  --text "这是克隆的声音" \
  --reference reference.wav \
  --reference-text "参考音频的文本内容" \
  --output cloned.wav \
  --model Base17B
```

参数说明：
- `--text`: 要合成的文本
- `--reference`: 参考音频文件路径
- `--reference-text`: 参考音频的文本内容（可选，提高质量）
- `--model`: 必须使用 Base 模型 (Base06B 或 Base17B)

### 4. 声音设计

```bash
cargo run --bin qwen3-tts -- design \
  --text "Hello from designed voice!" \
  --description "A warm, friendly female voice with medium pitch" \
  --output designed.wav \
  --model VoiceDesign17B
```

参数说明：
- `--text`: 要合成的文本
- `--description`: 声音描述（自然语言）
- `--model`: 必须使用 VoiceDesign17B

### 5. 下载模型

```bash
cargo run --bin qwen3-tts -- download \
  --model CustomVoice17B \
  --output checkpoints
```

### 6. 查看模型信息

```bash
cargo run --bin qwen3-tts -- info \
  --model-dir checkpoints/qwen3-tts
```

## 全局选项

- `--verbose`: 启用详细日志
- `--cpu`: 使用 CPU 而不是 GPU
- `--bf16`: 使用 BF16 精度（仅 GPU）

示例：
```bash
# CPU 模式，详细日志
cargo run --bin qwen3-tts -- --verbose --cpu synthesize \
  --text "Hello world" \
  --output output.wav
```

## 支持的语言

| 语言 | 代码 |
|------|------|
| 中文 | Chinese |
| 英语 | English |
| 日语 | Japanese |
| 韩语 | Korean |
| 德语 | German |
| 法语 | French |
| 俄语 | Russian |
| 葡萄牙语 | Portuguese |
| 西班牙语 | Spanish |
| 意大利语 | Italian |

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

## 性能提示

### GPU 加速

```bash
# 设置 CUDA compute capability
$env:CUDA_COMPUTE_CAP='90'  # RTX 5090
$env:CUDA_COMPUTE_CAP='86'  # RTX 3090

# 构建 release 版本
cargo build --release --features cuda
```

### 优化选项

```bash
# 高质量（较慢）
--temperature 0.7 --top-k 50 --top-p 0.95

# 快速（质量略低）
--temperature 0.9 --top-k 20 --top-p 0.9
```

## 故障排除

### 模型未找到

```
Error: Model directory does not exist
```

解决方法：
```bash
cargo run --bin qwen3-tts -- download --model CustomVoice17B
```

### 显存不足

```
Error: CUDA out of memory
```

解决方法：
- 使用较小的模型（0.6B 版本）
- 使用 CPU 模式：`--cpu`
- 减少 batch size

### 音频质量差

尝试调整参数：
```bash
--temperature 0.7 --top-p 0.9 --repetition-penalty 1.1
```

## 示例脚本

### PowerShell 批量合成

```powershell
$texts = @(
    "第一句话",
    "第二句话",
    "第三句话"
)

for ($i = 0; $i -lt $texts.Length; $i++) {
    cargo run --bin qwen3-tts -- synthesize `
      --text $texts[$i] `
      --speaker Vivian `
      --output "output_$i.wav"
}
```

### Bash 批量合成

```bash
#!/bin/bash

texts=(
    "第一句话"
    "第二句话"
    "第三句话"
)

for i in "${!texts[@]}"; do
    cargo run --bin qwen3-tts -- synthesize \
      --text "${texts[$i]}" \
      --speaker Vivian \
      --output "output_$i.wav"
done
```

## 许可证

Apache-2.0 License
