# SDKWork-TTS 快速开始指南

## 5 分钟快速上手

### 步骤 1: 安装 Rust

```bash
# Windows (PowerShell)
winget install Rustlang.Rust.MSVC
# 或访问 https://rustup.rs/ 下载安装

# 验证安装
rustc --version  # 应该显示 1.75+
```

### 步骤 2: 克隆项目

```bash
git clone https://github.com/sdkwork/sdkwork-tts.git
cd sdkwork-tts
```

### 步骤 3: 构建项目

```bash
# CPU 版本 (无需 GPU)
cargo build --release

# CUDA 版本 (需要 NVIDIA GPU)
$env:CUDA_COMPUTE_CAP='90'
cargo build --release --features cuda
```

### 步骤 4: 准备模型

```bash
# 创建 checkpoints 目录
mkdir checkpoints

# 下载 IndexTTS2 模型
# 方式 1: 使用 Python 脚本
python download_model.py

# 方式 2: 手动下载
# 访问 https://huggingface.co/IndexTeam/IndexTTS-2
# 下载所有文件到 checkpoints/indextts2/
```

### 步骤 5: 测试合成

```bash
# 列出可用引擎
./target/release/sdkwork-tts engines

# 使用 IndexTTS2 合成
./target/release/sdkwork-tts infer `
  --engine indextts2 `
  --speaker checkpoints/test_speaker.wav `
  --text "你好，这是 SDKWork-TTS 框架合成的声音" `
  --output output.wav

# 播放结果
./output.wav
```

## 命令行验证清单

### ✅ 基础验证

```bash
# 1. 检查 CLI 是否可用
./target/release/sdkwork-tts --version

# 2. 查看帮助
./target/release/sdkwork-tts --help

# 3. 列出引擎
./target/release/sdkwork-tts engines

# 4. 查看引擎详情
./target/release/sdkwork-tts engines --detailed
```

### ✅ 功能验证

```bash
# 1. IndexTTS2 基础合成
./target/release/sdkwork-tts infer `
  --engine indextts2 `
  --speaker checkpoints/speaker.wav `
  --text "测试文本" `
  --output test1.wav

# 2. Qwen3-TTS 合成
./target/release/sdkwork-tts infer `
  --engine qwen3-tts `
  --speaker checkpoints/speaker.wav `
  --text "Hello world" `
  --language en `
  --output test2.wav

# 3. 带情感控制
./target/release/sdkwork-tts infer `
  --engine indextts2 `
  --speaker checkpoints/speaker.wav `
  --emotion-alpha 0.8 `
  --text "这应该听起来很快乐" `
  --output test3.wav
```

### ✅ 性能验证

```bash
# 1. CPU 模式
./target/release/sdkwork-tts infer `
  --engine indextts2 `
  --cpu `
  --speaker checkpoints/speaker.wav `
  --text "CPU 模式测试" `
  --output cpu_test.wav

# 2. GPU 模式 (如果有 GPU)
./target/release/sdkwork-tts infer `
  --engine indextts2 `
  --speaker checkpoints/speaker.wav `
  --text "GPU 模式测试" `
  --output gpu_test.wav

# 3. 比较生成时间
# 查看输出中的处理时间信息
```

## 常见问题排查

### 问题 1: 找不到模型文件

**错误**: `Model file not found`

**解决**:
```bash
# 检查 checkpoints 目录
ls checkpoints/

# 确保模型文件存在
# config.yaml
# gpt.safetensors
# s2mel.safetensors
# bigvgan/bigvgan_generator.safetensors
```

### 问题 2: CUDA 不可用

**错误**: `CUDA error` 或 `No CUDA device`

**解决**:
```bash
# 1. 检查 CUDA 安装
nvcc --version

# 2. 使用 CPU 模式
./target/release/sdkwork-tts infer --cpu ...

# 3. 重新构建 (确保 CUDA 正确安装)
$env:CUDA_COMPUTE_CAP='90'
cargo build --release --features cuda
```

### 问题 3: 内存不足

**错误**: `Out of memory`

**解决**:
```bash
# 1. 使用较小的模型
./target/release/sdkwork-tts infer `
  --engine qwen3-tts `
  --model CustomVoice06B `
  ...

# 2. 减少 batch size
# 3. 关闭其他占用显存的程序
```

## 下一步

- 📖 阅读完整文档：`README_PERFECT.md`
- 💻 查看示例：`examples/`
- 📚 学习架构：`docs/ARCHITECTURE.md`
- 🔧 开发指南：`docs/DEVELOPMENT_PLAN.md`

## 获取帮助

- 📖 文档：`docs/`
- 🐛 问题：GitHub Issues
- 💬 讨论：GitHub Discussions
