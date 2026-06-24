> Migrated from `docs/SERVER_USAGE_GUIDE.md` on 2026-06-24.
> Owner: SDKWork maintainers

# TTS Server 使用指南

## 📖 目录

- [快速开始](#快速开始)
- [配置说明](#配置说明)
- [API 参考](#api 参考)
- [部署指南](#部署指南)
- [故障排查](#故障排查)

---

## 🚀 快速开始

### 1. 安装依赖

```bash
# 确保 Rust 1.75+ 已安装
rustc --version

# 如果没有安装 Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

### 2. 克隆项目

```bash
git clone https://github.com/Sdkwork-Cloud/sdkwork-tts.git
cd sdkwork-tts
```

### 3. 构建项目

```bash
# CPU 版本 (推荐用于测试)
cargo build --release --no-default-features --features cpu

# CUDA 版本 (需要 NVIDIA GPU)
$env:CUDA_COMPUTE_CAP='90'  # PowerShell
# export CUDA_COMPUTE_CAP='90'  # Linux/Mac
cargo build --release --features cuda
```

### 4. 启动服务器

```bash
# 使用启动脚本 (推荐)
./scripts/start_server.sh  # Linux/Mac
.\scripts\start_server.bat  # Windows

# 或直接运行
./target/release/sdkwork-tts server --mode local --port 8080
```

### 5. 验证运行

```bash
# 健康检查
curl http://localhost:8080/health

# 预期响应
{
  "status": "healthy",
  "version": "0.2.0",
  "mode": "local",
  "uptime": 10,
  "channels": ["local"],
  "speaker_count": 0
}
```

---

## ⚙️ 配置说明

### 配置文件

创建 `server.yaml` 配置文件：

```yaml
# 服务器模式：local, cloud, hybrid
mode: local

# 服务器地址和端口
host: "0.0.0.0"
port: 8080

# Local 模式配置
local:
  enabled: true
  checkpoints_dir: "checkpoints"
  default_engine: "indextts2"
  use_gpu: true
  use_fp16: false
  batch_size: 4
  max_concurrent: 10

# Speaker 库配置
speaker_lib:
  enabled: true
  local_path: "speaker_library"
  max_cache_size: 1000

# 日志配置
logging:
  level: "info"
  access_log: true
```

### 环境变量

可以使用环境变量覆盖配置：

```bash
# 服务器配置
export MODE=local
export PORT=8080
export HOST=0.0.0.0

# Cloud API Keys
export ALIYUN_API_KEY=your_key
export OPENAI_API_KEY=your_key
export VOLCANO_API_KEY=your_key
export MINIMAX_API_KEY=your_key

# 启动服务器
./scripts/start_server.sh
```

### 配置选项

#### 服务器模式

| 模式 | 说明 | 适用场景 |
|------|------|---------|
| **local** | 仅本地推理 | 有 GPU，注重隐私 |
| **cloud** | 仅云服务 | 无 GPU，快速部署 |
| **hybrid** | 混合模式 | 本地优先，云端备份 |

#### Local 配置

| 选项 | 默认值 | 说明 |
|------|--------|------|
| `enabled` | true | 启用 Local 模式 |
| `checkpoints_dir` | "checkpoints" | 模型目录 |
| `default_engine` | "indextts2" | 默认引擎 |
| `use_gpu` | true | 使用 GPU |
| `use_fp16` | false | FP16 精度 |
| `batch_size` | 4 | 批量大小 |
| `max_concurrent` | 10 | 最大并发 |

#### Cloud 配置

```yaml
cloud:
  enabled: false
  default_channel: null
  channels:
    - name: aliyun
      type: aliyun
      api_key: "${ALIYUN_API_KEY}"
      models: [tts-v1]
      default_model: tts-v1
      timeout: 30
      retries: 3
```

| 选项 | 默认值 | 说明 |
|------|--------|------|
| `name` | - | 渠道名称 |
| `type` | - | 渠道类型 |
| `api_key` | - | API Key |
| `models` | [] | 可用模型 |
| `timeout` | 30 | 超时 (秒) |
| `retries` | 3 | 重试次数 |

---

## 📡 API 参考

### 基础 API

#### 健康检查

```http
GET /health
```

**响应**:
```json
{
  "status": "healthy",
  "version": "0.2.0",
  "mode": "local",
  "uptime": 3600,
  "channels": ["local"],
  "speaker_count": 10
}
```

#### 服务器统计

```http
GET /api/v1/stats
```

**响应**:
```json
{
  "total_requests": 1000,
  "successful_requests": 998,
  "failed_requests": 2,
  "avg_processing_time_ms": 850.5,
  "active_connections": 5,
  "queue_size": 0,
  "memory_usage_mb": 256.5,
  "uptime": 3600
}
```

### 语音合成 API

#### 基础合成

```http
POST /api/v1/synthesis
Content-Type: application/json

{
  "text": "你好，这是测试文本",
  "speaker": "vivian",
  "channel": "local",
  "language": "zh",
  "parameters": {
    "speed": 1.0,
    "pitch": 0.0,
    "volume": 0.0,
    "emotion": null,
    "emotion_intensity": 1.0,
    "temperature": 0.8,
    "top_k": null,
    "top_p": null
  },
  "output_format": "wav",
  "streaming": false
}
```

**响应**:
```json
{
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "success",
  "audio": "base64_encoded_audio_data...",
  "duration": 2.5,
  "sample_rate": 24000,
  "format": "wav",
  "processing_time_ms": 850,
  "channel": "local",
  "model": "indextts2"
}
```

#### 流式合成

```http
POST /api/v1/synthesis/stream
Content-Type: application/json

{
  "text": "这是流式合成测试",
  "speaker": "vivian",
  "channel": "local"
}
```

**响应**: Server-Sent Events (SSE)

```
data: {"chunk": "base64_audio...", "index": 0}
data: {"chunk": "base64_audio...", "index": 1}
data: {"status": "complete"}
```

### 语音设计 API

```http
POST /api/v1/voice/design
Content-Type: application/json

{
  "text": "Hello from designed voice",
  "voice_design": {
    "description": "A warm, friendly female voice",
    "gender": "female",
    "age": "young",
    "accent": "american",
    "style": "friendly"
  },
  "output_format": "wav"
}
```

### 语音克隆 API

```http
POST /api/v1/voice/clone
Content-Type: application/json

{
  "text": "这是克隆的声音",
  "voice_clone": {
    "reference_audio": "data:audio/wav;base64,...",
    "reference_text": "参考音频的文本内容",
    "mode": "full"
  },
  "output_format": "wav"
}
```

**克隆模式**:

| 模式 | 说明 | 需要文本 | 质量 |
|------|------|---------|------|
| **quick** | 快速克隆 | 否 | 中 |
| **full** | 完整克隆 | 是 | 高 |
| **fine_tune** | 微调克隆 | 是 (多句) | 最高 |

### Speaker 管理 API

#### 列出 Speaker

```http
GET /api/v1/speakers?page=1&page_size=20&gender=female&language=zh
```

**响应**:
```json
{
  "total": 10,
  "speakers": [
    {
      "id": "vivian",
      "name": "Vivian",
      "description": "明亮、略带沙哑的年轻女声",
      "gender": "female",
      "age": "young",
      "languages": ["zh"],
      "source": "local",
      "preview_url": "/api/v1/speakers/vivian/preview",
      "tags": ["clear", "young", "female"],
      "created_at": "2026-02-21T00:00:00Z",
      "updated_at": "2026-02-21T00:00:00Z"
    }
  ],
  "pagination": {
    "page": 1,
    "page_size": 20,
    "total_pages": 1
  }
}
```

#### 获取 Speaker 详情

```http
GET /api/v1/speakers/:id
```

#### 添加 Speaker

```http
POST /api/v1/speakers
Content-Type: multipart/form-data

speaker_info: {...}
audio: <audio_file>
```

#### 删除 Speaker

```http
DELETE /api/v1/speakers/:id
```

### 渠道管理 API

#### 列出渠道

```http
GET /api/v1/channels
```

**响应**:
```json
{
  "channels": [
    {
      "name": "local",
      "type": "local",
      "enabled": true,
      "models": ["indextts2", "qwen3-tts"]
    },
    {
      "name": "aliyun",
      "type": "aliyun",
      "enabled": true,
      "models": ["tts-v1"]
    }
  ]
}
```

#### 列出渠道模型

```http
GET /api/v1/channels/:name/models
```

---

## 🚀 部署指南

### Docker 部署

```dockerfile
FROM rust:1.75 as builder

WORKDIR /app
COPY . .

RUN cargo build --release --no-default-features --features cpu

FROM debian:bullseye-slim

RUN apt-get update && apt-get install -y \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/target/release/sdkwork-tts /usr/local/bin/

EXPOSE 8080

CMD ["sdkwork-tts", "server", "--mode", "local", "--port", "8080"]
```

**构建和运行**:

```bash
docker build -t sdkwork-tts .
docker run -p 8080:8080 sdkwork-tts
```

### Kubernetes 部署

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: tts-server
spec:
  replicas: 3
  selector:
    matchLabels:
      app: tts-server
  template:
    metadata:
      labels:
        app: tts-server
    spec:
      containers:
      - name: tts-server
        image: sdkwork-tts:latest
        ports:
        - containerPort: 8080
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "2000m"
        env:
        - name: MODE
          value: "local"
        - name: PORT
          value: "8080"
---
apiVersion: v1
kind: Service
metadata:
  name: tts-server
spec:
  selector:
    app: tts-server
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8080
  type: LoadBalancer
```

### 系统服务 (systemd)

```ini
[Unit]
Description=SDKWork-TTS Server
After=network.target

[Service]
Type=simple
User=tts
WorkingDirectory=/opt/sdkwork-tts
ExecStart=/opt/sdkwork-tts/target/release/sdkwork-tts server --config /etc/sdkwork-tts/server.yaml
Restart=always
RestartSec=5

# 环境变量
Environment="MODE=local"
Environment="PORT=8080"

# 资源限制
LimitNOFILE=65535
LimitNPROC=4096

[Install]
WantedBy=multi-user.target
```

**安装服务**:

```bash
sudo cp tts-server.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable tts-server
sudo systemctl start tts-server
sudo systemctl status tts-server
```

---

## 🔧 故障排查

### 常见问题

#### 1. 服务器无法启动

**错误**: `Address already in use`

**解决**:
```bash
# 检查端口占用
lsof -i :8080

# 或使用不同端口
./target/release/sdkwork-tts server --port 8081
```

#### 2. 模型加载失败

**错误**: `Model file not found`

**解决**:
```bash
# 检查模型目录
ls checkpoints/

# 下载模型
huggingface-cli download IndexTeam/IndexTTS-2 --local-dir checkpoints/indextts2
```

#### 3. 内存不足

**错误**: `Out of memory`

**解决**:
```yaml
# 减小配置
local:
  batch_size: 1
  max_concurrent: 2
  
# 或使用 CPU 模式
local:
  use_gpu: false
```

#### 4. API 调用超时

**错误**: `Request timeout`

**解决**:
```yaml
# 增加超时时间
cloud:
  channels:
    - name: aliyun
      timeout: 60  # 增加到 60 秒
      retries: 5   # 增加到 5 次
```

### 日志分析

```bash
# 查看日志
tail -f /var/log/tts-server.log

# 搜索错误
grep "ERROR" /var/log/tts-server.log

# 查看访问日志
grep "POST /api/v1/synthesis" /var/log/tts-server.log
```

### 性能监控

```bash
# 查看服务器统计
curl http://localhost:8080/api/v1/stats

# 监控系统资源
top -p $(pgrep sdkwork-tts)

# 查看内存使用
ps -o pid,rss,command -p $(pgrep sdkwork-tts)
```

---

## 📞 支持

- **文档**: `/docs/`
- **问题**: [GitHub Issues](https://github.com/Sdkwork-Cloud/sdkwork-tts/issues)
- **讨论**: [GitHub Discussions](https://github.com/Sdkwork-Cloud/sdkwork-tts/discussions)

---

**文档版本**: 1.0.0  
**最后更新**: 2026-02-21

