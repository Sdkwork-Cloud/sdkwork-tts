> Migrated from `docs/DEPLOYMENT_GUIDE.md` on 2026-06-24.
> Owner: SDKWork maintainers

# SDKWork-TTS 部署指南

**版本**: 1.0.0  
**日期**: 2026 年 2 月 21 日

---

## 📖 目录

- [快速开始](#快速开始)
- [Docker 部署](#docker-部署)
- [本地安装](#本地安装)
- [库集成](#库集成)
- [跨平台启动](#跨平台启动)
- [生产部署](#生产部署)
- [故障排查](#故障排查)

---

## 🚀 快速开始

### 方式一：Docker (推荐)

```bash
# CPU 版本
docker compose --profile cpu up -d

# GPU 版本 (需要 NVIDIA GPU)
docker compose --profile gpu up -d

# 验证运行
curl http://localhost:8080/health
```

### 方式二：本地安装

```bash
# Linux/macOS
curl -fsSL https://github.com/Sdkwork-Cloud/sdkwork-tts/releases/latest/download/install.sh | bash

# Windows PowerShell
Invoke-WebRequest -Uri https://github.com/Sdkwork-Cloud/sdkwork-tts/releases/latest/download/install.ps1 -OutFile install.ps1
.\install.ps1
```

### 方式三：源码编译

```bash
# 克隆项目
git clone https://github.com/Sdkwork-Cloud/sdkwork-tts.git
cd sdkwork-tts

# 编译
cargo build --release

# 运行
./target/release/sdkwork-tts server --mode local
```

---

## 🐳 Docker 部署

### 1. 快速启动

```bash
# CPU 模式
docker compose --profile cpu up -d

# GPU 模式
docker compose --profile gpu up -d

# Cloud 模式
docker compose --profile cloud up -d
```

### 2. 环境变量配置

创建 `.env` 文件：

```bash
# OpenAI API
OPENAI_API_KEY=sk-your-api-key

# Aliyun API
ALIYUN_API_KEY=your-aliyun-key
ALIYUN_API_SECRET=your-aliyun-secret

# 服务器配置
MODE=local
PORT=8080
RUST_LOG=info
```

### 3. 数据卷挂载

```yaml
volumes:
  - ./checkpoints:/app/checkpoints:ro  # 模型文件 (只读)
  - ./speaker_library:/app/speaker_library  # Speaker 库
  - ./config:/app/config:ro  # 配置文件 (只读)
```

### 4. 构建自定义镜像

```bash
# CPU 版本
docker build -t sdkwork-tts:cpu --target runtime .

# GPU 版本
docker build -t sdkwork-tts:gpu -f Dockerfile.gpu .

# 运行自定义镜像
docker run -d -p 8080:8080 \
  -v ./checkpoints:/app/checkpoints \
  -v ./speaker_library:/app/speaker_library \
  sdkwork-tts:cpu
```

### 5. Docker 命令参考

```bash
# 查看日志
docker logs -f sdkwork-tts-cpu

# 进入容器
docker exec -it sdkwork-tts-cpu bash

# 重启服务
docker compose restart

# 停止服务
docker compose down

# 查看状态
docker compose ps
```

---

## 💻 本地安装

### Linux/macOS

```bash
# 1. 下载安装脚本
curl -fsSL https://raw.githubusercontent.com/Sdkwork-Cloud/sdkwork-tts/main/scripts/install.sh -o install.sh

# 2. 运行安装
chmod +x install.sh
./install.sh

# 3. 加载环境变量
source ~/.bashrc  # 或 source ~/.zshrc

# 4. 验证安装
sdkwork-tts --version

# 5. 启动服务器
sdkwork-tts server --mode local
```

### Windows

```powershell
# 1. 下载安装脚本
Invoke-WebRequest -Uri "https://raw.githubusercontent.com/Sdkwork-Cloud/sdkwork-tts/main/scripts/install.ps1" -OutFile "install.ps1"

# 2. 运行安装
.\install.ps1

# 3. 重启 PowerShell

# 4. 验证安装
sdkwork-tts --version

# 5. 启动服务器
sdkwork-tts server --mode local
```

### 自定义安装路径

```bash
# Linux/macOS
INSTALL_DIR=/opt/sdkwork-tts ./install.sh

# Windows
.\install.ps1 -InstallDir "C:\sdkwork-tts"
```

---

## 📚 库集成

### Cargo.toml 配置

```toml
[dependencies]
sdkwork-tts = { version = "1.0", features = ["server", "inference"] }
tokio = { version = "1", features = ["full"] }
anyhow = "1.0"
```

### 基本使用

```rust
use sdkwork_tts::server::{TtsServer, ServerConfig};
use anyhow::Result;

#[tokio::main]
async fn main() -> Result<()> {
    // 创建服务器配置
    let config = ServerConfig::default();
    
    // 创建并运行服务器
    let server = TtsServer::new(config);
    server.run().await?;
    
    Ok(())
}
```

### Local 模式集成

```rust
use sdkwork_tts::server::{ServerConfig, ServerMode, LocalConfig};

let config = ServerConfig {
    mode: ServerMode::Local,
    local: LocalConfig {
        enabled: true,
        checkpoints_dir: "checkpoints".into(),
        use_gpu: true,
        ..Default::default()
    },
    ..Default::default()
};

let server = TtsServer::new(config);
server.run().await?;
```

### Cloud 模式集成

```rust
use sdkwork_tts::server::{ServerConfig, ServerMode, CloudConfig, ChannelConfig};

let config = ServerConfig {
    mode: ServerMode::Cloud,
    cloud: CloudConfig {
        enabled: true,
        channels: vec![
            ChannelConfig {
                name: "openai".to_string(),
                channel_type: sdkwork_tts::server::ChannelTypeConfig::Openai,
                api_key: std::env::var("OPENAI_API_KEY")?,
                ..Default::default()
            }
        ],
        ..Default::default()
    },
    ..Default::default()
};
```

### 完整示例

查看 `examples/library_integration.rs` 获取完整示例代码。

---

## 🖥️ 跨平台启动

### Linux/macOS 脚本

```bash
#!/bin/bash
# start.sh

# 设置环境变量
export MODE=${MODE:-local}
export PORT=${PORT:-8080}
export RUST_LOG=${RUST_LOG:-info}

# 启动服务器
./sdkwork-tts server \
  --mode $MODE \
  --port $PORT \
  --config ${CONFIG_FILE:-server.yaml}
```

### Windows 批处理

```batch
@echo off
REM start.bat

REM 设置环境变量
set MODE=%MODE:~0,5%
if "%MODE%"=="" set MODE=local
set PORT=%PORT:~0,4%
if "%PORT%"=="" set PORT=8080

REM 启动服务器
sdkwork-tts server --mode %MODE% --port %PORT%
```

### PowerShell 脚本

```powershell
# start.ps1

param(
    [string]$Mode = "local",
    [string]$Port = "8080",
    [string]$Config = "server.yaml"
)

# 启动服务器
& sdkwork-tts server --mode $Mode --port $Port --config $Config
```

---

## 🏭 生产部署

### Kubernetes 部署

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: sdkwork-tts
spec:
  replicas: 3
  selector:
    matchLabels:
      app: sdkwork-tts
  template:
    metadata:
      labels:
        app: sdkwork-tts
    spec:
      containers:
      - name: tts
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
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: sdkwork-tts
spec:
  selector:
    app: sdkwork-tts
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8080
  type: LoadBalancer
```

### Systemd 服务

```ini
[Unit]
Description=SDKWork-TTS Server
After=network.target

[Service]
Type=simple
User=tts
WorkingDirectory=/opt/sdkwork-tts
ExecStart=/opt/sdkwork-tts/bin/sdkwork-tts server --config /opt/sdkwork-tts/config/server.yaml
Restart=always
RestartSec=5

# 环境变量
Environment="MODE=local"
Environment="PORT=8080"
Environment="RUST_LOG=info"

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

### Nginx 反向代理

```nginx
upstream tts_backend {
    server 127.0.0.1:8080;
    keepalive 32;
}

server {
    listen 80;
    server_name tts.example.com;

    location / {
        proxy_pass http://tts_backend;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        
        # 超时设置
        proxy_connect_timeout 60s;
        proxy_send_timeout 300s;
        proxy_read_timeout 300s;
    }
}
```

---

## 🔧 故障排查

### Docker 问题

**容器无法启动**:
```bash
# 查看日志
docker logs sdkwork-tts-cpu

# 检查配置
docker inspect sdkwork-tts-cpu

# 重新创建容器
docker compose down
docker compose up -d
```

**GPU 不可用**:
```bash
# 检查 NVIDIA 驱动
nvidia-smi

# 检查 NVIDIA Container Toolkit
docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi
```

### 本地安装问题

**命令找不到**:
```bash
# Linux/macOS
export PATH="$HOME/.sdkwork-tts/bin:$PATH"

# Windows
$env:Path += ";$HOME\sdkwork-tts\bin"
```

**权限问题**:
```bash
# Linux/macOS
chmod +x ~/.sdkwork-tts/bin/sdkwork-tts

# Windows (以管理员运行 PowerShell)
```

### 性能问题

**内存不足**:
```yaml
# 减少并发
local:
  max_concurrent: 2
  batch_size: 1
```

**响应慢**:
```bash
# 启用 GPU
export MODE=local
# 在配置中设置 use_gpu = true

# 或使用 Cloud 模式
export MODE=cloud
```

---

## 📞 支持

- **文档**: `/docs/`
- **问题**: [GitHub Issues](https://github.com/Sdkwork-Cloud/sdkwork-tts/issues)
- **讨论**: [GitHub Discussions](https://github.com/Sdkwork-Cloud/sdkwork-tts/discussions)

---

**文档版本**: 1.0.0  
**最后更新**: 2026-02-21

