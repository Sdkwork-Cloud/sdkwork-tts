# SDKWork-TTS 安装总结

**版本**: 1.0.0  
**日期**: 2026 年 2 月 21 日

---

## ✅ 安装完成！

感谢您选择 SDKWork-TTS。您的安装已完成。

---

## 📍 安装位置

```
安装目录：~/.sdkwork-tts/
二进制文件：~/.sdkwork-tts/bin/sdkwork-tts
数据目录：~/.sdkwork-tts/data/
配置目录：~/.sdkwork-tts/config/
```

---

## 🚀 快速启动

### 1. 加载环境变量

```bash
# Bash
source ~/.bashrc

# Zsh
source ~/.zshrc

# Fish
source ~/.config/fish/config.fish
```

### 2. 验证安装

```bash
sdkwork-tts --version
```

### 3. 启动服务器

```bash
# Local 模式 (推荐)
sdkwork-tts server --mode local

# Cloud 模式
sdkwork-tts server --mode cloud

# 使用配置文件
sdkwork-tts server --config ~/.sdkwork-tts/config/server.yaml
```

### 4. 测试 API

```bash
# 健康检查
curl http://localhost:8080/health

# 语音合成
curl -X POST http://localhost:8080/api/v1/synthesis \
  -H "Content-Type: application/json" \
  -d '{
    "text": "你好，这是测试文本",
    "speaker": "vivian",
    "channel": "local"
  }'
```

---

## 📚 下一步

### 阅读文档

- [快速入门](GETTING_STARTED.md) - 5 分钟上手
- [快速参考](QUICK_REFERENCE.md) - 常用命令
- [部署指南](docs/DEPLOYMENT_GUIDE.md) - 完整部署文档

### 配置服务器

```bash
# 生成配置文件
./scripts/generate_config.sh

# 编辑配置
nano ~/.sdkwork-tts/config/server.yaml
```

### 设置 Cloud API (可选)

```bash
# 编辑环境文件
nano ~/.sdkwork-tts/config/env

# 添加 API 密钥
export OPENAI_API_KEY=sk-...
export ALIYUN_API_KEY=...
```

---

## 🛠️ 常用命令

```bash
# 查看帮助
sdkwork-tts --help

# 列出引擎
sdkwork-tts engines

# 诊断工具
./scripts/diagnose.sh

# 升级
./scripts/upgrade.sh

# 卸载
./scripts/uninstall.sh
```

---

## 🐳 使用 Docker

```bash
# 启动
docker compose --profile cpu up -d

# 查看日志
docker logs -f sdkwork-tts

# 停止
docker compose down
```

---

## 📞 获取帮助

### 文档
- [GETTING_STARTED.md](GETTING_STARTED.md)
- [QUICK_REFERENCE.md](QUICK_REFERENCE.md)
- [docs/DEPLOYMENT_GUIDE.md](docs/DEPLOYMENT_GUIDE.md)

### 社区
- [GitHub Issues](https://github.com/Sdkwork-Cloud/sdkwork-tts/issues)
- [GitHub Discussions](https://github.com/Sdkwork-Cloud/sdkwork-tts/discussions)

### 诊断
```bash
# 运行诊断工具
./scripts/diagnose.sh

# 查看日志
tail -f ~/.sdkwork-tts/logs/server.log
```

---

## 🎉 恭喜！

您已成功安装 SDKWork-TTS！

**开始使用**: `sdkwork-tts server --mode local`

**查看文档**: `cat GETTING_STARTED.md`

---

**安装日期**: $(date)  
**版本**: 1.0.0  
**支持**: https://github.com/Sdkwork-Cloud/sdkwork-tts
