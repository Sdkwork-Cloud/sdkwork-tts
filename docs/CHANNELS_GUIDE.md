# TTS Cloud Channels 配置指南

**版本**: 1.0.0  
**日期**: 2026 年 2 月 21 日

---

## 📖 概述

SDKWork-TTS 支持多种云 TTS 服务提供商，每种都有其独特的特点和优势。

| 渠道 | 支持语言 | 声音数量 | 延迟 | 价格 | 最佳用途 |
|------|---------|---------|------|------|---------|
| **OpenAI** | 英语 | 6 | ~1s | $$ | 高质量英语 |
| **Google Cloud** | 220+ | 300+ | ~500ms | $$$ | 多语言支持 |
| **Aliyun** | 中文为主 | 50+ | ~500ms | $ | 中文场景 |
| **Volcano** | 中文/英语 | 20+ | ~400ms | $ | 性价比 |
| **Minimax** | 中文为主 | 30+ | ~600ms | $ | 情感表达 |

---

## 🔑 获取 API 密钥

### OpenAI

1. 访问 [OpenAI Platform](https://platform.openai.com/)
2. 注册/登录账号
3. 创建 API Key
4. 充值账户

```bash
export OPENAI_API_KEY=sk-...
```

**定价**: $0.015/1K characters (tts-1), $0.030/1K characters (tts-1-hd)

### Google Cloud

1. 访问 [Google Cloud Console](https://console.cloud.google.com/)
2. 创建项目
3. 启用 Text-to-Speech API
4. 创建 API Key 或服务账号

```bash
export GOOGLE_API_KEY=...
export GOOGLE_PROJECT_ID=your-project-id
```

**定价**: $4.00/1M characters (Neural2), $16.00/1M characters (Studio)

### Aliyun (阿里云)

1. 访问 [阿里云控制台](https://console.aliyun.com/)
2. 开通智能语音交互服务
3. 创建 AccessKey
4. 实名认证

```bash
export ALIYUN_API_KEY=your-access-key-id
export ALIYUN_API_SECRET=your-access-key-secret
```

**定价**: ¥0.008/次 (标准版), ¥0.02/次 (Premium)

### Volcano Engine (火山引擎)

1. 访问 [火山引擎控制台](https://console.volcengine.com/)
2. 开通语音合成服务
3. 创建 AccessKey
4. 创建应用获取 AppID

```bash
export VOLCANO_API_KEY=your-access-key
export VOLCANO_API_SECRET=your-secret-key
export VOLCANO_APP_ID=your-app-id
```

**定价**: ¥0.006/次 (标准), ¥0.012/次 (Premium)

### Minimax

1. 访问 [Minimax 平台](https://platform.minimaxi.com/)
2. 注册账号
3. 创建 API Key

```bash
export MINIMAX_API_KEY=...
```

**定价**: ¥0.008/次

---

## ⚙️ 配置文件

### server.yaml

```yaml
mode: cloud

cloud:
  enabled: true
  default_channel: openai
  
  channels:
    # OpenAI
    - name: openai
      type: openai
      api_key: "${OPENAI_API_KEY}"
      models:
        - tts-1
        - tts-1-hd
      default_model: tts-1
      timeout: 30
      retries: 3
    
    # Google Cloud
    - name: google
      type: google
      api_key: "${GOOGLE_API_KEY}"
      app_id: "${GOOGLE_PROJECT_ID}"
      models:
        - Standard
        - WaveNet
        - Neural2
        - Studio
      default_model: Neural2
      timeout: 30
      retries: 3
    
    # Aliyun
    - name: aliyun
      type: aliyun
      api_key: "${ALIYUN_API_KEY}"
      api_secret: "${ALIYUN_API_SECRET}"
      models:
        - tts-v1
      default_model: tts-v1
      timeout: 30
      retries: 3
    
    # Volcano Engine
    - name: volcano
      type: volcano
      api_key: "${VOLCANO_API_KEY}"
      api_secret: "${VOLCANO_API_SECRET}"
      app_id: "${VOLCANO_APP_ID}"
      models:
        - volcano_tts
        - volcano_tts_premium
      default_model: volcano_tts
      timeout: 30
      retries: 3
```

### .env 文件

```bash
# OpenAI
OPENAI_API_KEY=sk-...

# Google Cloud
GOOGLE_API_KEY=...
GOOGLE_PROJECT_ID=your-project-id

# Aliyun
ALIYUN_API_KEY=your-access-key-id
ALIYUN_API_SECRET=your-access-key-secret

# Volcano Engine
VOLCANO_API_KEY=your-access-key
VOLCANO_API_SECRET=your-secret-key
VOLCANO_APP_ID=your-app-id

# Minimax
MINIMAX_API_KEY=...
```

---

## 🎤 Speaker 参考

### OpenAI Voices

| Voice | Gender | 特点 | 适用场景 |
|-------|--------|------|---------|
| **alloy** | Neutral | 中性、多功能 | 通用场景 |
| **echo** | Male | 温暖、友好 | 客服、助手 |
| **fable** | Neutral | 英式口音、表现力强 | 故事讲述 |
| **onyx** | Male | 深沉、权威 | 新闻、纪录片 |
| **nova** | Female | 明亮、热情 | 营销、广告 |
| **shimmer** | Female | 柔和、温柔 | 冥想、放松 |

### Google Cloud Voices

| 系列 | 质量 | 价格 | 延迟 |
|------|------|------|------|
| **Standard** | 标准 | $ | 最低 |
| **WaveNet** | 高 | $$ | 低 |
| **Neural2** | 很高 | $$$ | 中 |
| **Studio** | 最高 | $$$$ | 最高 |

### Aliyun Voices

| 声音 | 语言 | 性别 | 特点 |
|------|------|------|------|
| **xiaoyun** | 中文 | 女 | 标准女声 |
| **aixia** | 中文 | 男 | 标准男声 |
| **aiqi** | 中文 | 女 | 温柔女声 |
| **aitong** | 中文 | 男 | 童声 |

### Volcano Engine Voices

| 声音 | 语言 | 性别 | 特点 |
|------|------|------|------|
| **BV001** | 中文 | 女 | 温柔知性 |
| **BV002** | 中文 | 男 | 沉稳磁性 |
| **BV005** | 英语 | 女 | 自然流畅 |
| **BV006** | 英语 | 男 | 自然清晰 |

---

## 🚀 使用示例

### 基础合成

```bash
# OpenAI
curl -X POST http://localhost:8080/api/v1/synthesis \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello world",
    "speaker": "alloy",
    "channel": "openai",
    "model": "tts-1"
  }'

# Google Cloud
curl -X POST http://localhost:8080/api/v1/synthesis \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello world",
    "speaker": "en-US-Neural2-F",
    "channel": "google",
    "model": "Neural2"
  }'

# Aliyun
curl -X POST http://localhost:8080/api/v1/synthesis \
  -H "Content-Type: application/json" \
  -d '{
    "text": "你好世界",
    "speaker": "xiaoyun",
    "channel": "aliyun"
  }'

# Volcano Engine
curl -X POST http://localhost:8080/api/v1/synthesis \
  -H "Content-Type: application/json" \
  -d '{
    "text": "你好世界",
    "speaker": "BV001",
    "channel": "volcano"
  }'
```

### 列出 Speaker

```bash
# OpenAI
curl http://localhost:8080/api/v1/channels/openai/speakers

# Google Cloud
curl http://localhost:8080/api/v1/channels/google/speakers

# Aliyun
curl http://localhost:8080/api/v1/channels/aliyun/speakers

# Volcano Engine
curl http://localhost:8080/api/v1/channels/volcano/speakers
```

### 列出模型

```bash
# OpenAI
curl http://localhost:8080/api/v1/channels/openai/models

# Google Cloud
curl http://localhost:8080/api/v1/channels/google/models
```

---

## 💰 成本优化

### 选择建议

| 场景 | 推荐渠道 | 原因 |
|------|---------|------|
| **英语高质量** | OpenAI | 最佳音质 |
| **多语言** | Google Cloud | 支持 220+ 语言 |
| **中文场景** | Aliyun/Volcano | 性价比高 |
| **大批量** | Volcano | 价格最低 |
| **情感表达** | Minimax | 情感丰富 |

### 优化技巧

1. **使用 Local 模式处理常用语音**
   - 缓存常用语音片段
   - 减少 API 调用

2. **选择合适的模型**
   - 非关键场景使用标准模型
   - 关键场景使用高质量模型

3. **批量处理**
   - 合并短文本
   - 减少 API 调用次数

4. **监控用量**
   - 设置预算告警
   - 定期审查用量

---

## 🔍 故障排查

### 认证错误

```bash
# OpenAI
# 错误：Invalid API key
# 解决：检查 OPENAI_API_KEY 是否正确

# Google Cloud
# 错误：API_KEY_INVALID
# 解决：检查 GOOGLE_API_KEY 和项目 ID

# Aliyun
# 错误：InvalidAccessKeyId
# 解决：检查 AccessKey ID 和 Secret

# Volcano Engine
# 错误：InvalidAccessKey
# 解决：检查 AccessKey 和 AppID
```

### 配额限制

```bash
# 查看用量
# OpenAI: https://platform.openai.com/usage
# Google: https://console.cloud.google.com/billing
# Aliyun: https://usercenter2.aliyun.com/bill
# Volcano: https://console.volcengine.com/bill
```

### 网络问题

```bash
# 测试连接
curl -I https://api.openai.com
curl -I https://texttospeech.googleapis.com
curl -I https://openspeech.bytedance.com

# 使用代理
export HTTPS_PROXY=http://proxy:port
```

---

## 📊 性能对比

| 渠道 | 平均延迟 | 成功率 | 并发限制 |
|------|---------|--------|---------|
| **OpenAI** | ~1000ms | 99.9% | 60 RPM |
| **Google Cloud** | ~500ms | 99.95% | 300 RPM |
| **Aliyun** | ~500ms | 99.9% | 100 QPS |
| **Volcano** | ~400ms | 99.9% | 200 QPS |

---

## 📞 支持链接

- **OpenAI**: https://platform.openai.com/docs
- **Google Cloud**: https://cloud.google.com/text-to-speech/docs
- **Aliyun**: https://help.aliyun.com/product/30421.html
- **Volcano Engine**: https://www.volcengine.com/docs/6561/79817

---

**文档版本**: 1.0.0  
**最后更新**: 2026-02-21
