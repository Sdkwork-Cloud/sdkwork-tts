# Qwen3-TTS 生产就绪路线图

## 概述

本文档提供了将当前 Qwen3-TTS Rust 实现变为生产就绪 TTS 引擎的详细路线图和实现指南。

**当前状态**: ✅ 架构完整，测试通过  
**目标状态**: 🎯 生产就绪，完整推理  
**预计工作量**: ~1,700 行代码，2-3 周开发时间

---

## 阶段 1: 模型权重加载 (500 行)

### 1.1 HuggingFace 权重下载

#### 任务
- [ ] 实现模型下载器
- [ ] 支持断点续传
- [ ] 验证文件完整性

#### 实现指南

```rust
// src/models/qwen3_tts/downloader.rs

use hf_hub::{api::sync::Api, Repo, RepoType};
use std::path::{Path, PathBuf};

pub struct ModelDownloader {
    api: Api,
    cache_dir: PathBuf,
}

impl ModelDownloader {
    pub fn new(cache_dir: Option<PathBuf>) -> Result<Self> {
        let api = Api::new()?;
        let cache_dir = cache_dir.unwrap_or_else(|| {
            dirs::home_dir()
                .unwrap()
                .join(".cache")
                .join("qwen3-tts")
        });
        Ok(Self { api, cache_dir })
    }

    pub fn download(&self, model_id: &str) -> Result<PathBuf> {
        let repo = Repo::with_revision(
            model_id.to_string(),
            RepoType::Model,
            "main".to_string(),
        );
        
        let model_dir = self.api.repo(repo).get(".")?;
        Ok(model_dir)
    }

    pub fn download_file(&self, model_id: &str, filename: &str) -> Result<PathBuf> {
        let repo = Repo::with_revision(
            model_id.to_string(),
            RepoType::Model,
            "main".to_string(),
        );
        
        let file_path = self.api.repo(repo).get(filename)?;
        Ok(file_path)
    }
}
```

#### 测试
```rust
#[test]
fn test_download() {
    let downloader = ModelDownloader::new(None).unwrap();
    let path = downloader.download("Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice").unwrap();
    assert!(path.exists());
}
```

### 1.2 权重映射

#### 任务
- [ ] 定义权重映射表
- [ ] 实现权重加载器
- [ ] 验证权重完整性

#### 权重映射表

```rust
// src/models/qwen3_tts/weights.rs

use std::collections::HashMap;

pub fn get_talker_weight_map() -> HashMap<String, String> {
    let mut map = HashMap::new();
    
    // Embedding
    map.insert("model.embed_tokens.weight".to_string(), "embed_tokens.weight".to_string());
    
    // Transformer layers
    for i in 0..28 {
        let prefix = format!("model.layers.{i}");
        
        // Self-attention
        map.insert(
            format!("{prefix}.self_attn.q_proj.weight"),
            format!("{prefix}.self_attn.wq.weight"),
        );
        map.insert(
            format!("{prefix}.self_attn.k_proj.weight"),
            format!("{prefix}.self_attn.wk.weight"),
        );
        map.insert(
            format!("{prefix}.self_attn.v_proj.weight"),
            format!("{prefix}.self_attn.wv.weight"),
        );
        map.insert(
            format!("{prefix}.self_attn.o_proj.weight"),
            format!("{prefix}.self_attn.wo.weight"),
        );
        
        // MLP
        map.insert(
            format!("{prefix}.mlp.gate_proj.weight"),
            format!("{prefix}.mlp.w1.weight"),
        );
        map.insert(
            format!("{prefix}.mlp.up_proj.weight"),
            format!("{prefix}.mlp.w3.weight"),
        );
        map.insert(
            format!("{prefix}.mlp.down_proj.weight"),
            format!("{prefix}.mlp.w2.weight"),
        );
        
        // Layer norms
        map.insert(
            format!("{prefix}.input_layernorm.weight"),
            format!("{prefix}.ln1.weight"),
        );
        map.insert(
            format!("{prefix}.post_attention_layernorm.weight"),
            format!("{prefix}.ln2.weight"),
        );
    }
    
    // Final norm
    map.insert("model.norm.weight".to_string(), "norm.weight".to_string());
    
    map
}
```

#### 权重加载器

```rust
use candle_core::{safetensors::load, Device, Tensor};
use std::collections::HashMap;
use std::path::Path;

pub struct WeightLoader {
    weights: HashMap<String, Tensor>,
}

impl WeightLoader {
    pub fn load<P: AsRef<Path>>(path: P, device: &Device) -> Result<Self> {
        let weights = load(path.as_ref(), device)?;
        Ok(Self { weights })
    }

    pub fn get(&self, name: &str) -> Result<&Tensor> {
        self.weights.get(name).ok_or_else(|| {
            anyhow::anyhow!("Weight not found: {}", name)
        })
    }

    pub fn remap(&self, mapping: &HashMap<String, String>) -> HashMap<String, Tensor> {
        let mut remapped = HashMap::new();
        
        for (hf_name, our_name) in mapping {
            if let Some(tensor) = self.weights.get(hf_name) {
                remapped.insert(our_name.clone(), tensor.clone());
            }
        }
        
        remapped
    }

    pub fn verify(&self, expected_keys: &[&str]) -> Result<()> {
        for key in expected_keys {
            if !self.weights.contains_key(*key) {
                return Err(anyhow::anyhow!("Missing weight: {}", key));
            }
        }
        Ok(())
    }
}
```

### 1.3 Safetensors 格式支持

#### 任务
- [ ] 实现 safetensors 加载
- [ ] 支持分片加载
- [ ] 内存映射优化

```rust
use safetensors::SafeTensors;
use memmap2::Mmap;
use std::fs::File;

pub fn load_safetensors_mmap<P: AsRef<Path>>(
    path: P,
    device: &Device,
) -> Result<HashMap<String, Tensor>> {
    let file = File::open(path.as_ref())?;
    let mmap = unsafe { Mmap::map(&file)? };
    
    let safetensor = SafeTensors::deserialize(&mmap)?;
    
    let mut tensors = HashMap::new();
    for name in safetensor.names() {
        let view = safetensor.tensor(&name)?;
        let tensor = Tensor::from_raw_buffer(
            view.data(),
            view.dtype().into(),
            view.shape(),
            device,
        )?;
        tensors.insert(name.to_string(), tensor);
    }
    
    Ok(tensors)
}
```

---

## 阶段 2: 完整推理循环 (300 行)

### 2.1 集成 TalkerModel + CodePredictor + Decoder

#### 任务
- [ ] 创建推理管道
- [ ] 实现数据流转换
- [ ] 错误处理

```rust
// src/models/qwen3_tts/pipeline.rs

pub struct Qwen3TTSPipeline {
    talker: TalkerModel,
    code_predictor: CodePredictor,
    decoder: Decoder12Hz,
    device: Device,
}

impl Qwen3TTSPipeline {
    pub fn new(
        talker: TalkerModel,
        code_predictor: CodePredictor,
        decoder: Decoder12Hz,
        device: Device,
    ) -> Self {
        Self {
            talker,
            code_predictor,
            decoder,
            device,
        }
    }

    pub fn synthesize(
        &self,
        text_ids: &[u32],
        speaker_condition: Option<&Tensor>,
        config: &GenerationConfig,
    ) -> Result<Vec<f32>> {
        // Step 1: TalkerModel - text to semantic tokens
        let semantic_tokens = self.talker.generate(text_ids, speaker_condition, config)?;
        
        // Step 2: CodePredictor - semantic to acoustic codes
        let acoustic_codes = self.code_predictor.generate(&semantic_tokens)?;
        
        // Step 3: Decoder - acoustic codes to audio
        let audio = self.decoder.decode(&acoustic_codes)?;
        
        Ok(audio)
    }
}
```

### 2.2 自回归生成循环

#### 任务
- [ ] 实现生成循环
- [ ] EOS 检测
- [ ] 最大长度限制

```rust
// src/models/qwen3_tts/generate.rs

pub fn generate_autoregressive(
    model: &mut TalkerModel,
    input_ids: &[u32],
    config: &GenerationConfig,
) -> Result<Vec<u32>> {
    let mut tokens = input_ids.to_vec();
    let mut kv_caches = model.new_kv_caches(config.max_new_tokens);
    let mut ctx = SamplingContext::new(config.seed);
    
    for _ in 0..config.max_new_tokens {
        // Forward pass
        let logits = model.forward(&tokens, &mut kv_caches)?;
        
        // Get last token logits
        let last_logits = logits.i((0, logits.dim(1)? - 1))?;
        
        // Sample next token
        let next_token = if config.temperature == 0.0 {
            argmax(&last_logits)?
        } else {
            sample(&last_logits, config, &mut ctx, &tokens)?
        };
        
        // Append token
        tokens.push(next_token);
        
        // Check EOS
        if Some(next_token) == config.eos_token_id {
            break;
        }
    }
    
    Ok(tokens)
}
```

### 2.3 流式输出支持

#### 任务
- [ ] 实现流式生成
- [ ] 音频块回调
- [ ] 低延迟优化

```rust
// src/models/qwen3_tts/streaming.rs

pub type AudioCallback = Box<dyn FnMut(&[f32]) -> Result<()> + Send>;

pub fn generate_streaming(
    pipeline: &Qwen3TTSPipeline,
    text_ids: &[u32],
    config: &GenerationConfig,
    mut callback: AudioCallback,
) -> Result<()> {
    let chunk_size = 50; // frames per chunk
    
    // Generate in chunks
    for chunk in (0..config.max_new_tokens).step_by(chunk_size) {
        // Generate chunk
        let chunk_tokens = &text_ids[chunk..chunk + chunk_size];
        let semantic_tokens = pipeline.talker.generate(chunk_tokens, None, config)?;
        
        // Convert to audio
        let acoustic_codes = pipeline.code_predictor.generate(&semantic_tokens)?;
        let audio = pipeline.decoder.decode(&acoustic_codes)?;
        
        // Callback
        callback(&audio)?;
    }
    
    Ok(())
}
```

---

## 阶段 3: Tokenizer 集成 (200 行)

### 3.1 HuggingFace Tokenizers

#### 任务
- [ ] 集成 tokenizers crate
- [ ] 加载 Qwen2 tokenizer
- [ ] 多语言支持

```rust
// src/models/qwen3_tts/tokenizer.rs

use tokenizers::{Tokenizer, Encoding};

pub struct QwenTokenizer {
    tokenizer: Tokenizer,
}

impl QwenTokenizer {
    pub fn from_pretrained(model_id: &str) -> Result<Self> {
        let tokenizer_path = download_tokenizer(model_id)?;
        let tokenizer = Tokenizer::from_file(tokenizer_path)?;
        Ok(Self { tokenizer })
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let encoding: Encoding = self.tokenizer.encode(text, true)?;
        Ok(encoding.get_ids().to_vec())
    }

    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        let text = self.tokenizer.decode(ids, true)?;
        Ok(text)
    }
}
```

### 3.2 语音 Tokenizer

#### 任务
- [ ] 实现语音编码器
- [ ] 支持 16 码本
- [ ] 音频→码本转换

```rust
// src/models/qwen3_tts/speech_tokenizer.rs

pub struct SpeechTokenizer {
    encoder: Encoder12Hz,
    device: Device,
}

impl SpeechTokenizer {
    pub fn encode(&self, audio: &[f32]) -> Result<Vec<Vec<u32>>> {
        // Convert audio to codes
        let audio_tensor = Tensor::from_vec(audio.to_vec(), (1, audio.len()), &self.device)?;
        let codes = self.encoder.encode(&audio_tensor)?;
        
        // Reshape to [seq, 16 codebooks]
        let codes_vec: Vec<u32> = codes.flatten_all()?.to_vec1()?;
        let seq_len = codes_vec.len() / 16;
        
        let mut result = Vec::with_capacity(seq_len);
        for i in 0..seq_len {
            let frame = codes_vec[i * 16..(i + 1) * 16].to_vec();
            result.push(frame);
        }
        
        Ok(result)
    }
}
```

---

## 阶段 4: 性能优化 (400 行)

### 4.1 FlashAttention 2

#### 任务
- [ ] 集成 candle-flash-attn
- [ ] 条件编译支持
- [ ] 性能基准测试

```toml
# Cargo.toml
[features]
flash-attn = ["cuda", "dep:candle-flash-attn"]

[dependencies]
candle-flash-attn = { version = "0.9", optional = true }
```

```rust
// src/models/qwen3_tts/attention.rs

#[cfg(feature = "flash-attn")]
use candle_flash_attn::flash_attn;

#[cfg(feature = "flash-attn")]
pub fn flash_attention_forward(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    softmax_scale: f32,
) -> Result<Tensor> {
    flash_attn(q, k, v, softmax_scale, true)
}

#[cfg(not(feature = "flash-attn"))]
pub fn flash_attention_forward(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    softmax_scale: f32,
) -> Result<Tensor> {
    // Fallback to standard attention
    standard_attention(q, k, v, softmax_scale)
}
```

### 4.2 KV 缓存优化

#### 任务
- [ ] 预分配 KV 缓存
- [ ] 原地更新
- [ ] 减少内存分配

```rust
// src/models/qwen3_tts/kv_cache_optimized.rs

pub struct OptimizedKVCache {
    k_cache: Tensor,  // Pre-allocated [max_seq, num_kv_heads, head_dim]
    v_cache: Tensor,
    current_pos: usize,
}

impl OptimizedKVCache {
    pub fn new(max_seq: usize, num_kv_heads: usize, head_dim: usize, device: &Device) -> Result<Self> {
        let k_cache = Tensor::zeros((max_seq, num_kv_heads, head_dim), DType::F32, device)?;
        let v_cache = Tensor::zeros((max_seq, num_kv_heads, head_dim), DType::F32, device)?;
        
        Ok(Self {
            k_cache,
            v_cache,
            current_pos: 0,
        })
    }

    pub fn update_inplace(&mut self, k_new: &Tensor, v_new: &Tensor) -> Result<()> {
        // Inplace update using slice_assign
        let seq_len = k_new.dim(0)?;
        
        self.k_cache = self.k_cache.slice_assign(
            &[self.current_pos..self.current_pos + seq_len, 0.., 0..],
            k_new,
        )?;
        self.v_cache = self.v_cache.slice_assign(
            &[self.current_pos..self.current_pos + seq_len, 0.., 0..],
            v_new,
        )?;
        
        self.current_pos += seq_len;
        Ok(())
    }
}
```

### 4.3 批处理支持

#### 任务
- [ ] 实现批处理推理
- [ ] 动态批处理
- [ ] 吞吐量优化

```rust
// src/models/qwen3_tts/batch.rs

pub fn batch_synthesize(
    pipeline: &Qwen3TTSPipeline,
    texts: &[Vec<u32>],
    config: &GenerationConfig,
) -> Result<Vec<Vec<f32>>> {
    let batch_size = texts.len();
    let mut results = Vec::with_capacity(batch_size);
    
    // Pad sequences to same length
    let max_len = texts.iter().map(|t| t.len()).max().unwrap();
    let padded_texts: Vec<Vec<u32>> = texts
        .iter()
        .map(|t| {
            let mut padded = t.clone();
            padded.extend(vec![0; max_len - t.len()]);
            padded
        })
        .collect();
    
    // Batch forward
    let batch_tensor = Tensor::from_vec(
        padded_texts.concat(),
        (batch_size, max_len),
        &pipeline.device,
    )?;
    
    let semantic_tokens = pipeline.talker.batch_generate(&batch_tensor, config)?;
    
    // Process each sequence
    for i in 0..batch_size {
        let seq_tokens = semantic_tokens.i((i, ..))?;
        let acoustic_codes = pipeline.code_predictor.generate(&seq_tokens)?;
        let audio = pipeline.decoder.decode(&acoustic_codes)?;
        results.push(audio);
    }
    
    Ok(results)
}
```

---

## 阶段 5: 流式推理 (300 行)

### 5.1 Dual-Track 流式架构

#### 任务
- [ ] 实现双流架构
- [ ] 文本流处理
- [ ] 音频流输出

```rust
// src/models/qwen3_tts/streaming_dual.rs

pub struct DualTrackStreamer {
    text_buffer: Vec<u32>,
    audio_buffer: Vec<f32>,
    pipeline: Arc<Qwen3TTSPipeline>,
}

impl DualTrackStreamer {
    pub fn new(pipeline: Arc<Qwen3TTSPipeline>) -> Self {
        Self {
            text_buffer: Vec::new(),
            audio_buffer: Vec::new(),
            pipeline,
        }
    }

    pub fn push_text(&mut self, text_ids: &[u32]) -> Result<()> {
        self.text_buffer.extend(text_ids);
        Ok(())
    }

    pub fn generate_audio(&mut self, config: &GenerationConfig) -> Result<Vec<f32>> {
        if self.text_buffer.is_empty() {
            return Ok(Vec::new());
        }
        
        // Generate from buffered text
        let semantic_tokens = self.pipeline.talker.generate(
            &self.text_buffer,
            None,
            config,
        )?;
        
        let acoustic_codes = self.pipeline.code_predictor.generate(&semantic_tokens)?;
        let audio = self.pipeline.decoder.decode(&acoustic_codes)?;
        
        self.text_buffer.clear();
        self.audio_buffer.extend(&audio);
        
        Ok(audio)
    }

    pub fn get_audio(&mut self) -> Vec<f32> {
        let audio = self.audio_buffer.clone();
        self.audio_buffer.clear();
        audio
    }
}
```

### 5.2 低延迟优化

#### 任务
- [ ] 增量生成
- [ ] 首包优化
- [ ] 延迟监控

```rust
// src/models/qwen3_tts/latency.rs

pub struct LatencyMonitor {
    start_time: Instant,
    first_token_time: Option<Instant>,
    first_audio_time: Option<Instant>,
}

impl LatencyMonitor {
    pub fn new() -> Self {
        Self {
            start_time: Instant::now(),
            first_token_time: None,
            first_audio_time: None,
        }
    }

    pub fn record_first_token(&mut self) {
        self.first_token_time = Some(Instant::now());
    }

    pub fn record_first_audio(&mut self) {
        self.first_audio_time = Some(Instant::now());
    }

    pub fn get_time_to_first_token(&self) -> Duration {
        self.first_token_time.unwrap() - self.start_time
    }

    pub fn get_time_to_first_audio(&self) -> Duration {
        self.first_audio_time.unwrap() - self.start_time
    }
}
```

---

## 测试计划

### 单元测试

```bash
# 运行所有测试
cargo test --lib

# 运行权重加载测试
cargo test --lib weights

# 运行推理测试
cargo test --lib inference
```

### 集成测试

```bash
# 运行端到端测试
cargo test --test e2e

# 运行性能测试
cargo bench
```

### 性能基准

```bash
# CPU 基准
cargo bench --no-default-features --features cpu

# CUDA 基准
cargo bench --features cuda
```

---

## 时间估算

| 阶段 | 任务 | 代码行数 | 时间 |
|------|------|---------|------|
| 1 | 权重加载 | 500 | 3-4 天 |
| 2 | 推理循环 | 300 | 2-3 天 |
| 3 | Tokenizer | 200 | 2 天 |
| 4 | 性能优化 | 400 | 4-5 天 |
| 5 | 流式推理 | 300 | 3-4 天 |
| **总计** | | **1,700** | **14-18 天** |

---

## 成功标准

- [ ] 完整推理循环工作
- [ ] 生成清晰音频
- [ ] RTF < 1.0 (GPU)
- [ ] 所有测试通过
- [ ] 文档完整
- [ ] 示例可运行

---

**创建日期**: 2026 年 2 月 21 日  
**版本**: 1.0.0
