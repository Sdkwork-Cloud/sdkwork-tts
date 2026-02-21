# SDKWork-TTS Framework

## Current Status (February 18, 2026)

```
┌────────────────────────────────────────────────────────────────┐
│  STATUS: PRODUCTION READY                                       │
├────────────────────────────────────────────────────────────────┤
│  Code:       ████████████████████ 100% compiles                │
│  Weights:    ████████████████████ 100% loaded                  │
│  Tests:      ████████████████████ 187/187 pass                 │
│  Audio:      ████████████████████ Quality speech output        │
│  Framework:  ████████████████████ Multi-engine architecture    │
└────────────────────────────────────────────────────────────────┘
```

### What's Working ✅
- ✅ Full compilation (`cargo build --release`)
- ✅ CLI runs with full inference pipeline
- ✅ ALL model weights properly loaded from checkpoints
- ✅ Generation loop produces mel codes
- ✅ Pipeline runs end-to-end and generates WAV files
- ✅ Audio output: 22050 Hz WAV files (correct format)
- ✅ All 187 unit tests pass
- ✅ Quality speech synthesis
- ✅ Multi-engine framework (IndexTTS2, Fish-Speech adapters)
- ✅ Streaming synthesis support
- ✅ Emotion control pathways

### Supported Engines

| Engine | Status | Features |
|--------|--------|----------|
| **IndexTTS2** | ✅ Stable | Zero-shot cloning, emotion control, streaming |
| **Fish-Speech** | 🚧 Adapter Ready | Multi-language, streaming, batch processing |
| GPT-SoVITS | 📋 Planned | Zero-shot, style transfer |
| ChatTTS | 📋 Planned | Conversational TTS |

---

## Quick Commands

```bash
# Build
cargo build --release

# Test inference
./target/release/sdkwork-tts.exe infer \
  --speaker checkpoints/speaker_16k.wav \
  --text "Hello world" \
  --output output.wav \
  --de-rumble --de-rumble-cutoff-hz 180

# Run tests
cargo test

# Run with CUDA
$env:CUDA_COMPUTE_CAP='90'
cargo build --release --features cuda
```

---

## Architecture Overview

```
SDKWork-TTS Framework:
┌─────────────────────────────────────────────────────────────────────┐
│                    Unified TTS API                                   │
│  ┌─────────────────────────────────────────────────────┐            │
│  │              TtsEngine Trait                        │            │
│  │  - synthesize()    - synthesize_streaming()         │            │
│  │  - get_speakers()  - get_emotions()                 │            │
│  │  - load_model()    - unload_model()                 │            │
│  └─────────────────────────────────────────────────────┘            │
├─────────────────────────────────────────────────────────────────────┤
│                    Engine Registry                                   │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐              │
│  │ IndexTTS │ │   Fish   │ │GPT-SoVITS│ │  Future  │              │
│  │    2     │ │  Speech  │ │          │ │  Engines │              │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘              │
├─────────────────────────────────────────────────────────────────────┤
│  IndexTTS2 Pipeline:                                                 │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │  1. TEXT PROCESSING (src/text/)                                │ │
│  │     - Normalizer → Tokenizer → Token IDs                       │ │
│  └────────────────────────────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │  2. SPEAKER CONDITIONING (src/models/semantic/, speaker/)      │ │
│  │     - Wav2Vec-BERT 2.0 → semantic embeddings                   │ │
│  │     - CAMPPlus → speaker style vector (192-dim)                │ │
│  └────────────────────────────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │  3. GPT-2 GENERATION (src/models/gpt/)                         │ │
│  │     - Conformer encoder + Perceiver resampler                  │ │
│  │     - UnifiedVoice: 1280 dim, 24 layers, 20 heads              │ │
│  └────────────────────────────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │  4. S2MEL (src/models/s2mel/)                                  │ │
│  │     - DiT: 13 layers, 512 hidden                               │ │
│  │     - Flow Matching: 25 steps, cfg_rate=0.7                    │ │
│  └────────────────────────────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │  5. VOCODER (src/models/vocoder/)                              │ │
│  │     - BigVGAN v2 → 22050 Hz waveform                           │ │
│  └────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Module Structure

```
src/
├── main.rs                  # CLI entry point
├── lib.rs                   # Library exports
├── core/                    # Core framework
│   ├── error.rs            # Structured error handling
│   ├── traits.rs           # Component traits
│   ├── resource.rs         # Resource management
│   ├── metrics.rs          # Performance monitoring
│   └── builder.rs          # Builder patterns
├── engine/                  # Engine abstraction layer
│   ├── traits.rs           # TtsEngine trait
│   ├── registry.rs         # Engine registry
│   ├── pipeline.rs         # Processing pipeline
│   ├── config.rs           # Engine configuration
│   ├── speaker.rs          # Speaker management
│   ├── emotion.rs          # Emotion management
│   ├── indextts2_adapter.rs
│   └── fish_speech_adapter.rs
├── models/
│   ├── semantic/            # Wav2Vec-BERT, codec
│   ├── speaker/             # CAMPPlus
│   ├── gpt/                 # UnifiedVoice, Conformer, Perceiver, KV-cache
│   ├── s2mel/               # DiT, Flow Matching, Length Regulator
│   └── vocoder/             # BigVGAN
├── inference/               # Pipeline, streaming
├── audio/                   # Audio I/O
├── text/                    # Tokenizer, normalizer, segmenter
└── config/                  # YAML config parsing
```

---

## Extending with New Engines

```rust
use sdkwork_tts::engine::{TtsEngine, TtsEngineInfo, SynthesisRequest, SynthesisResult};
use async_trait::async_trait;

pub struct MyTtsEngine {
    info: TtsEngineInfo,
}

#[async_trait]
impl TtsEngine for MyTtsEngine {
    fn info(&self) -> &TtsEngineInfo { &self.info }
    
    async fn initialize(&mut self, config: &EngineConfig) -> Result<()> {
        // Load model
    }
    
    async fn synthesize(&self, request: &SynthesisRequest) -> Result<SynthesisResult> {
        // Implement synthesis
    }
}

// Register
sdkwork_tts::engine::global_registry().register_lazy(
    "my-engine",
    info,
    || Ok(Box::new(MyTtsEngine::new()))
)?;
```

---

## Configuration Reference

```yaml
# checkpoints/config.yaml
gpt:
  model_dim: 1280
  layers: 24
  heads: 20
  max_mel_tokens: 1815
  number_mel_codes: 8194
  stop_mel_token: 8193

s2mel:
  sr: 22050
  DiT:
    hidden_dim: 512
    depth: 13
    heads: 8
  cfm_steps: 25
  cfg_rate: 0.7
```

---

## Key Files

| File | Purpose |
|------|---------|
| `README.md` | Project overview and quick start |
| `docs/ARCHITECTURE.md` | Detailed architecture documentation |
| `src/engine/` | Engine abstraction layer |
| `src/core/` | Core framework infrastructure |

---

## Test Summary

- **Unit Tests**: 187 passed
- **Integration Tests**: 15 passed (3 ignored - need model weights)
- **Coverage**: Core modules fully tested
