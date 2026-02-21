//! Qwen3-TTS Core Components - Simplified

use candle_core::{Device, Tensor, DType};
use candle_nn::{Linear, Module, VarBuilder};
use candle_core::Result;

use super::config::TalkerConfig;

/// RMSNorm
#[derive(Debug, Clone)]
pub struct RMSNorm {
    weight: Tensor,
    eps: f32,
}

impl RMSNorm {
    pub fn new(weight: Tensor, eps: f32) -> Self {
        Self { weight, eps }
    }

    pub fn load(vb: VarBuilder, hidden_size: usize) -> Result<Self> {
        let weight = vb.get(hidden_size, "weight")?;
        Ok(Self::new(weight, 1e-6))
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x_dtype = x.dtype();
        let x = x.to_dtype(DType::F32)?;
        let variance = x.sqr()?.mean_keepdim(candle_core::D::Minus1)?;
        let eps_tensor = Tensor::full(self.eps, variance.shape(), variance.device())?;
        let std = (variance + eps_tensor)?.sqrt()?;
        let x = x.broadcast_div(&std)?;
        let x = x.broadcast_mul(&self.weight.to_dtype(DType::F32)?)?;
        x.to_dtype(x_dtype)
    }
}

impl Module for RMSNorm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.forward(x)
    }
}

/// Rotary Position Embedding
#[derive(Debug, Clone)]
pub struct RotaryEmbedding {
    cos: Tensor,
    sin: Tensor,
    head_dim: usize,
}

impl RotaryEmbedding {
    pub fn new(head_dim: usize, max_pos: usize, theta: f64, device: &Device) -> Result<Self> {
        let inv_freq: Vec<f32> = (0..head_dim)
            .step_by(2)
            .map(|i| (1.0 / (theta.powf(i as f64 / head_dim as f64))) as f32)
            .collect();
        
        let inv_freq = Tensor::from_vec(inv_freq, (1, head_dim / 2), device)?;
        let positions: Vec<f32> = (0..max_pos).map(|i| i as f32).collect();
        let positions = Tensor::from_vec(positions, (max_pos, 1), device)?;
        let freqs = positions.matmul(&inv_freq)?;
        let freqs = Tensor::cat(&[&freqs, &freqs], 1)?;
        let cos = freqs.cos()?;
        let sin = freqs.sin()?;
        
        Ok(Self { cos, sin, head_dim })
    }

    pub fn apply(&self, q: &Tensor, k: &Tensor, _start_pos: usize) -> Result<(Tensor, Tensor)> {
        Ok((q.clone(), k.clone()))
    }
}

/// Causal Self-Attention
#[derive(Debug, Clone)]
pub struct CausalSelfAttention {
    wqkv: Linear,
    wo: Linear,
}

impl CausalSelfAttention {
    pub fn load(vb: VarBuilder, config: &TalkerConfig) -> Result<Self> {
        let head_dim = config.head_dim();
        let qkv_dim = config.num_attention_heads * head_dim;
        let kv_dim = config.num_kv_heads() * head_dim;
        let total_qkv_dim = qkv_dim + 2 * kv_dim;

        let wqkv = Linear::new(
            vb.get((total_qkv_dim, config.hidden_size), "wqkv.weight")?,
            None,
        );
        let wo = Linear::new(
            vb.get((config.hidden_size, config.hidden_size), "wo.weight")?,
            None,
        );

        Ok(Self { wqkv, wo })
    }

    pub fn forward(
        &self,
        x: &Tensor,
        _kv_cache: Option<(&mut Tensor, &mut Tensor)>,
        _start_pos: usize,
        _mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let qkv = self.wqkv.forward(x)?;
        self.wo.forward(&qkv)
    }
}

/// SwiGLU MLP
#[derive(Debug, Clone)]
pub struct SwiGLU {
    w1: Linear,
    w2: Linear,
    w3: Linear,
}

impl SwiGLU {
    pub fn load(vb: VarBuilder, config: &TalkerConfig) -> Result<Self> {
        let w1 = Linear::new(
            vb.get((config.intermediate_size, config.hidden_size), "w1.weight")?,
            None,
        );
        let w3 = Linear::new(
            vb.get((config.intermediate_size, config.hidden_size), "w3.weight")?,
            None,
        );
        let w2 = Linear::new(
            vb.get((config.hidden_size, config.intermediate_size), "w2.weight")?,
            None,
        );

        Ok(Self { w1, w2, w3 })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = self.w1.forward(x)?;
        let value = self.w3.forward(x)?;
        let sigmoid = candle_nn::ops::sigmoid(&gate)?;
        let swish = gate * sigmoid;
        let hidden = (swish * value)?;
        self.w2.forward(&hidden)
    }
}
