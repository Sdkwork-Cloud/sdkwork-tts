//! RVQ (Residual Vector Quantization) Module
//!
//! Implements Residual Vector Quantization for Qwen3-TTS tokenizer.
//! Qwen3-TTS uses 16 codebooks: 1 semantic + 15 acoustic RVQ.
//!
//! ## Architecture
//!
//! ```text
//! Input Audio Features
//!       ↓
//! ┌─────────────────┐
//! │  Semantic Path  │ → Codebook 0 (semantic, WavLM-guided)
//! └─────────────────┘
//!       ↓
//! ┌─────────────────┐
//! │  Acoustic Path  │ → Codebook 1-15 (RVQ, residual details)
//! │    (15 layers)  │
//! └─────────────────┘
//!       ↓
//! Output: 16 codebooks × 2048 codes
//! ```

use candle_core::{Device, Tensor, DType, D, IndexOp};
use candle_nn::{Embedding, Linear, Module, VarBuilder};
use anyhow::Result;

/// RVQ configuration
#[derive(Debug, Clone)]
pub struct RVQConfig {
    /// Number of codebooks (Qwen3-TTS: 16 = 1 semantic + 15 acoustic)
    pub num_codebooks: usize,
    /// Codebook size (Qwen3-TTS: 2048)
    pub codebook_size: usize,
    /// Codebook dimension (latent dimension per codebook)
    pub codebook_dim: usize,
    /// Input feature dimension
    pub input_dim: usize,
}

impl Default for RVQConfig {
    fn default() -> Self {
        Self {
            num_codebooks: 16,  // Qwen3-TTS: 16 codebooks
            codebook_size: 2048, // Qwen3-TTS: 2048 per codebook
            codebook_dim: 128,   // Typical latent dimension
            input_dim: 1024,     // Typical input feature dim
        }
    }
}

/// Single codebook for VQ
#[derive(Debug, Clone)]
pub struct Codebook {
    /// Codebook embeddings [codebook_size, codebook_dim]
    embedding: Embedding,
}

impl Codebook {
    /// Create new codebook
    pub fn new(embedding: Embedding) -> Self {
        Self { embedding }
    }

    /// Load codebook from VarBuilder
    pub fn load(vb: VarBuilder, codebook_size: usize, codebook_dim: usize) -> Result<Self> {
        let embedding = Embedding::new(
            vb.get((codebook_size, codebook_dim), "weight")?,
            codebook_dim,
        );
        Ok(Self { embedding })
    }

    /// Quantize input to nearest codebook vector
    /// 
    /// # Arguments
    /// * `x` - Input features [batch, seq, codebook_dim]
    /// 
    /// # Returns
    /// * (quantized features, code indices)
    pub fn quantize(&self, x: &Tensor) -> Result<(Tensor, Tensor)> {
        let (batch_size, seq_len, _) = x.dims3()?;
        
        // Flatten for efficient computation
        let x_flat = x.reshape((batch_size * seq_len, self.embedding.hidden_size()))?;
        
        // Compute distances to all codebook vectors
        // [batch*seq, dim] @ [dim, codebook_size] → [batch*seq, codebook_size]
        let codebook_t = self.embedding.embeddings().t()?;
        let distances = x_flat.matmul(&codebook_t)?;
        
        // Find nearest codebook vector (argmax of negative distance = min distance)
        let indices = distances.argmax(D::Minus1)?;
        
        // Lookup quantized vectors
        let quantized = self.embedding.forward(&indices)?;
        
        // Reshape back
        let quantized = quantized.reshape((batch_size, seq_len, self.embedding.hidden_size()))?;
        let indices = indices.reshape((batch_size, seq_len))?;
        
        Ok((quantized, indices))
    }

    /// Dequantize code indices to features
    /// 
    /// # Arguments
    /// * `indices` - Code indices [batch, seq]
    /// 
    /// # Returns
    /// * Quantized features [batch, seq, codebook_dim]
    pub fn dequantize(&self, indices: &Tensor) -> Result<Tensor> {
        Ok(self.embedding.forward(indices)?)
    }

    /// Get codebook embeddings
    pub fn embeddings(&self) -> &Tensor {
        self.embedding.embeddings()
    }
}

/// Residual Vector Quantization module
#[derive(Debug, Clone)]
pub struct RVQ {
    /// Codebooks for each RVQ layer
    codebooks: Vec<Codebook>,
    /// Projection from input dim to codebook dim
    project_in: Option<Linear>,
    /// Projection from codebook dim to output dim
    project_out: Option<Linear>,
    /// Configuration
    config: RVQConfig,
    /// Device
    device: Device,
}

impl RVQ {
    /// Create new RVQ module
    pub fn new(config: RVQConfig, device: &Device) -> Result<Self> {
        // Create codebooks
        let mut codebooks = Vec::with_capacity(config.num_codebooks);
        
        // Create dummy embeddings (will be loaded from weights)
        for _i in 0..config.num_codebooks {
            let embedding = Embedding::new(
                Tensor::zeros((config.codebook_size, config.codebook_dim), DType::F32, device)?,
                config.codebook_dim,
            );
            codebooks.push(Codebook::new(embedding));
        }

        Ok(Self {
            codebooks,
            project_in: None,
            project_out: None,
            config,
            device: device.clone(),
        })
    }

    /// Load RVQ from VarBuilder
    pub fn from_var_builder(vb: VarBuilder, config: &RVQConfig) -> Result<Self> {
        let device = vb.device();
        
        // Load input projection
        let project_in = if vb.contains_tensor("project_in.weight") {
            Some(Linear::new(
                vb.get((config.codebook_dim, config.input_dim), "project_in.weight")?,
                vb.get((config.codebook_dim,), "project_in.bias").ok(),
            ))
        } else {
            None
        };

        // Load output projection
        let project_out = if vb.contains_tensor("project_out.weight") {
            Some(Linear::new(
                vb.get((config.input_dim, config.codebook_dim), "project_out.weight")?,
                vb.get((config.input_dim,), "project_out.bias").ok(),
            ))
        } else {
            None
        };

        // Load codebooks
        let mut codebooks = Vec::with_capacity(config.num_codebooks);
        for i in 0..config.num_codebooks {
            let codebook = Codebook::load(
                vb.pp(&format!("codebook_{}", i)),
                config.codebook_size,
                config.codebook_dim,
            )?;
            codebooks.push(codebook);
        }

        Ok(Self {
            codebooks,
            project_in,
            project_out,
            config: config.clone(),
            device: device.clone(),
        })
    }

    /// Quantize input features to RVQ codes
    /// 
    /// # Arguments
    /// * `x` - Input features [batch, seq, input_dim]
    /// 
    /// # Returns
    /// * (all codebook indices [batch, seq, num_codebooks], residual)
    pub fn quantize(&self, x: &Tensor) -> Result<(Tensor, Tensor)> {
        let (_batch_size, _seq_len, _) = x.dims3()?;
        
        // Project to codebook dimension if needed
        let mut residual = if let Some(proj_in) = &self.project_in {
            proj_in.forward(x)?
        } else {
            x.clone()
        };

        let mut all_indices = Vec::with_capacity(self.config.num_codebooks);

        // RVQ: iteratively quantize residual
        for (i, codebook) in self.codebooks.iter().enumerate() {
            // Quantize current residual
            let (quantized, indices) = codebook.quantize(&residual)?;
            
            // Store indices
            all_indices.push(indices.unsqueeze(D::Minus1)?);
            
            // Update residual for next codebook
            if i < self.config.num_codebooks - 1 {
                residual = (residual - quantized)?;
            }
        }

        // Stack all indices: [batch, seq, num_codebooks]
        let all_indices = Tensor::cat(&all_indices, D::Minus1)?;

        Ok((all_indices, residual))
    }

    /// Dequantize RVQ codes to features
    /// 
    /// # Arguments
    /// * `codes` - All codebook indices [batch, seq, num_codebooks]
    /// 
    /// # Returns
    /// * Reconstructed features [batch, seq, input_dim]
    pub fn dequantize(&self, codes: &Tensor) -> Result<Tensor> {
        let (batch_size, seq_len, num_codebooks) = codes.dims3()?;
        
        if num_codebooks != self.config.num_codebooks {
            anyhow::bail!(
                "Expected {} codebooks, got {}",
                self.config.num_codebooks,
                num_codebooks
            );
        }

        // Initialize output with zeros
        let mut output = Tensor::zeros(
            (batch_size, seq_len, self.config.codebook_dim),
            DType::F32,
            &self.device,
        )?;

        // Sum quantized vectors from all codebooks
        for i in 0..self.config.num_codebooks {
            // Extract codebook i indices
            let indices = codes.i((.., .., i))?.contiguous()?;
            
            // Dequantize
            let quantized = self.codebooks[i].dequantize(&indices)?;
            
            // Accumulate
            output = (output + quantized)?;
        }

        // Project to output dimension if needed
        if let Some(proj_out) = &self.project_out {
            output = proj_out.forward(&output)?;
        }

        Ok(output)
    }

    /// Get number of codebooks
    pub fn num_codebooks(&self) -> usize {
        self.config.num_codebooks
    }

    /// Get codebook size
    pub fn codebook_size(&self) -> usize {
        self.config.codebook_size
    }

    /// Get codebook dimension
    pub fn codebook_dim(&self) -> usize {
        self.config.codebook_dim
    }
}

/// Semantic codebook (first codebook with WavLM guidance)
#[derive(Debug, Clone)]
pub struct SemanticCodebook {
    /// Base codebook
    codebook: Codebook,
    /// WavLM projection (aligns features with WavLM space)
    wavlm_projection: Option<Linear>,
}

impl SemanticCodebook {
    /// Create new semantic codebook
    pub fn new(codebook: Codebook, wavlm_projection: Option<Linear>) -> Self {
        Self {
            codebook,
            wavlm_projection,
        }
    }

    /// Load semantic codebook from VarBuilder
    pub fn load(vb: VarBuilder, codebook_size: usize, codebook_dim: usize) -> Result<Self> {
        let codebook = Codebook::load(vb.pp("codebook"), codebook_size, codebook_dim)?;
        
        let wavlm_projection = if vb.contains_tensor("wavlm_projection.weight") {
            Some(Linear::new(
                vb.get((codebook_dim, codebook_dim), "wavlm_projection.weight")?,
                vb.get((codebook_dim,), "wavlm_projection.bias").ok(),
            ))
        } else {
            None
        };

        Ok(Self {
            codebook,
            wavlm_projection,
        })
    }

    /// Quantize with WavLM guidance
    pub fn quantize(&self, x: &Tensor, wavlm_features: Option<&Tensor>) -> Result<(Tensor, Tensor)> {
        // Apply WavLM projection if provided
        let x = if let Some(proj) = &self.wavlm_projection {
            if let Some(wavlm) = wavlm_features {
                // Blend input with WavLM features
                let wavlm_proj = proj.forward(wavlm)?;
                ((x + wavlm_proj)? / 2.0)?  // Simple averaging
            } else {
                proj.forward(x)?
            }
        } else {
            x.clone()
        };

        // Quantize
        self.codebook.quantize(&x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rvq_config_default() {
        let config = RVQConfig::default();
        assert_eq!(config.num_codebooks, 16);
        assert_eq!(config.codebook_size, 2048);
        assert_eq!(config.codebook_dim, 128);
    }

    #[test]
    fn test_codebook_quantize() {
        let device = Device::Cpu;
        let config = RVQConfig::default();
        
        // Create codebook with random weights
        let embeddings = Tensor::randn(0.0, 1.0, (config.codebook_size, config.codebook_dim), &device).unwrap();
        let codebook = Codebook::new(Embedding::new(embeddings, config.codebook_dim));
        
        // Create test input
        let x = Tensor::randn(0.0, 1.0, (2, 10, config.codebook_dim), &device).unwrap();
        
        // Quantize
        let (quantized, indices) = codebook.quantize(&x).unwrap();
        
        // Check shapes
        assert_eq!(quantized.dims(), &[2, 10, config.codebook_dim]);
        assert_eq!(indices.dims(), &[2, 10]);
    }

    #[test]
    fn test_rvq_quantize_dequantize() {
        let device = Device::Cpu;
        let config = RVQConfig {
            num_codebooks: 4,  // Use smaller for testing
            codebook_size: 256,
            codebook_dim: 64,
            input_dim: 64,
        };
        
        let rvq = RVQ::new(config.clone(), &device).unwrap();
        
        // Create test input with F32 dtype
        let x = Tensor::randn(0.0f32, 1.0, (2, 10, config.input_dim), &device).unwrap();
        
        // Quantize
        let (codes, _residual) = rvq.quantize(&x).unwrap();
        
        // Check codes shape
        assert_eq!(codes.dims(), &[2, 10, config.num_codebooks]);
        
        // Dequantize
        let reconstructed = rvq.dequantize(&codes).unwrap();
        
        // Check reconstructed shape
        assert_eq!(reconstructed.dims(), &[2, 10, config.input_dim]);
    }
}
