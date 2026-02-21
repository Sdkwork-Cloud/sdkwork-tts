//! Generation - Simplified

use candle_core::{Device, Tensor, D};
use candle_core::Result;

/// Generation configuration
#[derive(Debug, Clone)]
pub struct GenerationConfig {
    pub max_new_tokens: usize,
    pub temperature: f64,
    pub top_k: Option<usize>,
    pub top_p: Option<f64>,
    pub repetition_penalty: f64,
    pub eos_token_id: Option<u32>,
    pub seed: u64,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_new_tokens: 2048,
            temperature: 0.8,
            top_k: Some(50),
            top_p: Some(0.95),
            repetition_penalty: 1.05,
            eos_token_id: None,
            seed: 42,
        }
    }
}

/// Sampling context
pub struct SamplingContext {
    rng: rand::rngs::StdRng,
}

impl SamplingContext {
    pub fn new(seed: u64) -> Self {
        Self {
            rng: rand::SeedableRng::seed_from_u64(seed),
        }
    }

    pub fn sample(&mut self, logits: &[f32]) -> Result<usize> {
        // Simple softmax sampling
        let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_logits: Vec<f32> = logits.iter().map(|&x| (x - max_logit).exp()).collect();
        let sum_exp: f32 = exp_logits.iter().sum();
        let probs: Vec<f32> = exp_logits.iter().map(|&x| x / sum_exp).collect();

        let r = rand::Rng::gen::<f32>(&mut self.rng);
        let mut cumsum = 0.0f32;
        for (i, &p) in probs.iter().enumerate() {
            cumsum += p;
            if r < cumsum {
                return Ok(i);
            }
        }
        Ok(probs.len() - 1)
    }
}

/// Generation output
#[derive(Debug, Clone)]
pub struct GenerationOutput {
    pub tokens: Vec<u32>,
    pub log_probs: Vec<f32>,
    pub num_tokens: usize,
    pub stopped_by_eos: bool,
}

/// Generator
pub struct Generator {
    config: GenerationConfig,
    #[allow(dead_code)]
    device: Device,
}

impl Generator {
    pub fn new(config: GenerationConfig, device: &Device) -> Self {
        Self {
            config,
            device: device.clone(),
        }
    }

    pub fn config(&self) -> &GenerationConfig {
        &self.config
    }
}

/// Argmax sampling
pub fn argmax(logits: &Tensor) -> Result<u32> {
    let token = logits.argmax(D::Minus1)?;
    token.to_scalar::<u32>()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generation_config_default() {
        let config = GenerationConfig::default();
        assert_eq!(config.max_new_tokens, 2048);
        assert_eq!(config.temperature, 0.8);
    }

    #[test]
    fn test_sampling_context() {
        let mut ctx = SamplingContext::new(42);
        let logits = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let token = ctx.sample(&logits).unwrap();
        assert!(token < 5);
    }
}
