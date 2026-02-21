//! STFT (Short-Time Fourier Transform) Module
//!
//! This module provides efficient STFT computation for audio processing
//! in Qwen3-TTS tokenizer.

use candle_core::{Device, Tensor, IndexOp};
use anyhow::Result;

/// STFT configuration
#[derive(Debug, Clone)]
pub struct STFTConfig {
    /// FFT size
    pub n_fft: usize,
    /// Hop length between frames
    pub hop_length: usize,
    /// Window length
    pub win_length: usize,
    /// Number of frequency bins in output (n_fft / 2 + 1)
    pub n_freqs: usize,
}

impl Default for STFTConfig {
    fn default() -> Self {
        Self {
            n_fft: 1024,
            hop_length: 256,
            win_length: 1024,
            n_freqs: 513,
        }
    }
}

impl STFTConfig {
    pub fn new(n_fft: usize, hop_length: usize, win_length: usize) -> Self {
        Self {
            n_fft,
            hop_length,
            win_length,
            n_freqs: n_fft / 2 + 1,
        }
    }

    pub fn validate(&self) -> Result<()> {
        if self.n_fft == 0 || self.hop_length == 0 || self.win_length == 0 {
            anyhow::bail!("Invalid STFT parameters");
        }
        if self.win_length > self.n_fft {
            anyhow::bail!("win_length must be <= n_fft");
        }
        Ok(())
    }
}

/// Window function
#[derive(Debug, Clone)]
pub struct WindowFunction {
    window: Tensor,
}

impl WindowFunction {
    pub fn hann(size: usize, device: &Device) -> Result<Self> {
        let window: Vec<f32> = (0..size)
            .map(|i| 0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / size as f32).cos()))
            .collect();
        Ok(Self {
            window: Tensor::from_vec(window, (size,), device)?,
        })
    }

    pub fn apply(&self, signal: &Tensor) -> Result<Tensor> {
        Ok((signal * &self.window)?)
    }

    pub fn window(&self) -> &Tensor {
        &self.window
    }
}

/// STFT processor
#[derive(Debug, Clone)]
pub struct STFTProcessor {
    config: STFTConfig,
    window: WindowFunction,
    device: Device,
}

impl STFTProcessor {
    pub fn new(config: STFTConfig, device: &Device) -> Result<Self> {
        config.validate()?;
        let window = WindowFunction::hann(config.win_length, device)?;
        
        Ok(Self {
            config,
            window,
            device: device.clone(),
        })
    }

    pub fn transform(&self, audio: &Tensor) -> Result<Tensor> {
        let audio = if audio.dims().len() == 1 {
            audio.unsqueeze(0)?
        } else {
            audio.clone()
        };

        let (batch_size, num_samples) = audio.dims2()?;
        let num_frames = (num_samples - self.config.win_length) / self.config.hop_length + 1;

        let mut stft_frames = Vec::with_capacity(batch_size);

        for b in 0..batch_size {
            let batch_audio = audio.i((b, ..))?;
            let frame_stft = self.transform_batch(&batch_audio, num_frames)?;
            stft_frames.push(frame_stft);
        }

        Ok(Tensor::stack(&stft_frames, 0)?)
    }

    fn transform_batch(&self, audio: &Tensor, num_frames: usize) -> Result<Tensor> {
        let mut frames = Vec::with_capacity(num_frames);

        for i in 0..num_frames {
            let start = i * self.config.hop_length;
            let frame = audio.narrow(0, start, self.config.win_length)?;
            let windowed = self.window.apply(&frame)?;
            let fft = self.compute_fft(&windowed)?;
            frames.push(fft);
        }

        Ok(Tensor::stack(&frames, 0)?)
    }

    fn compute_fft(&self, x: &Tensor) -> Result<Tensor> {
        let n = self.config.n_fft;
        let x_vec = x.to_vec1::<f32>()?;

        let mut padded = x_vec.clone();
        while padded.len() < n {
            padded.push(0.0);
        }

        let mut real_parts = Vec::with_capacity(self.config.n_freqs);
        let mut imag_parts = Vec::with_capacity(self.config.n_freqs);

        for k in 0..self.config.n_freqs {
            let mut real_sum = 0.0f32;
            let mut imag_sum = 0.0f32;

            for (n_idx, &x_n) in padded.iter().enumerate() {
                let angle = -2.0 * std::f32::consts::PI * k as f32 * n_idx as f32 / n as f32;
                real_sum += x_n * angle.cos();
                imag_sum += x_n * angle.sin();
            }

            real_parts.push(real_sum);
            imag_parts.push(imag_sum);
        }

        let real_tensor = Tensor::from_vec(real_parts, (self.config.n_freqs,), &self.device)?;
        let imag_tensor = Tensor::from_vec(imag_parts, (self.config.n_freqs,), &self.device)?;
        Ok(Tensor::stack(&[&real_tensor, &imag_tensor], 1)?)
    }

    pub fn magnitude(&self, stft: &Tensor) -> Result<Tensor> {
        let real = stft.narrow(3, 0, 1)?.squeeze(3)?;
        let imag = stft.narrow(3, 1, 1)?.squeeze(3)?;
        Ok((real.sqr()? + imag.sqr()?)?.sqrt()?)
    }

    pub fn power(&self, stft: &Tensor) -> Result<Tensor> {
        let real = stft.narrow(3, 0, 1)?.squeeze(3)?;
        let imag = stft.narrow(3, 1, 1)?.squeeze(3)?;
        Ok((real.sqr()? + imag.sqr()?)?)
    }

    pub fn log_magnitude(&self, stft: &Tensor, eps: f32) -> Result<Tensor> {
        let mag = self.magnitude(stft)?;
        let eps_tensor = Tensor::full(eps, mag.shape(), mag.device())?;
        Ok((mag + eps_tensor)?.log()?)
    }

    pub fn config(&self) -> &STFTConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::DType;

    #[test]
    fn test_stft_config_default() {
        let config = STFTConfig::default();
        assert_eq!(config.n_fft, 1024);
        assert_eq!(config.hop_length, 256);
        assert_eq!(config.win_length, 1024);
        assert_eq!(config.n_freqs, 513);
    }

    #[test]
    fn test_stft_config_validation() {
        let config = STFTConfig::default();
        assert!(config.validate().is_ok());

        let invalid_config = STFTConfig {
            n_fft: 0,
            ..Default::default()
        };
        assert!(invalid_config.validate().is_err());
    }

    #[test]
    fn test_window_function() {
        let device = Device::Cpu;
        let window = WindowFunction::hann(1024, &device).unwrap();
        assert_eq!(window.window().dims(), &[1024]);
    }

    #[test]
    fn test_stft_processor() {
        let device = Device::Cpu;
        let config = STFTConfig::default();
        let processor = STFTProcessor::new(config, &device).unwrap();

        let audio = Tensor::zeros((16000,), DType::F32, &device).unwrap();
        let stft = processor.transform(&audio.unsqueeze(0).unwrap()).unwrap();

        assert_eq!(stft.dims().len(), 4);
        assert_eq!(stft.dim(3).unwrap(), 2);
    }

    #[test]
    fn test_magnitude_computation() {
        let device = Device::Cpu;
        let config = STFTConfig::default();
        let processor = STFTProcessor::new(config, &device).unwrap();

        let audio = Tensor::zeros((1, 16000), DType::F32, &device).unwrap();
        let stft = processor.transform(&audio).unwrap();
        let mag = processor.magnitude(&stft).unwrap();

        assert_eq!(mag.dims().len(), 3);
    }
}
