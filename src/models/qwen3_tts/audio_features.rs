//! Audio Feature Extraction for Qwen3-TTS Tokenizer
//!
//! This module provides audio feature extraction capabilities:
//! - Mel spectrogram computation
//! - Audio resampling
//! - Frame processing at 12.5 Hz

use candle_core::{Device, Tensor};
use anyhow::Result;

/// Mel spectrogram configuration
#[derive(Debug, Clone)]
pub struct MelSpectrogramConfig {
    /// Sample rate of input audio
    pub sample_rate: u32,
    /// FFT size
    pub n_fft: usize,
    /// Hop length between frames
    pub hop_length: usize,
    /// Window length
    pub win_length: usize,
    /// Number of mel bands
    pub n_mels: usize,
    /// Minimum mel frequency
    pub fmin: f32,
    /// Maximum mel frequency
    pub fmax: Option<f32>,
}

impl Default for MelSpectrogramConfig {
    fn default() -> Self {
        Self {
            sample_rate: 24000,
            n_fft: 1024,
            hop_length: 256,
            win_length: 1024,
            n_mels: 80,
            fmin: 0.0,
            fmax: Some(12000.0),
        }
    }
}

/// Mel spectrogram extractor
#[derive(Debug, Clone)]
pub struct MelSpectrogramExtractor {
    config: MelSpectrogramConfig,
    mel_basis: Tensor,
    hann_window: Tensor,
    device: Device,
}

impl MelSpectrogramExtractor {
    /// Create new mel spectrogram extractor
    pub fn new(config: MelSpectrogramConfig, device: &Device) -> Result<Self> {
        let mel_basis = Self::compute_mel_basis(&config, device)?;
        let hann_window = Self::compute_hann_window(config.win_length, device)?;
        
        Ok(Self {
            config,
            mel_basis,
            hann_window,
            device: device.clone(),
        })
    }

    /// Compute mel basis matrix
    fn compute_mel_basis(config: &MelSpectrogramConfig, device: &Device) -> Result<Tensor> {
        let n_fft = config.n_fft;
        let n_mels = config.n_mels;
        let sample_rate = config.sample_rate as f32;
        let fmin = config.fmin;
        let fmax = config.fmax.unwrap_or(sample_rate / 2.0);

        let freqs: Vec<f32> = (0..n_fft / 2 + 1)
            .map(|i| i as f32 * sample_rate / n_fft as f32)
            .collect();
        
        let _mel_freqs: Vec<f32> = freqs.iter().map(|&f| Self::hz_to_mel(f)).collect();
        
        let mel_min = Self::hz_to_mel(fmin);
        let mel_max = Self::hz_to_mel(fmax);
        let mel_points: Vec<f32> = (0..n_mels + 2)
            .map(|i| mel_min + (mel_max - mel_min) * i as f32 / (n_mels + 1) as f32)
            .collect();
        
        let hz_points: Vec<f32> = mel_points.iter().map(|&m| Self::mel_to_hz(m)).collect();
        
        let mut mel_basis = vec![0.0f32; n_mels * (n_fft / 2 + 1)];
        
        for i in 0..n_mels {
            let left = hz_points[i];
            let center = hz_points[i + 1];
            let right = hz_points[i + 2];
            
            for (j, &freq) in freqs.iter().enumerate() {
                if freq >= left && freq <= center {
                    mel_basis[i * (n_fft / 2 + 1) + j] = (freq - left) / (center - left);
                } else if freq > center && freq <= right {
                    mel_basis[i * (n_fft / 2 + 1) + j] = (right - freq) / (right - center);
                }
            }
        }
        
        Ok(Tensor::from_vec(mel_basis, (n_mels, n_fft / 2 + 1), device)?)
    }

    /// Compute Hann window
    fn compute_hann_window(win_length: usize, device: &Device) -> Result<Tensor> {
        let window: Vec<f32> = (0..win_length)
            .map(|i| 0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / win_length as f32).cos()))
            .collect();
        
        Ok(Tensor::from_vec(window, (win_length,), device)?)
    }

    /// Convert Hz to mel scale
    pub fn hz_to_mel(hz: f32) -> f32 {
        2595.0 * (1.0 + hz / 700.0).log10()
    }

    /// Convert mel scale to Hz
    pub fn mel_to_hz(mel: f32) -> f32 {
        700.0 * (10.0_f32.powf(mel / 2595.0) - 1.0)
    }

    /// Get config
    pub fn config(&self) -> &MelSpectrogramConfig {
        &self.config
    }

    /// Get mel basis
    pub fn mel_basis(&self) -> &Tensor {
        &self.mel_basis
    }
}

/// Audio resampler
#[derive(Debug, Clone)]
pub struct AudioResampler {
    from_sample_rate: u32,
    to_sample_rate: u32,
}

impl AudioResampler {
    /// Create new resampler
    pub fn new(from_sample_rate: u32, to_sample_rate: u32) -> Self {
        Self {
            from_sample_rate,
            to_sample_rate,
        }
    }

    /// Resample audio
    pub fn resample(&self, audio: &Tensor) -> Result<Tensor> {
        if self.from_sample_rate == self.to_sample_rate {
            return Ok(audio.clone());
        }
        
        let ratio = self.to_sample_rate as f64 / self.from_sample_rate as f64;
        let new_length = (audio.dim(0)? as f64 * ratio) as usize;
        
        let mut resampled = Vec::with_capacity(new_length);
        let audio_vec = audio.to_vec1::<f32>()?;
        
        for i in 0..new_length {
            let src_idx = i as f64 / ratio;
            let src_idx_floor = src_idx.floor() as usize;
            let src_idx_ceil = src_idx.ceil() as usize;
            let frac = src_idx - src_idx_floor as f64;
            
            let sample = if src_idx_ceil >= audio_vec.len() {
                audio_vec[audio_vec.len() - 1]
            } else if src_idx_floor >= audio_vec.len() {
                0.0
            } else {
                (audio_vec[src_idx_floor] as f64 * (1.0 - frac) + audio_vec[src_idx_ceil] as f64 * frac) as f32
            };
            
            resampled.push(sample);
        }
        
        Ok(Tensor::from_vec(resampled, (new_length,), audio.device())?)
    }
}

/// Frame processor for 12.5 Hz frame rate
#[derive(Debug, Clone)]
pub struct FrameProcessor {
    sample_rate: u32,
    frame_rate: f32,
    hop_length: usize,
    win_length: usize,
}

impl FrameProcessor {
    /// Create new frame processor
    pub fn new(sample_rate: u32, frame_rate: f32) -> Self {
        let hop_length = (sample_rate as f32 / frame_rate) as usize;
        let win_length = hop_length.next_power_of_two();
        
        Self {
            sample_rate,
            frame_rate,
            hop_length,
            win_length,
        }
    }

    /// Get hop length
    pub fn hop_length(&self) -> usize {
        self.hop_length
    }

    /// Get window length
    pub fn win_length(&self) -> usize {
        self.win_length
    }

    /// Get frame rate
    pub fn frame_rate(&self) -> f32 {
        self.frame_rate
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::DType;

    #[test]
    fn test_mel_config_default() {
        let config = MelSpectrogramConfig::default();
        assert_eq!(config.sample_rate, 24000);
        assert_eq!(config.n_fft, 1024);
        assert_eq!(config.hop_length, 256);
        assert_eq!(config.n_mels, 80);
    }

    #[test]
    fn test_hz_to_mel() {
        let mel = MelSpectrogramExtractor::hz_to_mel(1000.0);
        assert!((mel - 1000.0).abs() < 100.0);
    }

    #[test]
    fn test_mel_to_hz() {
        let mel = MelSpectrogramExtractor::hz_to_mel(1000.0);
        let hz = MelSpectrogramExtractor::mel_to_hz(mel);
        assert!((hz - 1000.0).abs() < 100.0);
    }

    #[test]
    fn test_resampler_same_rate() {
        let device = Device::Cpu;
        let resampler = AudioResampler::new(24000, 24000);
        let audio = Tensor::zeros((1000,), DType::F32, &device).unwrap();
        let resampled = resampler.resample(&audio).unwrap();
        assert_eq!(resampled.dims(), &[1000]);
    }

    #[test]
    fn test_frame_processor() {
        let processor = FrameProcessor::new(24000, 12.5);
        assert_eq!(processor.hop_length(), 1920);
        assert_eq!(processor.win_length(), 2048);
        assert_eq!(processor.frame_rate(), 12.5);
    }
}
