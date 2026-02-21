//! Streaming Synthesis Module
//!
//! Provides real-time streaming TTS capabilities:
//! - Dual-Track architecture for streaming/non-streaming
//! - Ultra-low latency synthesis (target <100ms)
//! - Chunk-based audio streaming
//! - Automatic language detection
//!
//! Architecture inspired by Qwen3-TTS Dual-Track design

use std::sync::Arc;
use std::time::{Duration, Instant};

/// Streaming configuration
#[derive(Debug, Clone)]
pub struct StreamingConfig {
    /// Enable streaming mode
    pub enabled: bool,
    /// Target latency in milliseconds
    pub target_latency_ms: u64,
    /// Chunk size in samples
    pub chunk_size: usize,
    /// Sample rate
    pub sample_rate: u32,
    /// Enable automatic language detection
    pub auto_language: bool,
    /// Prefetch chunks ahead
    pub prefetch_chunks: usize,
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            target_latency_ms: 100,  // Target 100ms latency (Qwen3-TTS achieves 97ms)
            chunk_size: 4096,
            sample_rate: 22050,
            auto_language: true,
            prefetch_chunks: 2,
        }
    }
}

impl StreamingConfig {
    /// Create streaming config for ultra-low latency
    pub fn low_latency() -> Self {
        Self {
            enabled: true,
            target_latency_ms: 50,  // Aggressive 50ms target
            chunk_size: 2048,
            sample_rate: 22050,
            auto_language: true,
            prefetch_chunks: 1,
        }
    }

    /// Create streaming config for high quality
    pub fn high_quality() -> Self {
        Self {
            enabled: true,
            target_latency_ms: 200,  // More relaxed for quality
            chunk_size: 8192,
            sample_rate: 24000,
            auto_language: true,
            prefetch_chunks: 3,
        }
    }

    /// Get chunk duration in milliseconds
    pub fn chunk_duration_ms(&self) -> f64 {
        (self.chunk_size as f64 / self.sample_rate as f64) * 1000.0
    }
}

/// Audio chunk for streaming
#[derive(Debug, Clone)]
pub struct StreamChunk {
    /// Audio samples
    pub samples: Vec<f32>,
    /// Sample rate
    pub sample_rate: u32,
    /// Chunk sequence number
    pub sequence: usize,
    /// Is this the final chunk
    pub is_final: bool,
    /// Timestamp when chunk was generated
    pub generated_at: Instant,
    /// Text segment that generated this chunk
    pub text_segment: Option<String>,
}

impl StreamChunk {
    /// Create new stream chunk
    pub fn new(
        samples: Vec<f32>,
        sample_rate: u32,
        sequence: usize,
        is_final: bool,
    ) -> Self {
        Self {
            samples,
            sample_rate,
            sequence,
            is_final,
            generated_at: Instant::now(),
            text_segment: None,
        }
    }

    /// Get chunk duration in milliseconds
    pub fn duration_ms(&self) -> f64 {
        (self.samples.len() as f64 / self.sample_rate as f64) * 1000.0
    }

    /// Get latency from generation to now
    pub fn latency(&self) -> Duration {
        self.generated_at.elapsed()
    }
}

/// Streaming statistics
#[derive(Debug, Clone, Default)]
pub struct StreamingStats {
    /// Total chunks generated
    pub total_chunks: usize,
    /// Total audio duration in seconds
    pub total_audio_secs: f64,
    /// Average chunk latency in milliseconds
    pub avg_latency_ms: f64,
    /// Minimum latency in milliseconds
    pub min_latency_ms: f64,
    /// Maximum latency in milliseconds
    pub max_latency_ms: f64,
    /// First token latency (time to first chunk)
    pub first_token_latency_ms: f64,
    /// Real-time factor
    pub rtf: f64,
}

impl StreamingStats {
    /// Update statistics with new chunk
    pub fn update(&mut self, chunk: &StreamChunk, generation_start: Instant) {
        self.total_chunks += 1;
        self.total_audio_secs += chunk.duration_ms() / 1000.0;
        
        let latency = chunk.latency().as_secs_f64() * 1000.0;
        
        if self.total_chunks == 1 {
            self.first_token_latency_ms = latency;
            self.min_latency_ms = latency;
            self.max_latency_ms = latency;
            self.avg_latency_ms = latency;
        } else {
            self.min_latency_ms = self.min_latency_ms.min(latency);
            self.max_latency_ms = self.max_latency_ms.max(latency);
            // Running average
            self.avg_latency_ms = self.avg_latency_ms 
                + (latency - self.avg_latency_ms) / self.total_chunks as f64;
        }
        
        // Calculate RTF
        let total_generation_time = generation_start.elapsed().as_secs_f64();
        if total_generation_time > 0.0 {
            self.rtf = self.total_audio_secs / total_generation_time;
        }
    }
}

/// Streaming synthesizer trait
pub trait StreamSynthesizer: Send + Sync {
    /// Start streaming synthesis
    fn start_streaming(&self, text: &str) -> crate::core::error::Result<StreamHandle>;
    
    /// Get streaming configuration
    fn streaming_config(&self) -> &StreamingConfig;
    
    /// Get current statistics
    fn stats(&self) -> StreamingStats;
}

/// Handle for streaming synthesis
pub struct StreamHandle {
    /// Channel for receiving chunks
    receiver: std::sync::mpsc::Receiver<StreamChunk>,
    /// Statistics
    stats: Arc<std::sync::Mutex<StreamingStats>>,
    /// Generation start time
    generation_start: Instant,
    /// Is stream finished
    finished: bool,
}

impl StreamHandle {
    /// Create new stream handle
    pub fn new(
        receiver: std::sync::mpsc::Receiver<StreamChunk>,
        stats: Arc<std::sync::Mutex<StreamingStats>>,
    ) -> Self {
        Self {
            receiver,
            stats,
            generation_start: Instant::now(),
            finished: false,
        }
    }

    /// Get next chunk (blocking)
    pub fn next_chunk(&mut self) -> Option<StreamChunk> {
        if self.finished {
            return None;
        }

        match self.receiver.recv() {
            Ok(chunk) => {
                if chunk.is_final {
                    self.finished = true;
                }
                
                // Update statistics
                if let Ok(mut stats) = self.stats.lock() {
                    stats.update(&chunk, self.generation_start);
                }
                
                Some(chunk)
            }
            Err(_) => {
                self.finished = true;
                None
            }
        }
    }

    /// Try to get next chunk (non-blocking)
    pub fn try_next_chunk(&mut self) -> Option<StreamChunk> {
        if self.finished {
            return None;
        }

        match self.receiver.try_recv() {
            Ok(chunk) => {
                if chunk.is_final {
                    self.finished = true;
                }
                
                if let Ok(mut stats) = self.stats.lock() {
                    stats.update(&chunk, self.generation_start);
                }
                
                Some(chunk)
            }
            Err(std::sync::mpsc::TryRecvError::Empty) => None,
            Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                self.finished = true;
                None
            }
        }
    }

    /// Get current statistics
    pub fn stats(&self) -> StreamingStats {
        self.stats.lock().map(|s| s.clone()).unwrap_or_default()
    }

    /// Check if stream is finished
    pub fn is_finished(&self) -> bool {
        self.finished
    }

    /// Iterate over all chunks
    #[allow(clippy::should_implement_trait)]
    pub fn into_iter(self) -> StreamIterator {
        StreamIterator { handle: self }
    }
}

/// Iterator for stream chunks
pub struct StreamIterator {
    handle: StreamHandle,
}

impl Iterator for StreamIterator {
    type Item = StreamChunk;

    fn next(&mut self) -> Option<Self::Item> {
        self.handle.next_chunk()
    }
}

/// Language detection result
#[derive(Debug, Clone)]
pub struct LanguageDetection {
    /// Detected language code
    pub language: String,
    /// Confidence score (0.0 - 1.0)
    pub confidence: f32,
    /// Script detected
    pub script: Option<String>,
}

impl LanguageDetection {
    /// Detect language from text (simple heuristic)
    pub fn detect(text: &str) -> Self {
        let chars: Vec<char> = text.chars().collect();
        let total = chars.len() as f32;

        // Count character types
        let mut hanzi = 0f32;
        let mut hiragana = 0f32;
        let mut katakana = 0f32;
        let mut hangul = 0f32;
        let mut cyrillic = 0f32;
        let mut latin = 0f32;

        for &c in &chars {
            match c {
                '\u{4e00}'..='\u{9fff}' => hanzi += 1.0,
                '\u{3040}'..='\u{309f}' => hiragana += 1.0,
                '\u{30a0}'..='\u{30ff}' => katakana += 1.0,
                '\u{ac00}'..='\u{d7af}' => hangul += 1.0,
                '\u{0400}'..='\u{04ff}' => cyrillic += 1.0,
                'a'..='z' | 'A'..='Z' => latin += 1.0,
                _ => {}
            }
        }

        // Determine primary script
        let counts = [hanzi, hiragana + katakana, hangul, cyrillic, latin];
        let max_count = counts.iter().cloned().fold(0.0f32, f32::max);

        let (language, script, confidence) = if hanzi == max_count && hanzi > 0.0 {
            ("zh", Some("Hanzi"), hanzi / total)
        } else if (hiragana + katakana) == max_count && (hiragana + katakana) > 0.0 {
            ("ja", Some("Kana"), (hiragana + katakana) / total)
        } else if hangul == max_count && hangul > 0.0 {
            ("ko", Some("Hangul"), hangul / total)
        } else if cyrillic == max_count && cyrillic > 0.0 {
            ("ru", Some("Cyrillic"), cyrillic / total)
        } else {
            ("en", Some("Latin"), latin / total)
        };

        Self {
            language: language.to_string(),
            script: script.map(String::from),
            confidence,
        }
    }
}

/// Preset languages supported by the framework
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SupportedLanguage {
    Chinese,
    English,
    Japanese,
    Korean,
    German,
    French,
    Russian,
    Portuguese,
    Spanish,
    Italian,
    Auto,  // Automatic detection
}

impl SupportedLanguage {
    /// Get language code
    pub fn code(&self) -> &'static str {
        match self {
            SupportedLanguage::Chinese => "zh",
            SupportedLanguage::English => "en",
            SupportedLanguage::Japanese => "ja",
            SupportedLanguage::Korean => "ko",
            SupportedLanguage::German => "de",
            SupportedLanguage::French => "fr",
            SupportedLanguage::Russian => "ru",
            SupportedLanguage::Portuguese => "pt",
            SupportedLanguage::Spanish => "es",
            SupportedLanguage::Italian => "it",
            SupportedLanguage::Auto => "auto",
        }
    }

    /// Get all supported languages
    pub fn all() -> &'static [SupportedLanguage] {
        &[
            SupportedLanguage::Chinese,
            SupportedLanguage::English,
            SupportedLanguage::Japanese,
            SupportedLanguage::Korean,
            SupportedLanguage::German,
            SupportedLanguage::French,
            SupportedLanguage::Russian,
            SupportedLanguage::Portuguese,
            SupportedLanguage::Spanish,
            SupportedLanguage::Italian,
            SupportedLanguage::Auto,
        ]
    }

    /// Get language name
    pub fn name(&self) -> &'static str {
        match self {
            SupportedLanguage::Chinese => "Chinese",
            SupportedLanguage::English => "English",
            SupportedLanguage::Japanese => "Japanese",
            SupportedLanguage::Korean => "Korean",
            SupportedLanguage::German => "German",
            SupportedLanguage::French => "French",
            SupportedLanguage::Russian => "Russian",
            SupportedLanguage::Portuguese => "Portuguese",
            SupportedLanguage::Spanish => "Spanish",
            SupportedLanguage::Italian => "Italian",
            SupportedLanguage::Auto => "Auto-detect",
        }
    }
}

impl std::fmt::Display for SupportedLanguage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_language_detection_chinese() {
        let detection = LanguageDetection::detect("你好世界");
        assert_eq!(detection.language, "zh");
        assert!(detection.confidence > 0.9);
    }

    #[test]
    fn test_language_detection_english() {
        let detection = LanguageDetection::detect("Hello world");
        assert_eq!(detection.language, "en");
        assert!(detection.confidence > 0.9);
    }

    #[test]
    fn test_language_detection_japanese() {
        let detection = LanguageDetection::detect("こんにちは世界");
        assert_eq!(detection.language, "ja");
    }

    #[test]
    fn test_language_detection_korean() {
        let detection = LanguageDetection::detect("안녕하세요");
        assert_eq!(detection.language, "ko");
    }

    #[test]
    fn test_streaming_config() {
        let config = StreamingConfig::low_latency();
        assert!(config.enabled);
        assert_eq!(config.target_latency_ms, 50);
        assert!(config.chunk_duration_ms() > 0.0);
    }

    #[test]
    fn test_stream_chunk() {
        let chunk = StreamChunk::new(vec![0.0; 4096], 22050, 0, false);
        assert_eq!(chunk.sequence, 0);
        assert!(!chunk.is_final);
        assert!(chunk.duration_ms() > 0.0);
    }
}
