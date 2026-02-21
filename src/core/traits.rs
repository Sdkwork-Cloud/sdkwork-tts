//! Core traits defining the IndexTTS2 framework interfaces
//!
//! These traits establish contracts for all major components,
//! enabling loose coupling, testability, and extensibility.

use candle_core::Device;
use std::path::Path;

use super::error::Result;

/// Base trait for all model components
///
/// Provides common lifecycle methods and metadata for model components.
pub trait ModelComponent: Send + Sync {
    /// Component name for identification and logging
    fn name(&self) -> &str;

    /// Component version
    fn version(&self) -> &str {
        "1.0.0"
    }

    /// Check if the component is initialized and ready
    fn is_ready(&self) -> bool;

    /// Get memory usage in bytes (if applicable)
    fn memory_usage(&self) -> Option<usize> {
        None
    }

    /// Reset component state (for stateful components)
    fn reset(&mut self) -> Result<()> {
        Ok(())
    }
}

/// Trait for configurable components
///
/// Allows components to be configured from various sources.
pub trait Configurable {
    /// Configuration type for this component
    type Config;

    /// Create component with configuration
    fn with_config(config: &Self::Config, device: &Device) -> Result<Self>
    where
        Self: Sized;

    /// Update configuration at runtime (if supported)
    fn update_config(&mut self, config: &Self::Config) -> Result<()> {
        let _ = config;
        Ok(())
    }

    /// Get current configuration
    fn config(&self) -> Option<&Self::Config> {
        None
    }
}

/// Trait for loadable components (weights, vocabularies, etc.)
///
/// Standardizes resource loading across all components.
pub trait Loadable: ModelComponent {
    /// Load from a file path
    fn load<P: AsRef<Path>>(path: P, device: &Device) -> Result<Self>
    where
        Self: Sized;

    /// Load with optional configuration
    fn load_with_config<P: AsRef<Path>>(
        path: P,
        config: Option<&Path>,
        device: &Device,
    ) -> Result<Self>
    where
        Self: Sized,
    {
        let _ = config;
        Self::load(path, device)
    }

    /// Check if weights are loaded
    fn is_loaded(&self) -> bool;

    /// Unload weights to free memory
    fn unload(&mut self) -> Result<()> {
        Ok(())
    }

    /// Reload previously unloaded weights
    fn reload(&mut self) -> Result<()> {
        Ok(())
    }
}

/// Trait for initializable components
///
/// Components that need explicit initialization beyond construction.
pub trait Initializable: ModelComponent {
    /// Initialize the component
    fn initialize(&mut self) -> Result<()>;

    /// Check if initialized
    fn is_initialized(&self) -> bool;

    /// Deinitialize to free resources
    fn deinitialize(&mut self) -> Result<()> {
        Ok(())
    }
}

/// Trait for encoder components
///
/// Encoders transform input data into latent representations.
pub trait Encoder: ModelComponent {
    /// Input type
    type Input;
    /// Output type
    type Output;

    /// Encode input into latent representation
    fn encode(&self, input: &Self::Input) -> Result<Self::Output>;

    /// Encode in batches for efficiency
    fn encode_batch(&self, inputs: &[Self::Input]) -> Result<Vec<Self::Output>>
    where
        Self::Input: Clone,
        Self::Output: Clone,
    {
        inputs.iter().map(|i| self.encode(i)).collect()
    }

    /// Get output dimension
    fn output_dim(&self) -> usize;

    /// Get expected input shape (if fixed)
    fn input_shape(&self) -> Option<Vec<usize>> {
        None
    }
}

/// Trait for decoder components
///
/// Decoders transform latent representations into output data.
pub trait Decoder: ModelComponent {
    /// Input type (latent representation)
    type Input;
    /// Output type
    type Output;

    /// Decode latent representation into output
    fn decode(&self, input: &Self::Input) -> Result<Self::Output>;

    /// Decode in batches for efficiency
    fn decode_batch(&self, inputs: &[Self::Input]) -> Result<Vec<Self::Output>>
    where
        Self::Input: Clone,
        Self::Output: Clone,
    {
        inputs.iter().map(|i| self.decode(i)).collect()
    }

    /// Get input dimension
    fn input_dim(&self) -> usize;

    /// Get output shape (if fixed)
    fn output_shape(&self) -> Option<Vec<usize>> {
        None
    }
}

/// Trait for synthesizer components
///
/// Synthesizers generate output from conditioned inputs.
pub trait Synthesizer: ModelComponent {
    /// Conditioning input type
    type Condition;
    /// Output type
    type Output;
    /// Generation options
    type Options;

    /// Synthesize output from conditioning
    fn synthesize(
        &self,
        condition: &Self::Condition,
        options: &Self::Options,
    ) -> Result<Self::Output>;

    /// Synthesize with streaming output
    fn synthesize_streaming<F>(
        &self,
        condition: &Self::Condition,
        options: &Self::Options,
        callback: F,
    ) -> Result<()>
    where
        F: FnMut(Self::Output) -> Result<()>;
}

/// Trait for preprocessors
///
/// Preprocessors transform raw input into model-ready format.
pub trait Preprocessor: ModelComponent {
    /// Raw input type
    type RawInput;
    /// Processed output type
    type ProcessedOutput;

    /// Preprocess raw input
    fn preprocess(&self, input: &Self::RawInput) -> Result<Self::ProcessedOutput>;

    /// Validate input without processing
    fn validate(&self, input: &Self::RawInput) -> Result<()> {
        let _ = input;
        Ok(())
    }

    /// Get expected input format description
    fn input_format(&self) -> &'static str;
}

/// Trait for postprocessors
///
/// Postprocessors transform model output into final format.
pub trait Postprocessor: ModelComponent {
    /// Raw model output type
    type RawOutput;
    /// Final output type
    type FinalOutput;

    /// Postprocess raw output
    fn postprocess(&self, output: &Self::RawOutput) -> Result<Self::FinalOutput>;

    /// Postprocess with options
    fn postprocess_with_options(
        &self,
        output: &Self::RawOutput,
        options: &PostprocessOptions,
    ) -> Result<Self::FinalOutput> {
        let _ = options;
        self.postprocess(output)
    }
}

/// Postprocessing options
#[derive(Debug, Clone, Default)]
pub struct PostprocessOptions {
    /// Apply denormalization
    pub denormalize: bool,
    /// Apply filtering
    pub filter: Option<String>,
    /// Output format
    pub format: OutputFormat,
}

/// Output format enumeration
#[derive(Debug, Clone, Copy, Default)]
pub enum OutputFormat {
    #[default]
    Raw,
    Normalized,
    Compressed,
}

/// Trait for cacheable components
///
/// Components that can cache intermediate results.
pub trait Cacheable: ModelComponent {
    /// Cache key type
    type Key: std::hash::Hash + Eq + Clone;
    /// Cache value type
    type Value: Clone;

    /// Get from cache
    fn get_cached(&self, key: &Self::Key) -> Option<Self::Value>;

    /// Store in cache
    fn set_cached(&mut self, key: Self::Key, value: Self::Value);

    /// Clear cache
    fn clear_cache(&mut self);

    /// Get cache size
    fn cache_size(&self) -> usize;

    /// Set maximum cache size
    fn set_max_cache_size(&mut self, size: usize) {
        let _ = size;
    }
}

/// Trait for components with tunable parameters
///
/// Allows runtime adjustment of component behavior.
pub trait Tunable: ModelComponent {
    /// Parameter type
    type Parameter: Clone;

    /// Get current parameters
    fn parameters(&self) -> &Self::Parameter;

    /// Set parameters
    fn set_parameters(&mut self, params: Self::Parameter) -> Result<()>;

    /// Get parameter schema (for UI/documentation)
    fn parameter_schema(&self) -> ParameterSchema;
}

/// Parameter schema for documentation and validation
#[derive(Debug, Clone)]
pub struct ParameterSchema {
    /// Parameter name
    pub name: String,
    /// Parameter description
    pub description: String,
    /// Parameter type
    pub param_type: ParameterType,
    /// Default value
    pub default: Option<String>,
    /// Valid range (for numeric types)
    pub range: Option<(f64, f64)>,
    /// Valid options (for enum types)
    pub options: Option<Vec<String>>,
}

/// Parameter type enumeration
#[derive(Debug, Clone)]
pub enum ParameterType {
    Float,
    Integer,
    Boolean,
    String,
    Enum(Vec<String>),
    Array(Box<ParameterType>),
}

/// Trait for components that support batching
///
/// Optimizes processing of multiple inputs.
pub trait Batched: ModelComponent {
    /// Set batch size
    fn set_batch_size(&mut self, size: usize);

    /// Get current batch size
    fn batch_size(&self) -> usize;

    /// Get optimal batch size for current hardware
    fn optimal_batch_size(&self) -> usize {
        1
    }

    /// Process with dynamic batching
    fn process_batched<T, R>(
        &self,
        inputs: Vec<T>,
        processor: impl Fn(&[T]) -> Result<Vec<R>>,
    ) -> Result<Vec<R>> {
        processor(&inputs)
    }
}

/// Trait for components with device affinity
///
/// Manages device placement (CPU/GPU).
pub trait DeviceAffinity: ModelComponent {
    /// Get current device
    fn device(&self) -> &Device;

    /// Move to device
    fn to_device(&mut self, device: &Device) -> Result<()>;

    /// Check if on specific device type
    fn is_on_device(&self, device_type: DeviceType) -> bool;
}

/// Device type enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceType {
    Cpu,
    Cuda(u32),
    Metal,
}

/// Trait for versioned components
///
/// Supports serialization format versioning.
pub trait Versioned: ModelComponent {
    /// Get format version
    fn format_version(&self) -> u32;

    /// Check compatibility with version
    fn is_compatible(&self, version: u32) -> bool {
        version == self.format_version()
    }

    /// Migrate from older version
    fn migrate(&mut self, from_version: u32) -> Result<()> {
        if !self.is_compatible(from_version) {
            return Err(super::error::TtsError::Internal {
                message: format!(
                    "Cannot migrate from version {} to {}",
                    from_version,
                    self.format_version()
                ),
                location: Some(self.name().to_string()),
            });
        }
        Ok(())
    }
}

/// Trait for observable components
///
/// Allows external monitoring of component state.
pub trait Observable: ModelComponent {
    /// Metric type
    type Metric;

    /// Get current metrics
    fn metrics(&self) -> Vec<Self::Metric>;

    /// Subscribe to metric updates
    fn subscribe_metrics<F>(&self, callback: F)
    where
        F: Fn(&Self::Metric) + Send + 'static;
}

/// Trait for components supporting A/B testing
///
/// Enables comparison of different implementations.
pub trait AbTestable: ModelComponent {
    /// Variant name
    fn variant_name(&self) -> &str;

    /// Get baseline component
    fn baseline(&self) -> Option<&dyn ModelComponent>;

    /// Compare with baseline
    fn compare_with_baseline(&self) -> ComparisonResult {
        ComparisonResult::default()
    }
}

/// Comparison result
#[derive(Debug, Clone, Default)]
pub struct ComparisonResult {
    /// Accuracy difference (positive = better)
    pub accuracy_delta: f64,
    /// Speed difference (positive = faster)
    pub speed_delta: f64,
    /// Memory difference (positive = less memory)
    pub memory_delta: f64,
    /// Overall recommendation
    pub recommendation: Recommendation,
}

/// Recommendation enumeration
#[derive(Debug, Clone, Copy, Default)]
pub enum Recommendation {
    #[default]
    Neutral,
    UseVariant,
    UseBaseline,
    NeedsInvestigation,
}

#[cfg(test)]
mod tests {
    use super::*;

    struct MockEncoder {
        name: String,
        ready: bool,
        output_dim: usize,
    }

    impl ModelComponent for MockEncoder {
        fn name(&self) -> &str {
            &self.name
        }

        fn is_ready(&self) -> bool {
            self.ready
        }
    }

    impl Encoder for MockEncoder {
        type Input = Vec<f32>;
        type Output = Vec<f32>;

        fn encode(&self, input: &Self::Input) -> Result<Self::Output> {
            Ok(input.clone())
        }

        fn output_dim(&self) -> usize {
            self.output_dim
        }
    }

    #[test]
    fn test_model_component() {
        let encoder = MockEncoder {
            name: "test_encoder".to_string(),
            ready: true,
            output_dim: 128,
        };

        assert_eq!(encoder.name(), "test_encoder");
        assert!(encoder.is_ready());
        assert_eq!(encoder.output_dim(), 128);
    }

    #[test]
    fn test_encoder_batch() {
        let encoder = MockEncoder {
            name: "test".to_string(),
            ready: true,
            output_dim: 128,
        };

        let inputs = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let outputs = encoder.encode_batch(&inputs).unwrap();

        assert_eq!(outputs.len(), 2);
        assert_eq!(outputs[0], vec![1.0, 2.0]);
    }
}
