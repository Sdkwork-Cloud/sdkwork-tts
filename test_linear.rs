use candle_core::{Device, Tensor, DType};
use candle_nn::Linear;

fn main() -> anyhow::Result<()> {
    let device = Device::Cpu;
    
    // Create a linear layer: 160 -> 1024
    // PyTorch style: weight is (out_features, in_features) = (1024, 160)
    let weight = Tensor::randn(0.0f32, 0.02, (1024, 160), &device)?;
    let bias = Tensor::zeros((1024,), DType::F32, &device)?;
    let linear = Linear::new(weight, Some(bias));
    
    // Input: (batch=1, seq=50, in_features=160)
    let input = Tensor::randn(0.0f32, 1.0, (1, 50, 160), &device)?;
    
    println!("Input shape: {:?}", input.shape());
    
    // Forward pass
    let output = linear.forward(&input)?;
    println!("Output shape: {:?}", output.shape());
    
    Ok(())
}
