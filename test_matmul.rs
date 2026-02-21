use candle_core::{Device, Tensor, DType};

fn main() -> anyhow::Result<()> {
    let device = Device::Cpu;
    
    // Test 3D matmul
    let lhs = Tensor::randn(0.0f32, 1.0, (1, 50, 160), &device)?;
    let rhs = Tensor::randn(0.0f32, 0.02, (160, 1024), &device)?;
    
    println!("LHS shape: {:?}", lhs.shape());
    println!("RHS shape: {:?}", rhs.shape());
    
    let result = lhs.matmul(&rhs)?;
    println!("Result shape: {:?}", result.shape());
    
    Ok(())
}
