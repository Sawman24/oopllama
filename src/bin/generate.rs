use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{VarBuilder, VarMap};
use oopllama::custom_model::{GPT, Config};
use std::io::{Write, stdout};

fn main() -> Result<()> {
    let device = Device::new_cuda(0).unwrap_or(Device::Cpu);
    
    // 1. Setup Same Architecture
    let cfg = Config {
        vocab_size: 256,
        n_embd: 256,
        n_layer: 6,
        n_head: 8,
        max_seq_len: 128,
    };
    
    let mut varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = GPT::new(vb, &cfg)?;
    
    // 2. Load the Best Weights
    let weights_file = "nova_lean_weights.safetensors";
    if !std::path::Path::new(weights_file).exists() {
        println!("Error: Could not find weights file '{}'", weights_file);
        return Ok(());
    }
    varmap.load(weights_file)?;
    println!("🧠 Nova Brain Loaded Successfully!");

    // 3. Prompt the model
    let prompt = "Alice said ";
    let mut generated = prompt.as_bytes().iter().map(|&b| b as u32).collect::<Vec<u32>>();
    
    println!("Generating (Greedy Decoding)...");
    print!("{}", prompt);
    stdout().flush().unwrap();

    for _ in 0..500 {
        // Take the last max_seq_len tokens
        let start = if generated.len() > cfg.max_seq_len {
            generated.len() - cfg.max_seq_len
        } else {
            0
        };
        
        let context = &generated[start..];
        let input = Tensor::from_slice(context, (1, context.len()), &device)?;
        
        let logits = model.forward(&input)?;
        let seq_len = logits.dim(1)?;
        let logits_last = logits.narrow(1, seq_len - 1, 1)?.squeeze(1)?.squeeze(0)?;
        
        // Greedy decoding: take the most likely next character
        let next_token = logits_last.argmax(0)?.to_scalar::<u32>()?;
        
        generated.push(next_token);
        
        let c = next_token as u8 as char;
        print!("{}", c);
        stdout().flush().unwrap();
    }
    
    println!("\n\n--- End of Dream ---");
    Ok(())
}
