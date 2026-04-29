use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{VarBuilder, VarMap};
use oopllama::custom_model::{GPT, Config};
use std::io::{Write, stdout};

fn main() -> Result<()> {
    let device = Device::new_cuda(0).unwrap_or(Device::Cpu);
    
    // 1. Setup Architecture
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
    
    // 2. Load Weights
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
    let temperature = 0.2f64;
    
    println!("Generating (Sampling with Temp {})...", temperature);
    print!("{}", prompt);
    stdout().flush().unwrap();

    for _ in 0..500 {
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
        
        // --- SAMPLING WITH TEMPERATURE ---
        let prs = (&logits_last / temperature)?;
        let prs = candle_nn::ops::softmax(&prs, 0)?;
        
        let prs_vec: Vec<f32> = prs.to_vec1()?;
        let r = fastrand::f32();
        let mut cum = 0.0;
        let mut next_token = 0;
        for (idx, &p) in prs_vec.iter().enumerate() {
            cum += p;
            if r < cum {
                next_token = idx as u32;
                break;
            }
        }
        
        generated.push(next_token);
        
        let c = next_token as u8 as char;
        print!("{}", c);
        stdout().flush().unwrap();
    }
    
    println!("\n\n--- End of Dream ---");
    Ok(())
}
