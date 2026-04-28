use candle_core::{Device, Result, Tensor, DType};
use candle_nn::{AdamW, Optimizer, VarBuilder, VarMap, loss};

mod custom_model;
use custom_model::{GPT, Config};

fn check_temperature() -> u32 {
    let output = std::process::Command::new("nvidia-smi")
        .args(&["--query-gpu=temperature.gpu", "--format=csv,noheader"])
        .output();
        
    if let Ok(out) = output {
        let temp_str = String::from_utf8_lossy(&out.stdout);
        if let Ok(temp) = temp_str.trim().parse::<u32>() {
            return temp;
        }
    }
    0
}

fn main() -> Result<()> {
    println!("=====================================");
    println!("Initializing Custom NOVA Training...");
    println!("Architecture: GPT from scratch");
    println!("Safeguard: Thermal Throttling Active (Max 85°C)");
    println!("=====================================");

    let device = Device::new_cuda(0).unwrap_or(Device::Cpu);
    println!("Target Device: {:?}", device);

    // 1. Setup Model Architecture (Scaled up!)
    let cfg = Config {
        vocab_size: 256,
        n_embd: 512,      // 4x thicker embeddings
        n_layer: 8,       // 2x deeper
        n_head: 8,        // 2x more attention heads
        max_seq_len: 256, // 2x longer context window
    };
    
    let mut varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = GPT::new(vb, &cfg)?;
    
    let weights_file = "nova_large_weights.safetensors";
    if std::path::Path::new(weights_file).exists() {
        println!("Found existing weights! Resuming training from {}...", weights_file);
        varmap.load(weights_file)?;
    } else {
        println!("No existing weights found. Initializing fresh weights.");
    }
    
    println!("Model initialized successfully. Parameters: {}", varmap.all_vars().len());

    // 2. Setup Dataset
    println!("Loading TinyShakespeare dataset...");
    let dataset_string = std::fs::read_to_string("input.txt").unwrap_or_else(|_| {
        String::from("Fallback text! Could not load input.txt.")
    });
    let data_bytes = dataset_string.as_bytes();
    
    let batch_size = 16;
    let seq_len = cfg.max_seq_len;
    
    // 3. Setup Optimizer
    let mut opt = AdamW::new_lr(varmap.all_vars(), 3e-4)?;

    println!("Starting training loop...");
    let epochs = 5000; // Increased epochs since the dataset is much larger
    
    for epoch in 1..=epochs {
        // --- THERMAL SAFEGUARD ---
        if epoch % 50 == 0 {
            let temp = check_temperature();
            if temp >= 85 {
                println!("⚠️ CRITICAL: GPU Temperature reached {}°C! Pausing training for 60 seconds to cool down...", temp);
                std::thread::sleep(std::time::Duration::from_secs(60));
            }
        }

        // Create a random batch from our dataset
        let mut x_batch = Vec::new();
        let mut y_batch = Vec::new();
        
        for _ in 0..batch_size {
            // Randomly sample sequences from the large corpus
            let start_idx = fastrand::usize(..data_bytes.len().saturating_sub(seq_len + 1));
            
            let mut x_seq = Vec::new();
            let mut y_seq = Vec::new();
            for i in 0..seq_len {
                x_seq.push(data_bytes[start_idx + i] as u32);
                y_seq.push(data_bytes[start_idx + i + 1] as u32);
            }
            x_batch.extend_from_slice(&x_seq);
            y_batch.extend_from_slice(&y_seq);
        }
        
        let x = Tensor::from_slice(&x_batch, (batch_size, seq_len), &device)?;
        let y = Tensor::from_slice(&y_batch, (batch_size, seq_len), &device)?;
        
        // Forward Pass
        let logits = model.forward(&x)?;
        
        let logits_flat = logits.reshape((batch_size * seq_len, cfg.vocab_size))?;
        let y_flat = y.reshape((batch_size * seq_len,))?;
        
        let loss = loss::cross_entropy(&logits_flat, &y_flat)?;
        
        // Backward Pass
        opt.backward_step(&loss)?;
        
        if epoch % 100 == 0 || epoch == 1 {
            let loss_f32 = loss.to_vec0::<f32>()?;
            println!("Epoch {}/{} | Loss: {:.4}", epoch, epochs, loss_f32);
        }
    }

    println!("=====================================");
    println!("Training Complete!");
    println!("We have successfully backpropagated gradients and updated the neural weights of our custom model.");
    println!("Saving weights to '{}'...", weights_file);
    varmap.save(weights_file)?;
    println!("=====================================");

    println!("Testing Generative Output (Greedy Decoding)...");
    let mut generated = vec!['T' as u32, 'h' as u32, 'e' as u32, ' ' as u32];
    
    for _ in 0..200 {
        let input = Tensor::from_slice(&generated, (1, generated.len()), &device)?;
        let logits = model.forward(&input)?;
        
        let seq_len = logits.dim(1)?;
        let logits_last = logits.narrow(1, seq_len - 1, 1)?.squeeze(1)?.squeeze(0)?;
        
        // Pick the most likely next character
        let next_token = logits_last.argmax(0)?.to_scalar::<u32>()?;
        generated.push(next_token);
    }
    
    let generated_str: String = generated.into_iter().map(|c| c as u8 as char).collect();
    println!("\n🧠 Custom GPT Output: {}", generated_str);
    println!("=====================================");

    Ok(())
}

