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

    // 1. Setup Model Architecture
    let cfg = Config {
        vocab_size: 256, // Character-level model for quick training
        n_embd: 128,
        n_layer: 4,
        n_head: 4,
        max_seq_len: 128,
    };
    
    let mut varmap = VarMap::new();
    if std::path::Path::new("nova_weights.safetensors").exists() {
        println!("Found existing weights! Resuming training from nova_weights.safetensors...");
        varmap.load("nova_weights.safetensors")?;
    } else {
        println!("No existing weights found. Initializing fresh weights.");
    }

    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = GPT::new(vb, &cfg)?;
    
    println!("Model initialized successfully. Parameters: {}", varmap.all_vars().len());

    // 2. Setup Dataset (We will train it on a mini dataset)
    let dataset = "Hello! I am NOVA, your custom AI model. I was built from scratch in Rust using Candle. I am learning how to speak by looking at this text over and over again until I understand language!";
    let data_bytes = dataset.as_bytes();
    
    let batch_size = 4;
    let seq_len = cfg.max_seq_len;
    
    // 3. Setup Optimizer
    let mut opt = AdamW::new_lr(varmap.all_vars(), 1e-3)?;

    println!("Starting training loop...");
    let epochs = 500;
    
    for epoch in 1..=epochs {
        // --- THERMAL SAFEGUARD ---
        if epoch % 10 == 0 {
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
            // Because our dataset is tiny, we just pad or loop it, but here we just take the first seq_len bytes
            // In a real scenario, you would randomly sample chunks from a large corpus
            let mut x_seq = Vec::new();
            let mut y_seq = Vec::new();
            for i in 0..seq_len {
                let idx = (epoch + i) % data_bytes.len();
                let next_idx = (epoch + i + 1) % data_bytes.len();
                x_seq.push(data_bytes[idx] as u32);
                y_seq.push(data_bytes[next_idx] as u32);
            }
            x_batch.extend_from_slice(&x_seq);
            y_batch.extend_from_slice(&y_seq);
        }
        
        let x = Tensor::from_slice(&x_batch, (batch_size, seq_len), &device)?;
        let y = Tensor::from_slice(&y_batch, (batch_size, seq_len), &device)?;
        
        // Forward Pass
        let logits = model.forward(&x)?;
        
        // Calculate Cross Entropy Loss
        // Logits shape: [b, seq, vocab] -> [b * seq, vocab]
        // Target shape: [b, seq] -> [b * seq]
        let logits_flat = logits.reshape((batch_size * seq_len, cfg.vocab_size))?;
        let y_flat = y.reshape((batch_size * seq_len,))?;
        
        let loss = loss::cross_entropy(&logits_flat, &y_flat)?;
        
        // Backward Pass
        opt.backward_step(&loss)?;
        
        if epoch % 50 == 0 || epoch == 1 {
            let loss_f32 = loss.to_vec0::<f32>()?;
            println!("Epoch {}/{} | Loss: {:.4}", epoch, epochs, loss_f32);
        }
    }

    println!("=====================================");
    println!("Training Complete!");
    println!("We have successfully backpropagated gradients and updated the neural weights of our custom model.");
    println!("Saving weights to 'nova_weights.safetensors'...");
    varmap.save("nova_weights.safetensors")?;
    println!("=====================================");

    println!("Testing Generative Output (Greedy Decoding)...");
    let mut generated = vec!['H' as u32, 'e' as u32, 'l' as u32, 'l' as u32, 'o' as u32];
    
    for _ in 0..100 {
        let input = Tensor::from_slice(&generated, (1, generated.len()), &device)?;
        let logits = model.forward(&input)?;
        
        // Get logits for the last token: [1, seq_len, vocab_size] -> [vocab_size]
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

