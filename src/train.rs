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

    // 1. Setup Model Architecture (Back to F32 for ultimate stability)
    let dtype = DType::F32; 
    let cfg = Config {
        vocab_size: 256,
        n_embd: 512,
        n_layer: 8,
        n_head: 8,
        max_seq_len: 256,
    };
    
    let mut varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, dtype, &device);
    let model = GPT::new(vb, &cfg)?;
    
    let weights_file = "nova_large_weights.safetensors";
    if std::path::Path::new(weights_file).exists() {
        println!("Found existing weights! Resuming training from {}...", weights_file);
        varmap.load(weights_file)?;
    } else {
        println!("No existing weights found. Initializing fresh weights.");
    }

    // 2. Setup Dataset & MEGA-BATCHING (Keep this for speed!)
    println!("Preparing Mega-Batch on GPU for zero CPU latency...");
    let dataset_string = std::fs::read_to_string("alice.txt").unwrap_or_else(|_| String::from("Fallback text!"));
    let data_bytes = dataset_string.as_bytes();
    
    let batch_size = 32; // Safe batch size
    let seq_len = cfg.max_seq_len;
    let mega_batch_steps = 1000;
    
    let mut mega_x = Vec::with_capacity(mega_batch_steps * batch_size * seq_len);
    let mut mega_y = Vec::with_capacity(mega_batch_steps * batch_size * seq_len);
    
    for _ in 0..mega_batch_steps {
        for _ in 0..batch_size {
            let start_idx = fastrand::usize(..data_bytes.len().saturating_sub(seq_len + 1));
            mega_x.extend(data_bytes[start_idx..start_idx+seq_len].iter().map(|&b| b as u32));
            mega_y.extend(data_bytes[start_idx+1..start_idx+seq_len+1].iter().map(|&b| b as u32));
        }
    }
    
    let mega_x_tensor = Tensor::from_vec(mega_x, (mega_batch_steps, batch_size, seq_len), &device)?;
    let mega_y_tensor = Tensor::from_vec(mega_y, (mega_batch_steps, batch_size, seq_len), &device)?;

    // 3. Setup Optimizer
    let mut current_lr = 1e-4;
    let mut opt = AdamW::new(varmap.all_vars(), candle_nn::ParamsAdamW {
        lr: current_lr,
        weight_decay: 0.01,
        ..Default::default()
    })?;

    println!("Starting STABLE training loop (F32 + MegaBatch)...");
    let epochs = 50000;
    let mut smoothed_loss = 0.0;
    
    for epoch in 1..=epochs {
        if epoch % 500 == 0 {
            let temp = check_temperature();
            if temp >= 85 {
                println!("⚠️ CRITICAL: GPU Temperature {}°C! Cooling down...", temp);
                std::thread::sleep(std::time::Duration::from_secs(60));
            }
        }

        if epoch % 5000 == 0 {
            println!("💾 Auto-saving weights...");
            let _ = varmap.save(weights_file);
            current_lr *= 0.8; 
            opt.set_learning_rate(current_lr);
        }

        let step_idx = (epoch - 1) % mega_batch_steps;
        let x = mega_x_tensor.get(step_idx)?;
        let y = mega_y_tensor.get(step_idx)?;
        
        // Forward Pass
        let logits = model.forward(&x)?;
        let logits_flat = logits.reshape((batch_size * seq_len, cfg.vocab_size))?;
        let y_flat = y.reshape((batch_size * seq_len,))?;
        let loss = loss::cross_entropy(&logits_flat, &y_flat)?;
        
        // Backward Pass
        opt.backward_step(&loss)?;
        
        if epoch % 100 == 0 || epoch == 1 {
            let loss_val = loss.to_vec0::<f32>()?;
            if smoothed_loss == 0.0 { smoothed_loss = loss_val; }
            smoothed_loss = smoothed_loss * 0.9 + loss_val * 0.1;
            println!("Epoch {}/{} | Smoothed Loss: {:.4} | LR: {:.6}", epoch, epochs, smoothed_loss, current_lr);
        }
    }

    println!("=====================================");
    println!("Training Complete!");
    println!("We have successfully backpropagated gradients and updated the neural weights of our custom model.");
    println!("Saving weights to '{}'...", weights_file);
    varmap.save(weights_file)?;
    println!("=====================================");

    println!("Testing Generative Output (Greedy Decoding)...");
    let mut generated = vec!['A' as u32, 'l' as u32, 'i' as u32, 'c' as u32, 'e' as u32];
    
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

