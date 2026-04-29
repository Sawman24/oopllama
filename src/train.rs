use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{AdamW, Optimizer, VarBuilder, VarMap, loss};
use oopllama::custom_model::{GPT, Config};
use tokenizers::Tokenizer;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

fn check_temperature() -> u32 {
    let output = std::process::Command::new("nvidia-smi")
        .arg("--query-gpu=temperature.gpu")
        .arg("--format=csv,noheader,nounits")
        .output()
        .expect("failed to execute nvidia-smi");
    
    let temp_str = String::from_utf8_lossy(&output.stdout);
    temp_str.trim().parse::<u32>().unwrap_or(0)
}

fn main() -> Result<()> {
    println!("=====================================");
    println!("🚀 NOVA V2: PROJECT HAIL MARY EDITION");
    println!("Mode: Word-Level (BPE 4096 Vocab)");
    println!("=====================================");

    let device = Device::new_cuda(0).unwrap_or(Device::Cpu);
    
    // 1. Setup Tokenizer
    let tokenizer_path = "hail_mary_tokenizer.json";
    if !std::path::Path::new(tokenizer_path).exists() {
        println!("❌ Error: Run 'cargo run --release --bin train_tokenizer' first!");
        return Ok(());
    }
    let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(|e| candle_core::Error::Msg(e.to_string()))?;
    
    // 2. Load and Tokenize Dataset
    println!("Tokenizing Mega-Dataset...");
    let dataset_text = std::fs::read_to_string("master_training_data.txt").expect("Could not read master text");
    let encoding = tokenizer.encode(dataset_text, true).map_err(|e| candle_core::Error::Msg(e.to_string()))?;
    let tokens = encoding.get_ids();
    println!("✅ Dataset ready: {} tokens", tokens.len());

    // 3. Setup Model Architecture
    let cfg = Config {
        vocab_size: 4096, // Upgraded from 256!
        n_embd: 256,
        n_layer: 6,
        n_head: 8,
        max_seq_len: 128,
    };
    
    let mut varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = GPT::new(vb, &cfg)?;
    
    let weights_file = "nova_hail_mary_weights.safetensors";
    if std::path::Path::new(weights_file).exists() {
        println!("Resuming from existing weights...");
        varmap.load(weights_file)?;
    }

    // --- BACKGROUND THERMAL MONITOR ---
    let pause_flag = Arc::new(AtomicBool::new(false));
    let pause_clone = pause_flag.clone();
    std::thread::spawn(move || {
        loop {
            let temp = check_temperature();
            if temp >= 85 {
                pause_clone.store(true, Ordering::SeqCst);
            } else {
                pause_clone.store(false, Ordering::SeqCst);
            }
            std::thread::sleep(std::time::Duration::from_secs(5));
        }
    });

    // 4. Setup Optimizer
    let mut current_lr = 1e-3; // Slightly lower LR for BPE stability
    let mut opt = AdamW::new(varmap.all_vars(), candle_nn::ParamsAdamW {
        lr: current_lr,
        weight_decay: 0.01,
        ..Default::default()
    })?;

    println!("Starting HAIL MARY training loop...");
    let epochs = 200000;
    let batch_size = 64;
    let seq_len = cfg.max_seq_len;
    let mega_batch_steps = 1000;
    let mut smoothed_loss = 0.0;

    let mut mega_x_tensor: Option<Tensor> = None;
    let mut mega_y_tensor: Option<Tensor> = None;

    for epoch in 1..=epochs {
        // Async Thermal Check
        while pause_flag.load(Ordering::SeqCst) {
            println!("⚠️ Cooling down...");
            std::thread::sleep(std::time::Duration::from_secs(30));
        }

        // Adaptive LR Decay
        if epoch % 10000 == 0 {
            let _ = varmap.save(weights_file);
            current_lr *= 0.9; 
            opt.set_learning_rate(current_lr);
        }

        // MEGA-BATCH REFRESH (Now with Tokens!)
        if (epoch - 1) % mega_batch_steps == 0 {
            let mut mega_x = Vec::with_capacity(mega_batch_steps * batch_size * seq_len);
            let mut mega_y = Vec::with_capacity(mega_batch_steps * batch_size * seq_len);
            for _ in 0..mega_batch_steps {
                for _ in 0..batch_size {
                    let start_idx = fastrand::usize(..tokens.len().saturating_sub(seq_len + 1));
                    mega_x.extend_from_slice(&tokens[start_idx..start_idx+seq_len]);
                    mega_y.extend_from_slice(&tokens[start_idx+1..start_idx+seq_len+1]);
                }
            }
            mega_x_tensor = Some(Tensor::from_vec(mega_x, (mega_batch_steps, batch_size, seq_len), &device)?);
            mega_y_tensor = Some(Tensor::from_vec(mega_y, (mega_batch_steps, batch_size, seq_len), &device)?);
        }

        let step_idx = (epoch - 1) % mega_batch_steps;
        let x = mega_x_tensor.as_ref().unwrap().get(step_idx)?;
        let y = mega_y_tensor.as_ref().unwrap().get(step_idx)?;
        
        let logits = model.forward(&x)?;
        let logits_flat = logits.reshape((batch_size * seq_len, cfg.vocab_size))?;
        let y_flat = y.reshape((batch_size * seq_len,))?;
        let loss = loss::cross_entropy(&logits_flat, &y_flat)?;
        
        opt.backward_step(&loss)?;
        
        if epoch % 1000 == 0 || epoch == 1 {
            let loss_val = loss.to_vec0::<f32>()?;
            if smoothed_loss == 0.0 { smoothed_loss = loss_val; }
            smoothed_loss = smoothed_loss * 0.9 + loss_val * 0.1;
            println!("Epoch {}/{} | Smoothed Loss: {:.4} | LR: {:.6}", epoch, epochs, smoothed_loss, current_lr);
        }
    }

    varmap.save(weights_file)?;
    println!("✅ Training Complete!");
    Ok(())
}
