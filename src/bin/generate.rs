use candle_core::{DType, Device, Result, Tensor};
use oopllama::custom_model::{GPT, Config};
use tokenizers::Tokenizer;
use rand::distributions::{Distribution, WeightedIndex};
use rand::thread_rng;

fn main() -> Result<()> {
    let device = Device::new_cuda(0).unwrap_or(Device::Cpu);

    // 1. Load Tokenizer
    let tokenizer_path = "hail_mary_tokenizer.json";
    let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(|e| candle_core::Error::Msg(e.to_string()))?;

    // 2. Setup NOVA PRIME Model
    let cfg = Config {
        vocab_size: 32768,
        n_embd: 768,
        n_layer: 12,
        n_head: 12,
        max_seq_len: 256,
    };

    let weights_file = "nova_prime_best_weights.safetensors";
    if !std::path::Path::new(weights_file).exists() {
        println!("❌ Error: No weights found. Run training first!");
        return Ok(());
    }

    let mut varmap = candle_nn::VarMap::new();
    let vb = candle_nn::VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = GPT::new(vb, &cfg)?;
    varmap.load(weights_file)?;

    println!("🧠 Nova Brain Loaded Successfully (Word-Level)!");
    
    // 3. Inference Setup
    let prompt = "“What’s two plus two?”";
    let encoding = tokenizer.encode(prompt, true).map_err(|e| candle_core::Error::Msg(e.to_string()))?;
    let mut tokens = encoding.get_ids().to_vec();
    
    println!("Generating (Sampling with Temp 0.8)...");
    print!("{}", prompt);

    let temperature = 0.8;

    for _ in 0..100 {
        let input_tokens = if tokens.len() > cfg.max_seq_len {
            &tokens[tokens.len() - cfg.max_seq_len..]
        } else {
            &tokens[..]
        };

        let input = Tensor::new(input_tokens, &device)?.unsqueeze(0)?;
        let logits = model.forward(&input)?;
        let logits = logits.get(0)?.get(input_tokens.len() - 1)?;
        
        // Temperature Scaling
        let prs = candle_nn::ops::softmax(&(&logits / temperature as f64)?, 0)?;
        let probs: Vec<f32> = prs.to_vec1()?;
        
        // Weighted Sampling
        let dist = WeightedIndex::new(&probs).map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        let mut rng = thread_rng();
        let next_token = dist.sample(&mut rng) as u32;

        tokens.push(next_token);
        
        // Decode and Print
        let word = tokenizer.decode(&[next_token], true).unwrap_or_default();
        print!("{}", word);
        
        if next_token == 0 { break; } // Assuming 0 is EOS
    }

    println!("\n");
    Ok(())
}
