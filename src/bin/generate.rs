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
        max_seq_len: 128,
    };

    let weights_file = "nova_prime_personality_best_weights.safetensors";
    if !std::path::Path::new(weights_file).exists() {
        println!("❌ Error: No personality weights found yet. Wait for Epoch 200!");
        return Ok(());
    }

    let mut varmap = candle_nn::VarMap::new();
    let vb = candle_nn::VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = GPT::new(vb, &cfg)?;
    varmap.load(weights_file)?;

    println!("🧠 Nova Agent Loaded (IFT Personality Mode)!");
    println!("Type 'quit' or 'exit' to end the conversation.\n");

    let temperature = 0.2;
    let mut conversation_history = String::new();

    loop {
        use std::io::{self, Write};
        
        print!("User: ");
        io::stdout().flush().unwrap();
        
        let mut user_input = String::new();
        io::stdin().read_line(&mut user_input).unwrap();
        let user_input = user_input.trim();
        
        if user_input.eq_ignore_ascii_case("quit") || user_input.eq_ignore_ascii_case("exit") {
            break;
        }

        conversation_history.push_str(&format!("User: {}\nAssistant: ", user_input));
        
        // Keep context window manageable (take last 500 chars roughly)
        let context = if conversation_history.len() > 500 {
            let start = conversation_history.len() - 500;
            &conversation_history[start..]
        } else {
            &conversation_history
        };

        let encoding = tokenizer.encode(context, true).map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        let mut tokens = encoding.get_ids().to_vec();
        
        let mut assistant_response = String::new();

        for _ in 0..100 {
            let input_tokens = if tokens.len() > cfg.max_seq_len {
                &tokens[tokens.len() - cfg.max_seq_len..]
            } else {
                &tokens[..]
            };

            let input = Tensor::new(input_tokens, &device)?.unsqueeze(0)?;
            let logits = model.forward(&input)?;
            let mut logits = logits.get(0)?.get(input_tokens.len() - 1)?;
            
            // Repetition Penalty: Lowered to 1.1 for conversation flow
            let penalty = 1.1;
            let mut already_seen = std::collections::HashSet::new();
            let history_window = 30; 
            for &t in tokens.iter().rev().take(history_window) {
                if !already_seen.contains(&t) {
                    let current_val = logits.get(t as usize)?.to_vec0::<f32>()?;
                    let new_val = if current_val < 0.0 { current_val * penalty } else { current_val / penalty };
                    logits = logits.slice_assign(&[t as usize..t as usize + 1], &Tensor::new(&[new_val], &device)?)?;
                    already_seen.insert(t);
                }
            }

            // Temperature Scaling
            let prs = candle_nn::ops::softmax(&(&logits / temperature as f64)?, 0)?;
            let probs: Vec<f32> = prs.to_vec1()?;
            
            // Weighted Sampling
            let dist = WeightedIndex::new(&probs).map_err(|e| candle_core::Error::Msg(e.to_string()))?;
            let mut rng = thread_rng();
            let next_token = dist.sample(&mut rng) as u32;

            tokens.push(next_token);
            
            // Decode
            let word = tokenizer.decode(&[next_token], true).unwrap_or_default();
            
            // Stop if she tries to start a new User/Assistant block
            if word.contains("User:") || word.contains("Assistant:") { break; }
            
            print!("{}", word);
            io::stdout().flush().unwrap();
            assistant_response.push_str(&word);
            
            if next_token == 0 { break; } // Assuming 0 is EOS
        }
        
        println!();
        conversation_history.push_str(&assistant_response);
        conversation_history.push('\n');
    }

    Ok(())
}
