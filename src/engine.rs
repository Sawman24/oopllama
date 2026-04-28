use candle_core::{Device, Tensor, DType};
use candle_nn::VarBuilder;
use candle_transformers::models::llama::{Llama, LlamaConfig, Config, Cache};
use candle_transformers::generation::LogitsProcessor;
use tokenizers::Tokenizer;
use std::path::Path;

pub struct Telemetry {
    pub vram_total: u64,
    pub vram_used: u64,
    pub temperature: f32,
}

pub struct InferenceEngine {
    pub device: Device,
    pub model: Llama,
    pub tokenizer: Tokenizer,
    pub config: Config,
}

impl InferenceEngine {
    pub fn new() -> anyhow::Result<Self> {
        // Optimized for V100: Compute Capability 7.0
        let device = Device::new_cuda(0)?;
        
        // Define paths to downloaded models (using the persistent Docker volume)
        let model_dir = Path::new("/app/models");
        let config_path = model_dir.join("config.json");
        let tokenizer_path = model_dir.join("tokenizer.json");
        let weights_path = model_dir.join("model.safetensors");

        if !weights_path.exists() || !config_path.exists() || !tokenizer_path.exists() {
            tracing::warn!("Model files missing. Checking paths...");
            tracing::warn!("Sleeping to keep container alive for download...");
            while !weights_path.exists() || !config_path.exists() || !tokenizer_path.exists() {
                if !weights_path.exists() { tracing::warn!("Missing: {:?}", weights_path); }
                if !config_path.exists() { tracing::warn!("Missing: {:?}", config_path); }
                if !tokenizer_path.exists() { tracing::warn!("Missing: {:?}", tokenizer_path); }
                std::thread::sleep(std::time::Duration::from_secs(5));
            }
            tracing::info!("All model files found! Booting NOVA...");
        }

        tracing::info!("Loading NOVA Config...");
        let config_json: LlamaConfig = serde_json::from_reader(std::fs::File::open(&config_path)?)?;
        let config: Config = config_json.into_config(false); // false = no flash attention for now
        
        tracing::info!("Loading Tokenizer...");
        let tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(|e| anyhow::anyhow!(e.to_string()))?;

        tracing::info!("Loading Weights to VRAM (FP16)...");
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights_path], DType::F16, &device)? };
        
        tracing::info!("Initializing Transformer Engine...");
        let model = Llama::load(vb, &config)?;

        Ok(Self {
            device,
            model,
            tokenizer,
            config,
        })
    }

    pub fn get_telemetry(&self) -> Telemetry {
        Telemetry {
            vram_total: 16160,
            vram_used: 4096, // In a real setup, query via nvml
            temperature: 45.0,
        }
    }

    /// Actual Probabilistic Token Generation using Llama Architecture
    pub fn generate(&self, prompt: &str) -> anyhow::Result<String> {
        let mut cache = Cache::new(true, DType::F16, &self.config, &self.device)?;
        let mut logits_processor = LogitsProcessor::new(299792458, Some(0.7), Some(0.95));
        
        let tokens = self.tokenizer.encode(prompt, true).map_err(|e| anyhow::anyhow!(e.to_string()))?.get_ids().to_vec();
        
        let mut generated_tokens = vec![];
        let mut index_pos = 0;
        
        // 1. Initial Prompt Forward Pass (Prefill Phase)
        let input_tensor = Tensor::new(tokens.as_slice(), &self.device)?.unsqueeze(0)?;
        let logits = self.model.forward(&input_tensor, index_pos, &mut cache)?;
        
        // 2. Extract the logits for the very last token
        let logits = logits.squeeze(0)?; 
        let logits = if logits.rank() == 2 {
            // [seq_len, vocab_size] -> take the last token
            let seq_len = logits.dim(0)?;
            logits.narrow(0, seq_len - 1, 1)?.squeeze(0)?
        } else {
            // Already [vocab_size]
            logits
        };
        
        let mut next_token = logits_processor.sample(&logits)?;
        generated_tokens.push(next_token);
        index_pos += tokens.len();
        
        use std::io::Write;
        print!("\n🧠 NOVA: ");
        std::io::stdout().flush().unwrap();

        // 3. Decoding Loop
        for _ in 0..256 {
            let input_tensor = Tensor::new(&[next_token], &self.device)?.unsqueeze(0)?;
            let logits = self.model.forward(&input_tensor, index_pos, &mut cache)?;
            
            let logits = logits.squeeze(0)?;
            let logits = if logits.rank() == 2 {
                let seq_len = logits.dim(0)?;
                logits.narrow(0, seq_len - 1, 1)?.squeeze(0)?
            } else {
                logits
            };

            next_token = logits_processor.sample(&logits)?;
            generated_tokens.push(next_token);
            index_pos += 1;
            
            // Stream the token to the console in real-time
            if let Ok(t) = self.tokenizer.decode(&[next_token], true) {
                print!("{}", t);
                let _ = std::io::stdout().flush();
            }

            // Stop at EOS token (usually 2 for Llama models)
            if next_token == 2 || next_token == self.tokenizer.token_to_id("</s>").unwrap_or(2) {
                break;
            }
        }
        println!();
        
        let text = self.tokenizer.decode(&generated_tokens, true).map_err(|e| anyhow::anyhow!(e.to_string()))?;
        Ok(text)
    }
}
