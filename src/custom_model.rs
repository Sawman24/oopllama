use candle_core::{Device, Result, Tensor, IndexOp};
use candle_nn::{embedding, linear, layer_norm, Linear, LayerNorm, Embedding, VarBuilder, Module};

#[derive(Debug, Clone)]
pub struct Config {
    pub vocab_size: usize,
    pub n_embd: usize,
    pub n_layer: usize,
    pub n_head: usize,
    pub max_seq_len: usize,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            vocab_size: 32000, // TinyLlama vocab size
            n_embd: 256,
            n_layer: 4,
            n_head: 4,
            max_seq_len: 256,
        }
    }
}

pub struct CausalSelfAttention {
    c_attn: Linear,
    c_proj: Linear,
    n_head: usize,
    n_embd: usize,
}

impl CausalSelfAttention {
    pub fn new(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let c_attn = linear(cfg.n_embd, 3 * cfg.n_embd, vb.pp("c_attn"))?;
        let c_proj = linear(cfg.n_embd, cfg.n_embd, vb.pp("c_proj"))?;
        Ok(Self {
            c_attn,
            c_proj,
            n_head: cfg.n_head,
            n_embd: cfg.n_embd,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b_size, seq_len, n_embd) = x.dims3()?;
        let head_dim = self.n_embd / self.n_head;

        // Calculate Q, K, V
        let qkv = self.c_attn.forward(x)?;
        let qkv = qkv.reshape((b_size, seq_len, 3, self.n_head, head_dim))?;
        
        // Extract Q, K, V
        let q = qkv.i((.., .., 0, .., ..))?.transpose(1, 2)?; // [b, n_head, seq, head_dim]
        let k = qkv.i((.., .., 1, .., ..))?.transpose(1, 2)?.contiguous()?;
        let v = qkv.i((.., .., 2, .., ..))?.transpose(1, 2)?.contiguous()?;

        // Attention weights: Q * K^T / sqrt(head_dim)
        let att = q.matmul(&k.transpose(2, 3)?)?;
        let att = (att / (head_dim as f64).sqrt())?;
        
        // Causal Mask
        let mask = Self::get_mask(seq_len, x.device())?;
        let mask = mask.broadcast_as(att.shape())?;
        
        // Apply mask (replace 0s in mask with -inf)
        let att = mask.where_cond(&att, &(att.zeros_like()? - 1e4)?)?;
        let att = candle_nn::ops::softmax(&att, 3)?;

        // Output: att * V
        let y = att.matmul(&v)?;
        let y = y.transpose(1, 2)?.reshape((b_size, seq_len, n_embd))?;
        self.c_proj.forward(&y)
    }

    fn get_mask(size: usize, device: &Device) -> Result<Tensor> {
        let mask: Vec<_> = (0..size).flat_map(|i| (0..size).map(move |j| if j <= i { 1u8 } else { 0u8 })).collect();
        Tensor::from_slice(&mask, (size, size), device)
    }
}

pub struct MLP {
    c_fc: Linear,
    c_proj: Linear,
}

impl MLP {
    pub fn new(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let c_fc = linear(cfg.n_embd, 4 * cfg.n_embd, vb.pp("c_fc"))?;
        let c_proj = linear(4 * cfg.n_embd, cfg.n_embd, vb.pp("c_proj"))?;
        Ok(Self { c_fc, c_proj })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.c_fc.forward(x)?;
        // GELU activation
        let x = x.gelu()?;
        self.c_proj.forward(&x)
    }
}

pub struct Block {
    ln_1: LayerNorm,
    attn: CausalSelfAttention,
    ln_2: LayerNorm,
    mlp: MLP,
}

impl Block {
    pub fn new(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let ln_1 = layer_norm(cfg.n_embd, 1e-5, vb.pp("ln_1"))?;
        let attn = CausalSelfAttention::new(vb.pp("attn"), cfg)?;
        let ln_2 = layer_norm(cfg.n_embd, 1e-5, vb.pp("ln_2"))?;
        let mlp = MLP::new(vb.pp("mlp"), cfg)?;
        Ok(Self { ln_1, attn, ln_2, mlp })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = (x + self.attn.forward(&self.ln_1.forward(x)?)?)?;
        let x = (&x + self.mlp.forward(&self.ln_2.forward(&x)?)?)?;
        Ok(x)
    }
}

pub struct GPT {
    wte: Embedding,
    wpe: Embedding,
    blocks: Vec<Block>,
    ln_f: LayerNorm,
    lm_head: Linear,
    config: Config,
}

impl GPT {
    pub fn new(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let wte = embedding(cfg.vocab_size, cfg.n_embd, vb.pp("wte"))?;
        let wpe = embedding(cfg.max_seq_len, cfg.n_embd, vb.pp("wpe"))?;
        
        let mut blocks = Vec::new();
        for i in 0..cfg.n_layer {
            blocks.push(Block::new(vb.pp(&format!("h.{}", i)), cfg)?);
        }
        
        let ln_f = layer_norm(cfg.n_embd, 1e-5, vb.pp("ln_f"))?;
        let lm_head = linear(cfg.n_embd, cfg.vocab_size, vb.pp("lm_head"))?;
        
        Ok(Self { wte, wpe, blocks, ln_f, lm_head, config: cfg.clone() })
    }

    pub fn forward(&self, idx: &Tensor) -> Result<Tensor> {
        let (_b_size, seq_len) = idx.dims2()?;
        let pos: Vec<u32> = (0..seq_len as u32).collect();
        let pos = Tensor::from_slice(&pos, seq_len, idx.device())?.unsqueeze(0)?;

        let tok_emb = self.wte.forward(idx)?;
        let pos_emb = self.wpe.forward(&pos)?;
        
        let mut x = tok_emb.broadcast_add(&pos_emb)?;
        
        for block in &self.blocks {
            x = block.forward(&x)?;
        }
        
        let x = self.ln_f.forward(&x)?;
        self.lm_head.forward(&x)
    }
}
