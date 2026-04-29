use tokenizers::tokenizer::{Result, Tokenizer};
use tokenizers::models::bpe::{BPE, BpeTrainer};
use tokenizers::pre_tokenizers::byte_level::ByteLevel;

fn main() -> Result<()> {
    // 1. Initialize a BPE model
    let mut trainer = BpeTrainer::builder()
        .show_progress(true)
        .vocab_size(4096) 
        .min_frequency(2)
        .build();

    // In 0.21, we use the standard BPE model directly
    let mut tokenizer = Tokenizer::new(BPE::default());
    
    // Fix: with_pre_tokenizer now expects Some()
    tokenizer.with_pre_tokenizer(Some(ByteLevel::default()));

    // 2. Train on the perfect text
    let files = vec!["hail_mary_perfect.txt".to_string()];
    tokenizer.train_from_files(&mut trainer, files)?;

    // 3. Save the result
    tokenizer.save("hail_mary_tokenizer.json", true)?;
    println!("✅ Tokenizer trained and saved to hail_mary_tokenizer.json");
    Ok(())
}
