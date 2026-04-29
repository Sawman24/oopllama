use tokenizers::tokenizer::{Result, TokenizerBuilder};
use tokenizers::models::bpe::{BPE, BpeTrainer};
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::normalizers::utils::Sequence;

fn main() -> Result<()> {
    // 1. Setup the BPE Trainer
    let mut trainer = BpeTrainer::builder()
        .show_progress(true)
        .vocab_size(4096) 
        .min_frequency(2)
        .build();

    // 2. Build the Tokenizer with explicit (empty) Normalizer to satisfy type inference
    let mut tokenizer = TokenizerBuilder::new()
        .with_model(BPE::default())
        .with_normalizer(Some(Sequence::new(vec![]))) 
        .with_pre_tokenizer(Some(ByteLevel::default()))
        .build()?;

    // 3. Train on the perfect text
    let files = vec!["hail_mary_perfect.txt".to_string()];
    tokenizer.train_from_files(&mut trainer, files)?;

    // 4. Save the result
    tokenizer.save("hail_mary_tokenizer.json", true)?;
    println!("✅ Tokenizer trained and saved to hail_mary_tokenizer.json");
    Ok(())
}
