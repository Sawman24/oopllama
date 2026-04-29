use tokenizers::tokenizer::{Result, Tokenizer, Trainer};
use tokenizers::models::bpe::{BPE, BpeTrainer};
use tokenizers::pre_tokenizers::byte_level::ByteLevel;

fn main() -> Result<()> {
    // 1. Initialize a BPE model
    let mut trainer = BpeTrainer::builder()
        .show_progress(true)
        .vocab_size(4096) // 4k tokens is a sweet spot for a small model
        .min_frequency(2)
        .build();

    let mut tokenizer = Tokenizer::new(BPE::default());
    tokenizer.with_pre_tokenizer(ByteLevel::default());

    // 2. Train on the cleaned text
    let files = vec!["hail_mary_perfect.txt".to_string()];
    tokenizer.train_from_files(&mut trainer, files)?;

    // 3. Save the result
    tokenizer.save("hail_mary_tokenizer.json", true)?;
    println!("✅ Tokenizer trained and saved to hail_mary_tokenizer.json");
    Ok(())
}
