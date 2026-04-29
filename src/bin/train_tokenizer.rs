use tokenizers::tokenizer::{Result, Tokenizer, TrainerWrapper};
use tokenizers::models::bpe::{BPE, BpeTrainer};
use tokenizers::models::ModelWrapper;
use tokenizers::pre_tokenizers::byte_level::ByteLevel;

fn main() -> Result<()> {
    // 1. Initialize a BPE trainer and wrap it in the TrainerWrapper
    let bpe_trainer = BpeTrainer::builder()
        .show_progress(true)
        .vocab_size(4096) 
        .min_frequency(2)
        .build();
    let mut trainer = TrainerWrapper::BPE(bpe_trainer);

    // 2. Initialize the Tokenizer with a BPE model wrapped in ModelWrapper
    let mut tokenizer = Tokenizer::new(ModelWrapper::BPE(BPE::default()));
    
    // 3. Add the ByteLevel pre-tokenizer
    tokenizer.with_pre_tokenizer(Some(ByteLevel::default()));

    // 4. Train on the perfect text
    let files = vec!["hail_mary_perfect.txt".to_string()];
    tokenizer.train_from_files(&mut trainer, files)?;

    // 5. Save the result
    tokenizer.save("hail_mary_tokenizer.json", true)?;
    println!("✅ Tokenizer trained and saved to hail_mary_tokenizer.json");
    Ok(())
}
