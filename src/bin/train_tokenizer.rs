use tokenizers::tokenizer::{Result, TokenizerImpl};
use tokenizers::models::bpe::{BPE, BpeTrainer};
use tokenizers::normalizers::utils::Sequence as NormalizerSequence;
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::processors::sequence::Sequence as ProcessorSequence;
use tokenizers::decoders::byte_level::ByteLevel as DecoderByteLevel;

// Explicitly define the full type to satisfy the compiler
type MyTokenizer = TokenizerImpl<BPE, NormalizerSequence, ByteLevel, ProcessorSequence, DecoderByteLevel>;

fn main() -> Result<()> {
    // 1. Setup the BPE Trainer
    let mut trainer = BpeTrainer::builder()
        .show_progress(true)
        .vocab_size(4096) 
        .min_frequency(2)
        .build();

    // 2. Create the tokenizer with all components explicitly initialized
    let mut tokenizer = MyTokenizer::new(
        BPE::default(),
        NormalizerSequence::new(vec![]),
        ByteLevel::default(),
        ProcessorSequence::new(vec![]),
        DecoderByteLevel::default(),
    );

    // 3. Train on the perfect text
    let files = vec!["hail_mary_perfect.txt".to_string()];
    tokenizer.train_from_files(&mut trainer, files)?;

    // 4. Save the result
    tokenizer.save("hail_mary_tokenizer.json", true)?;
    println!("✅ Tokenizer trained and saved to hail_mary_tokenizer.json");
    Ok(())
}
