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

    // 2. Create the tokenizer with the Model
    // Because we use the 'MyTokenizer' type alias, the compiler knows the other 4 types!
    let mut tokenizer = MyTokenizer::new(BPE::default());
    
    // Attach the components (using Some() as required by v0.21)
    tokenizer.with_normalizer(Some(NormalizerSequence::new(vec![])));
    tokenizer.with_pre_tokenizer(Some(ByteLevel::default()));
    tokenizer.with_post_processor(Some(ProcessorSequence::new(vec![])));
    tokenizer.with_decoder(Some(DecoderByteLevel::default()));

    // 3. Train on the master training data
    let files = vec!["master_training_data.txt".to_string()];
    tokenizer.train_from_files(&mut trainer, files)?;

    // 4. Save the result
    tokenizer.save("hail_mary_tokenizer.json", true)?;
    println!("✅ Tokenizer trained and saved to hail_mary_tokenizer.json");
    Ok(())
}
