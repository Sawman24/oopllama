use redb::{Database, TableDefinition, ReadableTable};
use fastembed::{TextEmbedding, InitOptions, EmbeddingModel};

const FACTS_TABLE: TableDefinition<&str, &str> = TableDefinition::new("facts");

pub struct MemoryManager {
    db: Database,
    model: TextEmbedding,
}

impl MemoryManager {
    pub fn new(db_path: &str) -> anyhow::Result<Self> {
        let db = Database::builder()
            .create(db_path)?;

        // Initialize fastembed with a lightweight model for V100
        let model = TextEmbedding::try_new(
            InitOptions::new(EmbeddingModel::AllMiniLML6V2)
        )?;

        Ok(Self { db, model })
    }

    pub fn store_fact(&self, key: &str, value: &str) -> anyhow::Result<()> {
        let write_txn = self.db.begin_write()?;
        {
            let mut table = write_txn.open_table(FACTS_TABLE)?;
            table.insert(key, value)?;
        }
        write_txn.commit()?;
        Ok(())
    }

    /// Perform a vector-based similarity search (Linear Scan for simplicity)
    pub async fn search_relevant_context(&self, query: &str, top_k: usize) -> anyhow::Result<Vec<String>> {
        let query_embedding = self.model.embed(vec![query], None)?[0].clone();
        
        let read_txn = self.db.begin_read()?;
        let table = read_txn.open_table(FACTS_TABLE)?;
        
        let mut results = Vec::new();
        
        for entry in table.iter()? {
            let (_key, value) = entry?;
            let val_str = value.value();
            
            // In a production system, you'd store embeddings in a separate table/index.
            // Here we compute on-the-fly or retrieve pre-computed embeddings.
            let val_embedding = self.model.embed(vec![val_str], None)?[0].clone();
            
            let similarity = self.cosine_similarity(&query_embedding, &val_embedding);
            results.push((similarity, val_str.to_string()));
        }

        results.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        Ok(results.into_iter().take(top_k).map(|(_, v)| v).collect())
    }

    fn cosine_similarity(&self, a: &[f32], b: &[f32]) -> f32 {
        let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        dot_product / (norm_a * norm_b)
    }
}
