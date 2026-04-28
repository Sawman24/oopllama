use crate::engine::InferenceEngine;
use crate::tools::ToolDispatcher;
use crate::memory::MemoryManager;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::Mutex;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Message {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Default)]
pub struct SystemState {
    pub history: Vec<Message>,
    pub current_task: Option<String>,
    pub running: bool,
}

pub struct Agent {
    pub engine: Arc<InferenceEngine>,
    pub state: Arc<Mutex<SystemState>>,
    pub tools: Arc<ToolDispatcher>,
    pub memory: Arc<MemoryManager>,
}

impl Agent {
    pub fn new(
        engine: Arc<InferenceEngine>, 
        tools: Arc<ToolDispatcher>,
        memory: Arc<MemoryManager>,
    ) -> Self {
        Self {
            engine,
            state: Arc::new(Mutex::new(SystemState::default())),
            tools,
            memory,
        }
    }

    /// The core ReAct loop: Thought -> Action -> Observation
    pub async fn run_task(&mut self, task: String) -> anyhow::Result<String> {
        let relevant_facts = self.memory.search_relevant_context(&task, 3).await?;
        let context_str = relevant_facts.join("\n");
        if self.state.lock().await.history.is_empty() {
            let mut state = self.state.lock().await;
            let nova_persona = "You are NOVA, a helpful AI assistant. You answer questions accurately and concisely. \
NEVER roleplay or simulate a conversation. Only output YOUR response, then stop.";

            let context_block = if context_str.is_empty() {
                String::new()
            } else {
                format!("\nSmart Home Sensor Data:\n{}", context_str)
            };

            state.history.push(Message {
                role: "system".into(),
                content: format!("{}{}", nova_persona, context_block),
            });
        }

        // Add user message
        {
            let mut state = self.state.lock().await;
            state.history.push(Message {
                role: "user".into(),
                content: task.clone(),
            });
        }

        // 1. Format the prompt using the TinyLlama Chat template
        let mut prompt = String::new();
        for msg in self.state.lock().await.history.iter() {
            prompt.push_str(&format!("<|{}|>\n{}</s>\n", msg.role, msg.content));
        }
        prompt.push_str("<|assistant|>\n");

        // 2. Probabilistic Generation
        let response = self.engine.generate(&prompt)?; 
        
        // Trim standard hallucination markers if TinyLlama keeps generating
        let final_answer = response.split("<|user|>").next().unwrap_or(&response).trim();
        let final_answer = final_answer.split("</s>").next().unwrap_or(final_answer).trim();

        // 3. Store AI response in history
        {
            let mut state = self.state.lock().await;
            state.history.push(Message {
                role: "assistant".into(),
                content: final_answer.to_string(),
            });
        }

        Ok(final_answer.to_string())
    }
}
