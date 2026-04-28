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
        // 1. Context Retrieval (Phase 3)
        let relevant_facts = self.memory.search_relevant_context(&task, 3).await?;
        let context_str = relevant_facts.join("\n");

        {
            let mut state = self.state.lock().await;
            state.current_task = Some(task.clone());
            state.running = true;
            
            let nova_persona = "You are NOVA (Native On-device Virtual Agent), a highly advanced generative conversational AI. \
You use reasoning and probability to determine the best response and actions. \
You must ALWAYS respond using EXACTLY one of these two formats:
1. To use a tool, respond with: 'Thought: [your reasoning here]\nAction: [ToolName]'
2. To speak to the user, respond with: 'Thought: [your reasoning here]\nFinal Answer: [your response]'

Available Tools:
- GetTemperature: Returns the current temperature.
- SetLight: Turns a light on or off.";

            state.history.push(Message {
                role: "system".into(),
                content: format!("Relevant Context:\n{}\n{}", context_str, nova_persona),
            });

            state.history.push(Message {
                role: "user".into(),
                content: task,
            });
        }

        loop {
            // 1. Probabilistic Generation (NOVA's "Brain")
            // Format the prompt using the TinyLlama Chat template
            let mut prompt = String::new();
            for msg in self.state.lock().await.history.iter() {
                prompt.push_str(&format!("<|{}|>\n{}</s>\n", msg.role, msg.content));
            }
            // Prompt the AI to begin its response
            prompt.push_str("<|assistant|>\nThought:");

            let response = self.engine.generate(&prompt)?; 
            
            println!("\n🧠 NOVA: {}", response);

            // 2. ReAct Parser
            if response.contains("Action:") {
                let tool_call = response.split("Action: ").nth(1).unwrap_or("").trim();
                
                // 3. Dispatch Tool (Phase 2)
                let observation = match self.tools.dispatch(tool_call).await {
                    Ok(obs) => obs,
                    Err(e) => format!("Error: {}", e),
                };
                
                println!("🛠️ System: Observation: {}", observation);

                // 4. Memory Storage (Phase 3)
                // Store important observations as facts
                if observation.contains("22.5") {
                    self.memory.store_fact("living_room_temp", &format!("Living room temp is {}", observation))?;
                }

                let mut state = self.state.lock().await;
                state.history.push(Message {
                    role: "system".into(),
                    content: format!("Observation: {}", observation),
                });
            } else if response.contains("Final Answer:") {
                return Ok(response);
            }

            if self.state.lock().await.history.len() > 15 { break; }
        }

        Ok("Task completed.".into())
    }
}
