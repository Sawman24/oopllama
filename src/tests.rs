#[cfg(test)]
mod tests {
    use crate::engine::InferenceEngine;
    use crate::agent::Agent;
    use crate::tools::{ToolDispatcher, GetTemperature};
    use crate::memory::MemoryManager;
    use std::sync::Arc;

    #[tokio::test]
    async fn test_agent_memory_integration() {
        // Setup components (using mock paths)
        let engine = Arc::new(InferenceEngine::new().unwrap());
        let mut dispatcher = ToolDispatcher::new();
        dispatcher.register(Arc::new(GetTemperature));
        let dispatcher = Arc::new(dispatcher);
        let memory = Arc::new(MemoryManager::new("test_oopllama.redb").unwrap());

        // Store a fact manually
        memory.store_fact("user_pref", "The user likes the room at 22.5°C").unwrap();

        let mut agent = Agent::new(engine, dispatcher, memory);
        
        // Test task
        let result = agent.run_task("What is my temperature preference?".to_string()).await;
        
        assert!(result.is_ok());
        println!("Test Result: {}", result.unwrap());
    }
}
