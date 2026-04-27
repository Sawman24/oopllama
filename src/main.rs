mod engine;
mod agent;
mod tools;
mod memory;
mod tests;

use crate::engine::InferenceEngine;
use crate::agent::Agent;
use crate::tools::{ToolDispatcher, GetTemperature, SetLight};
use crate::memory::MemoryManager;

use std::sync::Arc;
use tokio::sync::mpsc;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

    // 1. Initialize Engine & Components
    let engine = Arc::new(InferenceEngine::new()?);
    let mut dispatcher = ToolDispatcher::new();
    dispatcher.register(Arc::new(GetTemperature));
    dispatcher.register(Arc::new(SetLight));
    let dispatcher = Arc::new(dispatcher);
    
    let memory = Arc::new(MemoryManager::new("oopllama.redb")?);

    // 2. Concurrency Model: Dedicated Inference Channel
    // Inference is GPU-bound and synchronous within its context; we wrap it.
    let (tx, mut rx) = mpsc::channel::<String>(32);

    // 3. Start Agent Task
    let agent_engine = engine.clone();
    let agent_dispatcher = dispatcher.clone();
    let agent_memory = memory.clone();
    
    tokio::spawn(async move {
        let mut agent = Agent::new(agent_engine, agent_dispatcher, agent_memory);
        
        while let Some(task) = rx.recv().await {
            match agent.run_task(task).await {
                Ok(res) => println!("Agent Output: {}", res),
                Err(e) => eprintln!("Agent Error: {}", e),
            }
        }
    });

    // 4. Telemetry Reporting Loop (Phase 4)
    let telemetry_engine = engine.clone();
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(std::time::Duration::from_secs(30));
        loop {
            interval.tick().await;
            let t = telemetry_engine.get_telemetry();
            tracing::info!("V100 Status: {}MB/{}MB VRAM | {}°C", t.vram_used, t.vram_total, t.temperature);
        }
    });

    // 5. Handle Async Events (Phase 4 & 5)
    println!("Oopllama System Online (V100 SXM2 Optimization Active)");
    
    // Simulate an external webhook event (e.g. from Home Assistant)
    let webhook_tx = tx.clone();
    tokio::spawn(async move {
        tokio::time::sleep(std::time::Duration::from_secs(2)).await;
        let event = "Automation Triggered: Security camera detected motion. What should I do?".to_string();
        webhook_tx.send(event).await.unwrap();
    });

    // Simulate user input
    tx.send("What is the current temperature and should I turn off the lights?".to_string()).await?;

    // Keep system alive
    tokio::signal::ctrl_c().await?;
    Ok(())
}
