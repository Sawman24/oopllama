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
                Ok(_) => {}, // Streamed by engine.rs
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

    // 6. Interactive CLI Loop
    use tokio::io::{self, AsyncBufReadExt, BufReader};
    
    println!("\n=========================================================");
    println!("NOVA is online. You can now type messages to talk to the AI.");
    println!("(Press Ctrl+C to exit)");
    println!("=========================================================\n");

    let mut reader = BufReader::new(io::stdin());
    let mut line = String::new();

    loop {
        line.clear();
        let bytes_read = reader.read_line(&mut line).await?;
        if bytes_read == 0 {
            break; // EOF
        }
        let input = line.trim();
        if !input.is_empty() {
            tx.send(input.to_string()).await?;
            // Add a small delay so logs don't immediately overwrite the prompt
            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        }
    }

    Ok(())
}
