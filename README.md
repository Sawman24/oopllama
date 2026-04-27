# Oopllama: Bare-Metal V100 AI Agent

Oopllama is a unified, high-performance AI agent system written in Rust, optimized for NVIDIA Tesla V100 (SXM2) hardware. It operates as an autonomous home assistant with manual KV Cache management, embedded vector memory, and a ReAct-based tool dispatcher.

## Key Features
- **Proprietary Inference Pipeline**: Built with `candle-core` for direct CUDA utilization.
- **Manual KV Cache**: Explicit tensor management for memory-efficient context persistence.
- **Embedded Memory**: Persistent fact storage using `redb` and semantic search via `fastembed`.
- **ReAct Controller**: Async agentic loop for complex task reasoning and hardware orchestration.
- **Hardware Telemetry**: Real-time VRAM and temperature monitoring for V100.

## Architecture
- `InferenceEngine`: CUDA-resident transformer forward pass.
- `MemoryManager`: Vector RAG and state persistence.
- `ToolDispatcher`: Type-safe home automation traits.

## Deployment (Docker)
This project is designed to run in a CUDA-enabled container.

### Prerequisites
- NVIDIA Tesla V100 (SXM2)
- NVIDIA Container Toolkit
- Docker & Docker Compose

### Quick Start
1. Clone the repo.
2. Place your `.safetensors` model weights in `./models`.
3. Run:
   ```bash
   docker-compose up --build
   ```

## Development
- `cargo build --release` (Requires CUDA toolkit installed on host).
- `cargo test` (Runs integration suite).
