use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::Arc;
use serde_json::Value;

#[async_trait]
pub trait Tool: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    async fn call(&self, args: Value) -> anyhow::Result<String>;
}

pub struct ToolDispatcher {
    tools: HashMap<String, Arc<dyn Tool>>,
}

impl ToolDispatcher {
    pub fn new() -> Self {
        Self {
            tools: HashMap::new(),
        }
    }

    pub fn register(&mut self, tool: Arc<dyn Tool>) {
        self.tools.insert(tool.name().to_string(), tool);
    }

    pub async fn dispatch(&self, call_str: &str) -> anyhow::Result<String> {
        // Simple parser: ToolName {"arg": "val"}
        let parts: Vec<&str> = call_str.splitn(2, ' ').collect();
        let name = parts[0];
        let args: Value = if parts.len() > 1 {
            serde_json::from_str(parts[1]).unwrap_or(Value::Null)
        } else {
            Value::Null
        };

        if let Some(tool) = self.tools.get(name) {
            tool.call(args).await
        } else {
            Err(anyhow::anyhow!("Tool not found: {}", name))
        }
    }
}

// Example Tool: GetTemperature
pub struct GetTemperature;

#[async_trait]
impl Tool for GetTemperature {
    fn name(&self) -> &str { "GetTemperature" }
    fn description(&self) -> &str { "Returns the current indoor temperature." }
    async fn call(&self, _args: Value) -> anyhow::Result<String> {
        // Mock hardware integration
        Ok("22.5°C".to_string())
    }
}

// Example Tool: SetLight
pub struct SetLight;

#[async_trait]
impl Tool for SetLight {
    fn name(&self) -> &str { "SetLight" }
    fn description(&self) -> &str { "Sets the light state. Args: { \"room\": \"string\", \"brightness\": 0-100, \"color\": \"string\" }" }
    async fn call(&self, args: Value) -> anyhow::Result<String> {
        let room = args["room"].as_str().unwrap_or("unknown");
        let brightness = args["brightness"].as_u64().unwrap_or(100);
        
        // Logic for MQTT or Home Assistant REST API
        tracing::info!("Setting light in {} to {}% brightness", room, brightness);
        
        Ok(format!("Light in {} set to {}%", room, brightness))
    }
}
