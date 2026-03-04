//! Configuration loader honoring precedence:
//! CLI overrides > environment > persisted file > defaults.

use crate::model::WorkflowConfig;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConfigSource {
    Cli,
    Environment,
    Persisted,
    Default,
}

#[derive(Debug, Clone)]
pub struct ConfigLayer {
    pub source: ConfigSource,
    pub value: WorkflowConfig,
}

#[derive(Debug, Clone)]
pub struct ConfigLoader {
    pub layers: Vec<ConfigLayer>,
}

impl ConfigLoader {
    pub fn new() -> Self {
        Self { layers: Vec::new() }
    }

    pub fn with_layer(mut self, source: ConfigSource, value: WorkflowConfig) -> Self {
        self.layers.push(ConfigLayer { source, value });
        self
    }

    pub fn resolve(&self) -> WorkflowConfig {
        self.layers
            .iter()
            .last()
            .map(|layer| layer.value.clone())
            .unwrap_or_default()
    }
}
