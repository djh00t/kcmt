//! Python-vs-Rust parity runner scaffold.

#[derive(Debug, Clone)]
pub struct ParityScenario {
    pub name: String,
    pub command: Vec<String>,
}

pub fn list_default_scenarios() -> Vec<ParityScenario> {
    vec![
        ParityScenario {
            name: "default-kcmt".to_string(),
            command: vec!["kcmt".to_string()],
        },
        ParityScenario {
            name: "status".to_string(),
            command: vec!["kcmt".to_string(), "status".to_string()],
        },
    ]
}
