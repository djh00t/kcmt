//! Provider adapter trait and registry.

use std::collections::HashMap;

pub trait ProviderAdapter: Send + Sync {
    fn id(&self) -> &'static str;
    fn display_name(&self) -> &'static str;
}

#[derive(Default)]
pub struct ProviderRegistry {
    adapters: HashMap<&'static str, Box<dyn ProviderAdapter>>,
}

impl ProviderRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register(&mut self, adapter: Box<dyn ProviderAdapter>) {
        self.adapters.insert(adapter.id(), adapter);
    }

    pub fn get(&self, provider_id: &str) -> Option<&dyn ProviderAdapter> {
        self.adapters.get(provider_id).map(|boxed| boxed.as_ref())
    }

    pub fn ids(&self) -> Vec<&'static str> {
        let mut ids: Vec<&'static str> = self.adapters.keys().copied().collect();
        ids.sort_unstable();
        ids
    }
}
