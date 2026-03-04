//! Shared provider traits and client plumbing.

pub mod clients;
pub mod error_map;
pub mod profile;
pub mod registry;
pub mod transport;

/// Placeholder provider list used by setup scaffolding.
pub const SUPPORTED_PROVIDERS: &[&str] = &["openai", "anthropic", "xai", "github"];
