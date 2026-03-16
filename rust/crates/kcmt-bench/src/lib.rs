//! Benchmark models and pipeline orchestration.

pub mod export;
pub mod model;
pub mod quality;
pub mod runner;

/// Placeholder schema version for setup validation.
pub const BENCHMARK_SCHEMA_VERSION: u32 = 1;
pub const RUNTIME_BENCHMARK_SCHEMA_VERSION: &str = "1.0.0";
