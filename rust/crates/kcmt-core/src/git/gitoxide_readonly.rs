//! Experimental read-only adapter surface for future gitoxide integration.
//!
//! This module is intentionally feature-gated at call sites and should not replace
//! parity-critical `git` CLI paths until explicit validation gates pass.

#[derive(Debug, Clone, Default)]
pub struct GitoxideReadonlyAdapter;

impl GitoxideReadonlyAdapter {
    pub fn describe_capabilities(&self) -> &'static str {
        "read-only repository metadata and diff pre-processing"
    }
}
