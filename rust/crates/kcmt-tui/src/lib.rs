//! Optional TUI scaffolding gated behind terminal checks.

use std::io::{self, IsTerminal};

/// Minimal TUI placeholder used until interactive workflows are implemented.
pub struct TuiApp;

impl TuiApp {
    /// Constructs a baseline TUI app shell.
    pub fn new() -> Self {
        Self
    }
}

pub fn should_enable_tui(explicit_no_tui: bool) -> bool {
    if explicit_no_tui {
        return false;
    }
    io::stdin().is_terminal() && io::stdout().is_terminal()
}
