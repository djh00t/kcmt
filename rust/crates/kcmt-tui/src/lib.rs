//! Optional Ratatui configure shell gated behind terminal checks.

use std::io::{self, IsTerminal};
use std::time::Duration;

use crossterm::event::{self, Event, KeyCode};
use crossterm::execute;
use crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
};
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Direction, Layout};
use ratatui::style::{Modifier, Style};
use ratatui::widgets::{Block, Borders, List, ListItem, Paragraph};
use ratatui::Terminal;

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

#[derive(Debug, Clone)]
pub struct ConfigureTuiState {
    pub provider: String,
    pub model: String,
    pub rule: String,
    pub credential_status: String,
}

pub fn run_configure_tui(state: ConfigureTuiState) -> anyhow::Result<()> {
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    if let Err(err) = execute!(stdout, EnterAlternateScreen) {
        let _ = disable_raw_mode();
        return Err(err.into());
    }
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = match Terminal::new(backend) {
        Ok(terminal) => terminal,
        Err(err) => {
            let _ = disable_raw_mode();
            let _ = execute!(io::stdout(), LeaveAlternateScreen);
            return Err(err.into());
        }
    };
    let result = run_loop(&mut terminal, &state);
    let raw_mode_result = disable_raw_mode();
    let screen_result = execute!(terminal.backend_mut(), LeaveAlternateScreen);
    let cursor_result = terminal.show_cursor();
    raw_mode_result?;
    screen_result?;
    cursor_result?;
    result
}

fn run_loop(
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    state: &ConfigureTuiState,
) -> anyhow::Result<()> {
    let items = [
        "Providers and credentials",
        "Provider model rules",
        "Models and defaults",
        "Prompt profiles",
        "Usage statistics",
        "Save summary",
    ];
    loop {
        terminal.draw(|frame| {
            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints([Constraint::Length(5), Constraint::Min(8), Constraint::Length(3)])
                .split(frame.area());
            let summary = Paragraph::new(format!(
                "Provider: {}\nModel: {}\nRule: {}\nCredentials: {}",
                state.provider, state.model, state.rule, state.credential_status
            ))
            .block(Block::default().title("kcmt configure").borders(Borders::ALL));
            frame.render_widget(summary, chunks[0]);
            let list = List::new(items.iter().map(|item| ListItem::new(*item)).collect::<Vec<_>>())
                .block(Block::default().title("Menu").borders(Borders::ALL))
                .style(Style::default())
                .highlight_style(Style::default().add_modifier(Modifier::BOLD));
            frame.render_widget(list, chunks[1]);
            let footer = Paragraph::new("This v1 configure TUI is read-oriented. Press q or Esc to exit; use CLI flags to save until editable TUI forms land.")
                .block(Block::default().borders(Borders::ALL));
            frame.render_widget(footer, chunks[2]);
        })?;
        if event::poll(Duration::from_millis(250))? {
            if let Event::Key(key) = event::read()? {
                if matches!(key.code, KeyCode::Char('q') | KeyCode::Esc) {
                    break;
                }
            }
        }
    }
    Ok(())
}
