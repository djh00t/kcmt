//! Optional Ratatui shells and testable state models.

use std::collections::BTreeMap;
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
use serde::{Deserialize, Serialize};

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

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct WorkflowTuiContext {
    pub repo_path: String,
    pub provider: String,
    pub model: String,
    pub mode: String,
    pub total_files: usize,
    pub last_screen: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct WorkflowTuiState {
    pub screen: String,
    pub repo_path: String,
    pub provider: String,
    pub model: String,
    pub mode: String,
    pub current_phase: String,
    pub total_files: usize,
    pub queued: usize,
    pub prepared: usize,
    pub committed: usize,
    pub failed: usize,
    pub push_state: String,
    pub active_file: Option<String>,
    pub files: BTreeMap<String, WorkflowTuiFileState>,
}

impl WorkflowTuiState {
    pub fn new(context: WorkflowTuiContext) -> Self {
        Self {
            screen: context
                .last_screen
                .filter(|screen| !screen.trim().is_empty())
                .unwrap_or_else(|| "workflow".to_string()),
            repo_path: context.repo_path,
            provider: context.provider,
            model: context.model,
            mode: context.mode,
            current_phase: "starting".to_string(),
            total_files: context.total_files,
            queued: 0,
            prepared: 0,
            committed: 0,
            failed: 0,
            push_state: "not triggered".to_string(),
            active_file: None,
            files: BTreeMap::new(),
        }
    }

    pub fn apply(&mut self, event: WorkflowTuiEvent) {
        match event {
            WorkflowTuiEvent::Discovered { total_files } => {
                self.total_files = total_files;
                self.current_phase = "discovered".to_string();
            }
            WorkflowTuiEvent::Queued { file_path, stage } => {
                self.queued += 1;
                self.active_file = Some(file_path.clone());
                self.current_phase = stage.clone();
                let file = self.file_mut(&file_path);
                file.stage = stage;
                file.status = "queued".to_string();
            }
            WorkflowTuiEvent::RequestSent { file_path } => {
                self.active_file = Some(file_path.clone());
                self.current_phase = "llm_wait".to_string();
                let file = self.file_mut(&file_path);
                file.stage = "llm_wait".to_string();
                file.status = "request_sent".to_string();
            }
            WorkflowTuiEvent::Prepared { file_path, subject } => {
                self.prepared += 1;
                self.active_file = Some(file_path.clone());
                self.current_phase = "prepared".to_string();
                let file = self.file_mut(&file_path);
                file.stage = "prepared".to_string();
                file.status = "ready".to_string();
                file.subject = Some(subject);
            }
            WorkflowTuiEvent::PrepareFailed { file_path, error } => {
                self.failed += 1;
                self.active_file = Some(file_path.clone());
                self.current_phase = "failed".to_string();
                let file = self.file_mut(&file_path);
                file.stage = "prepare_failed".to_string();
                file.status = "failed".to_string();
                file.error = Some(error);
            }
            WorkflowTuiEvent::CommitStarted { file_path } => {
                self.active_file = Some(file_path.clone());
                self.current_phase = "commit".to_string();
                let file = self.file_mut(&file_path);
                file.stage = "commit".to_string();
                file.status = "in_progress".to_string();
            }
            WorkflowTuiEvent::CommitSucceeded {
                file_path,
                subject,
                commit_hash,
            } => {
                self.committed += 1;
                self.active_file = Some(file_path.clone());
                self.current_phase = "committed".to_string();
                let file = self.file_mut(&file_path);
                file.stage = "done".to_string();
                file.status = "committed".to_string();
                file.subject = Some(subject);
                file.commit_hash = commit_hash;
            }
            WorkflowTuiEvent::CommitFailed { file_path, error } => {
                self.failed += 1;
                self.active_file = Some(file_path.clone());
                self.current_phase = "failed".to_string();
                let file = self.file_mut(&file_path);
                file.stage = "commit_failed".to_string();
                file.status = "failed".to_string();
                file.error = Some(error);
            }
            WorkflowTuiEvent::PushStarted => {
                self.current_phase = "push".to_string();
                self.push_state = "in_progress".to_string();
            }
            WorkflowTuiEvent::PushFinished { state } => {
                self.current_phase = "push".to_string();
                self.push_state = state;
            }
            WorkflowTuiEvent::Finished => {
                self.current_phase = if self.failed > 0 {
                    "complete_with_failures".to_string()
                } else {
                    "complete".to_string()
                };
                self.active_file = None;
            }
        }
    }

    pub fn render_lines(&self) -> Vec<String> {
        let mut lines = vec![
            format!(
                "kcmt workflow [{}] provider={} model={}",
                self.mode, self.provider, self.model
            ),
            format!(
                "phase={} files={} queued={} prepared={} committed={} failed={} push={}",
                self.current_phase,
                self.total_files,
                self.queued,
                self.prepared,
                self.committed,
                self.failed,
                self.push_state
            ),
        ];
        if let Some(active_file) = &self.active_file {
            lines.push(format!("active={active_file}"));
        }
        lines.extend(self.files.iter().map(|(path, file)| {
            let subject = file.subject.as_deref().unwrap_or("-");
            let error = file.error.as_deref().unwrap_or("-");
            format!(
                "{}\t{}\t{}\t{}\t{}",
                path, file.stage, file.status, subject, error
            )
        }));
        lines
    }

    pub fn to_json_line(&self) -> anyhow::Result<String> {
        Ok(serde_json::to_string(self)?)
    }

    fn file_mut(&mut self, path: &str) -> &mut WorkflowTuiFileState {
        self.files
            .entry(path.to_string())
            .or_insert_with(|| WorkflowTuiFileState {
                stage: "pending".to_string(),
                status: "pending".to_string(),
                subject: None,
                commit_hash: None,
                error: None,
            })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct WorkflowTuiFileState {
    pub stage: String,
    pub status: String,
    pub subject: Option<String>,
    pub commit_hash: Option<String>,
    pub error: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WorkflowTuiEvent {
    Discovered {
        total_files: usize,
    },
    Queued {
        file_path: String,
        stage: String,
    },
    RequestSent {
        file_path: String,
    },
    Prepared {
        file_path: String,
        subject: String,
    },
    PrepareFailed {
        file_path: String,
        error: String,
    },
    CommitStarted {
        file_path: String,
    },
    CommitSucceeded {
        file_path: String,
        subject: String,
        commit_hash: Option<String>,
    },
    CommitFailed {
        file_path: String,
        error: String,
    },
    PushStarted,
    PushFinished {
        state: String,
    },
    Finished,
}

#[derive(Debug, Clone)]
pub struct ConfigureTuiState {
    pub provider: String,
    pub model: String,
    pub rule: String,
    pub credential_status: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConfigureTuiOutcome {
    Save,
    Cancel,
}

pub fn run_configure_tui(state: ConfigureTuiState) -> anyhow::Result<ConfigureTuiOutcome> {
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
) -> anyhow::Result<ConfigureTuiOutcome> {
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
                .constraints([
                    Constraint::Length(5),
                    Constraint::Min(8),
                    Constraint::Length(3),
                ])
                .split(frame.area());
            let summary = Paragraph::new(format!(
                "Provider: {}\nModel: {}\nRule: {}\nCredentials: {}",
                state.provider, state.model, state.rule, state.credential_status
            ))
            .block(
                Block::default()
                    .title("kcmt configure")
                    .borders(Borders::ALL),
            );
            frame.render_widget(summary, chunks[0]);
            let list = List::new(
                items
                    .iter()
                    .map(|item| ListItem::new(*item))
                    .collect::<Vec<_>>(),
            )
            .block(Block::default().title("Menu").borders(Borders::ALL))
            .style(Style::default())
            .highlight_style(Style::default().add_modifier(Modifier::BOLD));
            frame.render_widget(list, chunks[1]);
            let footer = Paragraph::new("Press s to save this configuration, or q/Esc to cancel.")
                .block(Block::default().borders(Borders::ALL));
            frame.render_widget(footer, chunks[2]);
        })?;
        if event::poll(Duration::from_millis(250))? {
            if let Event::Key(key) = event::read()? {
                if matches!(key.code, KeyCode::Char('s') | KeyCode::Enter) {
                    return Ok(ConfigureTuiOutcome::Save);
                }
                if matches!(key.code, KeyCode::Char('q') | KeyCode::Esc) {
                    return Ok(ConfigureTuiOutcome::Cancel);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{ConfigureTuiOutcome, ConfigureTuiState};

    #[test]
    fn configure_tui_state_names_save_and_cancel_outcomes() {
        let state = ConfigureTuiState {
            provider: "anthropic".to_string(),
            model: "claude-test".to_string(),
            rule: "provider presets enabled".to_string(),
            credential_status: "keychain first, environment fallback".to_string(),
        };

        let outcome = ConfigureTuiOutcome::Save;

        assert_eq!(state.provider, "anthropic");
        assert!(matches!(outcome, ConfigureTuiOutcome::Save));
        assert_ne!(outcome, ConfigureTuiOutcome::Cancel);
    }
}

#[cfg(test)]
mod tests {
    use super::{WorkflowTuiContext, WorkflowTuiEvent, WorkflowTuiState};

    fn context(total_files: usize) -> WorkflowTuiContext {
        WorkflowTuiContext {
            repo_path: "/repo".to_string(),
            provider: "openai".to_string(),
            model: "gpt-test".to_string(),
            mode: "direct".to_string(),
            total_files,
            last_screen: None,
        }
    }

    #[test]
    fn workflow_model_tracks_commit_lifecycle() {
        let mut state = WorkflowTuiState::new(context(1));
        state.apply(WorkflowTuiEvent::Discovered { total_files: 1 });
        state.apply(WorkflowTuiEvent::Queued {
            file_path: "alpha.py".to_string(),
            stage: "diff".to_string(),
        });
        state.apply(WorkflowTuiEvent::RequestSent {
            file_path: "alpha.py".to_string(),
        });
        state.apply(WorkflowTuiEvent::Prepared {
            file_path: "alpha.py".to_string(),
            subject: "fix(alpha): update alpha".to_string(),
        });
        state.apply(WorkflowTuiEvent::CommitStarted {
            file_path: "alpha.py".to_string(),
        });
        state.apply(WorkflowTuiEvent::CommitSucceeded {
            file_path: "alpha.py".to_string(),
            subject: "fix(alpha): update alpha".to_string(),
            commit_hash: Some("abcdef1".to_string()),
        });
        state.apply(WorkflowTuiEvent::PushFinished {
            state: "not triggered".to_string(),
        });
        state.apply(WorkflowTuiEvent::Finished);

        assert_eq!(state.current_phase, "complete");
        assert_eq!(state.queued, 1);
        assert_eq!(state.prepared, 1);
        assert_eq!(state.committed, 1);
        assert_eq!(state.failed, 0);
        let file = state.files.get("alpha.py").expect("file state");
        assert_eq!(file.stage, "done");
        assert_eq!(file.status, "committed");
        assert_eq!(file.commit_hash.as_deref(), Some("abcdef1"));
        assert!(state.render_lines()[1].contains("committed=1"));
    }

    #[test]
    fn workflow_model_tracks_prepare_and_commit_failures_by_path() {
        let mut state = WorkflowTuiState::new(context(2));
        state.apply(WorkflowTuiEvent::PrepareFailed {
            file_path: "alpha.py".to_string(),
            error: "missing conventional commit header".to_string(),
        });
        state.apply(WorkflowTuiEvent::Prepared {
            file_path: "beta.py".to_string(),
            subject: "fix(beta): update beta".to_string(),
        });
        state.apply(WorkflowTuiEvent::CommitStarted {
            file_path: "beta.py".to_string(),
        });
        state.apply(WorkflowTuiEvent::CommitFailed {
            file_path: "beta.py".to_string(),
            error: "index lock".to_string(),
        });
        state.apply(WorkflowTuiEvent::Finished);

        assert_eq!(state.current_phase, "complete_with_failures");
        assert_eq!(state.failed, 2);
        assert_eq!(
            state.files["alpha.py"].error.as_deref(),
            Some("missing conventional commit header")
        );
        assert_eq!(state.files["beta.py"].stage, "commit_failed");
    }

    #[test]
    fn workflow_model_uses_persisted_last_screen() {
        let mut context = context(0);
        context.last_screen = Some("workflow".to_string());

        let state = WorkflowTuiState::new(context);

        assert_eq!(state.screen, "workflow");
    }
}
