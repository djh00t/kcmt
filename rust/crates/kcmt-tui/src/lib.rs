//! Optional Ratatui shells and testable state models.

use std::collections::BTreeMap;
use std::io::{self, IsTerminal};
use std::sync::{
    atomic::{AtomicBool, Ordering as AtomicOrdering},
    Arc, Mutex,
};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use crossterm::event::{self, Event, KeyCode};
use crossterm::execute;
use crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
};
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Alignment, Constraint, Direction, Layout};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, BorderType, Borders, List, ListItem, Paragraph, Wrap};
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkflowTuiCounter {
    pub label: &'static str,
    pub value: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkflowTuiFileRow {
    pub path: String,
    pub stage: String,
    pub status: String,
    pub subject: Option<String>,
    pub commit_hash: Option<String>,
    pub error: Option<String>,
    pub progress_percent: u16,
    pub is_active: bool,
    pub is_done: bool,
    pub is_failed: bool,
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
    pub diffs_collected: usize,
    pub requests_sent: usize,
    pub responses_received: usize,
    pub queued: usize,
    pub prepared: usize,
    pub committed: usize,
    pub failed: usize,
    pub push_state: String,
    pub active_file: Option<String>,
    pub files: BTreeMap<String, WorkflowTuiFileState>,
    #[serde(skip, default)]
    next_sequence: usize,
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
            diffs_collected: 0,
            requests_sent: 0,
            responses_received: 0,
            queued: 0,
            prepared: 0,
            committed: 0,
            failed: 0,
            push_state: "not triggered".to_string(),
            active_file: None,
            files: BTreeMap::new(),
            next_sequence: 0,
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
                self.diffs_collected += 1;
                self.active_file = Some(file_path.clone());
                self.current_phase = stage.clone();
                let file = self.file_mut(&file_path);
                file.stage = stage;
                file.status = "queued".to_string();
            }
            WorkflowTuiEvent::RequestSent { file_path } => {
                self.requests_sent += 1;
                self.active_file = Some(file_path.clone());
                self.current_phase = "llm_wait".to_string();
                let file = self.file_mut(&file_path);
                file.stage = "llm_wait".to_string();
                file.status = "request_sent".to_string();
            }
            WorkflowTuiEvent::Prepared { file_path, subject } => {
                self.responses_received += 1;
                self.prepared += 1;
                self.active_file = Some(file_path.clone());
                self.current_phase = "prepared".to_string();
                let file = self.file_mut(&file_path);
                file.stage = "prepared".to_string();
                file.status = "ready".to_string();
                file.subject = Some(subject);
            }
            WorkflowTuiEvent::PrepareFailed { file_path, error } => {
                self.responses_received += 1;
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

    pub fn stage_counters(&self) -> Vec<WorkflowTuiCounter> {
        vec![
            WorkflowTuiCounter {
                label: "Files discovered",
                value: self.total_files.to_string(),
            },
            WorkflowTuiCounter {
                label: "Diffs collected",
                value: self.diffs_collected.to_string(),
            },
            WorkflowTuiCounter {
                label: "LLM requests sent",
                value: self.requests_sent.to_string(),
            },
            WorkflowTuiCounter {
                label: "LLM responses received",
                value: self.responses_received.to_string(),
            },
            WorkflowTuiCounter {
                label: "Messages prepared",
                value: self.prepared.to_string(),
            },
            WorkflowTuiCounter {
                label: "Commits in progress",
                value: self.commit_rows().len().to_string(),
            },
            WorkflowTuiCounter {
                label: "Commits completed",
                value: self.committed.to_string(),
            },
            WorkflowTuiCounter {
                label: "Failures",
                value: self.failed.to_string(),
            },
            WorkflowTuiCounter {
                label: "Push",
                value: humanize_push_state(&self.push_state).to_string(),
            },
        ]
    }

    pub fn ordered_files(&self) -> Vec<WorkflowTuiFileRow> {
        let mut rows = self
            .files
            .iter()
            .map(|(path, file)| {
                let is_done = file.stage == "done";
                let is_failed = file.stage.contains("failed");
                let is_active = matches!(
                    file.stage.as_str(),
                    "diff" | "llm_wait" | "prepared" | "commit"
                );
                WorkflowTuiFileRow {
                    path: path.clone(),
                    stage: file.stage.clone(),
                    status: file.status.clone(),
                    subject: file.subject.clone(),
                    commit_hash: file.commit_hash.clone(),
                    error: file.error.clone(),
                    progress_percent: workflow_stage_progress_percent(&file.stage),
                    is_active,
                    is_done,
                    is_failed,
                }
            })
            .collect::<Vec<_>>();
        rows.sort_by(|left, right| {
            let rank_left = workflow_file_rank(left);
            let rank_right = workflow_file_rank(right);
            rank_left
                .cmp(&rank_right)
                .then_with(|| left.path.cmp(&right.path))
        });
        rows
    }

    pub fn active_now(&self) -> Option<String> {
        let active_file = self.active_file.as_ref()?;
        let row = self.files.get(active_file)?;
        let label = workflow_stage_label_text(&row.stage);
        Some(format!("Active now: {label} on {active_file}"))
    }

    pub fn viewport_summary(&self, visible_count: usize, total_visible: usize) -> Option<String> {
        if total_visible == 0 {
            None
        } else {
            Some(format!(
                "Visible {} of {} files",
                visible_count, total_visible
            ))
        }
    }

    pub fn footer_status(&self) -> String {
        if humanize_push_state(&self.push_state) == "Pushing" {
            return "Pushing commits to remote".to_string();
        }
        if self.current_phase.starts_with("complete") {
            if self.failed > 0 {
                return format!("Run completed with {} failures", self.failed);
            }
            return "Run complete".to_string();
        }
        let pending_requests = self.requests_sent.saturating_sub(self.responses_received);
        if pending_requests > 0 {
            return format!("Waiting on {} LLM requests", pending_requests);
        }
        let commits_in_progress = self.commit_rows().len();
        if commits_in_progress > 0 {
            return format!("Writing {} commits", commits_in_progress);
        }
        let pending_diffs = self.total_files.saturating_sub(self.diffs_collected);
        if pending_diffs > 0 {
            return format!("Collecting diffs for {} files", pending_diffs);
        }
        "Waiting for workflow activity".to_string()
    }

    pub fn overall_progress_pct(&self) -> u16 {
        let total = self.total_files.max(1) as f64;
        let mut pct = 0.0;
        pct += (self.diffs_collected.min(self.total_files) as f64 / total) * 20.0;
        pct += (self.prepared.min(self.total_files) as f64 / total) * 40.0;
        pct += ((self.committed + self.commit_rows().len()).min(self.total_files) as f64 / total)
            * 35.0;
        if humanize_push_state(&self.push_state) == "Pushing" {
            pct += 2.5;
        } else if self.current_phase.starts_with("complete")
            || (self.committed + self.failed >= self.total_files && self.total_files > 0)
        {
            pct += 5.0;
        }
        if self.current_phase.starts_with("complete") {
            100
        } else {
            pct.clamp(0.0, 100.0).round() as u16
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
        if !self.files.contains_key(path) {
            let sequence = self.next_sequence;
            self.next_sequence += 1;
            self.files.insert(
                path.to_string(),
                WorkflowTuiFileState {
                    stage: "pending".to_string(),
                    status: "pending".to_string(),
                    subject: None,
                    commit_hash: None,
                    error: None,
                    sequence,
                },
            );
        }
        self.files.get_mut(path).expect("file state should exist")
    }

    fn commit_rows(&self) -> Vec<&WorkflowTuiFileState> {
        self.files
            .values()
            .filter(|file| file.stage == "commit")
            .collect()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct WorkflowTuiFileState {
    pub stage: String,
    pub status: String,
    pub subject: Option<String>,
    pub commit_hash: Option<String>,
    pub error: Option<String>,
    #[serde(skip, default)]
    sequence: usize,
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

#[derive(Debug)]
pub struct WorkflowTuiSession {
    stop: Arc<AtomicBool>,
    handle: Option<JoinHandle<anyhow::Result<()>>>,
}

impl Drop for WorkflowTuiSession {
    fn drop(&mut self) {
        self.stop.store(true, AtomicOrdering::SeqCst);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

pub fn spawn_workflow_tui(
    state: Arc<Mutex<WorkflowTuiState>>,
) -> anyhow::Result<WorkflowTuiSession> {
    let stop = Arc::new(AtomicBool::new(false));
    let worker_stop = Arc::clone(&stop);
    let handle = thread::Builder::new()
        .name("kcmt-workflow-tui".to_string())
        .spawn(move || run_workflow_tui_loop(state, worker_stop))?;
    Ok(WorkflowTuiSession {
        stop,
        handle: Some(handle),
    })
}

fn run_workflow_tui_loop(
    state: Arc<Mutex<WorkflowTuiState>>,
    stop: Arc<AtomicBool>,
) -> anyhow::Result<()> {
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

    let result = run_workflow_tui_loop_inner(&mut terminal, state, stop);
    let raw_mode_result = disable_raw_mode();
    let screen_result = execute!(terminal.backend_mut(), LeaveAlternateScreen);
    let cursor_result = terminal.show_cursor();
    raw_mode_result?;
    screen_result?;
    cursor_result?;
    result
}

fn run_workflow_tui_loop_inner(
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    state: Arc<Mutex<WorkflowTuiState>>,
    stop: Arc<AtomicBool>,
) -> anyhow::Result<()> {
    let mut scroll_offset = 0usize;
    loop {
        let snapshot = state
            .lock()
            .map_err(|err| anyhow::anyhow!("workflow TUI state lock poisoned: {err}"))?
            .clone();
        let terminal_rows = terminal.size()?.height as usize;
        let visible_count = workflow_visible_file_count(terminal_rows);
        let files = snapshot.ordered_files();
        let max_scroll = files.len().saturating_sub(visible_count);
        scroll_offset = scroll_offset.min(max_scroll);
        terminal.draw(|frame| {
            render_workflow_screen(frame, &snapshot, &files, scroll_offset, visible_count)
        })?;

        if snapshot.current_phase.starts_with("complete") {
            return Ok(());
        }
        if stop.load(AtomicOrdering::SeqCst) {
            return Ok(());
        }
        if event::poll(Duration::from_millis(120))? {
            if let Event::Key(key) = event::read()? {
                if key.code == KeyCode::Char('q')
                    || key.code == KeyCode::Esc
                    || (key.code == KeyCode::Char('c')
                        && key
                            .modifiers
                            .contains(crossterm::event::KeyModifiers::CONTROL))
                {
                    stop.store(true, AtomicOrdering::SeqCst);
                    return Ok(());
                }
                match key.code {
                    KeyCode::Down | KeyCode::Char('j') => {
                        scroll_offset = scroll_offset.saturating_add(1).min(max_scroll);
                    }
                    KeyCode::Up | KeyCode::Char('k') => {
                        scroll_offset = scroll_offset.saturating_sub(1);
                    }
                    KeyCode::PageDown => {
                        scroll_offset = scroll_offset.saturating_add(visible_count).min(max_scroll);
                    }
                    KeyCode::PageUp => {
                        scroll_offset = scroll_offset.saturating_sub(visible_count);
                    }
                    KeyCode::Home => {
                        scroll_offset = 0;
                    }
                    KeyCode::End | KeyCode::Char('G') => {
                        scroll_offset = max_scroll;
                    }
                    KeyCode::Char('g') => {
                        if !key
                            .modifiers
                            .contains(crossterm::event::KeyModifiers::SHIFT)
                        {
                            scroll_offset = 0;
                        }
                    }
                    _ => {}
                }
            }
        }
    }
}

fn workflow_visible_file_count(terminal_rows: usize) -> usize {
    let used_rows = 6 + 5 + 4 + 2;
    (terminal_rows.saturating_sub(used_rows) / 2).max(1)
}

fn render_workflow_screen(
    frame: &mut ratatui::Frame<'_>,
    state: &WorkflowTuiState,
    files: &[WorkflowTuiFileRow],
    scroll_offset: usize,
    visible_count: usize,
) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(6),
            Constraint::Length(5),
            Constraint::Min(10),
            Constraint::Length(4),
        ])
        .split(frame.area());

    let header = Paragraph::new(vec![
        Line::from(vec![
            Span::styled(
                "kcmt",
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                " workflow",
                Style::default()
                    .fg(Color::White)
                    .add_modifier(Modifier::BOLD),
            ),
        ]),
        Line::from(Span::styled(
            format!(
                "provider={}  model={}  repo={}",
                state.provider, state.model, state.repo_path
            ),
            Style::default().fg(Color::DarkGray),
        )),
        Line::from(Span::styled(
            format!(
                "phase={}  files={}  push={}",
                workflow_phase_label(state),
                state.total_files,
                humanize_push_state(&state.push_state)
            ),
            Style::default().fg(Color::DarkGray),
        )),
        if let Some(active_now) = state.active_now() {
            Line::from(Span::styled(active_now, Style::default().fg(Color::Cyan)))
        } else {
            Line::from(Span::styled(
                "Waiting for workflow activity",
                Style::default().fg(Color::DarkGray),
            ))
        },
    ])
    .alignment(Alignment::Center)
    .block(
        Block::default()
            .title("Workflow")
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .border_style(Style::default().fg(Color::Cyan)),
    );
    frame.render_widget(header, chunks[0]);

    let counters = state.stage_counters();
    let counter_lines = counters
        .chunks(3)
        .map(|group| {
            let mut spans = Vec::new();
            for (index, counter) in group.iter().enumerate() {
                if index > 0 {
                    spans.push(Span::styled(" │ ", Style::default().fg(Color::DarkGray)));
                }
                spans.push(Span::styled(
                    format!("{:<24}", counter.label),
                    Style::default()
                        .fg(Color::DarkGray)
                        .add_modifier(Modifier::BOLD),
                ));
                spans.push(Span::raw(" "));
                spans.push(counter_value_span(counter));
            }
            Line::from(spans)
        })
        .collect::<Vec<_>>();

    let counters_panel = Paragraph::new(counter_lines)
        .wrap(Wrap { trim: true })
        .block(
            Block::default()
                .title("Stage counters")
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded)
                .border_style(Style::default().fg(Color::Magenta)),
        );
    frame.render_widget(counters_panel, chunks[1]);

    let list_area = chunks[2];
    let list_visible_count = visible_count.max(1).min(files.len().max(1));
    let max_scroll = files.len().saturating_sub(list_visible_count);
    let clamped_scroll = scroll_offset.min(max_scroll);
    let visible_files = files
        .iter()
        .skip(clamped_scroll)
        .take(list_visible_count)
        .collect::<Vec<_>>();

    let file_items = if visible_files.is_empty() {
        vec![ListItem::new(vec![Line::from(Span::styled(
            "Waiting for workflow activity...",
            Style::default().fg(Color::DarkGray),
        ))])]
    } else {
        visible_files
            .into_iter()
            .map(|row| render_workflow_file_item(row))
            .collect::<Vec<_>>()
    };

    let file_list = List::new(file_items).block(
        Block::default()
            .title(format!(
                "Files {}-{} of {}",
                if files.is_empty() {
                    0
                } else {
                    clamped_scroll + 1
                },
                if files.is_empty() {
                    0
                } else {
                    (clamped_scroll + list_visible_count).min(files.len())
                },
                files.len()
            ))
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .border_style(Style::default().fg(Color::Green)),
    );
    frame.render_widget(file_list, list_area);

    let progress_bar = workflow_progress_bar(state.overall_progress_pct(), list_area.width.into());
    let footer = Paragraph::new(vec![
        Line::from(vec![
            Span::styled(progress_bar, Style::default().fg(Color::Green)),
            Span::raw(" "),
            Span::styled(
                format!(
                    "{:>3}% {}",
                    state.overall_progress_pct(),
                    workflow_phase_label(state)
                ),
                Style::default().fg(Color::DarkGray),
            ),
        ]),
        Line::from(vec![Span::styled(
            "j/k, arrows, PgUp/PgDn, g/G to scroll | q or Esc to exit",
            Style::default().fg(Color::DarkGray),
        )]),
    ])
    .alignment(Alignment::Center)
    .wrap(Wrap { trim: true })
    .block(
        Block::default()
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .border_style(Style::default().fg(Color::DarkGray)),
    );
    frame.render_widget(footer, chunks[3]);
}

fn render_workflow_file_item(row: &WorkflowTuiFileRow) -> ListItem<'static> {
    let highlight = if row.is_active {
        Color::Cyan
    } else if row.is_failed {
        Color::Red
    } else if row.is_done {
        Color::Green
    } else {
        Color::White
    };
    let stage_style = Style::default().fg(workflow_stage_color(&row.stage));
    let mut top_spans = vec![
        Span::styled(
            row.path.clone(),
            Style::default()
                .fg(highlight)
                .add_modifier(if row.is_active {
                    Modifier::BOLD
                } else {
                    Modifier::empty()
                }),
        ),
        Span::raw(" "),
        Span::styled(
            format!("[{}]", workflow_stage_label_text(&row.stage)),
            stage_style.add_modifier(Modifier::BOLD),
        ),
        Span::raw(" "),
        Span::styled(
            format!("{:>3}%", row.progress_percent),
            stage_style.add_modifier(Modifier::BOLD),
        ),
    ];
    if let Some(hash) = row.commit_hash.as_deref() {
        top_spans.push(Span::raw(" "));
        top_spans.push(Span::styled(
            truncate_hash(hash),
            Style::default().fg(Color::DarkGray),
        ));
    }
    let detail = if let Some(error) = row.error.as_deref() {
        Span::styled(error.to_string(), Style::default().fg(Color::Red))
    } else if let Some(subject) = row.subject.as_deref() {
        Span::styled(subject.to_string(), Style::default().fg(Color::Green))
    } else {
        Span::styled(
            workflow_stage_label_text(&row.stage).to_string(),
            Style::default().fg(Color::DarkGray),
        )
    };
    ListItem::new(vec![
        Line::from(top_spans),
        Line::from(vec![
            Span::styled("  ", Style::default().fg(Color::DarkGray)),
            detail,
        ]),
    ])
}

fn counter_value_span(counter: &WorkflowTuiCounter) -> Span<'static> {
    let value_style = match counter.label {
        "Files discovered" => Style::default().fg(Color::Cyan),
        "Diffs collected" => Style::default().fg(Color::Blue),
        "LLM requests sent" | "LLM responses received" => Style::default().fg(Color::Magenta),
        "Messages prepared" | "Commits completed" => Style::default().fg(Color::Green),
        "Commits in progress" => Style::default().fg(Color::Yellow),
        "Failures" => Style::default().fg(Color::Red),
        "Push" => push_state_style(&counter.value),
        _ => Style::default().fg(Color::White),
    };
    Span::styled(counter.value.clone(), value_style)
}

fn push_state_style(value: &str) -> Style {
    match value {
        "Pushing" => Style::default().fg(Color::Yellow),
        "Done" => Style::default().fg(Color::Green),
        "Failed" => Style::default().fg(Color::Red),
        _ => Style::default().fg(Color::DarkGray),
    }
}

fn workflow_stage_color(stage: &str) -> Color {
    match stage {
        "diff" => Color::Blue,
        "llm_wait" => Color::Cyan,
        "prepared" => Color::Green,
        "commit" => Color::Yellow,
        "done" => Color::Green,
        stage if stage.contains("failed") => Color::Red,
        _ => Color::DarkGray,
    }
}

fn workflow_stage_label_text(stage: &str) -> &'static str {
    match stage {
        "pending" => "pending",
        "diff" => "collecting diff",
        "llm_wait" => "waiting for LLM response",
        "prepared" => "message prepared",
        "commit" => "writing commit",
        "done" => "commit complete",
        "prepare_failed" => "prepare failed",
        "commit_failed" => "commit failed",
        stage if stage.contains("failed") => "failed",
        _ => "pending",
    }
}

fn workflow_stage_label(state: &WorkflowTuiState) -> String {
    let active_stage = state
        .active_file
        .as_ref()
        .and_then(|path| state.files.get(path))
        .map(|file| file.stage.as_str());
    match state.current_phase.as_str() {
        "push" if humanize_push_state(&state.push_state) == "Pushing" => "PUSH".to_string(),
        phase if phase.starts_with("complete") => "COMPLETE".to_string(),
        "commit" | "committed" => "COMMIT".to_string(),
        "prepared" | "llm_wait" | "prepare_failed" => "PREPARE".to_string(),
        "diff" | "discovered" | "starting" => "DIFF".to_string(),
        "failed" => match active_stage {
            Some("commit_failed") | Some("commit") => "COMMIT".to_string(),
            Some("prepare_failed") | Some("llm_wait") | Some("prepared") => "PREPARE".to_string(),
            Some("diff") => "DIFF".to_string(),
            _ => "FAILED".to_string(),
        },
        _ => state.current_phase.to_ascii_uppercase(),
    }
}

fn workflow_phase_label(state: &WorkflowTuiState) -> String {
    workflow_stage_label(state)
}

fn humanize_push_state(push_state: &str) -> &'static str {
    match push_state.to_ascii_lowercase().as_str() {
        "in_progress" | "pushing" => "Pushing",
        "done" | "pushed" | "success" => "Done",
        "error" | "failed" => "Failed",
        _ => "Idle",
    }
}

fn workflow_stage_progress_percent(stage: &str) -> u16 {
    match stage {
        "pending" => 0,
        "diff" => 20,
        "llm_wait" => 45,
        "prepared" => 80,
        "commit" => 90,
        "done" => 100,
        stage if stage.contains("failed") => 100,
        _ => 0,
    }
}

fn workflow_file_rank(row: &WorkflowTuiFileRow) -> usize {
    if row.is_active {
        match row.stage.as_str() {
            "commit" => 0,
            "prepared" => 1,
            "llm_wait" => 2,
            "diff" => 3,
            _ => 0,
        }
    } else if row.is_failed {
        4
    } else if row.is_done {
        5
    } else {
        6
    }
}

fn workflow_progress_bar(percent: u16, width: usize) -> String {
    let width = width.saturating_sub(10).clamp(10, 36);
    let filled = ((percent as usize * width) + 50) / 100;
    let empty = width.saturating_sub(filled);
    format!("{}{}", "█".repeat(filled), "░".repeat(empty))
}

fn truncate_hash(hash: &str) -> String {
    hash.chars().take(7).collect()
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
                    Constraint::Min(10),
                    Constraint::Length(4),
                ])
                .split(frame.area());
            let header = Paragraph::new(vec![
                Line::from(vec![
                    Span::styled(
                        "kcmt",
                        Style::default()
                            .fg(Color::Cyan)
                            .add_modifier(Modifier::BOLD),
                    ),
                    Span::styled(
                        " configure",
                        Style::default()
                            .fg(Color::White)
                            .add_modifier(Modifier::BOLD),
                    ),
                ]),
                Line::from(Span::styled(
                    "Rust interactive setup",
                    Style::default().fg(Color::DarkGray),
                )),
                Line::from(Span::styled(
                    "Provider presets, model selection, and credential routing",
                    Style::default().fg(Color::DarkGray),
                )),
            ])
            .alignment(Alignment::Center)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_type(BorderType::Rounded)
                    .border_style(Style::default().fg(Color::Cyan)),
            );
            frame.render_widget(header, chunks[0]);

            let body_chunks = Layout::default()
                .direction(Direction::Horizontal)
                .constraints([Constraint::Percentage(46), Constraint::Percentage(54)])
                .split(chunks[1]);

            let summary = Paragraph::new(vec![
                kv_line("Provider", &state.provider, Color::Cyan),
                kv_line("Model", &state.model, Color::Green),
                kv_line("Rule", &state.rule, Color::Magenta),
                kv_line("Credentials", &state.credential_status, Color::Yellow),
                Line::from(""),
                Line::from(Span::styled(
                    "This screen mirrors the Python setup flow:",
                    Style::default().add_modifier(Modifier::BOLD),
                )),
                Line::from("providers, model defaults, credentials, and save summary"),
            ])
            .wrap(Wrap { trim: true })
            .block(
                Block::default()
                    .title("Current selection")
                    .borders(Borders::ALL)
                    .border_type(BorderType::Rounded),
            );
            frame.render_widget(summary, body_chunks[0]);

            let list = List::new(
                items
                    .iter()
                    .enumerate()
                    .map(|(idx, item)| {
                        let is_active = idx == 0;
                        let prefix = if is_active { "❯ " } else { "  " };
                        ListItem::new(Line::from(vec![
                            Span::styled(
                                prefix,
                                Style::default().fg(if is_active {
                                    Color::Cyan
                                } else {
                                    Color::DarkGray
                                }),
                            ),
                            Span::styled(
                                *item,
                                Style::default()
                                    .fg(if is_active { Color::White } else { Color::Gray })
                                    .add_modifier(if is_active {
                                        Modifier::BOLD
                                    } else {
                                        Modifier::empty()
                                    }),
                            ),
                        ]))
                    })
                    .collect::<Vec<_>>(),
            )
            .block(
                Block::default()
                    .title("Navigation")
                    .borders(Borders::ALL)
                    .border_type(BorderType::Rounded)
                    .border_style(Style::default().fg(Color::Magenta)),
            )
            .highlight_style(
                Style::default()
                    .fg(Color::Black)
                    .bg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            );
            frame.render_widget(list, body_chunks[1]);

            let footer = Paragraph::new(vec![
                Line::from(vec![
                    Span::styled(
                        "s",
                        Style::default()
                            .fg(Color::Black)
                            .bg(Color::Green)
                            .add_modifier(Modifier::BOLD),
                    ),
                    Span::raw(" save "),
                    Span::styled(
                        "Enter",
                        Style::default()
                            .fg(Color::Black)
                            .bg(Color::Cyan)
                            .add_modifier(Modifier::BOLD),
                    ),
                    Span::raw(" save "),
                    Span::styled(
                        "q",
                        Style::default()
                            .fg(Color::Black)
                            .bg(Color::Red)
                            .add_modifier(Modifier::BOLD),
                    ),
                    Span::raw(" cancel "),
                    Span::styled(
                        "Esc",
                        Style::default()
                            .fg(Color::Black)
                            .bg(Color::Red)
                            .add_modifier(Modifier::BOLD),
                    ),
                    Span::raw(" cancel"),
                ]),
                Line::from(Span::styled(
                    "Press s or Enter to save this configuration.",
                    Style::default().fg(Color::DarkGray),
                )),
            ])
            .alignment(Alignment::Center)
            .wrap(Wrap { trim: true })
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_type(BorderType::Rounded)
                    .border_style(Style::default().fg(Color::DarkGray)),
            );
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

fn kv_line(label: &str, value: &str, value_color: Color) -> Line<'static> {
    Line::from(vec![
        Span::styled(
            format!("{label:<12}"),
            Style::default()
                .fg(Color::DarkGray)
                .add_modifier(Modifier::BOLD),
        ),
        Span::raw(" "),
        Span::styled(value.to_string(), Style::default().fg(value_color)),
    ])
}

#[cfg(test)]
mod tests {
    use super::{
        ConfigureTuiOutcome, ConfigureTuiState, WorkflowTuiContext, WorkflowTuiEvent,
        WorkflowTuiState,
    };

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
    fn workflow_model_sorts_active_files_first_and_reports_counters() {
        let mut state = WorkflowTuiState::new(context(2));
        state.apply(WorkflowTuiEvent::Queued {
            file_path: "zeta.py".to_string(),
            stage: "diff".to_string(),
        });
        state.apply(WorkflowTuiEvent::RequestSent {
            file_path: "zeta.py".to_string(),
        });
        state.apply(WorkflowTuiEvent::Prepared {
            file_path: "zeta.py".to_string(),
            subject: "feat(zeta): update zeta".to_string(),
        });
        state.apply(WorkflowTuiEvent::CommitStarted {
            file_path: "zeta.py".to_string(),
        });
        state.apply(WorkflowTuiEvent::Queued {
            file_path: "alpha.py".to_string(),
            stage: "diff".to_string(),
        });
        let rows = state.ordered_files();

        assert_eq!(rows.first().map(|row| row.path.as_str()), Some("zeta.py"));
        assert_eq!(state.stage_counters()[0].value, "2");
        assert_eq!(state.stage_counters()[1].value, "2");
        assert_eq!(state.stage_counters()[2].value, "1");
        assert_eq!(state.stage_counters()[3].value, "1");
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
