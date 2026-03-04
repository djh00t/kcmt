use clap::{Parser, Subcommand};

#[derive(Debug, Parser, Clone)]
#[command(name = "kcmt", about = "AI-assisted conventional commits")]
pub struct CliArgs {
    #[arg(long)]
    pub oneshot: bool,

    #[arg(long)]
    pub file: Option<String>,

    #[arg(long)]
    pub configure: bool,

    #[command(subcommand)]
    pub command: Option<CliCommand>,
}

#[derive(Debug, Subcommand, Clone)]
pub enum CliCommand {
    Status,
    Benchmark,
}
