//! Shared CLI entrypoint wiring for kcmt aliases.

pub mod args;
pub mod commands;

use clap::Parser;
use std::path::PathBuf;

use kcmt_core::git::repo::find_git_repo_root;

use args::{CliArgs, CliCommand, StatusArgs};

/// Executes a named entrypoint.
pub fn run_entrypoint(name: &str) -> i32 {
    let args = CliArgs::parse();
    dispatch(name, args)
}

fn require_git_repo(requested_repo_path: &PathBuf) -> Result<PathBuf, String> {
    find_git_repo_root(requested_repo_path).ok_or_else(|| {
        format!("Not a Git repository: {}", requested_repo_path.display())
    })
}

fn dispatch(_entrypoint: &str, args: CliArgs) -> i32 {
    if args.configure {
        return commands::configure::run_configure();
    }

    let requested_repo_path = PathBuf::from(&args.repo_path);
    let repo_path = find_git_repo_root(&requested_repo_path)
        .unwrap_or_else(|| requested_repo_path.clone());

    match args.command {
        Some(CliCommand::Status(StatusArgs { raw })) => {
            match commands::status::render_status(Some(repo_path), raw) {
                Ok(output) => {
                    print!("{output}");
                    0
                }
                Err(err) => {
                    println!("{err}");
                    1
                }
            }
        }
        Some(CliCommand::Benchmark(benchmark)) => {
            commands::benchmark::run_benchmark_command(repo_path, benchmark)
        }
        None => {
            if args.oneshot {
                let repo_path = match require_git_repo(&requested_repo_path) {
                    Ok(repo_path) => repo_path,
                    Err(err) => {
                        eprintln!("{err}");
                        return 1;
                    }
                };
                return match commands::workflow::run_oneshot_workflow(repo_path) {
                    Ok(output) => {
                        print!("{output}");
                        0
                    }
                    Err(err) => {
                        eprintln!("{err}");
                        1
                    }
                };
            }
            if let Some(path) = args.file {
                let repo_path = match require_git_repo(&requested_repo_path) {
                    Ok(repo_path) => repo_path,
                    Err(err) => {
                        eprintln!("{err}");
                        return 1;
                    }
                };
                return match commands::workflow::run_file_workflow(repo_path, &path) {
                    Ok(output) => {
                        print!("{output}");
                        0
                    }
                    Err(err) => {
                        eprintln!("{err}");
                        1
                    }
                };
            }
            eprintln!(
                "{}",
                kcmt_core::error::KcmtError::Message(
                    "Rust runtime default workflow is not implemented yet.".to_string()
                )
            );
            1
        }
    }
}
