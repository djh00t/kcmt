//! Shared CLI entrypoint wiring for kcmt aliases.

pub mod args;
pub mod commands;

use clap::Parser;
use std::path::PathBuf;

use kcmt_core::config::loader::ConfigOverrides;
use kcmt_core::git::repo::find_git_repo_root;

use args::{CliArgs, CliCommand, StatusArgs};

/// Executes a named entrypoint.
pub fn run_entrypoint(name: &str) -> i32 {
    let args = CliArgs::parse();
    dispatch(name, args)
}

fn require_git_repo(requested_repo_path: &PathBuf) -> Result<PathBuf, String> {
    find_git_repo_root(requested_repo_path)
        .ok_or_else(|| format!("Not a Git repository: {}", requested_repo_path.display()))
}

fn config_overrides(args: &CliArgs, repo_path: PathBuf) -> ConfigOverrides {
    ConfigOverrides {
        provider: args.provider.clone(),
        model: args.model.clone(),
        endpoint: args.endpoint.clone(),
        api_key_env: args.api_key_env.clone(),
        repo_path: Some(repo_path),
        max_commit_length: args.max_commit_length,
        auto_push: args.auto_push_override(),
        use_batch: args.use_batch_override(),
        batch_model: args.batch_model.clone(),
        batch_timeout_seconds: args.batch_timeout_seconds,
        file_limit: args.limit,
    }
}

fn dispatch(_entrypoint: &str, args: CliArgs) -> i32 {
    if args.list_models {
        return commands::configure::run_list_models();
    }

    if args.verify_keys {
        return commands::configure::run_verify_keys();
    }

    let requested_repo_path = PathBuf::from(&args.repo_path);
    let repo_path =
        find_git_repo_root(&requested_repo_path).unwrap_or_else(|| requested_repo_path.clone());

    if args.configure || args.configure_all {
        let overrides = config_overrides(&args, repo_path.clone());
        return commands::configure::run_configure(repo_path, overrides);
    }

    if args.benchmark {
        return commands::benchmark::run_provider_benchmark(repo_path, &args);
    }

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
                let overrides = config_overrides(&args, repo_path.clone());
                return match commands::workflow::run_oneshot_workflow(repo_path, overrides) {
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
            if let Some(path) = args.file.clone() {
                let repo_path = match require_git_repo(&requested_repo_path) {
                    Ok(repo_path) => repo_path,
                    Err(err) => {
                        eprintln!("{err}");
                        return 1;
                    }
                };
                let overrides = config_overrides(&args, repo_path.clone());
                return match commands::workflow::run_file_workflow(repo_path, &path, overrides) {
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
