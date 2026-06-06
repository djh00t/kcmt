//! Shared CLI entrypoint wiring for kcmt aliases.

pub mod args;
pub mod commands;

use clap::Parser;
use std::io::{self, Read};
use std::path::PathBuf;
use std::time::Instant;

use kcmt_core::config::loader::ConfigOverrides;
use kcmt_core::git::repo::find_git_repo_root;
use kcmt_core::preferences::{default_keychain_account, OsKeychainStore, SecretStore};

use args::{CliArgs, CliCommand, StatusArgs};
use commands::configure::ConfigureMode;
use commands::workflow::WorkflowOutputOptions;

#[derive(Debug, Clone)]
struct RepoDiscovery {
    requested_repo_path: PathBuf,
    repo_path: PathBuf,
    found_repo_path: Option<PathBuf>,
    duration_ms: f64,
}

/// Executes a named entrypoint.
pub fn run_entrypoint(name: &str) -> i32 {
    let parse_start = Instant::now();
    let args = CliArgs::parse();
    let arg_parse_ms = parse_start.elapsed().as_secs_f64() * 1000.0;
    dispatch(name, args, arg_parse_ms)
}

fn config_overrides(args: &CliArgs, repo_path: PathBuf) -> ConfigOverrides {
    ConfigOverrides {
        provider: args.provider.clone(),
        model: args.model.clone(),
        endpoint: args.endpoint.clone(),
        api_key_env: args.api_key_env.clone(),
        keychain_account: None,
        repo_path: Some(repo_path),
        max_commit_length: args.max_commit_length,
        auto_push: args.auto_push_override(),
        use_batch: args.use_batch_override(),
        batch_model: args.batch_model.clone(),
        batch_timeout_seconds: args.batch_timeout_seconds,
        file_limit: args.limit,
        max_retries: args.max_retries,
        prepare_workers: args.workers,
    }
}

fn output_options(args: &CliArgs) -> WorkflowOutputOptions {
    WorkflowOutputOptions {
        compact: args.compact,
        verbose: args.verbose || args.debug,
        no_progress: args.no_progress,
        profile_startup: args.profile_startup || args.debug,
        startup_stages: Vec::new(),
    }
}

fn dispatch(_entrypoint: &str, args: CliArgs, arg_parse_ms: f64) -> i32 {
    let dispatch_start = Instant::now();
    if let Some(token) = args
        .github_token
        .as_ref()
        .filter(|token| !token.trim().is_empty())
    {
        std::env::set_var("GITHUB_TOKEN", token);
    }
    let stdin_api_key = if args.api_key_stdin {
        read_secret_from_stdin()
    } else {
        None
    };
    let explicit_api_key = args
        .api_key
        .as_ref()
        .filter(|token| !token.trim().is_empty())
        .cloned()
        .or(stdin_api_key);
    if let Some(api_key) = explicit_api_key.as_ref() {
        std::env::set_var("KCMT_EXPLICIT_API_KEY", api_key);
        let provider = args.provider.as_deref().unwrap_or("openai");
        std::env::set_var("KCMT_EXPLICIT_API_KEY_PROVIDER", provider);
    }

    if args.list_models {
        return commands::configure::run_list_models(args.debug);
    }

    if args.verify_keys {
        return commands::configure::run_verify_keys();
    }

    let repo_discovery = discover_repo(&args);
    let repo_path = repo_discovery.repo_path.clone();

    if args.configure || args.configure_all {
        let overrides = config_overrides(&args, repo_path.clone());
        if args.tui && kcmt_tui::should_enable_tui(false) {
            let state = kcmt_tui::ConfigureTuiState {
                provider: args
                    .provider
                    .clone()
                    .unwrap_or_else(|| "auto/default".to_string()),
                model: args
                    .model
                    .clone()
                    .unwrap_or_else(|| "auto/default".to_string()),
                rule: "provider presets enabled".to_string(),
                credential_status: "keychain first, environment fallback".to_string(),
            };
            match kcmt_tui::run_configure_tui(state) {
                Ok(kcmt_tui::ConfigureTuiOutcome::Save) => {}
                Ok(kcmt_tui::ConfigureTuiOutcome::Cancel) => {
                    println!("Configuration unchanged");
                    return 0;
                }
                Err(err) => {
                    eprintln!("{err}");
                    return 1;
                }
            }
        }
        if args.save_api_key {
            let Some(api_key) = explicit_api_key
                .as_deref()
                .filter(|value| !value.trim().is_empty())
            else {
                eprintln!("--save-api-key requires --api-key or --api-key-stdin");
                return 1;
            };
            let provider = args.provider.as_deref().unwrap_or("openai");
            let account = default_keychain_account(provider);
            if let Err(err) = OsKeychainStore.set_secret(&account, api_key) {
                eprintln!("failed to save API key to OS keychain: {err}");
                return 1;
            }
            println!("Saved {provider} credentials to OS keychain account {account}");
        }
        let mode = if args.configure_all {
            ConfigureMode::All
        } else {
            ConfigureMode::Single
        };
        return commands::configure::run_configure(repo_path, overrides, mode);
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
        Some(CliCommand::Stats(stats)) => commands::stats::render_stats_command(repo_path, stats),
        Some(CliCommand::Benchmark(benchmark)) => {
            commands::benchmark::run_benchmark_command(repo_path, benchmark)
        }
        None => {
            if let Some(path) = args.file.clone() {
                let repo_path = match repo_discovery.found_repo_path.clone() {
                    Some(repo_path) => repo_path,
                    None => {
                        eprintln!(
                            "Not a Git repository: {}",
                            repo_discovery.requested_repo_path.display()
                        );
                        return 1;
                    }
                };
                let overrides = config_overrides(&args, repo_path.clone());
                let mut output_options = output_options(&args);
                add_dispatch_telemetry(
                    &mut output_options,
                    &repo_discovery,
                    dispatch_start,
                    arg_parse_ms,
                );
                return match commands::workflow::run_file_workflow(
                    repo_path,
                    &path,
                    overrides,
                    output_options,
                ) {
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
            if args.oneshot {
                let repo_path = match repo_discovery.found_repo_path.clone() {
                    Some(repo_path) => repo_path,
                    None => {
                        eprintln!(
                            "Not a Git repository: {}",
                            repo_discovery.requested_repo_path.display()
                        );
                        return 1;
                    }
                };
                let overrides = config_overrides(&args, repo_path.clone());
                let mut output_options = output_options(&args);
                add_dispatch_telemetry(
                    &mut output_options,
                    &repo_discovery,
                    dispatch_start,
                    arg_parse_ms,
                );
                return match commands::workflow::run_oneshot_workflow(
                    repo_path,
                    overrides,
                    output_options,
                ) {
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
            let repo_path = match repo_discovery.found_repo_path.clone() {
                Some(repo_path) => repo_path,
                None => {
                    eprintln!(
                        "Not a Git repository: {}",
                        repo_discovery.requested_repo_path.display()
                    );
                    return 1;
                }
            };
            let overrides = config_overrides(&args, repo_path.clone());
            let mut output_options = output_options(&args);
            add_dispatch_telemetry(
                &mut output_options,
                &repo_discovery,
                dispatch_start,
                arg_parse_ms,
            );
            match commands::workflow::run_default_workflow(repo_path, overrides, output_options) {
                Ok(output) => {
                    print!("{output}");
                    0
                }
                Err(err) => {
                    eprintln!("{err}");
                    1
                }
            }
        }
    }
}

fn read_secret_from_stdin() -> Option<String> {
    let mut value = String::new();
    io::stdin().read_to_string(&mut value).ok()?;
    let value = value.trim().to_string();
    if value.is_empty() {
        None
    } else {
        Some(value)
    }
}

fn discover_repo(args: &CliArgs) -> RepoDiscovery {
    let requested_repo_path = PathBuf::from(&args.repo_path);
    let repo_start = Instant::now();
    let found_repo_path = find_git_repo_root(&requested_repo_path);
    let duration_ms = repo_start.elapsed().as_secs_f64() * 1000.0;
    let repo_path = found_repo_path
        .clone()
        .unwrap_or_else(|| requested_repo_path.clone());
    RepoDiscovery {
        requested_repo_path,
        repo_path,
        found_repo_path,
        duration_ms,
    }
}

fn add_dispatch_telemetry(
    output_options: &mut WorkflowOutputOptions,
    repo_discovery: &RepoDiscovery,
    dispatch_start: Instant,
    arg_parse_ms: f64,
) {
    output_options.record_startup_stage("arg_parse", arg_parse_ms, 1);
    output_options.record_startup_stage(
        "repo_discovery",
        repo_discovery.duration_ms,
        usize::from(repo_discovery.found_repo_path.is_some()),
    );
    let dispatch_ms =
        (dispatch_start.elapsed().as_secs_f64() * 1000.0 - repo_discovery.duration_ms).max(0.0);
    output_options.record_startup_stage("dispatch", dispatch_ms, 1);
}
