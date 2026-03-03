//! Shared CLI entrypoint wiring for kcmt aliases.

pub mod args;
pub mod commands;

use clap::Parser;

use args::{CliArgs, CliCommand};

/// Executes a named entrypoint.
pub fn run_entrypoint(name: &str) -> i32 {
    let args = CliArgs::parse();
    dispatch(name, args)
}

fn dispatch(_entrypoint: &str, args: CliArgs) -> i32 {
    if args.configure {
        return commands::configure::run_configure();
    }

    match args.command {
        Some(CliCommand::Status) => match commands::status::render_status(None) {
            Ok(output) => {
                print!("{output}");
                0
            }
            Err(err) => {
                eprintln!("{err}");
                1
            }
        },
        Some(CliCommand::Benchmark) => commands::benchmark::run_benchmark_command(),
        None => {
            let _ = kcmt_core::package_name();
            if args.oneshot {
                println!("(placeholder) oneshot mode");
            }
            if let Some(path) = args.file {
                println!("(placeholder) file-scoped mode: {path}");
            }
            0
        }
    }
}
