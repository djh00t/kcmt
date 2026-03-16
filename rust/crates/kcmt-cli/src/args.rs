use clap::{Args, Parser, Subcommand, ValueEnum};

#[derive(Debug, Parser, Clone)]
#[command(about = "AI-assisted conventional commits")]
pub struct CliArgs {
    #[arg(long, global = true, default_value = ".")]
    pub repo_path: String,

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
    Status(StatusArgs),
    Benchmark(BenchmarkArgs),
}

#[derive(Debug, Args, Clone, Default)]
pub struct StatusArgs {
    #[arg(long)]
    pub raw: bool,
}

#[derive(Debug, Args, Clone, Default)]
pub struct BenchmarkArgs {
    #[command(subcommand)]
    pub command: Option<BenchmarkCommand>,
}

#[derive(Debug, Subcommand, Clone)]
pub enum BenchmarkCommand {
    Runtime(RuntimeBenchmarkArgs),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum BenchmarkRuntime {
    Python,
    Rust,
    Both,
}

#[derive(Debug, Args, Clone)]
pub struct RuntimeBenchmarkArgs {
    #[arg(long, value_enum, default_value_t = BenchmarkRuntime::Both)]
    pub runtime: BenchmarkRuntime,

    #[arg(long, default_value_t = 3)]
    pub iterations: usize,

    #[arg(long)]
    pub json: bool,

    #[arg(long)]
    pub rust_bin: Option<String>,
}

#[cfg(test)]
mod tests {
    use clap::Parser;

    use super::{BenchmarkCommand, BenchmarkRuntime, CliArgs, CliCommand};

    #[test]
    fn parses_global_repo_path_after_status_subcommand() {
        let args = CliArgs::parse_from(["kcmt", "status", "--repo-path", "/tmp/repo"]);

        assert_eq!(args.repo_path, "/tmp/repo");
        assert!(matches!(args.command, Some(CliCommand::Status(_))));
    }

    #[test]
    fn parses_status_raw_flag() {
        let args = CliArgs::parse_from(["commit", "status", "--repo-path", ".", "--raw"]);

        match args.command {
            Some(CliCommand::Status(status)) => assert!(status.raw),
            other => panic!("expected status subcommand, got {other:?}"),
        }
    }

    #[test]
    fn invalid_flag_returns_clap_error() {
        let err = CliArgs::try_parse_from(["kcmt", "--definitely-invalid-flag"])
            .expect_err("invalid flag should fail");

        assert_eq!(err.kind(), clap::error::ErrorKind::UnknownArgument);
    }

    #[test]
    fn parses_runtime_benchmark_subcommand() {
        let args = CliArgs::parse_from([
            "kcmt",
            "benchmark",
            "runtime",
            "--repo-path",
            "/tmp/repo",
            "--runtime",
            "both",
            "--iterations",
            "2",
            "--json",
        ]);

        match args.command {
            Some(CliCommand::Benchmark(benchmark)) => match benchmark.command {
                Some(BenchmarkCommand::Runtime(runtime)) => {
                    assert_eq!(args.repo_path, "/tmp/repo");
                    assert_eq!(runtime.runtime, BenchmarkRuntime::Both);
                    assert_eq!(runtime.iterations, 2);
                    assert!(runtime.json);
                }
                other => panic!("expected runtime benchmark command, got {other:?}"),
            },
            other => panic!("expected benchmark command, got {other:?}"),
        }
    }
}
