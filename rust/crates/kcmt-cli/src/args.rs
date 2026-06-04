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

    #[arg(long = "no-progress")]
    pub no_progress: bool,

    #[arg(long)]
    pub limit: Option<usize>,

    #[arg(long)]
    pub workers: Option<usize>,

    #[arg(long, short = 'v')]
    pub verbose: bool,

    #[arg(long = "profile-startup")]
    pub profile_startup: bool,

    #[arg(long = "max-retries")]
    pub max_retries: Option<usize>,

    #[arg(long = "github-token")]
    pub github_token: Option<String>,

    #[arg(long, alias = "summary")]
    pub compact: bool,

    #[arg(long)]
    pub configure: bool,

    #[arg(long = "configure-all")]
    pub configure_all: bool,

    #[arg(long = "list-models")]
    pub list_models: bool,

    #[arg(long)]
    pub benchmark: bool,

    #[arg(long = "benchmark-limit", default_value_t = 0)]
    pub benchmark_limit: usize,

    #[arg(long = "benchmark-timeout")]
    pub benchmark_timeout: Option<f64>,

    #[arg(long = "benchmark-json")]
    pub benchmark_json: bool,

    #[arg(long = "benchmark-csv")]
    pub benchmark_csv: bool,

    #[arg(long)]
    pub debug: bool,

    #[arg(long = "verify-keys")]
    pub verify_keys: bool,

    #[arg(long)]
    pub provider: Option<String>,

    #[arg(long)]
    pub model: Option<String>,

    #[arg(long)]
    pub endpoint: Option<String>,

    #[arg(long)]
    pub api_key_env: Option<String>,

    #[arg(long = "batch", conflicts_with = "no_batch")]
    pub batch: bool,

    #[arg(long = "no-batch")]
    pub no_batch: bool,

    #[arg(long)]
    pub batch_model: Option<String>,

    #[arg(long = "batch-timeout")]
    pub batch_timeout_seconds: Option<u64>,

    #[arg(long = "auto-push", conflicts_with = "no_auto_push")]
    pub auto_push: bool,

    #[arg(long = "no-auto-push")]
    pub no_auto_push: bool,

    #[arg(long)]
    pub max_commit_length: Option<usize>,

    #[command(subcommand)]
    pub command: Option<CliCommand>,
}

impl CliArgs {
    pub fn use_batch_override(&self) -> Option<bool> {
        if self.batch {
            Some(true)
        } else if self.no_batch {
            Some(false)
        } else {
            None
        }
    }

    pub fn auto_push_override(&self) -> Option<bool> {
        if self.auto_push {
            Some(true)
        } else if self.no_auto_push {
            Some(false)
        } else {
            None
        }
    }
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
    #[arg(long)]
    pub repo_path: Option<String>,

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
                    assert_eq!(runtime.repo_path.as_deref(), Some("/tmp/repo"));
                    assert_eq!(runtime.runtime, BenchmarkRuntime::Both);
                    assert_eq!(runtime.iterations, 2);
                    assert!(runtime.json);
                }
                other => panic!("expected runtime benchmark command, got {other:?}"),
            },
            other => panic!("expected benchmark command, got {other:?}"),
        }
    }

    #[test]
    fn parses_legacy_provider_benchmark_flags() {
        let args = CliArgs::parse_from([
            "kcmt",
            "--benchmark",
            "--benchmark-json",
            "--benchmark-csv",
            "--benchmark-limit",
            "2",
            "--benchmark-timeout",
            "4.5",
            "--provider",
            "openai",
            "--model",
            "gpt-test",
        ]);

        assert!(args.benchmark);
        assert!(args.benchmark_json);
        assert!(args.benchmark_csv);
        assert_eq!(args.benchmark_limit, 2);
        assert_eq!(args.benchmark_timeout, Some(4.5));
        assert_eq!(args.provider.as_deref(), Some("openai"));
        assert_eq!(args.model.as_deref(), Some("gpt-test"));
    }

    #[test]
    fn parses_legacy_workflow_override_flags() {
        let args = CliArgs::parse_from([
            "kcmt",
            "--oneshot",
            "--provider",
            "anthropic",
            "--model",
            "claude-test",
            "--batch",
            "--batch-model",
            "gpt-batch-test",
            "--batch-timeout",
            "1000",
            "--no-auto-push",
            "--max-commit-length",
            "68",
        ]);

        assert_eq!(args.provider.as_deref(), Some("anthropic"));
        assert_eq!(args.model.as_deref(), Some("claude-test"));
        assert_eq!(args.use_batch_override(), Some(true));
        assert_eq!(args.batch_model.as_deref(), Some("gpt-batch-test"));
        assert_eq!(args.batch_timeout_seconds, Some(1000));
        assert_eq!(args.auto_push_override(), Some(false));
        assert_eq!(args.max_commit_length, Some(68));
    }

    #[test]
    fn parses_legacy_non_tui_control_flags() {
        let args = CliArgs::parse_from([
            "kcmt",
            "--oneshot",
            "--no-progress",
            "--limit",
            "2",
            "--workers",
            "4",
            "--verbose",
            "--debug",
            "--profile-startup",
            "--max-retries",
            "5",
            "--github-token",
            "gh-test",
            "--compact",
        ]);

        assert!(args.no_progress);
        assert_eq!(args.limit, Some(2));
        assert_eq!(args.workers, Some(4));
        assert!(args.verbose);
        assert!(args.debug);
        assert!(args.profile_startup);
        assert_eq!(args.max_retries, Some(5));
        assert_eq!(args.github_token.as_deref(), Some("gh-test"));
        assert!(args.compact);
    }

    #[test]
    fn no_batch_and_auto_push_flags_override_boolean_defaults() {
        let args = CliArgs::parse_from(["commit", "--oneshot", "--no-batch", "--auto-push"]);

        assert_eq!(args.use_batch_override(), Some(false));
        assert_eq!(args.auto_push_override(), Some(true));
    }
}
