use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone, Copy)]
struct ProviderCase {
    provider: &'static str,
    model: &'static str,
    batch_model: Option<&'static str>,
    endpoint: &'static str,
    api_key_env: &'static str,
    supports_batch: bool,
}

#[derive(Debug, Clone, Copy)]
enum RunMode {
    Standard,
    Batch,
}

impl RunMode {
    fn label(self) -> &'static str {
        match self {
            Self::Standard => "standard",
            Self::Batch => "batch",
        }
    }
}

#[derive(Debug)]
struct MatrixResult {
    provider: &'static str,
    mode: &'static str,
    model: &'static str,
    status: String,
    reason: String,
    wall_time_ms: f64,
    committed_files: usize,
    ignored_ok: bool,
    stage_timings: BTreeMap<String, f64>,
}

struct TempDir {
    path: PathBuf,
}

impl TempDir {
    fn new(label: &str) -> Self {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock should be after unix epoch")
            .as_nanos();
        let suffix = COUNTER.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!("kcmt-live-{label}-{nanos}-{suffix}"));
        fs::create_dir_all(&path).expect("temp dir should be created");
        Self { path }
    }

    fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

const PROVIDERS: &[ProviderCase] = &[
    ProviderCase {
        provider: "openai",
        model: "gpt-5.4-mini",
        batch_model: Some("gpt-5.4-mini"),
        endpoint: "https://api.openai.com/v1",
        api_key_env: "OPENAI_API_KEY",
        supports_batch: true,
    },
    ProviderCase {
        provider: "anthropic",
        model: "claude-sonnet-4-20250514",
        batch_model: None,
        endpoint: "https://api.anthropic.com",
        api_key_env: "ANTHROPIC_API_KEY",
        supports_batch: false,
    },
    ProviderCase {
        provider: "xai",
        model: "grok-code-fast",
        batch_model: Some("grok-4.3"),
        endpoint: "https://api.x.ai/v1",
        api_key_env: "XAI_API_KEY",
        supports_batch: true,
    },
    ProviderCase {
        provider: "github",
        model: "openai/gpt-4.1-mini",
        batch_model: None,
        endpoint: "https://models.github.ai/inference",
        api_key_env: "GITHUB_TOKEN",
        supports_batch: false,
    },
];

const EXPECTED_COMMITTED_FILES: &[&str] = &[
    "root_modified.txt",
    "root_deleted.txt",
    "root_new.txt",
    "nested/deep/modified.rs",
    "nested/deep/deleted.md",
    "nested/deep/new.py",
    "staged_only.txt",
];

const EXPECTED_IGNORED_FILES: &[&str] = &["root_ignored.log", "nested/deep/ignored.tmp"];

#[test]
#[ignore = "live LLM provider matrix requires real API keys and may take minutes"]
fn live_llm_provider_matrix_commits_all_git_states() {
    if std::env::var("KCMT_LIVE_LLM_MATRIX").as_deref() != Ok("1") {
        println!("skipped: set KCMT_LIVE_LLM_MATRIX=1 to run the live provider matrix");
        return;
    }

    let mut results = Vec::new();
    for provider in PROVIDERS {
        results.push(run_or_skip(*provider, RunMode::Standard));
        if provider.supports_batch {
            results.push(run_or_skip(*provider, RunMode::Batch));
        }
    }

    print_scoreboard(&results);

    let failures = results
        .iter()
        .filter(|result| result.status == "failed")
        .collect::<Vec<_>>();
    assert!(
        failures.is_empty(),
        "live LLM matrix failures: {failures:#?}"
    );
}

fn run_or_skip(provider: ProviderCase, mode: RunMode) -> MatrixResult {
    if std::env::var(provider.api_key_env)
        .ok()
        .filter(|value| !value.trim().is_empty())
        .is_none()
    {
        return MatrixResult {
            provider: provider.provider,
            mode: mode.label(),
            model: model_for_mode(provider, mode),
            status: "skipped".to_string(),
            reason: format!("missing {}", provider.api_key_env),
            wall_time_ms: 0.0,
            committed_files: 0,
            ignored_ok: false,
            stage_timings: BTreeMap::new(),
        };
    }

    let repo = TempDir::new(provider.provider);
    let seed_hash = seed_repo(repo.path());
    let config_home = TempDir::new("config-home");
    let started = Instant::now();
    let output = run_kcmt(repo.path(), config_home.path(), provider, mode);
    let wall_time_ms = started.elapsed().as_secs_f64() * 1000.0;
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    let stage_timings = parse_stage_timings(&stdout);

    if !output.status.success() {
        return MatrixResult {
            provider: provider.provider,
            mode: mode.label(),
            model: model_for_mode(provider, mode),
            status: "failed".to_string(),
            reason: first_error_line(&stdout, &stderr),
            wall_time_ms,
            committed_files: count_committed_files(repo.path(), &seed_hash),
            ignored_ok: ignored_files_are_uncommitted(repo.path()),
            stage_timings,
        };
    }

    let dirty = git(repo.path(), &["status", "--short"]);
    let committed_files = count_committed_files(repo.path(), &seed_hash);
    let ignored_ok = ignored_files_are_uncommitted(repo.path());
    let committed_ok = committed_files == EXPECTED_COMMITTED_FILES.len();
    let status = if dirty.is_empty() && committed_ok && ignored_ok {
        "passed"
    } else {
        "failed"
    };
    let reason = if status == "passed" {
        "all expected files committed; ignored files preserved".to_string()
    } else {
        format!("dirty={dirty:?}; committed_files={committed_files}; ignored_ok={ignored_ok}")
    };

    MatrixResult {
        provider: provider.provider,
        mode: mode.label(),
        model: model_for_mode(provider, mode),
        status: status.to_string(),
        reason,
        wall_time_ms,
        committed_files,
        ignored_ok,
        stage_timings,
    }
}

fn seed_repo(repo: &Path) -> String {
    git(repo, &["init", "-q"]);
    git(repo, &["config", "user.name", "kcmt live matrix"]);
    git(repo, &["config", "user.email", "kcmt-live@example.com"]);
    fs::create_dir_all(repo.join("nested/deep")).expect("nested dir");
    fs::write(
        repo.join(".gitignore"),
        "*.log\nnested/deep/*.tmp\nignored_dir/\n",
    )
    .expect("gitignore");
    fs::write(repo.join("root_modified.txt"), "root before\n").expect("root modified seed");
    fs::write(repo.join("root_deleted.txt"), "root delete me\n").expect("root deleted seed");
    fs::write(repo.join("staged_only.txt"), "staged before\n").expect("staged seed");
    fs::write(
        repo.join("nested/deep/modified.rs"),
        "pub fn value() -> i32 { 1 }\n",
    )
    .expect("nested modified seed");
    fs::write(repo.join("nested/deep/deleted.md"), "# Delete me\n").expect("nested deleted seed");
    git(repo, &["add", "."]);
    git(repo, &["commit", "-m", "chore(repo): seed live matrix"]);
    let seed_hash = git(repo, &["rev-parse", "HEAD"]);

    fs::write(repo.join("root_modified.txt"), "root after\n").expect("root modified");
    fs::remove_file(repo.join("root_deleted.txt")).expect("root deleted");
    fs::write(repo.join("root_new.txt"), "root new\n").expect("root new");
    fs::write(repo.join("root_ignored.log"), "root ignored\n").expect("root ignored");
    fs::write(
        repo.join("nested/deep/modified.rs"),
        "pub fn value() -> i32 { 2 }\n",
    )
    .expect("nested modified");
    fs::remove_file(repo.join("nested/deep/deleted.md")).expect("nested deleted");
    fs::write(repo.join("nested/deep/new.py"), "print('nested new')\n").expect("nested new");
    fs::write(repo.join("nested/deep/ignored.tmp"), "nested ignored\n").expect("nested ignored");
    fs::write(repo.join("staged_only.txt"), "staged after\n").expect("staged update");
    git(repo, &["add", "staged_only.txt"]);
    seed_hash
}

fn run_kcmt(repo: &Path, config_home: &Path, provider: ProviderCase, mode: RunMode) -> Output {
    let mut command = Command::new(env!("CARGO_BIN_EXE_kcmt"));
    command
        .current_dir(repo)
        .env("KCMT_CONFIG_HOME", config_home)
        .env("KCMT_GIT_COMMIT_BACKEND", "gix")
        .arg("--provider")
        .arg(provider.provider)
        .arg("--api-key-env")
        .arg(provider.api_key_env)
        .arg("--endpoint")
        .arg(provider.endpoint)
        .arg("--model")
        .arg(provider.model)
        .arg("--debug")
        .arg("--no-auto-push")
        .arg("--repo-path")
        .arg(repo);
    if matches!(mode, RunMode::Batch) {
        let batch_model = provider.batch_model.unwrap_or(provider.model);
        command
            .arg("--batch")
            .arg("--batch-model")
            .arg(batch_model)
            .arg("--batch-timeout")
            .arg("900");
    }
    command.output().expect("kcmt binary should run")
}

fn model_for_mode(provider: ProviderCase, mode: RunMode) -> &'static str {
    match mode {
        RunMode::Standard => provider.model,
        RunMode::Batch => provider.batch_model.unwrap_or(provider.model),
    }
}

fn parse_stage_timings(stdout: &str) -> BTreeMap<String, f64> {
    let mut stages = BTreeMap::new();
    for line in stdout.lines() {
        let Some(rest) = line.strip_prefix("[kcmt-profile] ") else {
            continue;
        };
        let Some((stage, timing)) = rest.split_once(": ") else {
            continue;
        };
        let Some((duration, _items)) = timing.split_once(" ms items=") else {
            continue;
        };
        if let Ok(ms) = duration.parse::<f64>() {
            stages.insert(stage.to_string(), ms);
        }
    }
    stages
}

fn count_committed_files(repo: &Path, seed_hash: &str) -> usize {
    let names = git(
        repo,
        &[
            "log",
            "--name-only",
            "--pretty=format:",
            "--diff-filter=ACDMRT",
            &format!("{seed_hash}..HEAD"),
            "--",
        ],
    );
    EXPECTED_COMMITTED_FILES
        .iter()
        .filter(|path| names.lines().any(|line| line == **path))
        .count()
}

fn ignored_files_are_uncommitted(repo: &Path) -> bool {
    let ignored = git(repo, &["status", "--short", "--ignored"]);
    EXPECTED_IGNORED_FILES
        .iter()
        .all(|path| ignored.lines().any(|line| line == format!("!! {path}")))
        && EXPECTED_IGNORED_FILES.iter().all(|path| {
            !git(repo, &["log", "--name-only", "--pretty=format:", "--"])
                .lines()
                .any(|line| line == *path)
        })
}

fn first_error_line(stdout: &str, stderr: &str) -> String {
    stdout
        .lines()
        .chain(stderr.lines())
        .find(|line| line.contains("provider") || line.contains("✗") || line.contains("failed"))
        .unwrap_or("command failed without a provider error line")
        .to_string()
}

fn print_scoreboard(results: &[MatrixResult]) {
    const STAGES: &[&str] = &[
        "status_scan",
        "diff_preparation",
        "llm_enqueue",
        "llm_wait",
        "commit_create",
        "commit",
        "snapshot",
        "workflow_total",
    ];
    println!();
    println!("Live LLM Provider Matrix Scoreboard");
    println!(
        "| provider | mode | model | status | files | ignored | wall_ms | {} | reason |",
        STAGES.join(" | ")
    );
    println!(
        "|---|---|---|---|---:|---|---:|{}|---|",
        STAGES.iter().map(|_| "---:").collect::<Vec<_>>().join("|")
    );
    for result in results {
        let stages = STAGES
            .iter()
            .map(|stage| {
                result
                    .stage_timings
                    .get(*stage)
                    .map(|value| format!("{value:.1}"))
                    .unwrap_or_else(|| "-".to_string())
            })
            .collect::<Vec<_>>();
        println!(
            "| {} | {} | {} | {} | {} | {} | {:.1} | {} | {} |",
            result.provider,
            result.mode,
            result.model,
            result.status,
            result.committed_files,
            result.ignored_ok,
            result.wall_time_ms,
            stages.join(" | "),
            result.reason.replace('|', "/")
        );
    }

    println!();
    println!("Provider/Mode Stats");
    println!("| provider | mode | status | wall_ms | llm_wait_ms | workflow_total_ms | files |");
    println!("|---|---|---|---:|---:|---:|---:|");
    for result in results {
        println!(
            "| {} | {} | {} | {:.1} | {} | {} | {} |",
            result.provider,
            result.mode,
            result.status,
            result.wall_time_ms,
            format_optional_stage(&result.stage_timings, "llm_wait"),
            format_optional_stage(&result.stage_timings, "workflow_total"),
            result.committed_files
        );
    }
}

fn format_optional_stage(stages: &BTreeMap<String, f64>, stage: &str) -> String {
    stages
        .get(stage)
        .map(|value| format!("{value:.1}"))
        .unwrap_or_else(|| "-".to_string())
}

fn git(repo: &Path, args: &[&str]) -> String {
    let output = Command::new("git")
        .current_dir(repo)
        .args(args)
        .output()
        .expect("git command should run");
    assert!(
        output.status.success(),
        "git {:?} failed: {}",
        args,
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8_lossy(&output.stdout).trim().to_string()
}
