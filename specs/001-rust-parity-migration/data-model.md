# Data Model: Rust Migration with Feature Parity

## Entity: WorkflowConfig

Represents user/runtime configuration controlling command behavior and provider resolution.

| Field | Type | Required | Validation Rules |
|---|---|---|---|
| provider | string | yes | Must be one of supported providers |
| model | string | yes | Non-empty; provider-compatible |
| llm_endpoint | string | yes | Valid URI |
| api_key_env | string | yes | Uppercase env var style; must resolve at runtime for provider calls |
| git_repo_path | string | yes | Path must resolve to a git repository root |
| max_commit_length | integer | yes | Positive integer; default aligns with current behavior |
| auto_push | boolean | yes | Default parity with current config |
| providers | map | yes | Must include known provider entries and endpoint/key metadata |
| model_priority | list | yes | Ordered provider/model pairs; no duplicates |
| use_batch | boolean | yes | OpenAI-specific feature flag |
| batch_model | string/null | no | Required when batch mode enabled |
| batch_timeout_seconds | integer | yes | Must be >= minimum timeout threshold |

## Entity: ProviderProfile

Provider-specific runtime metadata resolved from config and environment.

| Field | Type | Required | Validation Rules |
|---|---|---|---|
| provider_id | string | yes | Unique key (`openai`, `anthropic`, `xai`, `github`) |
| display_name | string | yes | Non-empty |
| endpoint | string | yes | Valid URI |
| api_key_env | string | yes | Must reference environment variable |
| preferred_model | string/null | no | Provider-supported model id |
| timeout_policy | object | yes | Includes connect/read/request timeout bounds |
| retry_policy | object | yes | Must define max attempts and backoff strategy |

## Entity: ChangeSet

Represents file-level repository changes used for commit generation.

| Field | Type | Required | Validation Rules |
|---|---|---|---|
| file_path | string | yes | Repository-relative normalized path |
| change_type | enum | yes | One of `A`, `M`, `D` |
| diff_content | string | yes | May be empty only for unsupported/binary diff cases |
| staged | boolean | yes | Indicates staged vs working tree source |
| ignored | boolean | yes | Must mirror `.gitignore` handling |

## Entity: CommitRecommendation

Generated commit message candidate and validation metadata.

| Field | Type | Required | Validation Rules |
|---|---|---|---|
| subject | string | yes | Conventional Commit prefix/scope; <= max length |
| body | string/null | no | Optional explanatory text |
| raw_message | string | yes | Concatenated final message |
| provider_id | string | yes | Source provider identifier |
| model | string | yes | Source model identifier |
| success | boolean | yes | False requires `error` |
| error | string/null | no | Required on generation failure |

## Entity: BenchmarkRun

Captures a benchmark execution, results, exclusions, and output artifacts.

| Field | Type | Required | Validation Rules |
|---|---|---|---|
| timestamp | string | yes | ISO-8601 UTC timestamp |
| params | object | yes | Includes filters, limits, timeout inputs |
| results | list<BenchmarkResult> | yes | May be empty only when exclusions are present |
| exclusions | list<BenchmarkExclusion> | yes | Structured reason/detail records |
| details | list<BenchmarkSampleDetail> | no | Present only in detailed mode |
| schema_version | integer | yes | Positive, monotonically incremented on contract changes |

## Entity: BenchmarkResult

Per provider/model aggregate metrics.

| Field | Type | Required | Validation Rules |
|---|---|---|---|
| provider | string | yes | Known provider id |
| model | string | yes | Non-empty |
| avg_latency_ms | number | yes | >= 0 |
| avg_cost_usd | number | yes | >= 0 |
| quality | number | yes | 0..100 |
| success_rate | number | yes | 0..1 |
| runs | integer | yes | > 0 |

## Relationships

- `WorkflowConfig` 1..N `ProviderProfile`
- `WorkflowConfig` influences generation of N `CommitRecommendation`
- `ChangeSet` 1..N contributes to `CommitRecommendation` generation context
- `BenchmarkRun` 1..N `BenchmarkResult`
- `BenchmarkRun` 1..N `BenchmarkExclusion`

## State Transitions

### CommitRecommendation Lifecycle

1. `Draft` -> message generated from diff context
2. `Validated` -> conventional format and length constraints pass
3. `Applied` -> commit successfully created (or simulated in dry path)
4. `Failed` -> generation or git apply step returned error

### BenchmarkRun Lifecycle

1. `Initialized` -> provider/model matrix prepared
2. `Executing` -> samples processed and metrics collected
3. `Completed` -> aggregates persisted and reports emitted
4. `CompletedWithExclusions` -> finished with skipped providers/models
5. `Failed` -> unrecoverable benchmark pipeline error
