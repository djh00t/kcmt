# Configuration Contract

## Scope

Defines compatibility requirements for runtime configuration resolution.

## Sources

1. CLI overrides (highest precedence)
2. Environment variables
3. Persisted config file (`config.json` under config home)
4. Built-in defaults

## Required Compatibility Rules

- Existing provider identifiers remain valid.
- Existing environment variable names remain valid.
- Provider endpoint and model override behavior remains backward compatible.
- Missing/invalid provider credentials produce actionable errors without leaking secret values.

## Persisted Config Requirements

The persisted configuration object must continue to support at minimum:

- provider
- model
- llm_endpoint
- api_key_env
- git_repo_path
- max_commit_length
- auto_push
- providers (provider metadata map)
- model_priority (ordered fallback list)
- use_batch
- batch_model
- batch_timeout_seconds

## Validation Guarantees

- Unknown legacy fields are ignored safely where possible.
- Backward-compatible defaulting is applied for missing optional fields.
- Invalid type/value combinations return deterministic validation errors.
