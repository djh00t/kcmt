# Exit/Error Baseline Matrix

This matrix defines expected automation-facing contracts for success and failure classes.

| Scenario | Expected Exit Code | stderr Contract | Notes |
|---|---:|---|---|
| Successful commit generation | 0 | empty or informational only | No fatal error output |
| Argument/usage error | 2 | includes usage + parse detail | Parser contract |
| Configuration error | 1 | actionable guidance without secrets | Missing/invalid provider settings |
| Provider timeout/rate-limit/malformed response | 1 | normalized provider error text | Deterministic failure class |
| Git operation failure | 1 | git error summary with command context | Includes pathscope when relevant |

## Validation Rules

- stderr output MUST NOT include secret/token material.
- Matrix comparisons MUST normalize dynamic timestamps and absolute paths.
