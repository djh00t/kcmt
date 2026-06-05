//! Provider prompt and commit-message post-processing.

const CONVENTIONAL_TYPES: &[&str] = &[
    "feat", "fix", "docs", "style", "refactor", "test", "chore", "perf", "ci", "build", "revert",
];

pub fn build_prompt(diff: &str, context: &str, style: &str) -> String {
    let mut parts = vec![
        "Generate a conventional commit message for these changes:".to_string(),
        String::new(),
        "DIFF:".to_string(),
        diff.to_string(),
    ];
    if !context.is_empty() {
        parts.extend([String::new(), "CONTEXT:".to_string(), context.to_string()]);
    }
    match style {
        "conventional" => parts.extend([
            String::new(),
            "STRICT REQUIREMENTS:".to_string(),
            "- MUST use format: type(scope): description".to_string(),
            "- Scope REQUIRED (api ui auth db config tests etc.)".to_string(),
            "- If diff shows substantial changes (>5 lines), add body".to_string(),
            "- Body should explain what changed and why".to_string(),
            "- Keep subject line under 50 characters".to_string(),
            "- Only output the commit message (no backticks / quotes)".to_string(),
        ]),
        "simple" => parts.extend([
            String::new(),
            "Keep it simple but include mandatory scope.".to_string(),
        ]),
        _ => {}
    }
    parts.push(String::new());
    parts.push("Analyze the changes carefully and be specific.".to_string());
    parts.join("\n")
}

pub fn sanitize_commit_output(raw: &str) -> Result<String, String> {
    let mut text = raw.trim().to_string();
    if text.is_empty() {
        return Err("Empty LLM output (no heuristic fallback)".to_string());
    }

    if is_wrapped(&text, '"') || is_wrapped(&text, '\'') {
        text = text[1..text.len() - 1].trim().to_string();
    }
    if text.starts_with("```") {
        for part in text.split("```") {
            let candidate = part.trim();
            if !candidate.is_empty()
                && !candidate.starts_with("yaml")
                && !candidate.starts_with("json")
            {
                text = candidate.to_string();
                break;
            }
        }
    }

    let lines: Vec<String> = text
        .lines()
        .map(str::trim_end)
        .filter(|line| !line.trim().is_empty())
        .map(ToOwned::to_owned)
        .collect();

    let mut header = None;
    let mut body_lines = Vec::new();
    for (index, line) in lines.iter().enumerate() {
        let cleaned = clean_candidate_header(line);
        if validate_conventional_commit(&cleaned) {
            header = Some(cleaned);
            body_lines = lines[index + 1..].to_vec();
            break;
        }
    }

    let Some(mut header) = header else {
        return Err("LLM output missing conventional commit header (no fallback)".to_string());
    };
    header = header.trim_end_matches('.').to_string();

    let body: Vec<String> = body_lines
        .into_iter()
        .filter(|line| {
            let trimmed = line.trim();
            !(trimmed.starts_with("```")
                || trimmed.starts_with("---")
                || trimmed.starts_with("==="))
        })
        .take(13)
        .collect();
    if body.is_empty() {
        Ok(header)
    } else {
        Ok(format!("{header}\n\n{}", body.join("\n")))
    }
}

pub fn validate_conventional_commit(message: &str) -> bool {
    let header = message.trim().lines().next().unwrap_or_default();
    let Some((prefix, description)) = header.split_once(": ") else {
        return false;
    };
    if description.trim().is_empty() {
        return false;
    }
    if let Some((commit_type, scope)) = prefix.split_once('(') {
        scope.ends_with(')')
            && !scope.trim_end_matches(')').is_empty()
            && is_conventional_type(commit_type)
            && scope
                .trim_end_matches(')')
                .chars()
                .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '_' | '-'))
    } else {
        is_conventional_type(prefix)
    }
}

fn clean_candidate_header(line: &str) -> String {
    let stripped = line
        .trim_start_matches(|ch| matches!(ch, '-' | '*' | '•' | ' '))
        .trim()
        .trim_matches('`')
        .trim();
    stripped.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn is_conventional_type(value: &str) -> bool {
    CONVENTIONAL_TYPES.contains(&value)
}

fn is_wrapped(value: &str, delimiter: char) -> bool {
    value.starts_with(delimiter) && value.ends_with(delimiter) && value.len() >= 2
}

#[cfg(test)]
mod tests {
    use super::{build_prompt, sanitize_commit_output, validate_conventional_commit};

    #[test]
    fn build_prompt_matches_python_conventional_shape() {
        let prompt = build_prompt(
            "diff --git a/app.py b/app.py",
            "File: app.py",
            "conventional",
        );

        assert!(prompt.contains("Generate a conventional commit message"));
        assert!(prompt.contains("DIFF:\ndiff --git a/app.py b/app.py"));
        assert!(prompt.contains("CONTEXT:\nFile: app.py"));
        assert!(prompt.contains("- MUST use format: type(scope): description"));
        assert!(prompt.ends_with("Analyze the changes carefully and be specific."));
    }

    #[test]
    fn sanitize_extracts_conventional_header_from_wrapped_output() {
        let sanitized = sanitize_commit_output(
            "```text\nHere is the commit:\n- `fix(core): handle retries.`\n\nExplains the fix.\n```",
        )
        .expect("wrapped provider output should sanitize");

        assert_eq!(sanitized, "fix(core): handle retries\n\nExplains the fix.");
    }

    #[test]
    fn sanitize_accepts_unscoped_conventional_header() {
        assert_eq!(
            sanitize_commit_output("\"feat: add provider dispatch\"").unwrap(),
            "feat: add provider dispatch"
        );
    }

    #[test]
    fn sanitize_rejects_unrecognizable_provider_output() {
        let err = sanitize_commit_output("This changes some files").expect_err("invalid output");

        assert!(err.contains("missing conventional commit header"));
    }

    #[test]
    fn validates_conventional_commit_headers() {
        assert!(validate_conventional_commit(
            "feat(core): add provider dispatch"
        ));
        assert!(validate_conventional_commit("fix: repair retry handling"));
        assert!(!validate_conventional_commit("repair retry handling"));
    }
}
