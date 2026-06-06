//! Provider prompt and commit-message post-processing.

use crate::preferences::{Preferences, PromptProfile};

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

pub fn selected_prompt_profile(preferences: &Preferences) -> PromptProfile {
    preferences
        .prompt_profiles
        .iter()
        .find(|profile| profile.id == preferences.default_prompt_profile)
        .cloned()
        .or_else(|| preferences.prompt_profiles.first().cloned())
        .unwrap_or_else(PromptProfile::default_commit)
}

pub fn build_prompt_with_profile(diff: &str, context: &str, profile: &PromptProfile) -> String {
    build_prompt_with_profile_style(diff, context, "conventional", profile)
}

pub fn build_prompt_with_profile_style(
    diff: &str,
    context: &str,
    style: &str,
    profile: &PromptProfile,
) -> String {
    let prepared_diff = prepare_diff_for_prompt(diff, context);
    let mut prompt = build_prompt(&prepared_diff, context, style);
    let instruction = profile.user_instruction.trim();
    if !instruction.is_empty() {
        prompt.push_str("\n\nUSER PREFERENCES:\n");
        prompt.push_str(instruction);
    }
    prompt
}

pub fn prepare_diff_for_prompt(diff: &str, context: &str) -> String {
    let file_path_hint = context
        .split_once("File:")
        .map(|(_, value)| value.trim())
        .unwrap_or_default();
    let mut diff_for_prompt = diff.to_string();
    let binary_summary = if is_binary_diff(diff) && !looks_like_text_file(file_path_hint) {
        let snippet = diff.chars().take(400).collect::<String>();
        Some(format!(
            "Binary diff detected.\nFile hint: {}\nGit reported: {}",
            if file_path_hint.is_empty() {
                "unknown"
            } else {
                file_path_hint
            },
            if snippet.trim().is_empty() {
                "<no additional details>"
            } else {
                snippet.trim()
            }
        ))
    } else {
        None
    };

    if diff_for_prompt.len() > 12_000 {
        let head = take_chars(&diff_for_prompt, 8_000);
        let tail = take_last_chars(&diff_for_prompt, 2_000);
        diff_for_prompt = format!("{head}\n...\n{tail}");
    }

    let cleaned_diff = clean_diff_for_llm(&diff_for_prompt);
    if let Some(summary) = binary_summary {
        format!("{summary}\n\n{cleaned_diff}").trim().to_string()
    } else {
        cleaned_diff
    }
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

fn is_binary_diff(diff: &str) -> bool {
    diff.lines().any(|line| {
        (line.starts_with("Binary files") && line.contains(" differ"))
            || line.contains("GIT binary patch")
    })
}

fn looks_like_text_file(file_path: &str) -> bool {
    if file_path.is_empty() {
        return false;
    }
    const TEXT_EXTENSIONS: &[&str] = &[
        ".py", ".pyi", ".pyx", ".pxd", ".js", ".ts", ".jsx", ".tsx", ".css", ".scss", ".sass",
        ".less", ".html", ".htm", ".xml", ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".md",
        ".rst", ".txt", ".csv", ".tsv", ".java", ".c", ".cpp", ".h", ".hpp", ".go", ".rs", ".rb",
        ".php", ".sh", ".bash", ".zsh", ".ps1", ".bat", ".cmd", ".gradle", ".make", ".mk",
        ".cmake",
    ];
    let lower = file_path.to_ascii_lowercase();
    TEXT_EXTENSIONS
        .iter()
        .any(|extension| lower.ends_with(extension))
        || ["src/", "lib/", "app/", "kcmt/"]
            .iter()
            .any(|directory| lower.starts_with(directory))
        || ["/src/", "/lib/", "/app/", "/kcmt/"]
            .iter()
            .any(|directory| lower.contains(directory))
}

fn clean_diff_for_llm(diff: &str) -> String {
    diff.trim()
        .lines()
        .map(|line| {
            if line.contains("--- /dev/null") {
                "--- (new file)"
            } else if line.contains("+++ /dev/null") {
                "+++ (deleted)"
            } else {
                line
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn take_chars(value: &str, count: usize) -> String {
    value.chars().take(count).collect()
}

fn take_last_chars(value: &str, count: usize) -> String {
    let mut tail = value.chars().rev().take(count).collect::<Vec<_>>();
    tail.reverse();
    tail.into_iter().collect()
}

fn is_conventional_type(value: &str) -> bool {
    CONVENTIONAL_TYPES.contains(&value)
}

fn is_wrapped(value: &str, delimiter: char) -> bool {
    value.starts_with(delimiter) && value.ends_with(delimiter) && value.len() >= 2
}

#[cfg(test)]
mod tests {
    use super::{
        build_prompt, build_prompt_with_profile, prepare_diff_for_prompt, sanitize_commit_output,
        validate_conventional_commit,
    };
    use crate::preferences::PromptProfile;

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
    fn build_prompt_with_profile_appends_user_instruction() {
        let profile = PromptProfile {
            id: "team".to_string(),
            name: "Team".to_string(),
            system_instruction: "system".to_string(),
            user_instruction: "Prefer repo-specific scopes.".to_string(),
        };

        let prompt = build_prompt_with_profile("diff", "File: app.py", &profile);

        assert!(prompt.contains("STRICT REQUIREMENTS"));
        assert!(prompt.contains("USER PREFERENCES:\nPrefer repo-specific scopes."));
    }

    #[test]
    fn prepare_diff_summarizes_binary_changes_for_prompt() {
        let prepared = prepare_diff_for_prompt(
            "Binary files a/assets/logo.png and b/assets/logo.png differ",
            "File: assets/logo.png",
        );

        assert!(prepared.contains("Binary diff detected."));
        assert!(prepared.contains("File hint: assets/logo.png"));
        assert!(prepared.contains("Git reported: Binary files"));
    }

    #[test]
    fn prepare_diff_preserves_text_file_binary_markers() {
        let diff = "Binary files a/src/generated.py and b/src/generated.py differ";
        let prepared = prepare_diff_for_prompt(diff, "File: src/generated.py");

        assert!(!prepared.contains("Binary diff detected."));
        assert_eq!(prepared, diff);
    }

    #[test]
    fn prepare_diff_treats_extensionless_code_dir_files_as_text() {
        let diff = "Binary files a/src/tool and b/src/tool differ";
        let prepared = prepare_diff_for_prompt(diff, "File: src/tool");

        assert!(!prepared.contains("Binary diff detected."));
        assert_eq!(prepared, diff);
    }

    #[test]
    fn prepare_diff_truncates_large_diffs_and_cleans_null_paths() {
        let large = format!("--- /dev/null\n+++ b/generated.txt\n{}", "x".repeat(13_000));
        let prepared = prepare_diff_for_prompt(&large, "File: generated.txt");

        assert!(prepared.len() < large.len());
        assert!(prepared.contains("--- (new file)"));
        assert!(prepared.contains("\n...\n"));
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
