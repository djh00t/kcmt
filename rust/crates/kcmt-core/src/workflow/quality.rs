const CONVENTIONAL_TYPES: &[&str] = &[
    "build", "chore", "ci", "config", "deps", "docs", "feat", "fix", "infra", "perf", "refactor",
    "schema", "security", "style", "test",
];

pub fn is_conventional_commit(message: &str) -> bool {
    let header = match message
        .lines()
        .next()
        .map(str::trim)
        .filter(|line| !line.is_empty())
    {
        Some(header) => header,
        None => return false,
    };
    let Some((prefix, subject)) = header.split_once(": ") else {
        return false;
    };
    if subject.trim().is_empty() {
        return false;
    }
    let prefix = prefix.strip_suffix('!').unwrap_or(prefix);
    let commit_type = prefix
        .split_once('(')
        .map(|(commit_type, scope)| {
            if !scope.ends_with(')') || scope.trim_end_matches(')').is_empty() {
                return "";
            }
            commit_type
        })
        .unwrap_or(prefix);
    CONVENTIONAL_TYPES.contains(&commit_type)
}

pub fn sanitize_commit_message(raw: &str) -> Option<String> {
    let cleaned = raw
        .replace("```text", "")
        .replace("```plaintext", "")
        .replace("```", "");
    let lines: Vec<&str> = cleaned.lines().collect();
    for (index, line) in lines.iter().enumerate() {
        let candidate = line.trim();
        if !is_conventional_commit(candidate) {
            continue;
        }
        let mut selected = vec![candidate.to_string()];
        for body_line in lines.iter().skip(index + 1) {
            let trimmed = body_line.trim_end();
            if trimmed.starts_with("Here is") {
                continue;
            }
            selected.push(trimmed.to_string());
        }
        while selected.last().is_some_and(|line| line.trim().is_empty()) {
            selected.pop();
        }
        return Some(selected.join("\n"));
    }
    None
}

#[cfg(test)]
mod tests {
    use super::{is_conventional_commit, sanitize_commit_message};

    #[test]
    fn accepts_valid_conventional_commit_headers() {
        assert!(is_conventional_commit("feat(cli): add batch queue"));
        assert!(is_conventional_commit("fix!: change config precedence"));
        assert!(is_conventional_commit("docs: update README"));
    }

    #[test]
    fn rejects_invalid_or_empty_headers() {
        assert!(!is_conventional_commit(""));
        assert!(!is_conventional_commit("updated the thing"));
        assert!(!is_conventional_commit("feat cli add queue"));
    }

    #[test]
    fn sanitizes_verbose_provider_output_to_first_valid_header() {
        let raw = "Here is the commit message:\n\n```text\nfeat(core): add telemetry\n\n- records stage timings\n```";

        let sanitized = sanitize_commit_message(raw).expect("valid header should be recovered");

        assert_eq!(
            sanitized,
            "feat(core): add telemetry\n\n- records stage timings"
        );
    }

    #[test]
    fn returns_none_when_no_valid_header_can_be_recovered() {
        assert_eq!(sanitize_commit_message("no useful commit here"), None);
    }
}
