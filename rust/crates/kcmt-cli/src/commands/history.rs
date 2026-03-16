use std::env;
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

pub(crate) fn snapshot_path(repo_path: &Path) -> PathBuf {
    state_dir(repo_path).join("last_run.json")
}

pub(crate) fn state_dir(repo_path: &Path) -> PathBuf {
    config_home().join("repos").join(repo_namespace(repo_path))
}

fn config_home() -> PathBuf {
    if let Ok(home) = env::var("KCMT_CONFIG_HOME") {
        if !home.trim().is_empty() {
            return PathBuf::from(home);
        }
    }

    if let Ok(home) = env::var("XDG_CONFIG_HOME") {
        if !home.trim().is_empty() {
            return PathBuf::from(home).join("kcmt");
        }
    }

    if let Ok(home) = env::var("HOME") {
        if !home.trim().is_empty() {
            return PathBuf::from(home).join(".config").join("kcmt");
        }
    }

    PathBuf::from(".kcmt")
}

fn repo_namespace(repo_path: &Path) -> String {
    let normalized = normalize_repo_path(repo_path.to_path_buf());
    let digest = Sha256::digest(normalized.to_string_lossy().as_bytes());
    let digest_hex = format!("{digest:x}");
    let safe_tail = normalized
        .file_name()
        .and_then(|name| name.to_str())
        .map(sanitize_tail)
        .filter(|tail| !tail.is_empty())
        .unwrap_or_else(|| "repo".to_string());
    format!("{}-{}", safe_tail, &digest_hex[..8])
}

fn sanitize_tail(raw: &str) -> String {
    raw.chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || matches!(ch, '.' | '_' | '-') {
                ch
            } else {
                '-'
            }
        })
        .collect()
}

pub(crate) fn normalize_repo_path(repo_path: PathBuf) -> PathBuf {
    if repo_path.is_absolute() {
        repo_path
    } else {
        env::current_dir()
            .unwrap_or_else(|_| PathBuf::from("."))
            .join(repo_path)
    }
}

#[cfg(test)]
mod tests {
    use super::repo_namespace;
    use std::path::PathBuf;

    #[test]
    fn uses_python_compatible_snapshot_namespace() {
        let repo = PathBuf::from("/tmp/example repo");
        assert_eq!(repo_namespace(&repo), "example-repo-d305539e");
    }
}
