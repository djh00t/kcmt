def format_subject(kind: str, scope: str) -> str:
    if scope:
        return f"{kind}({scope}): update"
    return f"{kind}: update"
