import ast
from pathlib import Path

def run_ast_analysis():
    public_api = []
    modules = []

    for py in Path(".").rglob("*.py"):
        if ".venv" in str(py):
            continue
        tree = ast.parse(py.read_text())
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                if not node.name.startswith("_"):
                    public_api.append(f"{py}:{node.name}")
        modules.append(str(py))

    return {
        "public_api": public_api,
        "module_count": len(modules),
    }