import ast
import networkx as nx
from pathlib import Path

def run_dependency_graph():
    graph = nx.DiGraph()

    for py in Path(".").rglob("*.py"):
        tree = ast.parse(py.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for name in node.names:
                    graph.add_edge(py.name, name.name)

    return {
        "nodes": len(graph.nodes),
        "edges": len(graph.edges),
    }