import ast
import pathlib


def test_chat_memory_initialized_once():
    source = pathlib.Path("agent/core.py").read_text()
    tree = ast.parse(source)
    assignments = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Attribute)
        and getattr(target.value, "id", "") == "_short_term_memory"
        and target.attr == "chat_memory"
    ]
    assert len(assignments) == 1
