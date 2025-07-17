# agent/__init__.py
from importlib import import_module
import os

_selected = os.getenv("AGENT_VERSION", "v1").lower()   # v1 | v2
if _selected == "v2":
    core2 = import_module(".core2", __name__)
    handle_query = core2.handle_query
    agent_workflow = core2.agent_workflow
else:
    core = import_module(".core", __name__)
    handle_query = core.handle_query
    agent_executor = core.agent_executor
    ResearchResponse = core.ResearchResponse
    parser = core.parser

__all__ = [
    "handle_query",
    "agent_executor",
    "agent_workflow",
    "ResearchResponse",
    "parser",
]
