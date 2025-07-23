"""LangChain tool exposing services.self_inspection.agent_introspect."""
from __future__ import annotations

from langchain.tools import Tool

from services.self_inspection import agent_introspect


self_inspection_tool = Tool(
    name="self_inspection",
    func=lambda: agent_introspect(),
    description=(
        "Provide a concise summary of the agent's architecture, capabilities and limitations."
    ),
    handle_tool_error=True,
)

__all__ = ["self_inspection_tool"]
