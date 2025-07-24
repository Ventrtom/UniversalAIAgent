from __future__ import annotations

"""LangChain tool exposing :func:`services.clear_memory`."""

from langchain.tools import Tool

from services import clear_memory


def _clear(mem: str = "all") -> str:
    """Wrapper for :func:`clear_memory` used by the agent."""
    return clear_memory(mem)


clear_rag_memory_tool = Tool(
    name="clear_rag_memory",
    func=_clear,
    description=(
        "Delete persistent vector-store memory. Argument can be 'kb_docs',"
        " 'chat_memory' or 'all'."
    ),
    handle_tool_error=True,
)

__all__ = ["clear_rag_memory_tool"]
