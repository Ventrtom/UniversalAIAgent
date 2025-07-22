"""LangChain Tool for updating the kb_docs memory."""
from langchain.tools import Tool
from services.kb_loader import update_kb
import asyncio


async def _update_kb(arg: str = "") -> str:
    return await asyncio.to_thread(update_kb, arg)

def _update_kb_sync(arg: str = "") -> str:
    return asyncio.run(_update_kb(arg))

kb_loader_tool = Tool(
    name="kb_loader",
    func=_update_kb_sync,
    coroutine=_update_kb,
    description=(
        "Synchronise the long-term knowledge base (`kb_docs`) with data from "
        "Confluence and files dropped in `./input`.\n"
        "Call without arguments to load both sources.\n"
        "Use 'confluence' to sync only Confluence pages or 'files' (optionally "
        "'files:<filter>') to import local files. The filter accepts the same"
        " options as `process_input_files`."
    ),
    handle_tool_error=True,
)

__all__ = ["kb_loader_tool"]
