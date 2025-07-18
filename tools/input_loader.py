# tools/input_loader.py
"""
LangChain Tool: process_input_files
-----------------------------------
Wrapper, který zpracuje volitelný `arg` od agenta a přepošle ho do služební
funkce.  Díky tomu může agent jemně řídit import (viz docstring v services).
"""
from langchain.tools import Tool
from services.input_loader import process_input_files


def _process_input_files(arg: str = "") -> str:
    """Proxy, která předá argument dál do business-logiky."""
    return process_input_files(arg)


process_input_tool = Tool(
    name="process_input_files",
    func=_process_input_files,
    description=(
        """
        Purpose
        -------
        Scan the local **`./input`** drop-folder, import every file that matches the
        given filter into the Chroma vector database so RAG queries can reference
        them, and return a one-sentence Markdown summary per file.

        Parameters
        ----------
        arg : str, optional — controls behaviour

            • ""            → default full scan (skip already-indexed files)  
            • "force"       → re-index **everything** (ignore fingerprints)  
            • "pdf" / ".pdf"→ import *only* files with this extension  
            • "subdir/foo"  → limit scan to the given (sub)directory  
            • "file.ext"    → import exactly the specified file  

        Returns
        -------
        Markdown report, e.g.:

            ✅ Imported 2 files:
            - report_Q3.pdf: Overview of KPI results for Q3 2024 …
            - shifts.xlsx  : Daily OEE and breakdown statistics …

        Use this tool whenever the user wants to make newly dropped local files
        available to the retrieval pipeline or asks “what did I just upload?”.
        """
    ),
    handle_tool_error=True,
)

__all__ = ["process_input_tool"]
