from __future__ import annotations

import os
import openai
from datetime import datetime
from pathlib import Path
from typing import List

from langchain.tools import Tool
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document

# Opt‑out from OpenAI telemetry
os.environ.setdefault("OPENAI_TELEMETRY", "0")
if hasattr(openai, "telemetry") and hasattr(openai.telemetry, "TelemetryClient"):
    openai.telemetry.TelemetryClient.capture = lambda *_, **__: None

# ───────────────────────── Helper: Robust TXT Saver ──────────────────────────
CHROMA_DIR = os.getenv("CHROMA_DIR_V2", "data")
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)


def _save_to_txt(data: str, prefix: str = "research") -> str:
    """Save *data* into a new TXT file and report the path."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{prefix}_{timestamp}.txt"
    path = OUTPUT_DIR / filename
    path.write_text(data, encoding="utf-8")
    return f"Data saved to {path.as_posix()}"


save_tool = Tool(
    name="save_text_to_file",
    func=_save_to_txt,
    description=(
        """
        Purpose
        -------
        Persist any piece of plain text to a timestamped *.txt* file under ./output/.
        Ideal for jotting down idea drafts, SWOT analyses, user-stories, or scratch notes
        that you might want to share later.

        Parameters
        ----------
        data   : str   (required) – the full body of text to save.
        prefix : str   (optional, default "research") – filename prefix; the final name
                is <prefix>_YYYY-MM-DD_HH-MM-SS.txt.

        Returns
        -------
        Confirmation string with the relative file path.  No file object is returned.

        Caveats
        -------
        • Simply writes to disk – it does NOT version existing files or push anything
        back to Jira/Drive.
        """
    ),
    handle_tool_error=True,
)

# ───────────────────────── RAG Retriever over Chroma ─────────────────────────
_embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
_retriever = Chroma(persist_directory=CHROMA_DIR, embedding_function=_embeddings).as_retriever(search_kwargs={"k": 4})


def _rag_lookup(query: str) -> str:
    docs: List[Document] = _retriever.invoke(query)
    if not docs:
        return "Žádné interní dokumenty se k dotazu nenašly."
    return "\n\n".join(d.page_content for d in docs)


rag_tool = Tool(
    name="rag_retriever",
    func=_rag_lookup,
    description=(
        """
        Purpose
        -------
        Semantic Retrieval-Augmented Generation over Productoo’s **internal knowledge**
        base (embedded in a Chroma DB).  Sources include P4 user documentation,
        roadmap pages, and archived conversation snippets.

        Parameters
        ----------
        query : str – a natural-language question or keyword string.

        Returns
        -------
        Up to 4 highly relevant passages concatenated together.

        Typical use cases
        -----------------
        P4 application feature details, implementation scenarios, or roadmap justifications
        to ground an answer before calling an LLM.
        """
    ),
    handle_tool_error=True,
)

__all__ = ["save_tool", "rag_tool"]
