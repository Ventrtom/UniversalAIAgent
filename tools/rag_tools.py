from __future__ import annotations

import os
import openai
from datetime import datetime
from pathlib import Path
from typing import List
from dotenv import load_dotenv
load_dotenv()

from langchain.tools import Tool
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document
from services.self_inspection import _count_tokens

# Opt‑out from OpenAI telemetry
os.environ.setdefault("OPENAI_TELEMETRY", "0")
if hasattr(openai, "telemetry") and hasattr(openai.telemetry, "TelemetryClient"):
    openai.telemetry.TelemetryClient.capture = lambda *_, **__: None

# ───────────────────────── Helper: Robust TXT Saver ──────────────────────────
CHROMA_DIR = os.getenv("CHROMA_DIR_V2", "data")
OUTPUT_DIR = Path("files")
OUTPUT_DIR.mkdir(exist_ok=True)


def _save_to_txt(
    data: str,
    prefix: str = "research",
    filename: str | None = None,
) -> str:
    """Save *data* into a new TXT file and report the path.

    Parameters
    ----------
    data : str
        The text content to persist.
    prefix : str, optional
        Fallback prefix if *filename* is not provided.
    filename : str | None, optional
        Explicit file name (with or without ``.txt`` extension).
    """

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if filename:
        base = filename if filename.endswith(".txt") else f"{filename}.txt"
    else:
        base = f"{prefix}_{timestamp}.txt"
    path = OUTPUT_DIR / base
    path.write_text(data, encoding="utf-8")
    return f"Data saved to {path.as_posix()}"


save_tool = Tool(
    name="save_text_to_file",
    func=_save_to_txt,
    description=(
        """
        Purpose
        -------
        Persist any piece of plain text to a timestamped *.txt* file under ./files/.
        Ideal for jotting down idea drafts, SWOT analyses, user-stories, or scratch notes
        that you might want to share later.

        Parameters
        ----------
        data      : str   (required) – the full body of text to save.
        prefix    : str   (optional, default "research") – fallback prefix for auto naming.
        filename  : str   (optional) – desired file name; ``.txt`` will be appended
                if missing. When omitted, the name defaults to
                ``<prefix>_YYYY-MM-DD_HH-MM-SS.txt``.

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
_vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=_embeddings)


def _rag_lookup(query: str, max_tokens: int = 1200) -> str:
    k = max(2, min(8, 2 + len(query) // 50))
    docs: List[Document] = _vectorstore.max_marginal_relevance_search(
        query,
        k=k,
        fetch_k=max(20, k * 4),
        lambda_mult=0.5,
    )
    if not docs:
        return "Žádné interní dokumenty se k dotazu nenašly."

    seen: set[str] = set()
    unique: List[Document] = []
    for d in docs:
        meta = d.metadata or {}
        doc_id = meta.get("document_id") or meta.get("page_id") or meta.get("file_id") or meta.get("id")
        if doc_id and doc_id in seen:
            continue
        if doc_id:
            seen.add(doc_id)
        unique.append(d)

    remaining = max_tokens
    docs_left = len(unique)
    parts: list[str] = []
    for d in unique:
        if remaining <= 0:
            break
        tokens_share = max(1, remaining // docs_left)
        content_tokens = _count_tokens(d.page_content)
        if content_tokens <= tokens_share:
            snippet = d.page_content
        else:
            keep_ratio = tokens_share / content_tokens
            char_limit = max(1, int(len(d.page_content) * keep_ratio))
            snippet = d.page_content[:char_limit]
        parts.append(snippet)
        used = _count_tokens(snippet)
        remaining -= used
        docs_left -= 1

    return "\n\n".join(parts)


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
