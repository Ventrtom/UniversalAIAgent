"""Utilities for working with the RAG vector store."""
from __future__ import annotations

import os
import json
from pathlib import Path
from datetime import datetime
from typing import List

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import DirectoryLoader, ConfluenceLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document
from .self_inspection import _count_tokens


# ───────────────────────── Helper: Robust TXT Saver ───────────────────────────
CHROMA_DIR = os.getenv("CHROMA_DIR_V2", "data")
OUTPUT_DIR = Path("files")
OUTPUT_DIR.mkdir(exist_ok=True)


def save_text_to_file(
    data: str,
    prefix: str = "research",
    filename: str | None = None,
) -> str:
    """Save *data* into a new TXT file and report the path.

    If *filename* is provided, it will be used (``.txt`` appended when missing).
    Otherwise a name ``<prefix>_YYYY-MM-DD_HH-MM-SS.txt`` is generated.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if filename:
        base = filename if filename.endswith(".txt") else f"{filename}.txt"
    else:
        base = f"{prefix}_{timestamp}.txt"
    path = OUTPUT_DIR / base
    path.write_text(data, encoding="utf-8")
    return f"Data saved to {path.as_posix()}"


# ───────────────────────── Vector store builder ─────────────────────────────--

def build_vectorstore(docs_path: str = "./data", persist_directory: str = CHROMA_DIR) -> None:
    """Create a Chroma vector store from text documents."""
    load_dotenv()
    loader = DirectoryLoader(docs_path, glob="**/*.txt")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    Chroma.from_documents(documents=chunks, embedding=embedding_model, persist_directory=persist_directory)


# ───────────────────────── RAG Retriever over Chroma ─────────────────────────-
_embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
_vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=_embeddings)


def rag_lookup(query: str, max_tokens: int = 1200) -> str:
    """Return adaptively selected relevant documents from the vector store."""
    k = max(2, min(8, 2 + len(query) // 50))
    docs: List[Document] = _vectorstore.max_marginal_relevance_search(
        query,
        k=k,
        fetch_k=max(20, k * 4),
        lambda_mult=0.5,
    )
    if not docs:
        return "Žádné interní dokumenty se k dotazu nenašly."

    # Deduplicate by known document identifiers
    seen: set[str] = set()
    unique: List[Document] = []
    for d in docs:
        meta = d.metadata or {}
        doc_id = meta.get("document_id") or meta.get("page_id") or meta.get(
            "file_id"
        ) or meta.get("id")
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
            # approximate trimming by character ratio
            keep_ratio = tokens_share / content_tokens
            char_limit = max(1, int(len(d.page_content) * keep_ratio))
            snippet = d.page_content[:char_limit]
        parts.append(snippet)
        used = _count_tokens(snippet)
        remaining -= used
        docs_left -= 1

    return "\n\n".join(parts)


# ───────────────────────── Confluence loader ----------------------------------

def load_confluence_pages(config_path: Path | str = "config.json") -> None:
    """Load Confluence pages referenced in *config_path* into the vector store."""
    load_dotenv()
    token = os.getenv("JIRA_AUTH_TOKEN")
    if not token:
        raise RuntimeError("Missing JIRA_AUTH_TOKEN in environment")

    cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
    conf_cfg = cfg.get("confluence", {})
    base_url = conf_cfg.get("base_url")
    user = conf_cfg.get("user")
    ancestor_ids = conf_cfg.get("ancestor_ids", [])
    if not ancestor_ids:
        return

    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
    existing_ids = {
        str(m.get("page_id") or m.get("id"))
        for m in vectorstore._collection.get(include=["metadatas"])["metadatas"]
        if m.get("page_id") or m.get("id")
    }

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    new_chunks: List[Document] = []

    for anc in ancestor_ids:
        loader = ConfluenceLoader(
            url=base_url,
            username=user,
            api_key=token,
            cql=f"ancestor={anc}",
            include_archived_content=False,
            include_restricted_content=False,
        )
        docs = loader.load()
        fresh = []
        for doc in docs:
            pid = str(doc.metadata.get("id") or doc.metadata.get("page_id"))
            if pid in existing_ids:
                continue
            doc.metadata["page_id"] = pid
            fresh.append(doc)
        if not fresh:
            continue
        new_chunks.extend(splitter.split_documents(fresh))

    if new_chunks:
        vectorstore.add_documents(new_chunks)

