#!/usr/bin/env python3
"""
rag_confluence_loader.py

Load Confluence pages (and their descendants) into a Chroma vectorstore for RAG.
Skips pages already present in the vectorstore by page_id metadata.
"""

import os
import json
from pathlib import Path

from dotenv import load_dotenv

# ── Suppress Chroma telemetry errors ──────────────────────────────────────────
# Must be set before importing Chroma
os.environ["CHROMA_TELEMETRY"] = "0"

from langchain_community.document_loaders import ConfluenceLoader
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter


def load_config(config_path: Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_existing_page_ids(vectorstore: Chroma) -> set:
    """Retrieve page_id metadata from all documents currently in the vectorstore."""
    col = vectorstore._collection
    metadatas = col.get(include=["metadatas"])["metadatas"]
    return { str(m.get("page_id") or m.get("id")) for m in metadatas if m.get("page_id") or m.get("id") }


def main():
    # ── Load environment and config ─────────────────────────────────────────────
    load_dotenv()
    token = os.getenv("JIRA_AUTH_TOKEN")
    if not token:
        raise RuntimeError("Missing JIRA_AUTH_TOKEN in environment")

    config_path = Path(__file__).parent / "config.json"
    cfg = load_config(config_path)
    conf_cfg = cfg.get("confluence", {})
    base_url = conf_cfg["base_url"]
    user = conf_cfg["user"]
    ancestor_ids = conf_cfg.get("ancestor_ids", [])
    if not ancestor_ids:
        print("No ancestor_ids configured. Nothing to load.")
        return

    # ── Prepare embeddings & vectorstore ────────────────────────────────────────
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma(
        persist_directory="rag_chroma_db",
        embedding_function=embeddings
    )

    existing_ids = get_existing_page_ids(vectorstore)
    print(f"{len(existing_ids)} pages already in vectorstore.")

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    new_chunks = []

    # ── Load & filter Confluence pages ──────────────────────────────────────────
    for anc in ancestor_ids:
        print(f"→ Fetching descendants of Confluence page {anc} …")
        loader = ConfluenceLoader(
            url=base_url,
            username=user,
            api_key=token,
            cql=f"ancestor={anc}",
            include_archived_content=False,
            include_restricted_content=False
        )
        docs = loader.load()
        print(f"   • Retrieved {len(docs)} pages under ancestor {anc}")

        fresh = []
        for doc in docs:
            pid = str(doc.metadata.get("id") or doc.metadata.get("page_id"))
            if pid in existing_ids:
                print(f"     – Skipping already indexed page {pid}")
                continue
            doc.metadata["page_id"] = pid
            fresh.append(doc)

        if not fresh:
            continue

        chunks = splitter.split_documents(fresh)
        print(f"Chunked into {len(chunks)} pieces")
        new_chunks.extend(chunks)

    # ── Add only new chunks to Chroma ───────────────────────────────────────────
    if new_chunks:
        vectorstore.add_documents(new_chunks)
        print(f"Added {len(new_chunks)} new document chunks.")
    else:
        print("No new documents to add.")

if __name__ == "__main__":
    main()
