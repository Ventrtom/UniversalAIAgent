from __future__ import annotations

"""Utilities for updating the kb_docs collection."""

import hashlib
import json
import os
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.document_loaders import ConfluenceLoader
from langchain_unstructured import UnstructuredLoader
from langchain_community.vectorstores.utils import filter_complex_metadata

CHROMA_DIR = os.getenv("CHROMA_DIR_V2", "data")
INPUT_DIR = Path("files")
INPUT_DIR.mkdir(exist_ok=True)

_embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
_store = Chroma(
    collection_name="kb_docs",
    embedding_function=_embeddings,
    persist_directory=CHROMA_DIR,
)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _sanitize_metadata(doc: Document) -> None:
    """Ensure metadata contains only simple scalars."""
    try:
        filter_complex_metadata(doc)
    except Exception:
        try:
            doc.metadata = filter_complex_metadata(doc.metadata)  # type: ignore[arg-type]
        except Exception:
            doc.metadata = {
                k: (str(v) if isinstance(v, (list, dict, set, tuple)) else v)
                for k, v in doc.metadata.items()
            }


def _existing_file_map() -> dict[str, str]:
    """Return mapping of file_path to fingerprint."""
    col = _store._collection
    metas = col.get(include=["metadatas"])["metadatas"]
    result = {}
    for m in metas:
        fp = m.get("file_id")
        path = m.get("file_path")
        if fp and path:
            result[str(path)] = str(fp)
    return result


def _file_id(path: Path) -> str:
    h = hashlib.md5(path.read_bytes()).hexdigest()  # noqa: S324 – local hash
    return f"{path.as_posix()}::{h}"


def _existing_pages() -> dict[str, str]:
    col = _store._collection
    metas = col.get(include=["metadatas"])["metadatas"]
    result = {}
    for m in metas:
        pid = m.get("page_id") or m.get("id")
        ch = m.get("content_hash")
        if pid:
            result[str(pid)] = str(ch or "")
    return result


def _delete_by(where: dict) -> None:
    try:
        _store.delete(where=where)
    except Exception:
        pass


def _summarise(text: str, llm: ChatOpenAI) -> str:
    prompt = "Shrň následující dokument do max 1 věty (čeština):\n" + text[:4000]
    return llm.invoke(prompt).content.strip()


# ---------------------------------------------------------------------------
# Local file loader
# ---------------------------------------------------------------------------

def _load_files(arg: str = "") -> List[str]:
    arg = arg.strip()
    force_reindex = arg.lower() == "force"

    ext_filter = None
    single_target: Path | None = None
    dir_filter: Path | None = None

    if arg and not force_reindex:
        if arg.lower().lstrip(".") in {"pdf", "docx", "pptx", "xlsx", "txt", "md", "html", "csv"}:
            ext_filter = arg.lower().lstrip(".")
        else:
            maybe_path = INPUT_DIR / arg
            if maybe_path.exists():
                single_target = maybe_path if maybe_path.is_file() else None
                dir_filter = maybe_path if maybe_path.is_dir() else None

    paths: List[Path] = [single_target] if single_target else [p for p in (dir_filter or INPUT_DIR).rglob("*") if p.is_file()]
    if ext_filter:
        paths = [p for p in paths if p.suffix.lower().lstrip(".") == ext_filter]
    if not paths:
        return []

    existing_map = {} if force_reindex else _existing_file_map()
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    reports: List[str] = []
    new_docs: List[Document] = []

    for path in paths:
        fid = _file_id(path)
        old_fid = existing_map.get(str(path))
        if old_fid == fid:
            continue
        if old_fid and old_fid != fid:
            _delete_by({"file_path": str(path)})
        try:
            raw_docs = UnstructuredLoader(str(path)).load()
        except Exception as exc:  # pragma: no cover
            reports.append(f"- {path.name}: ❌ Nepodařilo se načíst ({exc})")
            continue

        docs: List[Document] = []
        for item in raw_docs:
            if isinstance(item, Document):
                docs.append(item)
            elif isinstance(item, tuple) and len(item) == 2:
                content, meta = item
                docs.append(Document(page_content=str(content), metadata=dict(meta or {})))
            else:
                docs.append(Document(page_content=str(item), metadata={}))

        base_meta = {"source": "input_folder", "file_path": str(path), "file_id": fid}
        for d in docs:
            d.metadata.update(base_meta)
            _sanitize_metadata(d)

        new_docs.extend(splitter.split_documents(docs))
        summary = _summarise(" ".join(d.page_content for d in docs)[:8000], llm)
        reports.append(f"- {path.name}: {summary}")

    if new_docs:
        _store.add_documents(new_docs)
    return reports


# ---------------------------------------------------------------------------
# Confluence loader
# ---------------------------------------------------------------------------

def _load_confluence(config_path: Path | str = "config.json", force: bool = False) -> List[str]:
    load_dotenv()
    token = os.getenv("JIRA_AUTH_TOKEN")
    if not token:
        return ["❌ Missing JIRA_AUTH_TOKEN"]

    cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
    conf_cfg = cfg.get("confluence", {})
    base_url = conf_cfg.get("base_url")
    user = conf_cfg.get("user")
    ancestor_ids = conf_cfg.get("ancestor_ids", [])
    if not ancestor_ids:
        return []

    existing = {} if force else _existing_pages()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    reports: List[str] = []
    new_chunks: List[Document] = []

    for anc in ancestor_ids:
        loader = ConfluenceLoader(
            url=base_url,
            username=user,
            api_key=token,
            cql=f"(id={anc} OR ancestor={anc})",
            include_archived_content=False,
            include_restricted_content=False,
        )
        docs = loader.load()
        for doc in docs:
            pid = str(doc.metadata.get("id") or doc.metadata.get("page_id"))
            content_hash = hashlib.md5(doc.page_content.encode("utf-8")).hexdigest()
            doc.metadata["page_id"] = pid
            doc.metadata["content_hash"] = content_hash
            old = existing.get(pid)
            if old == content_hash:
                continue
            if old and old != content_hash:
                _delete_by({"page_id": pid})
            new_chunks.extend(splitter.split_documents([doc]))
            reports.append(f"- page {pid}")

    if new_chunks:
        _store.add_documents(new_chunks)
    return reports


# ---------------------------------------------------------------------------
# Public orchestrator
# ---------------------------------------------------------------------------

def update_kb(arg: str = "") -> str:
    """Update kb_docs from Confluence and/or local files.

    Parameters
    ----------
    arg : str, optional
        "confluence" – only Confluence pages
        "files" or "files:<filter>" – only local files
        "force" – reindex everything
        any other value is passed as file filter
    """
    arg = (arg or "").strip()
    force = arg.lower() == "force"

    run_conf = arg == "" or arg.startswith("confluence") or force
    file_arg = "" if arg in ("", "confluence") else arg.split(":", 1)[1] if arg.startswith("files:") else arg if not arg.startswith("confluence") else ""

    reports = []
    if run_conf:
        rep = _load_confluence(force=force)
        if rep:
            reports.append("### Confluence\n" + "\n".join(rep))

    if arg == "" or arg.startswith("files") or (arg and not arg.startswith("confluence")) or force:
        rep = _load_files(file_arg if not force else "force")
        if rep:
            reports.append("### Files\n" + "\n".join(rep))

    if not reports:
        return "Žádné nové dokumenty nebyly nahrány."
    return "\n\n".join(reports)

__all__ = ["update_kb"]
