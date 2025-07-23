# services/input_loader.py

"""
Input Loader – import files from ./files do Chroma vektorové DB
---------------------------------------------------------------
➤ Přetahuje soubory z lokální složky `./files` (případně podmnožinu podle
  argumentu) a indexuje je pro RAG dotazy.

Argumenty (string):
    ""            → plný sken (přeskočí už zaindexované soubory)
    "force"       → znovu zaindexuje všechno (ignoruje fingerprinty)
    "pdf" / ".pdf"→ jen danou příponu
    "subdir/foo"  → jen daný (pod)adresář
    "file.ext"    → jen konkrétní soubor

Vrací:
    Markdown report – jedna řádka na každý nově naimportovaný soubor.
"""
from __future__ import annotations

import hashlib
import inspect
from pathlib import Path
import os
from typing import List

from langchain_chroma import Chroma
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores.utils import filter_complex_metadata

# ---- konfigurace -----------------------------------------------------------
CHROMA_DIR = os.getenv("CHROMA_DIR_V2", "data")
INPUT_DIR = Path("files")
INPUT_DIR.mkdir(exist_ok=True)

_embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
_vectorstore = Chroma(
    collection_name="rag_store",
    embedding_function=_embeddings,
    persist_directory=CHROMA_DIR,
)

# ---- utilitky --------------------------------------------------------------
def _file_id(path: Path) -> str:
    """Fingerprint souboru = cesta::md5(content)."""
    h = hashlib.md5(path.read_bytes()).hexdigest()  # noqa: S324 – lokální hash
    return f"{path.as_posix()}::{h}"


def _already_indexed(file_id: str) -> bool:
    col = _vectorstore._collection
    return file_id in (m.get("file_id") for m in col.get(include=["metadatas"])["metadatas"])


def _summarise(text: str, llm: ChatOpenAI) -> str:
    prompt = "Shrň následující dokument do max 1 věty (čeština):\n" + text[:4000]
    return llm.invoke(prompt).content.strip()


# ---- robustní sanitizace metadat ------------------------------------------
# API detekujeme dynamicky – pokud selže volání s Documentem, zkusíme dict
def _sanitize_metadata(doc: Document) -> None:
    """Zaručí, že v metadata zůstanou jen skaláry a zavolá korektní variantu
    filter_complex_metadata bez ohledu na verzi knihovny."""
    try:
        # 🅰️ novější API (bere Document, mutuje in-place)
        filter_complex_metadata(doc)
    except Exception:
        try:
            # 🅱️ starší API (bere dict, vrací dict)
            doc.metadata = filter_complex_metadata(doc.metadata)  # type: ignore[arg-type]
        except Exception:
            # 🆘 poslední záchrana – ručně převést neskalární hodnoty na str
            doc.metadata = {
                k: (str(v) if isinstance(v, (list, dict, set, tuple)) else v)
                for k, v in doc.metadata.items()
            }


# ----------------------------------------------------------------------------
def process_input_files(arg: str = "") -> str:
    """Hlavní API volané LangChain toolem."""
    arg = arg.strip()
    force_reindex = arg.lower() == "force"

    # -- vyhodnocení filtru ---------------------------------------------------
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

    paths: List[Path] = (
        [single_target]
        if single_target
        else [p for p in (dir_filter or INPUT_DIR).rglob("*") if p.is_file()]
    )
    if ext_filter:
        paths = [p for p in paths if p.suffix.lower().lstrip(".") == ext_filter]
    if not paths:
        return "Nenalezeny žádné soubory odpovídající zadání."

    # -- načítání & chunkování ------------------------------------------------
    from langchain_unstructured import UnstructuredLoader  # lazy import

    new_docs: List[Document] = []
    reports: List[str] = []
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    for path in paths:
        fid = _file_id(path)
        if not force_reindex and _already_indexed(fid):
            continue

        # 1️⃣ načti dokument(y)
        try:
            raw_docs = UnstructuredLoader(str(path)).load()
        except Exception as exc:  # pragma: no cover
            reports.append(f"- {path.name}: ❌ Nepodařilo se načíst ({exc})")
            continue

        # 2️⃣ normalizuj na Document
        docs: List[Document] = []
        for item in raw_docs:
            if isinstance(item, Document):
                docs.append(item)
            elif isinstance(item, tuple) and len(item) == 2:
                content, meta = item
                docs.append(Document(page_content=str(content), metadata=dict(meta or {})))
            else:
                docs.append(Document(page_content=str(item), metadata={}))

        # 3️⃣ metadata & sanitizace
        base_meta = {"source": "input_folder", "file_path": str(path), "file_id": fid}
        for d in docs:
            d.metadata.update(base_meta)
            _sanitize_metadata(d)

        # 4️⃣ split + přidej do sbírky
        new_docs.extend(splitter.split_documents(docs))

        # 5️⃣ shrnutí pro report
        summary = _summarise(" ".join(d.page_content for d in docs)[:8000], llm)
        reports.append(f"- {path.name}: {summary}")

    if new_docs:
        _vectorstore.add_documents(new_docs)  # Chroma se uloží automaticky

    if not reports:
        return "Žádné nové soubory nebyly naimportovány."
    return f"✅ Naimportováno {len(reports)} souborů:\n" + "\n".join(reports)
