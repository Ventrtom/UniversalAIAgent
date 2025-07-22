from __future__ import annotations

"""Utilities for listing and reading files in the ``output`` folder."""

from pathlib import Path

from langchain_unstructured import UnstructuredLoader

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

ALLOWED_EXTS = {".txt", ".md", ".csv", ".pdf"}


def list_output_files() -> str:
    """Return Markdown-formatted list of available output files."""
    files = [p.name for p in OUTPUT_DIR.iterdir() if p.is_file() and p.suffix.lower() in ALLOWED_EXTS]
    if not files:
        return "Žádné soubory ve složce output nenalezeny."
    lines = [f"- {name}" for name in sorted(files)]
    return "\n".join(lines)


def read_output_file(name: str) -> str:
    """Return textual content of *name* from ``output`` if supported."""
    name = name.strip()
    if not name:
        return "Nebyl zadán název souboru."
    path = OUTPUT_DIR / name
    if not path.exists() or not path.is_file():
        return f"Soubor '{name}' neexistuje."
    ext = path.suffix.lower()
    if ext not in ALLOWED_EXTS:
        return f"Nepodporovaný typ souboru: {ext}"
    if ext == ".pdf":
        try:
            docs = UnstructuredLoader(str(path)).load()
            return "\n".join(d.page_content for d in docs)
        except Exception as exc:  # pragma: no cover - external call
            return f"Chyba při čtení PDF: {exc}"
    try:
        return path.read_text(encoding="utf-8")
    except Exception as exc:  # pragma: no cover - unexpected errors
        return f"Chyba při čtení souboru '{name}': {exc}"

