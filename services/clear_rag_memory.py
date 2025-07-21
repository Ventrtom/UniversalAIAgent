#!/usr/bin/env python3
"""Utilities for wiping the persistent RAG memory.

The module originally served only as a manual script.  It now exposes the
``clear_memory`` function so it can be invoked programmatically (for example via
a LangChain tool) while keeping a simple CLI for ad‑hoc use.

The persistent Chroma DB lives in ``$CHROMA_DIR_V2`` (defaults to ``data/``) and
contains two collections used by :mod:`agent.core2`:

``kb_docs`` – knowledge base documents.
``chat_memory`` – long‑term chat transcript.

``clear_memory`` can wipe either collection individually or remove the whole DB
directory (optionally creating a timestamped backup).  If ``chromadb`` is not
installed the validation falls back to a crude file‑size based check.
"""

from __future__ import annotations

import argparse
import os
import datetime as _dt
from pathlib import Path
import shutil as _sh
import sys as _sys
from typing import Optional

try:
    import chromadb
except ImportError:  # pragma: no cover
    chromadb = None  # type: ignore

DB_DIR = Path(os.getenv("CHROMA_DIR_V2", "data"))


# ------------------------------------------------------------------------------
# Utility: počet vektorů v DB
# ------------------------------------------------------------------------------

def _count_vectors(collection: Optional[str] = None) -> int:
    """Return number of embeddings in *collection* or all collections."""
    if not DB_DIR.exists():
        return 0

    if chromadb is None:
        # Fallback: count files larger than 1 KiB (best effort)
        return sum(1 for p in DB_DIR.rglob("*") if p.is_file() and p.stat().st_size > 1024)

    client = chromadb.PersistentClient(path=str(DB_DIR))
    if collection:
        try:
            col = client.get_collection(collection)
            return col.count()
        except Exception:
            return 0

    total = 0
    for meta in client.list_collections():
        col = client.get_collection(meta.name)
        total += col.count()
    return total


def _validate_empty(collection: Optional[str] = None) -> bool:
    """Return ``True`` when there are no vectors left."""
    leftover = _count_vectors(collection)
    if leftover == 0:
        print("✅ RAG paměť je prázdná, hotovo.")
        return True

    print(f"❌ V paměti zůstalo ještě {leftover} vektorů!", file=_sys.stderr)
    return False


# ------------------------------------------------------------------------------
# Hlavní operace (wipe / backup)
# ------------------------------------------------------------------------------

def _wipe_rag_db(force: bool, backup: bool) -> None:
    """Delete or backup the entire DB directory and recreate it."""
    if not DB_DIR.exists():
        print(f"Adresář '{DB_DIR.name}' neexistuje – RAG paměť je už prázdná.")
        return

    if not force:
        prompt = f"Toto {'zálohuje a poté ' if backup else ''}SMAŽE '{DB_DIR}'. Pokračovat? [y/N] "
        if input(prompt).strip().lower() not in {"y", "yes"}:
            print("Operace zrušena.")
            return

    if backup:
        stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        dest = DB_DIR.with_name(f"{DB_DIR.name}_backup_{stamp}")
        _sh.move(str(DB_DIR), dest)
        print(f"Vektorová DB byla přesunuta do: {dest}")
    else:
        _sh.rmtree(DB_DIR)
        print("Vektorová DB byla nenávratně smazána.")

    DB_DIR.mkdir(exist_ok=True)
    print(f"Vytvořen nový prázdný '{DB_DIR.name}'.")


def _clear_collection(name: str) -> None:
    if chromadb is None:
        raise RuntimeError("chromadb not installed")
    client = chromadb.PersistentClient(path=str(DB_DIR))
    try:
        client.delete_collection(name)
    except Exception:
        pass
    client.get_or_create_collection(name)


def clear_memory(target: str = "all", backup: bool = False) -> str:
    """Clear selected persistent memory.

    Parameters
    ----------
    target : str
        ``"kb_docs"``, ``"chat_memory"`` or ``"all"``.
    backup : bool
        When clearing the whole DB, move the directory to a timestamped backup
        before recreating it.
    """

    t = target.lower().strip()
    mapping = {
        "kb": "kb_docs",
        "kb_docs": "kb_docs",
        "chat": "chat_memory",
        "chat_memory": "chat_memory",
    }

    if t in {"all", "db", "database"}:
        _wipe_rag_db(force=True, backup=backup)
        _validate_empty()
        return "Vector store wiped"

    name = mapping.get(t)
    if not name:
        raise ValueError(f"Unknown memory '{target}'")

    _clear_collection(name)
    _validate_empty(name)
    return f"Collection '{name}' cleared"


# ------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------

def main() -> None:  # noqa: D401 – krátký název je OK
    parser = argparse.ArgumentParser(
        description="Vymaže (nebo zazálohuje) perzistentní RAG paměť a ověří, že je prázdná."
    )
    parser.add_argument("memory", nargs="?", default="all", help="kb_docs, chat_memory or all")
    parser.add_argument("-y", "--yes", action="store_true", help="Provede akci bez interaktivního potvrzení.")
    parser.add_argument("--backup", action="store_true", help="Před smazáním vytvoří timestampovanou zálohu.")
    parser.add_argument("--check", action="store_true", help="Pouze zkontroluje, zda je paměť prázdná (nic nemaže).")
    args = parser.parse_args()

    if args.check:
        ok = _validate_empty(args.memory if args.memory != "all" else None)
        _sys.exit(0 if ok else 1)

    if not args.yes:
        confirm = input(f"Opravdu chcete smazat '{args.memory}'? [y/N] ").strip().lower()
        if confirm not in {"y", "yes"}:
            print("Operace zrušena.")
            return

    print(clear_memory(args.memory, backup=args.backup))
    ok = _validate_empty(args.memory if args.memory != "all" else None)
    _sys.exit(0 if ok else 1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        _sys.exit("\nPřerušeno uživatelem.")
