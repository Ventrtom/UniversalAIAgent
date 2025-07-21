#!/usr/bin/env python3
"""
clear_rag_memory.py
===================

Jednorázové promazání a/nebo validace Chroma DB (adresář daný `CHROMA_DIR_V2`,
výchozí `data/`) používané jako RAG paměť.

• Smaže nebo zazálohuje DB a ověří, že po operaci nezůstaly žádné vektory.
• Závislost na 'chromadb' je volitelná – bez ní proběhne fallback validace přes velikost souborů.

Mazání + automatická validace	
    python clear_rag_memory.py -y	
    Po smazání ověří, jestli je DB opravdu prázdná; pokud ne, skript skončí s Neo (1).

Záloha + validace	
    python clear_rag_memory.py --backup -y	
    Přesune DB do zálohy, vytvoří čistý adresář, zkontroluje.

Pouhá kontrola (např. v CI)	
    python clear_rag_memory.py --check	
    Nemění data, jen vrátí 0/1 podle stavu (lze použít v bash if …).

"""

from __future__ import annotations

import argparse
import os
import datetime as _dt
import pathlib as _pl
import shutil as _sh
import sys as _sys
from typing import Optional

try:
    import chromadb
except ImportError:  # pragma: no cover
    chromadb = None  # type: ignore

ROOT = _pl.Path(__file__).resolve().parent
DB_DIR = ROOT / os.getenv("CHROMA_DIR_V2", "data")


# ------------------------------------------------------------------------------
# Utility: počet vektorů v DB
# ------------------------------------------------------------------------------

def _count_vectors() -> int:
    """
    Sečte celkový počet embeddingů ve všech kolekcích.

    Vrací 0, pokud:
      • DB adresář neexistuje
      • Není dostupná knihovna chromadb a současně je adresář prázdný
    """
    if not DB_DIR.exists():
        return 0

    if chromadb is None:
        # Fallback: spočítáme soubory větší než 1 KiB (pravděp. data)
        return sum(1 for p in DB_DIR.rglob("*") if p.is_file() and p.stat().st_size > 1024)

    client = chromadb.PersistentClient(path=str(DB_DIR))
    total = 0
    for col_meta in client.list_collections():
        col = client.get_collection(col_meta.name)
        total += col.count()
    return total


def _validate_empty() -> bool:
    """Vrátí True, když vektorů == 0."""
    leftover = _count_vectors()
    if leftover == 0:
        print("✅ RAG paměť je prázdná, hotovo.")
        return True

    print(f"❌ V paměti zůstalo ještě {leftover} vektorů!", file=_sys.stderr)
    return False


# ------------------------------------------------------------------------------
# Hlavní operace (wipe / backup)
# ------------------------------------------------------------------------------

def _wipe_rag_db(force: bool, backup: bool) -> None:
    """Smaže nebo zálohuje adresář daný `CHROMA_DIR_V2` a pak jej znovu vytvoří."""
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


# ------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------

def main() -> None:  # noqa: D401 – krátký název je OK
    parser = argparse.ArgumentParser(
        description="Vymaže (nebo zazálohuje) perzistentní RAG paměť a ověří, že je prázdná."
    )
    parser.add_argument("-y", "--yes", action="store_true", help="Provede akci bez interaktivního potvrzení.")
    parser.add_argument("--backup", action="store_true", help="Před smazáním vytvoří timestampovanou zálohu.")
    parser.add_argument("--check", action="store_true", help="Pouze zkontroluje, zda je paměť prázdná (nic nemaže).")
    args = parser.parse_args()

    if args.check:
        ok = _validate_empty()
        _sys.exit(0 if ok else 1)

    _wipe_rag_db(force=args.yes, backup=args.backup)
    ok = _validate_empty()
    _sys.exit(0 if ok else 1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        _sys.exit("\nPřerušeno uživatelem.")
