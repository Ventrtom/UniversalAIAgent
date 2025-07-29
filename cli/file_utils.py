"""Utility functions for working with shared files."""

from __future__ import annotations

import logging
import os
from typing import List, Optional, Tuple

import gradio as gr

from .config import FILES_DIR

logger = logging.getLogger(__name__)


def list_files() -> List[str]:
    """Return sorted list of filenames in ``FILES_DIR``."""
    try:
        return sorted(p.name for p in FILES_DIR.iterdir() if p.is_file())
    except Exception as exc:
        logger.error("Error listing files: %s", exc)
        return []


def read_file(fname: str) -> str:
    """Return text content of ``fname`` or diagnostic message."""
    if not fname:
        return ""
    path = FILES_DIR / fname
    try:
        return path.read_text("utf-8")
    except UnicodeDecodeError:
        return f"[Nelze zobrazit: binární soubor] {fname}"
    except FileNotFoundError:
        return f"[Soubor neexistuje] {fname}"
    except Exception as exc:
        logger.error("Error reading file %s: %s", fname, exc)
        return f"[Chyba při čtení souboru] {fname}"


def file_path(fname: Optional[str]) -> Optional[str]:
    """Return absolute path to ``fname`` if it exists."""
    if not fname:
        return None
    p = FILES_DIR / fname
    return str(p) if p.exists() else None


def refresh_choices() -> gr.Dropdown:
    """Update dropdown choices with current file list."""
    choices = list_files()
    return gr.update(choices=choices, value=choices[0] if choices else None)


def file_selected(fname: Optional[str]) -> Tuple[str, gr.File, str]:
    """Handle selection of a file from dropdown."""
    if not fname:
        return "", gr.File(visible=False), ""
    path = file_path(fname)
    preview = read_file(fname)
    return preview, gr.File(value=path, visible=bool(path)), fname


def trigger_download(fname: Optional[str]) -> gr.File:
    """Return file component to download the selected file."""
    if not fname:
        return gr.update(visible=False)
    path = file_path(fname)
    return gr.update(value=path, visible=bool(path))


def save_file(
    content: str, new_name: str, original_name: str
) -> Tuple[gr.Dropdown, gr.File, str]:
    """Persist edited content and optionally rename the file."""
    if not original_name:
        return refresh_choices(), gr.update(visible=False), ""
    old_path = FILES_DIR / original_name
    target_name = new_name.strip() or original_name
    if not os.path.splitext(target_name)[1]:
        target_name += old_path.suffix
    new_path = FILES_DIR / target_name
    if old_path != new_path:
        try:
            old_path.rename(new_path)
        except Exception as exc:
            logger.error("Error renaming file: %s", exc)
            new_path = old_path
    try:
        new_path.write_text(content, encoding="utf-8")
    except Exception as exc:
        logger.error("Error writing file: %s", exc)
    dropdown = refresh_choices()
    return dropdown, gr.update(value=str(new_path), visible=True), new_path.name


def delete_file(fname: str) -> gr.Dropdown:
    """Delete ``fname`` and refresh dropdown."""
    if fname:
        try:
            (FILES_DIR / fname).unlink()
        except Exception as exc:
            logger.error("Error deleting file %s: %s", fname, exc)
    return refresh_choices()


__all__ = [
    "list_files",
    "read_file",
    "file_path",
    "refresh_choices",
    "file_selected",
    "trigger_download",
    "save_file",
    "delete_file",
]
