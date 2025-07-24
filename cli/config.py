"""Shared configuration constants for the local UI."""

from __future__ import annotations

import pathlib

# Directory where user files are stored
FILES_DIR = pathlib.Path("files")
FILES_DIR.mkdir(exist_ok=True)

# Maximum number of conversation messages kept in memory
MAX_HISTORY_LENGTH = 50
