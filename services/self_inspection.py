from __future__ import annotations

"""Self‑inspection utilities for the AI agent.

Generates a concise textual snapshot of the repository and feeds it to
an LLM so that the agent can report *its own* architecture.

───────────────────────────────────────────────────────────────────────────────
### Additions in this build

* **Regex komentářový filtr** – řádky začínající `#` a inline komentáře po
  kódu jsou odstraněny (kromě `#!` she‑bangu a mnemotechniky
  `# …truncated…`). Typicky šetří ≈ 20‑30 % tokenů.
* **Zásobník důležitosti** – soubory jsou seřazeny podle heuristických
  priorit ↓; méně důležité padají první, pokud dochází rozpočet.
  Prioritu/patterny lze změnit v konst. `PRIORITY_PATTERNS` nebo přes env
  `AI_SNAPSHOT_PRIORITY_PATTERNS` (čárkami oddělené regexy – přidají se na
  začátek seznamu).
* Veřejné API beze změny (`agent_introspect()`), takže downstream kód je
  stále kompatibilní.
"""

from pathlib import Path
from io import StringIO
from datetime import datetime
import os
import re
from typing import Iterable, Set, List, Tuple

try:
    import tiktoken  # type: ignore
except ModuleNotFoundError:  # pragma: no cover – optional dependency
    tiktoken = None  # noqa: N816 – keep the name for simplicity

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

__all__: list[str] = ["agent_introspect"]

###############################################################################
# Configuration
###############################################################################

EXCLUDE_FOLDERS: set[str] = {
    ".git",
    "__pycache__",
    ".venv",
    "venv",
    "rag_chroma_db_v2",
    "files",
    "data",
}

EXCLUDE_FILES: set[str] = {
    "project_snapshot.md",
    "merge_project.py",
    ".env",
    ".png",
    ".json",
}

# Importance ranking – regexes evaluated in order; first match ⇒ highest prio
PRIORITY_PATTERNS: List[str] = [
    r"(^|/)agent/",
    r"(^|/)cli/",
    r"(^|/)services/", 
    r"(^|/)tools/", 
    r"run\.py$",
    r"__init__\.py$",
]

###############################################################################
# ENV overrides
###############################################################################

if (_env_folders := os.getenv("AI_SNAPSHOT_EXCLUDE_FOLDERS")):
    EXCLUDE_FOLDERS |= {p.strip() for p in _env_folders.split(",") if p.strip()}
if (_env_files := os.getenv("AI_SNAPSHOT_EXCLUDE_FILES")):
    EXCLUDE_FILES |= {p.strip() for p in _env_files.split(",") if p.strip()}
if (_env_priority := os.getenv("AI_SNAPSHOT_PRIORITY_PATTERNS")):
    # prepend custom patterns to take precedence
    PRIORITY_PATTERNS = [p.strip() for p in _env_priority.split(",") if p.strip()] + PRIORITY_PATTERNS

###############################################################################
# Token counting helpers
###############################################################################

MAX_SNAPSHOT_TOKENS: int = int(os.getenv("AI_SNAPSHOT_MAX_TOKENS", "12000"))
MAX_LINES_PER_FILE: int = int(os.getenv("AI_SNAPSHOT_MAX_LINES_PER_FILE", "1000"))
_CHARS_PER_TOKEN_APPROX: int = 4  # heuristic fallback


def _count_tokens(text: str) -> int:  # pragma: no cover – utility
    """Return *approximate* number of tokens in *text* (cheap heuristic fallback)."""
    if tiktoken is None:
        return max(1, len(text) // _CHARS_PER_TOKEN_APPROX)
    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text, disallowed_special={"<|endoftext|>"}))

###############################################################################
# Comment stripping
###############################################################################

_COMMENT_INLINE_RE = re.compile(r"\s+#.*(?=$)")


def _strip_comments(code: str) -> str:
    """Remove full‑line and trailing `#` comments to save tokens.

    *Preserves* she‑bang (`#!/usr/bin/env python` etc.), encoding hints and the
    artificial marker `# …truncated…`.
    """
    cleaned_lines: list[str] = []
    for ln in code.splitlines():
        stripped = ln.lstrip()
        if stripped.startswith("#"):
            # keep she‑bang / encoding and our own trunc marker
            if stripped.startswith("#!") or stripped.startswith("# -*-") or "…truncated…" in stripped:
                cleaned_lines.append(ln)
            # else drop full‑line comment entirely
            continue
        # remove trailing comments unless inside quotes (heuristic)
        if "#" in ln and ln.count("\"") % 2 == 0 and ln.count("'", 0, ln.find("#")) % 2 == 0:
            ln = _COMMENT_INLINE_RE.sub("", ln)
        cleaned_lines.append(ln.rstrip())
    return "\n".join(cleaned_lines)

###############################################################################
# Helper utilities
###############################################################################


def _should_exclude(path: Path, folders: Set[str], files: Set[str]) -> bool:
    if path.name.startswith(".") and path.name != ".env":
        return True
    if any(part in folders for part in path.parts):
        return True
    return path.name in files


def _priority_key(rel_path: str) -> Tuple[int, int, str]:
    """Return (rank, parts, path) where *lower* means *higher* priority."""
    for rank, pattern in enumerate(PRIORITY_PATTERNS):
        if re.search(pattern, rel_path):
            return (0, rank, rel_path)
    return (1, len(rel_path.split("/")), rel_path)  # secondary: folder depth

###############################################################################
# Snapshot generation
###############################################################################


def generate_project_snapshot(
    root_dir: Path,
    *,
    exclude_folders: Iterable[str] | None = None,
    exclude_files: Iterable[str] | None = None,
    max_tokens: int | None = None,
    max_lines_per_file: int | None = None,
) -> str:
    """Return a Markdown snapshot of *root_dir* within the token budget."""

    folders = EXCLUDE_FOLDERS | set(exclude_folders or ())
    files = EXCLUDE_FILES | set(exclude_files or ())
    token_budget = max_tokens or MAX_SNAPSHOT_TOKENS
    line_cap = max_lines_per_file or MAX_LINES_PER_FILE

    buf = StringIO()
    total_tokens = 0

    # 1 Directory tree – sorted by importance first
    dir_header = "## Struktura\n\n```\n"
    buf.write(dir_header)
    total_tokens += _count_tokens(dir_header)

    all_files = [p for p in root_dir.rglob("*") if p.is_file() and not _should_exclude(p, folders, files)]
    all_files.sort(key=lambda p: _priority_key(str(p.relative_to(root_dir))))

    for path in all_files:
        rel = f"{path.relative_to(root_dir)}\n"
        new_tokens = _count_tokens(rel)
        if total_tokens + new_tokens > token_budget:
            break
        buf.write(rel)
        total_tokens += new_tokens

    buf.write("```\n\n## Obsahy\n")
    total_tokens += _count_tokens("```\n\n## Obsahy\n")

    # 2 File contents (Python only) – same importance ordering
    py_files = [p for p in all_files if p.suffix == ".py"]
    for path in py_files:
        try:
            raw = path.read_text("utf-8")
        except Exception:
            continue

        raw = _strip_comments(raw)

        lines = raw.splitlines()
        if len(lines) > line_cap:
            raw = "\n".join(lines[:line_cap]) + "\n# …truncated…"

        header = f"\n---\n### `{path.relative_to(root_dir)}`\n```python\n"
        footer = "\n```\n"
        chunk_tokens = _count_tokens(header + raw + footer)
        if total_tokens + chunk_tokens > token_budget:
            break

        buf.write(header)
        buf.write(raw)
        buf.write(footer)
        total_tokens += chunk_tokens

    return buf.getvalue()

###############################################################################
# LLM‑powered summary
###############################################################################


def _summarise_snapshot(snapshot: str) -> str:
    llm = ChatOpenAI(model="gpt-4o", temperature=0.2)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a software architecture analyst. Summarise the architecture, "
                "capabilities and tools of the AI Agentic assistant based on this code snapshot.",
            ),
            ("human", "{snapshot}"),
        ]
    )
    chain = prompt | llm
    return chain.invoke({"snapshot": snapshot}).content

###############################################################################
# Public API
###############################################################################


def agent_introspect() -> str:
    """Return a Markdown summary of the project – always within the token budget."""
    root = Path(__file__).parent.parent  # project root
    snapshot = generate_project_snapshot(root)
    summary = _summarise_snapshot(snapshot)
    timestamp = datetime.now().isoformat()
    return f"# Agent Introspection ({timestamp})\n\n{summary}"
