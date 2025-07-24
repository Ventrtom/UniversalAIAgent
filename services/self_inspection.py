# services/self_inspection.py

from pathlib import Path
from io import StringIO
from datetime import datetime
import os
from typing import Iterable, Set
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

__all__ = ["agent_introspect"]

# Folder / file names (not globs) that should never be included in the snapshot.
EXCLUDE_FOLDERS: set[str] = {".git", "__pycache__", ".venv", "venv"}
EXCLUDE_FILES: set[str] = {"project_snapshot.md", "merge_project.py", ".env", ".png", ".json"}

_env_folders = os.getenv("AI_SNAPSHOT_EXCLUDE_FOLDERS")
if _env_folders:
    EXCLUDE_FOLDERS |= {item.strip() for item in _env_folders.split(",") if item.strip()}

_env_files = os.getenv("AI_SNAPSHOT_EXCLUDE_FILES")
if _env_files:
    EXCLUDE_FILES |= {item.strip() for item in _env_files.split(",") if item.strip()}

def _should_exclude(path: Path, folders: Set[str], files: Set[str]) -> bool:
    """Return *True* if *path* (a file or directory) is in the exclusion sets."""
    # Early exit for the most common cases
    if path.name.startswith(".") and path.name != ".env":  # ignore other dotfiles
        return True

    # Check explicit folder exclusions
    if any(part in folders for part in path.parts):
        return True

    # Check explicit file exclusions
    return path.name in files

def generate_project_snapshot(
    root_dir: Path,
    *,
    exclude_folders: Iterable[str] | None = None,
    exclude_files: Iterable[str] | None = None,
) -> str:
    """Return a Markdown‑formatted snapshot of *root_dir*.

    Parameters
    ----------
    root_dir:
        The directory whose contents will be summarised.
    exclude_folders, exclude_files:
        Optional iterables with additional folder / file names to exclude.
        When *None*, the function uses the global :pydata:`EXCLUDE_FOLDERS`
        / :pydata:`EXCLUDE_FILES`.
    """
    folders = EXCLUDE_FOLDERS | set(exclude_folders or ())
    files = EXCLUDE_FILES | set(exclude_files or ())

    result = StringIO()

    # 1 Directory tree
    result.write("## Struktura\n\n```\n")

    for path in sorted(root_dir.rglob("*")):
        if _should_exclude(path, folders, files):
            continue
        if path.is_file():
            rel_path = path.relative_to(root_dir)
            result.write(f"{rel_path}\n")

    result.write("```\n\n## Obsahy\n")

    # 2 File contents – only *.py (extend if needed)
    for path in sorted(root_dir.rglob("*.py")):
        if _should_exclude(path, folders, files):
            continue
        try:
            content = path.read_text("utf-8")
        except Exception:
            # Ignore binary / unreadable files silently
            continue
        rel_path = path.relative_to(root_dir)
        result.write(
            f"\n---\n### `{rel_path}`\n```python\n{content}\n```\n"
        )

    return result.getvalue()

def summarize_agent(snapshot: str) -> str:
    llm = ChatOpenAI(model="gpt-4o", temperature=0.2)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a software architecture analyst. Summarize the architecture, capabilities and tools of the AI Agentic assistant based on this code snapshot."),
        ("human", "{snapshot}")
    ])
    chain = prompt | llm
    return chain.invoke({"snapshot": snapshot}).content

def agent_introspect() -> str:
    """Return a short high level summary of this project."""
    root = Path(__file__).parent.parent  # root of project
    snapshot = generate_project_snapshot(root)
    summary = summarize_agent(snapshot)
    timestamp = datetime.now().isoformat()
    return f"# Agent Introspection ({timestamp})\n\n{summary}"
