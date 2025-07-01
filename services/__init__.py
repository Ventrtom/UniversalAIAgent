# services/__init__.py
"""Service layer consolidating Jira, RAG and web helpers."""
from .jira_service import JiraClient, JiraClientError, find_duplicate_ideas, _extract_text_from_adf
from .rag_service import (
    save_text_to_file,
    build_vectorstore,
    rag_lookup,
    load_confluence_pages,
)
from .web_service import search_web, wiki_snippet, tavily_search
from .jira_content_service import (
    enhance_idea,
    epic_from_idea,
    user_stories_for_epic,
)
from .input_loader import process_input_files

__all__ = [
    "JiraClient",
    "JiraClientError",
    "find_duplicate_ideas",
    "save_text_to_file",
    "build_vectorstore",
    "rag_lookup",
    "load_confluence_pages",
    "search_web",
    "wiki_snippet",
    "tavily_search",
    "_extract_text_from_adf",
    "jira_content_tools",
    "process_input_files",
]
