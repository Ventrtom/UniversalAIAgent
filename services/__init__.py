"""Service layer consolidating Jira, RAG and web helpers."""
from .jira_service import JiraClient, JiraClientError, find_duplicate_ideas
from .rag_service import (
    save_text_to_file,
    build_vectorstore,
    rag_lookup,
    load_confluence_pages,
)
from .web_service import search_web, wiki_snippet, tavily_search

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
]
