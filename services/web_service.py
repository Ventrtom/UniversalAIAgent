"""Web search helpers split from the legacy tools module."""
from __future__ import annotations

import os
import openai
from tavily import TavilyClient
from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

# Opt‑out from OpenAI telemetry
os.environ.setdefault("OPENAI_TELEMETRY", "0")
if hasattr(openai, "telemetry") and hasattr(openai.telemetry, "TelemetryClient"):
    openai.telemetry.TelemetryClient.capture = lambda *_, **__: None


# ───────────────────────── General Web Search ────────────────────────────────
_duck = DuckDuckGoSearchRun()


def search_web(query: str) -> str:
    """Return DuckDuckGo search snippets for *query*."""
    return _duck.run(query)


# ───────────────────────── Wikipedia Snippet ----------------------------------
_api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=400)
_wiki_runner = WikipediaQueryRun(api_wrapper=_api_wrapper)


def wiki_snippet(query: str) -> str:
    """Return a short Wikipedia summary for *query*."""
    return _wiki_runner.run(query)


# ───────────────────────── Tavily Semantic Search ────────────────────────────
_TAVILY_KEY = os.getenv("TAVILY_API_KEY")
_client: TavilyClient | None = TavilyClient(api_key=_TAVILY_KEY) if _TAVILY_KEY else None


def tavily_search(query: str) -> str:
    """Perform semantic search via Tavily if configured."""
    if _client is None:
        return "Tavily není nakonfigurováno (chybí TAVILY_API_KEY)."
    try:
        raw = _client.search(query=query, max_results=6)
    except Exception as exc:  # pragma: no cover - external call
        return f"Tavily search selhalo: {exc}"

    results = raw.get("results", [])
    if not results:
        return "Tavily nic nenašlo."
    snippets = [f"- {r['url']}\n  {r['content'][:400].strip()}…" for r in results]
    return "\n\n".join(snippets)
