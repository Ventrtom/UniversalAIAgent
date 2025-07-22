from __future__ import annotations

import os
import openai
from langchain.tools import Tool
from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from tavily import TavilyClient
import httpx

# Opt‑out from OpenAI telemetry
os.environ.setdefault("OPENAI_TELEMETRY", "0")
if hasattr(openai, "telemetry") and hasattr(openai.telemetry, "TelemetryClient"):
    openai.telemetry.TelemetryClient.capture = lambda *_, **__: None

# ───────────────────────── General Web Search ────────────────────────────────
_duck = DuckDuckGoSearchRun()
search_tool = Tool(
    name="searchWeb",
    func=_duck.run,
    description=(
        """
        Purpose
        -------
        Fast, general-purpose web lookup powered by DuckDuckGo.  Best for definitions,
        quick facts, fresh news articles, or blog posts.

        Parameters
        ----------
        query : str – the search phrase.

        Returns
        -------
        DDG’s textual snippet(s); no rich metadata or full RSS feed.

        When *not* to use
        -----------------
        For in-depth competitor or market research prefer tavily_search, which does
        semantic ranking.
        """
    ),
    handle_tool_error=True,
)

# ───────────────────────── Wikipedia Snippet ─────────────────────────────────
api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=400)
wiki_tool = Tool(
    name="wikipedia_query",
    func=WikipediaQueryRun(api_wrapper=api_wrapper).run,
    description=(
        """
        Purpose
        -------
        Pull a concise (< 400 chars) summary paragraph from Wikipedia for quick
        background, historical context, industry standards, or terminology.

        Parameters
        ----------
        query : str – topic name or phrase.

        Returns
        -------
        Plain-text summary (no infobox tables or references).
        """
    ),
    handle_tool_error=True,
)

# ───────────────────────── Tavily Semantic Search ────────────────────────────
_TAVILY_KEY = os.getenv("TAVILY_API_KEY")
_client: TavilyClient | None = None
if _TAVILY_KEY:
    _client = TavilyClient(api_key=_TAVILY_KEY)


async def _tavily_search(query: str) -> str:
    if _client is None:
        return "Tavily není nakonfigurováno (chybí TAVILY_API_KEY)."

    try:
        async with httpx.AsyncClient(base_url=_client.base_url, headers=_client.headers, proxies=_client.proxies) as hc:
            resp = await hc.post("/search", json={"query": query, "max_results": 6})
            resp.raise_for_status()
            raw = resp.json()
    except Exception as exc:  # pragma: no cover - external call
        return f"Tavily search selhalo: {exc}"

    results = raw.get("results", [])
    if not results:
        return "Tavily nic nenašlo."

    snippets = [f"- {r['url']}\n  {r['content'][:400].strip()}…" for r in results]
    return "\n\n".join(snippets)


tavily_tool = Tool(
    name="tavily_search",
    coroutine=_tavily_search,
    description=(
        """
        Purpose
        -------
        LLM-backed **semantic** web search via the Tavily API – tailored for competitive
        intelligence, white-papers, press releases, or discovering market trends that
        keyword search might miss.

        Parameters
        ----------
        query : str – question or topic.

        Returns
        -------
        Up to 6 items: each entry shows the URL plus a ~400-character snippet.

        Prerequisite
        ------------
        Environment variable TAVILY_API_KEY must be set; otherwise the tool politely
        reports it is unavailable.
        """
    ),
    handle_tool_error=True,
)

__all__ = ["search_tool", "wiki_tool", "tavily_tool"]
