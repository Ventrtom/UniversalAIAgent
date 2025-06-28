"""tools.py — Unified LangChain Tools for the P4 Assistant
====================================================
• Strongly‑typed, fully documented LangChain tools.
• JIRA retriever converted to **StructuredTool** ⇒ už žádná chyba při volání bez argumentů.
• Tavily & RAG unchanged, jen drobné kosmetické fixy.
"""
from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import openai
from pydantic import BaseModel, Field
from langchain.tools import Tool, StructuredTool
from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document
from jira_retriever import fetch_jira_issues  # local helper for JIRA REST API
from tavily import TavilyClient

# ───────────────────────── Telemetry opt‑out ──────────────────────────────────
os.environ.setdefault("OPENAI_TELEMETRY", "0")
if hasattr(openai, "telemetry") and hasattr(openai.telemetry, "TelemetryClient"):
    openai.telemetry.TelemetryClient.capture = lambda *_, **__: None

# ───────────────────────── Helper: Robust TXT Saver ───────────────────────────
OUTPUT_DIR = Path("output"); OUTPUT_DIR.mkdir(exist_ok=True)

def _save_to_txt(data: str, prefix: str = "research") -> str:
    """
    Uloží *data* do nového TXT souboru v ./output a vrátí textovou hlášku.
    Název souboru = <prefix>_YYYY-MM-DD_HH-MM-SS.txt
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename  = f"{prefix}_{timestamp}.txt"
    path      = OUTPUT_DIR / filename

    path.write_text(data, encoding="utf-8")

    return f"✅ Data saved to {path.as_posix()}"

save_tool = Tool(
    name="save_text_to_file",
    func=_save_to_txt,
    description=(
        "Uloží libovolný text (např. rozpracovanou Idea, SWOT, user‑story) do souboru v ./output.\n"
        "Params → data (string, povinný); filename (volitelný). Pokud filename chybí, vytvoří se research_YYYY‑MM‑DD.txt. "
        "Vrací potvrzovací zprávu s cestou."
    ),
)

# ───────────────────────── General Web Search ─────────────────────────────────
_duck = DuckDuckGoSearchRun()
search_tool = Tool(
    name="searchWeb",
    func=_duck.run,
    description=(
        "Rychlé obecné vyhledávání (DuckDuckGo). Použij pro definice, novinky, blog‑posty.\n"
        "Nevhodné pro detailní konkurenční analýzu – tam je lepší tavily_search."
    ),
)

# ───────────────────────── Wikipedia Snippet ──────────────────────────────────
api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=400)
wiki_tool = Tool(
    name="wikipedia_query",
    func=WikipediaQueryRun(api_wrapper=api_wrapper).run,
    description="Získá krátké shrnutí z Wikipedie – vhodné pro historické pozadí, standardy, pojmy.",
)

# ───────────────────────── RAG Retriever over Chroma ──────────────────────────
CHROMA_DIR = "rag_chroma_db"
_embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
_retriever = Chroma(persist_directory=CHROMA_DIR, embedding_function=_embeddings).as_retriever(search_kwargs={"k": 4})

def _rag_lookup(query: str) -> str:
    docs: List[Document] = _retriever.invoke(query)
    if not docs:
        return "Žádné interní dokumenty se k dotazu nenašly."
    return "\n\n".join(d.page_content for d in docs)

rag_tool = Tool(
    name="rag_retriever",
    func=_rag_lookup,
    description=(
        "Prohledá interní znalostní bázi Productoo (uživatelská dokumentace P4, Roadmapa, archiv konverzací).\n"
        "Vrací až 4 nejrelevantnější úryvky. Vhodné pro: funkce APS/CMMS, scénáře výroby, Roadmapu."
    ),
)

# ───────────────────────── JIRA Ideas Retriever (Structured) ──────────────────
class JiraIdeasInput(BaseModel):
    """Optional keyword filter for JIRA Ideas."""
    keyword: Optional[str] = Field(default=None, description="Klíčové slovo pro filtrování Ideas podle summary/description. Pokud None, vrátí vše.")


def _jira_ideas_struct(keyword: Optional[str] = None) -> str:
    try:
        issues = fetch_jira_issues()
    except Exception as exc:
        return f"❌ Chyba při načítání JIRA: {exc}"

    if not issues:
        return "Nenalezeny žádné JIRA Ideas."

    if keyword:
        issues = [iss for iss in issues if keyword.lower() in (iss["summary"] + iss["description"]).lower()]
        if not issues:
            return f"🔎 Žádné Ideas neobsahují klíčové slovo '{keyword}'."

    lines = [
        f"{iss['key']} | {iss['status']} | {iss['summary']}\n{iss['description'] or '- žádný popis -'}"
        for iss in issues
    ]
    return "\n\n".join(lines)

jira_ideas = StructuredTool.from_function(
    func=_jira_ideas_struct,
    name="jira_ideas_retriever",
    description=(
        "Načte backlog *Ideas* z JIRA (projekt=P4). Vhodné pro: \n"
        " • ověření duplicitních nápadů\n"
        " • rozpracování existujících Ideas do detailu\n"
        " • návrh akceptačních kritérií\n"
        "Volitelný parametr `keyword` umožňuje filtrovat Ideas podle textu. Pokud keyword chybí, vrátí celé backlog."
    ),
    args_schema=JiraIdeasInput,
)

# ───────────────────────── Tavily Semantic Search ─────────────────────────────
_TAVILY_KEY = os.getenv("TAVILY_API_KEY"); _client: TavilyClient | None = None
if _TAVILY_KEY:
    _client = TavilyClient(api_key=_TAVILY_KEY)

def _tavily_search(query: str) -> str:
    if _client is None:
        return "❌ Tavily není nakonfigurováno (chybí TAVILY_API_KEY)."

    try:
        raw = _client.search(query=query, max_results=6)
    except Exception as exc:
        return f"❌ Tavily search selhalo: {exc}"

    results = raw.get("results", [])
    if not results:
        return "⛔️ Tavily nic nenašlo."

    snippets = [f"- {r['url']}\n  {r['content'][:400].strip()}…" for r in results]
    return "\n\n".join(snippets)

tavily_tool = Tool(
    name="tavily_search",
    func=_tavily_search,
    description=(
        "Pokročilé sémantické vyhledávání (LLM). Vhodné pro: \n"
        " • konkurenční analýzu APS/CMMS\n        • white‑papers, tiskové zprávy, trendy trhu\n"
        "Vrací až 6 výsledků (URL + snippet). Vyžaduje proměnnou prostředí TAVILY_API_KEY."
    ),
)

# ───────────────────────── Exports ────────────────────────────────────────────
__all__ = [
    "search_tool",
    "wiki_tool",
    "rag_tool",
    "jira_ideas",
    "tavily_tool",
    "save_tool",
]
