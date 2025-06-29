"""tools.py — Unified LangChain Tools for the P4 Assistant
====================================================
• Strongly‑typed, fully documented LangChain tools.
• JIRA retriever converted to **StructuredTool** ⇒ už žádná chyba při volání bez argumentů.
• Tavily & RAG unchanged, jen drobné kosmetické fixy.
"""
from __future__ import annotations

import os
import re
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
from jira_client import JiraClient  # NEW: switched from legacy jira_retriever
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

    return f"Data saved to {path.as_posix()}"

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

# Create a single, re‑usable client instance – avoids repeated handshakes
_JIRA = JiraClient()

class JiraIdeasInput(BaseModel):
    """Optional keyword filter for JIRA Ideas."""

    keyword: Optional[str] = Field(
        default=None,
        description=(
            "Klíčové slovo pro filtrování Ideas podle summary/description. Pokud None, vrátí vše."
        ),
    )


def _jira_ideas_struct(keyword: Optional[str] = None) -> str:
    try:
        # Pull backlog (project=P4) via the new client
        issues = _JIRA.search_issues("project = P4 ORDER BY created DESC", max_results=100)
    except Exception as exc:  # pragma: no cover – user‑facing path only
        return f"Chyba při načítání JIRA: {exc}"

    if not issues:
        return "Nenalezeny žádné JIRA Ideas."

    # Convert & filter
    def _plain(issue):
        f = {**issue.get("fields", {}), **{k: v for k, v in issue.items() if k != "fields"}}
        return {
            "key": issue["key"],
            "summary": f.get("summary", ""),
            "description": f.get("description_plain") or f.get("description", ""),
            "status": f.get("status", {}).get("name", ""),
        }

    ideas = [_plain(i) for i in issues]

    if keyword:
        kw = keyword.lower()
        ideas = [i for i in ideas if kw in (i["summary"] + i["description"]).lower()]
        if not ideas:
            return f"Žádné Ideas neobsahují klíčové slovo '{keyword}'."

    # Human‑readable formatting
    lines = [
        f"{i['key']} | {i['status']} | {i['summary']}\n" f"{i['description'] or '- žádný popis -'}"
        for i in ideas
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
        return "Tavily není nakonfigurováno (chybí TAVILY_API_KEY)."

    try:
        raw = _client.search(query=query, max_results=6)
    except Exception as exc:
        return f"Tavily search selhalo: {exc}"

    results = raw.get("results", [])
    if not results:
        return "Tavily nic nenašlo."

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




# ───────────────────────── Jira Issue Detail (Structured) ────────────────────

class JiraIssueDetailInput(BaseModel):
    """Schema for ``jira_issue_detail`` tool."""

    key: str = Field(..., description="Jira key, e.g. P4-123")


def _format_acceptance_criteria(text: str) -> str:
    """Extract *Given / When / Then* lines from description."""
    pattern = re.compile(r"^\s*(?:\*|-)?\s*(Given|When|Then)\b.*", re.IGNORECASE)
    items = [ln.strip() for ln in text.splitlines() if pattern.match(ln)]
    return "\n".join(f"- {ln.lstrip('*- ').strip()}" for ln in items)


def _jira_issue_detail(key: str) -> str:
    """Return a single Jira issue with rich context in markdown."""
    try:
        issue = _JIRA.get_issue(
            key,
            fields=[
                "summary",
                "status",
                "issuetype",
                "labels",
                "description",
                "subtasks",
                "comment",
            ],
        )
    except Exception as exc:  # noqa: BLE001 – user-facing error branch
        msg = str(exc)
        if "404" in msg:
            return f"Issue {key} not found"
        raise

    f = {**issue.get("fields", {}), **{k: v for k, v in issue.items() if k != "fields"}}

    # Summary block ----------------------------------------------------------
    title = f.get("summary", "—")
    status = f.get("status", {}).get("name", "—")
    itype = f.get("issuetype", {}).get("name", "—")
    labels = ", ".join(f.get("labels") or []) or "—"

    # Description ------------------------------------------------------------
    from jira_client import _extract_text_from_adf  # local import to avoid cycles

    raw_desc = f.get("description")
    description_src = f.get("description_plain") or f.get("description")
    description = (
        _extract_text_from_adf(raw_desc)
        if isinstance(description_src, (dict, list))
        else (raw_desc or "—")
    ).strip()

    # Acceptance criteria ----------------------------------------------------
    ac_block = _format_acceptance_criteria(description)

    # Sub-tasks --------------------------------------------------------------
    subtasks = f.get("subtasks") or []
    sub_lines = [
        f"- {st['key']} – {st.get('fields', {}).get('summary', '')}".rstrip()
        for st in subtasks
    ]

    # Latest comments (max 3) -------------------------------------------------
    comments = sorted(
        (f.get("comment", {}).get("comments") or []),
        key=lambda c: c.get("created", ""),
        reverse=True,
    )[:3]

    def _fmt_date(ts: str) -> str:
        return ts.split("T")[0] if ts else "—"

    com_lines = [
        f"- **{c.get('author', {}).get('displayName', 'Unknown')}** "
        f"({_fmt_date(c.get('created'))}): {c.get('body', '').strip()}"
        for c in comments
    ]

    # Assemble markdown ------------------------------------------------------
    parts: list[str] = [
        f"**{key} – {title}**",
        f"Status: {status} | Type: {itype} | Labels: {labels}",
        "",
        "### Description",
        description or "—",
    ]
    if ac_block:
        parts += ["", "### Acceptance Criteria", ac_block]
    if sub_lines:
        parts += ["", "### Sub-tasks", *sub_lines]
    if com_lines:
        parts += ["", "### Latest Comments", *com_lines]

    return "\n".join(parts).strip()


jira_issue_detail = StructuredTool.from_function(
    func=_jira_issue_detail,
    name="jira_issue_detail",
    description="Return a single Jira issue (summary, description, subtasks, latest comments).",
    args_schema=JiraIssueDetailInput,
)





# ───────────────────────── Exports ────────────────────────────────────────────
__all__ = [
    "search_tool",
    "wiki_tool",
    "rag_tool",
    "jira_ideas",
    "jira_issue_detail",
    "tavily_tool",
    "save_tool",
]
