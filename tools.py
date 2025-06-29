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
from jira_client import JiraClient, find_duplicate_ideas
from requests.exceptions import HTTPError
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
        """
        Purpose
        -------
        Persist any piece of plain text to a timestamped *.txt* file under ./output/.
        Ideal for jotting down idea drafts, SWOT analyses, user-stories, or scratch notes
        that you might want to share later.

        Parameters
        ----------
        data   : str   (required) – the full body of text to save.
        prefix : str   (optional, default "research") – filename prefix; the final name
                is <prefix>_YYYY-MM-DD_HH-MM-SS.txt.

        Returns
        -------
        Confirmation string with the relative file path.  No file object is returned.

        Caveats
        -------
        • Simply writes to disk – it does NOT version existing files or push anything
        back to Jira/Drive.
        """
    ),
)

# ───────────────────────── General Web Search ─────────────────────────────────
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
)

# ───────────────────────── Wikipedia Snippet ──────────────────────────────────
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
        """
        Purpose
        -------
        Semantic Retrieval-Augmented Generation over Productoo’s **internal knowledge
        base** (embedded in a Chroma DB).  Sources include P4 user documentation,
        roadmap pages, and archived conversation snippets.

        Parameters
        ----------
        query : str – a natural-language question or keyword string.

        Returns
        -------
        Up to 4 highly relevant passages concatenated together.

        Typical use cases
        -----------------
        P4 application feature details, implementation scenarios, or roadmap justifications
        to ground an answer before calling an LLM.
        """
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
        """
        Purpose
        -------
        List items from the P4 Jira “Ideas” backlog in a readable table-like text form.
        Useful for spotting duplicates manually, expanding an idea into deeper detail,
        or extracting candidate acceptance criteria.

        Parameters
        ----------
        keyword : str | None – optional free-text filter applied to summary + description
                (case-insensitive).  If None, returns the entire backlog (max 100).

        Returns
        -------
        For each match: "KEY | Status | Summary" on one line, followed by the description
        on the next line.
        """
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
    except HTTPError as http_exc:
        # 404 → neexistuje, 403 → bez práv – obojí chceme vrátit jako info
        code = getattr(http_exc.response, "status_code", None)
        if code in (403, 404) or "does not exist" in str(http_exc).lower():
            return f"Issue {key} neexistuje nebo k němu nemáte přístup."
        return f"HTTP chyba při načítání {key}: {http_exc}"
    except Exception as exc:  # pragma: no cover – jiná neočekávaná chyba
        return f"Chyba při načítání issue {key}: {exc}"

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
    description=(
        """
        Purpose
        -------
        Fetch a single Jira issue and format it as rich Markdown, giving a 360° snapshot
        for rapid context-building.

        Parameters
        ----------
        key : str – Jira key, e.g. "P4-24".

        Output sections
        ---------------
        • Summary line
        • Status | Type | Labels
        • Full Description (ADF converted to text)
        • Acceptance Criteria (auto-extracted Given/When/Then, if present)
        • Sub-tasks list
        • Up to three latest comments with authors and dates
        """
        ),
    args_schema=JiraIssueDetailInput,
)

# ───────────────────── Jira Duplicate-Idea Checker (Structured) ──────────────

class DuplicateIdeasInput(BaseModel):
    summary: str = Field(..., description="Krátký popis/summary nové Idea.")
    description: str | None = Field(
        default=None,
        description="(Volitelné) Delší popis nápadu – zahrne se do kontroly duplicity.",
    )
    threshold: float = Field(
        default=0.8,
        ge=0,
        le=1,
        description="Práh kosinové podobnosti 0-1 (výchozí 0.8).",
    )

def _duplicate_ideas(summary: str,
                     description: str | None = None,
                     threshold: float = 0.8) -> str:
    try:
        matches = find_duplicate_ideas(summary, description, threshold)
    except ValueError as exc:  # invalid threshold
        return f"Neplatný parametr `threshold`: {exc}"

    if not matches:
        return "Žádné potenciální duplicity nad daným prahem nenalezeny."
    return "Možné duplicitní nápady: " + ", ".join(matches)


jira_duplicates = StructuredTool.from_function(
    func=_duplicate_ideas,
    name="jira_duplicate_idea_checker",
    description=(
        """
        Purpose
        -------
        Detect whether a *new* idea is likely a **duplicate** of an existing P4 Jira Idea
        using cosine similarity of OpenAI embeddings.

        Parameters
        ----------
        summary     : str   (required) – one-line headline of the proposed idea.
        description : str | None (optional) – longer text; concatenated with summary for
                    embedding.
        threshold   : float       (optional, 0-1, default 0.8) – similarity cutoff;
                    lower for fuzzier matches.

        Returns
        -------
        Either "No potential duplicates above threshold"
        OR a comma-separated list of Jira keys ordered by similarity (highest first).

        Implementation notes
        --------------------
        • Uses text-embedding-3-small.
        • Performs acronym expansion (e.g. 2FA → "two factor authentication").
        • Considers *summary + description* on both sides.
        """
    ),
    args_schema=DuplicateIdeasInput,
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
    "jira_duplicates",
]
