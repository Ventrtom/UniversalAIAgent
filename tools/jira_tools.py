from __future__ import annotations

import os
import re
import openai
from typing import List, Optional

from langchain.tools import StructuredTool
from pydantic import BaseModel, Field
from requests.exceptions import HTTPError

from services import JiraClient, find_duplicate_ideas

# Opt‑out from OpenAI telemetry
os.environ.setdefault("OPENAI_TELEMETRY", "0")
if hasattr(openai, "telemetry") and hasattr(openai.telemetry, "TelemetryClient"):
    openai.telemetry.TelemetryClient.capture = lambda *_, **__: None

# Single Jira client instance ---------------------------------------------------
_JIRA = JiraClient()


# ───────────────────────── JIRA Ideas Retriever ─────────────────────────────--
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
        issues = _JIRA.search_issues("project = P4 ORDER BY created DESC", max_results=100)
    except Exception as exc:  # pragma: no cover – user-facing path only
        return f"Chyba při načítání JIRA: {exc}"

    if not issues:
        return "Nenalezeny žádné JIRA Ideas."

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

# ───────────────────────── Jira Issue Detail ─────────────────────────────────--
class JiraIssueDetailInput(BaseModel):
    """Schema for ``jira_issue_detail`` tool."""

    key: str = Field(..., description="Jira key, e.g. P4-123")


def _format_acceptance_criteria(text: str) -> str:
    pattern = re.compile(r"^\s*(?:\*|-)?\s*(Given|When|Then)\b.*", re.IGNORECASE)
    items = [ln.strip() for ln in text.splitlines() if pattern.match(ln)]
    return "\n".join(f"- {ln.lstrip('*- ').strip()}" for ln in items)


def _jira_issue_detail(key: str) -> str:
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
        code = getattr(http_exc.response, "status_code", None)
        if code in (403, 404) or "does not exist" in str(http_exc).lower():
            return f"Issue {key} neexistuje nebo k němu nemáte přístup."
        return f"HTTP chyba při načítání {key}: {http_exc}"
    except Exception as exc:  # pragma: no cover – other unexpected error
        return f"Chyba při načítání issue {key}: {exc}"

    f = {**issue.get("fields", {}), **{k: v for k, v in issue.items() if k != "fields"}}

    title = f.get("summary", "—")
    status = f.get("status", {}).get("name", "—")
    itype = f.get("issuetype", {}).get("name", "—")
    labels = ", ".join(f.get("labels") or []) or "—"

    from services import _extract_text_from_adf

    raw_desc = f.get("description")
    description_src = f.get("description_plain") or f.get("description")
    description = (
        _extract_text_from_adf(raw_desc)
        if isinstance(description_src, (dict, list))
        else (raw_desc or "—")
    ).strip()

    ac_block = _format_acceptance_criteria(description)

    subtasks = f.get("subtasks") or []
    sub_lines = [
        f"- {st['key']} – {st.get('fields', {}).get('summary', '')}".rstrip()
        for st in subtasks
    ]

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

# ─────────────── Jira Duplicate-Idea Checker (Structured) --------------------
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


def _duplicate_ideas(
    summary: str, description: str | None = None, threshold: float = 0.8
) -> str:
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

__all__ = [
    "jira_ideas",
    "jira_issue_detail",
    "jira_duplicates",
    "_JIRA",
    "_jira_issue_detail",
]
