# tools/jira_content_tools.py
# ═══════════════════════════
"""
Tool-layer wrapper for the OpenAI-powered Jira content helpers.

These tools turn fuzzy inputs into production-ready Jira tickets in
professional English, following agile / software-engineering best-practices.

Public attribute
----------------
ALL_TOOLS : list[StructuredTool]
    Import and add to your agent's tool list, e.g.:

    from tools.jira_content_tools import ALL_TOOLS
    agent = initialize_agent(
        tools=ALL_TOOLS + other_tools,
        ...
    )
"""
from __future__ import annotations

from typing import List, Literal
import asyncio

from langchain.tools import StructuredTool
from pydantic import BaseModel, Field

from services import (
    enhance_idea_adf as _enhance_idea, # původní verze je "enhance_idea" tak stačí jen přepsat.
    epic_from_idea as _epic_from_idea,
    user_stories_for_epic as _user_stories_for_epic,
)

# ──────────────────────────────────────────────────────────────────────────────
# 1. Enhance Idea → Jira Idea
# ──────────────────────────────────────────────────────────────────────────────
class EnhanceIdeaInput(BaseModel):
    """Arguments for the *enhance_idea* tool."""
    summary: str = Field(
        ...,
        description="One-line product idea summary. Keep it short and action-oriented.",
        examples=["Predictive maintenance for packaging line"],
    )
    description: str | None = Field(
        default=None,
        description="Optional free-form description copied from stakeholder notes.",
        examples=[
            "Operators complain about frequent unplanned stops. "
            "We should investigate sensor trends, forecast failures and schedule maintenance."
        ],
    )
    audience: Literal["business", "technical", "mixed"] = Field(
        default="mixed",
        description="Target audience that will read the Idea (affects tone).",
        examples=["business"],
    )
    max_words: int | None = Field(
        default=360,
        description="Hard limit for total word count of the generated description.",
        examples=[100],
    )


async def _enhance_idea_tool(
    *,
    summary: str,
    description: str | None = None,
    audience: str = "mixed",
    max_words: int | None = 120,
    ) -> str:
    """Convert a rough product idea into a polished **Jira Idea** ticket body.

    Output format (Markdown):
    - ## Problem
    - ## Proposed solution
    - ## Business value (bullets)
    - ## Acceptance criteria (G/W/T bullets)

    Always writes in professional British English, avoids fluff, and
    adheres to company style (second-level headings, max 5 ACs)."""

    return await asyncio.to_thread(
        _enhance_idea,
        summary=summary,
        description=description,
        audience=audience,
        max_words=max_words or 120,
    )


enhance_idea_tool = StructuredTool.from_function(
    name="enhance_idea",
    coroutine=_enhance_idea_tool,
    description=(
        "Transform a **raw product idea** (summary + optional notes) into a concise, "
        "board-ready *Jira Idea* body in Markdown.\n\n"
        "**When to call**  • Any time stakeholder wording is informal, incomplete, "
        "in Czech, or otherwise unfit for an executive audience.\n\n"
        "**Output**  • Four second-level headings — *Problem*, *Proposed solution*, "
        "*Business value*, *Acceptance criteria* — in professional British English, "
        "capped at `max_words` (default 360).\n\n"
        "**Args**\n"
        " • `summary` (STR, required) – one-line tagline.\n"
        " • `description` (STR, optional) – stakeholder context.\n"
        " • `audience` (ENUM) – \"business\", \"technical\", or \"mixed\"; "
        "controls tone and terminology.\n"
        " • `max_words` (INT) – hard length limit.\n\n"
        "Avoid jargon, keep headings exactly as above, respect the word limit."
    ),
    func=_enhance_idea_tool,
    args_schema=EnhanceIdeaInput,
    handle_tool_error=True,
)

# ──────────────────────────────────────────────────────────────────────────────
# 2. Idea → Epic
# ──────────────────────────────────────────────────────────────────────────────
class EpicFromIdeaInput(BaseModel):
    """Arguments for the *epic_from_idea* tool."""
    summary: str = Field(
        ...,
        description="Idea summary that will become the epic goal.",
        examples=["Predictive maintenance for packaging line"],
    )
    description: str | None = Field(
        default=None,
        description="Optional idea description giving context/problem statement.",
    )


async def _epic_from_idea_tool(*, summary: str, description: str | None = None) -> str:
    """Produce a complete **Jira Epic** template in Markdown.

    Sections:
    - Epic Goal
    - Context
    - Definition of Done (bullets)
    - Acceptance criteria (G/W/T, ≤7)
    - Out of scope

    The text is direct, neutral, no marketing adjectives."""
    return await asyncio.to_thread(_epic_from_idea, summary, description)


epic_from_idea_tool = StructuredTool.from_function(
    name="epic_from_idea",
    description=(
        "Expand a validated Idea into a comprehensive *Epic* draft "
        "ready for backlog refinement, with DoD & ACs."
    ),
    coroutine=_epic_from_idea_tool,
    args_schema=EpicFromIdeaInput,
    handle_tool_error=True,
)

# ──────────────────────────────────────────────────────────────────────────────
# 3. Epic → User Stories
# ──────────────────────────────────────────────────────────────────────────────
class UserStoriesForEpicInput(BaseModel):
    """Arguments for the *user_stories_for_epic* tool."""
    epic_name: str = Field(
        ...,
        description="Exact Jira Epic name / goal.",
        examples=["Predictive maintenance platform (Phase I)"],
    )
    epic_description: str = Field(
        ...,
        description="Full epic description (context, DoD, ACs).",
    )
    count: int = Field(
        default=5,
        ge=1,
        le=15,
        description="How many user stories to generate. Default 5.",
    )


async def _user_stories_for_epic_tool(
    *, epic_name: str, epic_description: str, count: int = 5
) -> str:
    """Generate INVEST-compliant **User Stories** for the given epic in czech language.

    For each story:
    - Title
    - One-paragraf abstract
    - ‘As a … I want … so that …’
    - Acceptance criteria (G/W/T)
    - T-shirt estimate"""
    return await asyncio.to_thread(_user_stories_for_epic, epic_name, epic_description, count)


user_stories_for_epic_tool = StructuredTool.from_function(
    name="user_stories_for_epic",
    description=(
        """Generate INVEST-compliant **User Stories** for the given epic in czech language.

        For each story:
        - Title
        - One-paragraf abstract
        - ‘As a … I want … so that …’
        - Acceptance criteria (G/W/T)
        - T-shirt estimate"""
    ),
    coroutine=_user_stories_for_epic_tool,
    args_schema=UserStoriesForEpicInput,
    handle_tool_error=True,
)

# ──────────────────────────────────────────────────────────────────────────────
# Convenience export
# ──────────────────────────────────────────────────────────────────────────────
ALL_TOOLS: List[StructuredTool] = [
    enhance_idea_tool,
    epic_from_idea_tool,
    user_stories_for_epic_tool,
]

__all__ = [
    "enhance_idea_tool",
    "epic_from_idea_tool",
    "user_stories_for_epic_tool",
    "ALL_TOOLS",
]
