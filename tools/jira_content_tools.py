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

from typing import List

from langchain.tools import StructuredTool
from pydantic import BaseModel, Field

from services.jira_content_service import (
    enhance_idea as _enhance_idea,
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


def _enhance_idea_tool(*, summary: str, description: str | None = None) -> str:
    """Convert a rough product idea into a polished **Jira Idea** ticket body.

    Output format (Markdown):
    - ## Problem
    - ## Proposed solution
    - ## Business value (bullets)
    - ## Acceptance criteria (G/W/T bullets)

    Always writes in professional British English, avoids fluff, and
    adheres to company style (second-level headings, max 5 ACs)."""
    return _enhance_idea(summary, description)


enhance_idea_tool = StructuredTool.from_function(
    name="enhance_idea",
    description=(
        "Draft a well-structured *Idea* description for Jira in Markdown. "
        "Use when you have a short, raw idea that needs to be turned into a "
        "ticket the team can evaluate."
    ),
    func=_enhance_idea_tool,
    args_schema=EnhanceIdeaInput,
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


def _epic_from_idea_tool(*, summary: str, description: str | None = None) -> str:
    """Produce a complete **Jira Epic** template in Markdown.

    Sections:
    - Epic Goal
    - Context
    - Definition of Done (bullets)
    - Acceptance criteria (G/W/T, ≤7)
    - Out of scope

    The text is direct, neutral, no marketing adjectives."""
    return _epic_from_idea(summary, description)


epic_from_idea_tool = StructuredTool.from_function(
    name="epic_from_idea",
    description=(
        "Expand a validated Idea into a comprehensive *Epic* draft "
        "ready for backlog refinement, with DoD & ACs."
    ),
    func=_epic_from_idea_tool,
    args_schema=EpicFromIdeaInput,
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


def _user_stories_for_epic_tool(
    *, epic_name: str, epic_description: str, count: int = 5
) -> str:
    """Generate INVEST-compliant **User Stories** for the given epic in czech language.

    For each story:
    - Title
    - ‘As a … I want … so that …’
    - Acceptance criteria (G/W/T)
    - T-shirt estimate"""
    return _user_stories_for_epic(epic_name, epic_description, count)


user_stories_for_epic_tool = StructuredTool.from_function(
    name="user_stories_for_epic",
    description=(
        "Break an Epic into a list of INVEST-ready User Stories, each with an "
        "estimate and acceptance criteria."
    ),
    func=_user_stories_for_epic_tool,
    args_schema=UserStoriesForEpicInput,
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
