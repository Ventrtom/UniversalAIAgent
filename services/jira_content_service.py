# services/jira_content_service.py
# ════════════════════════════════
"""OpenAI-powered helpers for drafting high-quality Jira issue content.

Public API
----------
enhance_idea(...)           -> Markdown for an *Idea* ticket
epic_from_idea(...)         -> Markdown for an *Epic* ticket
user_stories_for_epic(...)  -> Markdown list of INVEST-ready User Stories

"""

from __future__ import annotations

import os
import openai
from typing import List

from langchain_openai import ChatOpenAI

# ── Disable OpenAI telemetry ───────────────────────────────────────────────────
os.environ.setdefault("OPENAI_TELEMETRY", "0")
if hasattr(openai, "telemetry") and hasattr(openai.telemetry, "TelemetryClient"):
    openai.telemetry.TelemetryClient.capture = lambda *_, **__: None

# ── LLM configuration ─────────────────────────────────────────────────────────
_MODEL = os.getenv("JIRA_CONTENT_MODEL", "gpt-4o")
_TEMPERATURE = float(os.getenv("JIRA_CONTENT_TEMPERATURE", "0.2"))

_llm = ChatOpenAI(model=_MODEL, temperature=_TEMPERATURE)


# ── Internal helper ───────────────────────────────────────────────────────────
def _run(prompt: str) -> str:
    """Call the LLM synchronously and return trimmed text."""
    return _llm.invoke(prompt).content.strip()


# ── Prompt templates ────────────────────────────────────────────────────────

# Few-shot příklad – udává strukturu i tón
_IDEA_EXAMPLE: str = """
## Problem
Our maintenance costs grew 15 % YoY due to frequent unplanned packaging-line stops.  

## Proposed solution
Monitor vibration and temperature, predict failures 48 h ahead and auto-schedule maintenance.  

## Business value
- Reduce downtime by 10 h / month  
- Save €30 k per quarter  

## Acceptance criteria
- **Given** line sensors are online  
- **When** a failure probability > 70% is detected  
- **Then** a work order is created and maintenance slot reserved
"""  
# (konec example)


# ── High-level generators ─────────────────────────────────────────────────────
def enhance_idea(
    summary: str,
    description: str | None = None,
    audience: str = "mixed",
    max_words: int = 360,
    ) -> str:

    # ── dynamické systémové instrukce podle publika a délky ───────────────
    _audience_map = {
        "business": "executives and sales",
        "technical": "engineering teams",
        "mixed": "cross-functional stakeholders",
    }
    audience_phrase = _audience_map.get(audience, "cross-functional stakeholders")

    _IDEA_SYSTEM_TMPL = (
        "You are a senior product manager writing short, board-ready Jira Idea "
        "descriptions. Audience: {audience_phrase}. Tone: plain, concise, "
        "British English. Limit the whole output to ≤ {max_words} words. "
        "Highlight the core insight in the first sentence. Use exactly the "
        "section headings shown in the example."
    )
    idea_system = _IDEA_SYSTEM_TMPL.format(
        audience_phrase=audience_phrase, max_words=max_words
    )

    """
    Turn a raw *Idea* (summary + optional description) into
    a polished, well-structured Markdown body ready for Jira.
    """
    prompt = f"""
        {idea_system}

        {_IDEA_EXAMPLE}

        ### Raw Idea
        SUMMARY: {summary}
        DESCRIPTION: {description or '(none)'}

        ### Task
        Rewrite the Raw Idea into a polished Jira Idea description using the **same four headings** "
        (Problem, Proposed solution, Business value, Acceptance criteria). "
        Use active voice, avoid jargon and filler, keep total length ≤ {max_words} words.
        """

    return _run(prompt)


def epic_from_idea(summary: str, description: str | None = None) -> str:
    """
    Scale an Idea into a **Jira Epic**.  Outputs Markdown containing:
      • Epic Goal
      • Context
      • Definition of Done
      • Acceptance criteria
      • Out of scope
    """
    prompt = f"""
    You are an agile product owner.

    Create a **Jira Epic** draft in English from the Idea below.
    Use clear, specific language (avoid adjectives like "great").
    Return Markdown in this order:

    **Epic Goal** – one sentence  
    **Context** – 1-2 paragraphs linking problem & solution  
    **Definition of Done** – bullet list  
    **Acceptance criteria** – Given / When / Then bullets (≤ 7)  
    **Out of scope** – bullets of exclusions

    Idea basis
    ----------
    SUMMARY: {summary}
    DESCRIPTION: {description or '(none)'}
    """
    return _run(prompt)


def user_stories_for_epic(
    epic_name: str,
    epic_description: str,
    count: int = 5,
) -> str:
    """
    Generate *count* INVEST-compliant User Stories for the given Epic.
    Each story includes title, user-story sentence, acceptance criteria
    and a T-shirt-size estimate.
    """
    prompt = f"""
You are an agile coach.

Draft **{count} independent user stories** derived from the Epic below.

EPIC NAME: {epic_name}

EPIC DESCRIPTION:
{epic_description}

Output for each story:

### <Story Title>
**User story**: As a <persona> I want … so that …  
**Acceptance criteria**
- Given / When / Then bullets  
**Estimate**: <S / M / L>

Ensure every story is INVEST-ready and uses professional English.
"""
    return _run(prompt)


__all__: List[str] = [
    "enhance_idea",
    "epic_from_idea",
    "user_stories_for_epic",
]



