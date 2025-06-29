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


# ── High-level generators ─────────────────────────────────────────────────────
def enhance_idea(summary: str, description: str | None = None) -> str:
    """
    Turn a raw *Idea* (summary + optional description) into
    a polished, well-structured Markdown body ready for Jira.
    """
    prompt = f"""
You are a senior software product manager.

Transform the raw idea below into a **Jira Idea** description
written in professional British English.  
Return Markdown with these sections:

## Problem
<one concise paragraph>

## Proposed solution
<1-3 paragraphs, technical outline>

## Business value
- <bullet list of measurable benefits>

## Acceptance criteria
- Given / When / Then bullets (max 5)

Raw Idea
--------
SUMMARY: {summary}
DESCRIPTION: {description or '(none)'}
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
