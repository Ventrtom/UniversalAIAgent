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
import json
import openai
from pathlib import Path
from typing import List, Dict, Any
from jinja2 import Environment, FileSystemLoader, select_autoescape

from langchain_openai import ChatOpenAI

# ── Disable OpenAI telemetry ───────────────────────────────────────────────────
os.environ.setdefault("OPENAI_TELEMETRY", "0")
if hasattr(openai, "telemetry") and hasattr(openai.telemetry, "TelemetryClient"):
    openai.telemetry.TelemetryClient.capture = lambda *_, **__: None

# ── LLM configuration ─────────────────────────────────────────────────────────
_MODEL = os.getenv("JIRA_CONTENT_MODEL", "gpt-4o")
_TEMPERATURE = float(os.getenv("JIRA_CONTENT_TEMPERATURE", "0.2"))

_llm = ChatOpenAI(model=_MODEL, temperature=_TEMPERATURE)

# ── Jinja2 prostředí pro šablony ────────────────────────────────────────────
_PROMPT_ENV = Environment(
    loader=FileSystemLoader(Path(__file__).resolve().parent.parent / "prompts"),
    autoescape=select_autoescape(disabled_extensions=("jinja2",)),
    trim_blocks=True,
    lstrip_blocks=True,
)

def _render(name: str, **kwargs) -> str:
    """Vyrenderuj zadanou .jinja2 šablonu s parametry."""
    return _PROMPT_ENV.get_template(name).render(**kwargs)

# ── Internal helper ───────────────────────────────────────────────────────────
def _run(prompt: str) -> str:
    """Call the LLM synchronously and return plain text (Markdown‑variant)."""
    return _llm.invoke(prompt).content.strip()

# ADF helper ---------------------------------------------------------------
def _run_json(prompt: str) -> Dict[str, Any]:
    """Call LLM and parse response as JSON/ADF."""
    raw = _llm.invoke(prompt).content.strip()
    return json.loads(raw)

# ── High-level generators ─────────────────────────────────────────────────────
def enhance_idea(
    summary: str,
    description: str | None = None,
    audience: str = "mixed",
    max_words: int = 360,
    ) -> str:

    prompt = _render(
        "idea.jinja2",
        summary=summary,
        description=description,
        audience=audience,
        max_words=max_words,
    )
    return _run(prompt)

# ---------------------------- NEW ADF variant ----------------------------
def enhance_idea_adf(
    summary: str,
    description: str | None = None,
    audience: str = "mixed",
    max_words: int = 360,
) -> Dict[str, Any]:
    """
    Generate Jira‑Idea description **directly in ADF**.
    """
    prompt = _render(
        "idea_adf.jinja2",
        summary=summary,
        description=description,
        audience=audience,
        max_words=max_words,
    )
    return _run_json(prompt)

def epic_from_idea(
        summary: str,
        description: str | None = None
        ) -> str:
    """
    Scale an Idea into a **Jira Epic**.  Outputs Markdown containing:
      • Epic Goal
      • Context
      • Definition of Done
      • Acceptance criteria
      • Out of scope
    """
    prompt = _render(
        "epic.jinja2",
        summary=summary,
        description=description,
    )
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
    prompt = _render(
        "user_stories.jinja2",
        epic_name=epic_name,
        epic_description=epic_description,
        count=count,
    )
    return _run(prompt)


__all__: List[str] = [
    "enhance_idea",
    "enhance_idea_adf",
    "epic_from_idea",
    "user_stories_for_epic",
]



