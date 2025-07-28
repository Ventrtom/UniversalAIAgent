"""Helpers for microphone recording and audio transcription."""

from __future__ import annotations

import logging
from typing import Tuple

import gradio as gr
from openai import OpenAI

from .js_snippets import AUTO_STOP_START_JS, AUTO_STOP_STOP_JS, MIC_TOGGLE_RECORD_JS, START_ONE_SHOT_RECORD_JS, TOGGLE_WAKE_LISTENER_JS

logger = logging.getLogger(__name__)

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    """Create the OpenAI client lazily to avoid import-time failures."""
    global _client
    if _client is None:
        _client = OpenAI()
    return _client


def transcribe_audio(path: str) -> str:
    """Transcribe ``path`` using OpenAI Whisper."""
    if not path:
        return ""
    try:
        with open(path, "rb") as f:
            result = _get_client().audio.transcriptions.create(
                model="whisper-1", file=f
            )
        return result.text.strip()
    except Exception as exc:  # pragma: no cover - network
        logger.error("Audio transcription failed: %s", exc)
        raise


def show_mic() -> Tuple[gr.Audio, gr.Markdown]:
    """Reveal microphone input and reset status."""
    return gr.update(visible=True, value=None), gr.update(value="", visible=False)


def handle_recording(path: str) -> Tuple[gr.Audio, str, gr.Markdown]:
    """Transcribe recording and populate the message box."""
    if not path:
        return gr.update(visible=False, value=None), "", gr.update(visible=False)
    try:
        text = transcribe_audio(path)
        return (
            gr.update(visible=False, value=None),
            text,
            gr.update(value="", visible=False),
        )
    except Exception as exc:  # pragma: no cover - network
        return (
            gr.update(visible=False, value=None),
            "",
            gr.update(value=f"❌ Přepis selhal: {exc}", visible=True),
        )


def cancel_recording() -> Tuple[gr.Audio, gr.Markdown]:
    """Cancel ongoing recording and hide recorder widgets."""
    return gr.update(visible=False, value=None), gr.update(value="", visible=False)


__all__ = [
    "START_ONE_SHOT_RECORD_JS",
    "TOGGLE_WAKE_LISTENER_JS",
    "MIC_TOGGLE_RECORD_JS",
    "AUTO_STOP_START_JS",
    "AUTO_STOP_STOP_JS",
    "transcribe_audio",
    "show_mic",
    "handle_recording",
    "cancel_recording",
]
