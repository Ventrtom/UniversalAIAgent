# cli/ui.py
"""Lightweight local front-end for the Universal AI Agent."""
from __future__ import annotations

import asyncio
import gradio as gr
import textwrap
import json
import os
import pathlib
import re
import warnings
import logging
from typing import Optional, List, Dict, Any, Tuple

from agent import handle_query, handle_query_stream, ResearchResponse

warnings.filterwarnings("ignore", message=".*LangChainDeprecationWarning.*")
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "false")

OUTPUT_DIR = pathlib.Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)


def list_files() -> list[str]:
    return sorted(p.name for p in OUTPUT_DIR.iterdir() if p.is_file())


def read_file(fname: str) -> str:
    path = OUTPUT_DIR / fname
    if not fname:
        return ""
    try:
        return path.read_text("utf-8")
    except (UnicodeDecodeError, FileNotFoundError):
        return f"[Nelze zobrazit: binární nebo neexistuje] {fname}"


def file_path(fname: str | None) -> str | None:
    """Vrátí plnou cestu k výstupnímu souboru, nebo None, pokud není zadán."""
    if not fname:
        return None
    p = OUTPUT_DIR / fname
    return str(p) if p.exists() else None


def refresh_choices():
    try:
        return gr.Dropdown.update(choices=list_files())
    except AttributeError:
        return gr.update(choices=list_files())


def pretty(raw: str) -> str:
    clean = raw.strip().removeprefix("```json").removesuffix("```").strip("` \n")
    m = re.search(r"\{.*\}", clean, re.DOTALL)
    if not m:
        return raw
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return raw
    summary = data.get("summary", raw).strip()
    extras = []
    if src := data.get("sources"):
        extras.append("_**Zdroje:**_ " + ", ".join(map(str, src)))
    if tools := data.get("tools_used"):
        extras.append("_**Nástroje:**_ " + ", ".join(map(str, tools)))
    return summary + ("\n\n" + "\n".join(extras) if extras else "")


async def chat_fn(msg, history, reveal=False):
    history = history or []
    history.append({"role": "user", "content": msg})

    bot = {"role": "assistant", "content": ""}
    history.append(bot)

    async for tok in handle_query_stream(msg):
        bot["content"] += tok
        yield history, history

    raw_json = bot["content"].splitlines()[-1]
    parsed = ResearchResponse.parse_raw(raw_json)
    bot["content"] = textwrap.fill(parsed.answer, 100)
    if reveal and parsed.intermediate_steps:
        bot["content"] += "\n\n**Intermediate steps:**\n" + "\n".join(parsed.intermediate_steps)
    yield history, history


def file_selected(fname):
    """Callback dropdownu: naplní náhled a zobrazí tlačítko pro stažení."""
    if not fname:                               # nic nevybráno → vyčisti UI
        return "", gr.update(visible=False)
    path = file_path(fname)
    preview = read_file(fname)
    try:
        file_update = gr.File.update(value=path, visible=bool(path))
    except AttributeError:                      # Gradio < 4.0
        file_update = gr.update(value=path, visible=bool(path))
    return preview, file_update


def trigger_download(fname):
    if not fname:
        return gr.update(visible=False)
    path = file_path(fname)
    try:
        return gr.File.update(value=path, visible=bool(path))
    except AttributeError:
        return gr.update(value=path, visible=bool(path))


def launch() -> None:
    with gr.Blocks(title="Universal AI Agent") as demo:
        gr.Markdown("## AI Agent • Output soubory")

        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(type="messages", height=500)
                with gr.Row():
                    msg = gr.Textbox(lines=2, scale=10, placeholder="Zadej dotaz… (Shift+Enter)")
                    reveal = gr.Checkbox(label="🔍 Reveal steps")
                msg.submit(
                        chat_fn, [msg, chatbot, reveal], [chatbot, chatbot]
                    ).then(lambda: "", None, msg)

            with gr.Column():
                files = gr.Dropdown(choices=list_files(), label="Output soubory", interactive=True)
                content = gr.Textbox(label="Náhled obsahu", lines=14, interactive=False, show_copy_button=True)
                download_file = gr.File(label="Klikni pro stažení", visible=False)

                files.change(file_selected, files, [content, download_file])

                with gr.Row():
                    gr.Button("↻ Refresh").click(refresh_choices, None, files).then(
                        file_selected, files, [content, download_file]
                    )
                    gr.Button("⬇ Stáhnout").click(trigger_download, files, download_file)

        if list_files():
            demo.load(file_selected, inputs=files, outputs=[content, download_file])

        demo.queue()
        demo.launch()


if __name__ == "__main__":
    launch()
