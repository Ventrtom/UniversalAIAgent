# cli/ui.py
"""Lightweight local front-end for the Universal AI Agent."""
from __future__ import annotations

import asyncio
import json
import os
import pathlib
import re
import warnings

import gradio as gr

from agent import handle_query

warnings.filterwarnings("ignore", message=".*LangChainDeprecationWarning.*")
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "false")

OUTPUT_DIR = pathlib.Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)


def list_files() -> list[str]:
    return sorted(p.name for p in OUTPUT_DIR.iterdir() if p.is_file())


def read_file(fname: str) -> str:
    path = OUTPUT_DIR / fname
    try:
        return path.read_text("utf-8")
    except (UnicodeDecodeError, FileNotFoundError):
        return f"[Nelze zobrazit: binární nebo neexistuje] {fname}"


def file_path(fname: str) -> str | None:
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


async def chat_fn(msg, history):
    history = history or []
    history.append({"role": "user", "content": msg})

    loop = asyncio.get_event_loop()
    raw = await loop.run_in_executor(None, lambda: handle_query(msg))
    history.append({"role": "assistant", "content": pretty(raw)})
    return history, history


def file_selected(fname):
    path = file_path(fname)
    preview = read_file(fname)
    try:
        file_update = gr.File.update(value=path, visible=bool(path))
    except AttributeError:
        file_update = gr.update(value=path, visible=bool(path))
    return preview, file_update


def trigger_download(fname):
    path = file_path(fname)
    try:
        return gr.File.update(value=path, visible=bool(path))
    except AttributeError:
        return gr.update(value=path, visible=bool(path))


def launch() -> None:
    with gr.Blocks(title="Universal AI Agent") as demo:
        gr.Markdown("## 💬 AI Agent • 🗂 Output soubory")

        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(type="messages", height=420)
                msg = gr.Textbox(lines=2, placeholder="Zadej dotaz… (Ctrl+Enter)")
                msg.submit(chat_fn, [msg, chatbot], [chatbot, chatbot]).then(lambda: "", None, msg)

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
