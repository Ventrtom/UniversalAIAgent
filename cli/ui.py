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
from typing import Optional, List, Dict, Any, Tuple, AsyncGenerator

from agent import handle_query, handle_query_stream, ResearchResponse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings("ignore", message=".*LangChainDeprecationWarning.*")
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "false")

# Constants
OUTPUT_DIR = pathlib.Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)
MAX_HISTORY_LENGTH = 50  # Limit chat history to prevent memory issues


def list_files() -> List[str]:
    """Get sorted list of files in output directory."""
    try:
        return sorted(p.name for p in OUTPUT_DIR.iterdir() if p.is_file())
    except Exception as e:
        logger.error(f"Error listing files: {e}")
        return []


def read_file(fname: str) -> str:
    """Read file content with proper error handling."""
    if not fname:
        return ""

    path = OUTPUT_DIR / fname
    try:
        return path.read_text("utf-8")
    except UnicodeDecodeError:
        return f"[Nelze zobrazit: binární soubor] {fname}"
    except FileNotFoundError:
        return f"[Soubor neexistuje] {fname}"
    except Exception as e:
        logger.error(f"Error reading file {fname}: {e}")
        return f"[Chyba při čtení souboru] {fname}"


def file_path(fname: Optional[str]) -> Optional[str]:
    """Return full path to output file, or None if not specified."""
    if not fname:
        return None
    p = OUTPUT_DIR / fname
    return str(p) if p.exists() else None


def refresh_choices() -> gr.Dropdown:
    """Refresh dropdown choices with current files."""
    choices = list_files()
    return gr.Dropdown(choices=choices, value=choices[0] if choices else None)


def pretty_format_response(raw: str) -> str:
    """Format raw response for better display."""
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


def limit_history(
    history: List[Dict[str, Any]], max_length: int = MAX_HISTORY_LENGTH
) -> List[Dict[str, Any]]:
    """Limit chat history length to prevent memory issues."""
    if len(history) <= max_length:
        return history

    # Keep system messages and recent messages
    system_messages = [msg for msg in history if msg.get("role") == "system"]
    recent_messages = history[-(max_length - len(system_messages)) :]

    return system_messages + recent_messages


def format_intermediate_steps(steps: List[Any]) -> str:
    """Format intermediate steps for better readability."""
    if not steps:
        return ""

    formatted_steps = []
    for i, step in enumerate(steps, 1):
        if hasattr(step, "tool") and hasattr(step, "tool_input"):
            # LangChain ToolAgentAction
            tool_name = step.tool
            tool_input = step.tool_input

            # Format tool input nicely
            if isinstance(tool_input, dict):
                input_str = ", ".join(f"{k}: {v}" for k, v in tool_input.items())
            else:
                input_str = str(tool_input)

            formatted_steps.append(f"**{i}.** 🔧 **{tool_name}**({input_str})")

        elif isinstance(step, tuple) and len(step) == 2:
            # (action, result) tuple
            action, result = step
            if hasattr(action, "tool"):
                tool_name = action.tool
                tool_input = action.tool_input
                if isinstance(tool_input, dict):
                    input_str = ", ".join(f"{k}: {v}" for k, v in tool_input.items())
                else:
                    input_str = str(tool_input)

                formatted_steps.append(f"**{i}.** 🔧 **{tool_name}**({input_str})")

                # Add result if it's meaningful and not too long
                if result and isinstance(result, str) and len(result) < 200:
                    formatted_steps.append(f"   📋 Result: {result}")
            else:
                formatted_steps.append(f"**{i}.** {str(step)}")
        else:
            # Fallback for other types
            step_str = str(step)
            if len(step_str) > 300:
                step_str = step_str[:300] + "..."
            formatted_steps.append(f"**{i}.** {step_str}")

    return "\n".join(formatted_steps)
    """Limit chat history length to prevent memory issues."""
    if len(history) <= max_length:
        return history

    # Keep system messages and recent messages
    system_messages = [msg for msg in history if msg.get("role") == "system"]
    recent_messages = history[-(max_length - len(system_messages)) :]

    return system_messages + recent_messages


async def chat_fn(
    msg: str,
    history: Optional[List[Dict[str, Any]]],
    reveal: bool = False,
    steps: Optional[List[str]] = None,
    live_log_markdown: gr.Markdown | None = None,
) -> AsyncGenerator[Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[str]], None]:
    """Handle chat interaction with proper error handling."""
    if not msg.strip():
        yield history or [], history or [], steps or []
        return

    history = history or []
    steps = steps or []
    steps.clear()

    if live_log_markdown is not None:
        live_log_markdown.value = ""
        live_log_markdown.visible = True

    history.append({"role": "user", "content": msg})

    bot = {"role": "assistant", "content": ""}
    history.append(bot)

    try:
        async for tok in handle_query_stream(msg):
            if tok.startswith("§STEP§"):
                steps.append(tok[7:])
                if live_log_markdown is not None:
                    live_log_markdown.value = "\n".join(
                        f"**{i}.** {s}" for i, s in enumerate(steps, 1)
                    )
                await asyncio.sleep(0)
                continue

            bot["content"] += tok
            # Limit history to prevent memory issues
            current_history = limit_history(history)
            yield current_history, current_history, steps

        # Parse final response
        try:
            raw_json = bot["content"].splitlines()[-1]
            parsed = ResearchResponse.parse_raw(raw_json)
            bot["content"] = parsed.answer

            if reveal and parsed.intermediate_steps:
                formatted_steps = format_intermediate_steps(parsed.intermediate_steps)
                bot["content"] += f"\n\n**🔍 Intermediate steps:**\n{formatted_steps}"
        except Exception as e:
            logger.error(f"Error parsing response: {e}")
            # Keep the raw content if parsing fails
            pass

    except Exception as e:
        logger.error(f"Error in chat function: {e}")
        bot["content"] = f"Omlouváme se, došlo k chybě: {str(e)}"

    final_history = limit_history(history)

    if not reveal and live_log_markdown is not None:
        live_log_markdown.value = ""
        live_log_markdown.visible = False
        steps.clear()

    yield final_history, final_history, steps


def file_selected(fname: Optional[str]) -> Tuple[str, gr.File]:
    """Handle file selection with improved error handling."""
    if not fname:
        return "", gr.File(visible=False)

    path = file_path(fname)
    preview = read_file(fname)

    return preview, gr.File(value=path, visible=bool(path))


def trigger_download(fname: Optional[str]) -> gr.File:
    """Trigger file download."""
    if not fname:
        return gr.File(visible=False)

    path = file_path(fname)
    return gr.File(value=path, visible=bool(path))


def clear_chat() -> Tuple[List, str]:
    """Clear chat history."""
    return [], ""


def launch() -> None:
    """Launch the Gradio interface."""
    # Custom CSS for better styling
    custom_css = """
    .gradio-container {
        max-width: 1200px !important;
    }
    .chat-message {
        margin: 10px 0;
    }
    .file-preview {
        font-family: monospace;
        font-size: 12px;
    }
    """

    with gr.Blocks(
        title="Universal AI Agent", theme=gr.themes.Soft(), css=custom_css
    ) as demo:
        gr.Markdown(
            "## 🤖 Universal AI Agent\n"
            "Inteligentní asistent pro výzkum a analýzu dat"
        )

        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    type="messages",
                    height=500,
                    label="Konverzace",
                    show_copy_button=True,
                    elem_classes=["chat-message"],
                    line_breaks=True,
                    sanitize_html=False,
                )

                with gr.Row():
                    msg = gr.Textbox(
                        lines=2,
                        scale=10,
                        placeholder="Zadejte svůj dotaz zde... (Shift+Enter pro odeslání)",
                        label="Váš dotaz",
                        show_copy_button=True,
                    )
                    with gr.Column(scale=1):
                        submit_btn = gr.Button("📤 Odeslat", variant="primary")
                        clear_btn = gr.Button("🗑️ Vyčistit", variant="secondary")

                with gr.Row():
                    reveal = gr.Checkbox(
                        label="🔍 Zobrazit kroky zpracování",
                        info="Ukáže detaily o tom, jak agent zpracovával váš dotaz",
                    )

            with gr.Column(scale=2):
                gr.Markdown("### 📁 Výstupní soubory")

                live_log_markdown = gr.Markdown("", label="Živý log")
                steps_state = gr.State([])

                files = gr.Dropdown(
                    choices=list_files(),
                    label="Dostupné soubory",
                    interactive=True,
                    info="Vyberte soubor pro náhled a stažení",
                )

                content = gr.Textbox(
                    label="Náhled obsahu",
                    lines=14,
                    interactive=False,
                    show_copy_button=True,
                    elem_classes=["file-preview"],
                )

                download_file = gr.File(
                    label="📥 Stáhnout vybraný soubor", visible=False
                )

                with gr.Row():
                    refresh_btn = gr.Button("🔄 Obnovit seznam", variant="secondary")
                    download_btn = gr.Button("⬇️ Stáhnout", variant="primary")

        # Chat interactions
        msg_submit = msg.submit(
            chat_fn,
            inputs=[msg, chatbot, reveal, steps_state, live_log_markdown],
            outputs=[chatbot, chatbot, steps_state],
        ).then(lambda: "", outputs=[msg])

        submit_btn.click(
            chat_fn,
            inputs=[msg, chatbot, reveal, steps_state, live_log_markdown],
            outputs=[chatbot, chatbot, steps_state],
        ).then(lambda: "", outputs=[msg])

        clear_btn.click(clear_chat, outputs=[chatbot, msg]).then(
            lambda: [], outputs=[steps_state]
        ).then(lambda: gr.update(value="", visible=False), outputs=[live_log_markdown])

        # File interactions
        files.change(file_selected, inputs=[files], outputs=[content, download_file])

        refresh_btn.click(refresh_choices, outputs=[files]).then(
            file_selected, inputs=[files], outputs=[content, download_file]
        )

        download_btn.click(trigger_download, inputs=[files], outputs=[download_file])

        # Load initial file if available
        if list_files():
            demo.load(file_selected, inputs=[files], outputs=[content, download_file])

        # Configure demo
        demo.queue(max_size=10)
        demo.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            debug=False,
            show_error=True,
            inbrowser=True,
        )


if __name__ == "__main__":
    launch()
