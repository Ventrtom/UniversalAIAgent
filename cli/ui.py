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
import functools
from typing import Optional, List, Dict, Any, Tuple, AsyncGenerator

from agent import handle_query, handle_query_stream, ResearchResponse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings("ignore", message=".*LangChainDeprecationWarning.*")
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "false")

# Constants
FILES_DIR = pathlib.Path("files")
FILES_DIR.mkdir(exist_ok=True)
MAX_HISTORY_LENGTH = 50  # Limit chat history to prevent memory issues


def list_files() -> List[str]:
    """Get sorted list of files in the shared directory."""
    try:
        return sorted(p.name for p in FILES_DIR.iterdir() if p.is_file())
    except Exception as e:
        logger.error(f"Error listing files: {e}")
        return []


def read_file(fname: str) -> str:
    """Read file content with proper error handling."""
    if not fname:
        return ""

    path = FILES_DIR / fname
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
    """Return full path to file in ``files`` or ``None`` if not specified."""
    if not fname:
        return None
    p = FILES_DIR / fname
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


def file_selected(fname: Optional[str]) -> Tuple[str, gr.File, str]:
    """Handle file selection with improved error handling."""
    if not fname:
        return "", gr.File(visible=False), ""

    path = file_path(fname)
    preview = read_file(fname)

    return preview, gr.File(value=path, visible=bool(path)), fname


def trigger_download(fname: Optional[str]) -> gr.File:
    """Trigger file download."""
    if not fname:
        return gr.File(visible=False)

    path = file_path(fname)
    return gr.File(value=path, visible=bool(path))


def save_file(
    content: str, new_name: str, original_name: str
) -> Tuple[gr.Dropdown, gr.File, str]:
    """Persist edited content and optionally rename the file."""
    if not original_name:
        return refresh_choices(), gr.File(visible=False), ""

    old_path = FILES_DIR / original_name
    target_name = new_name.strip() or original_name
    if not os.path.splitext(target_name)[1]:
        target_name += old_path.suffix
    new_path = FILES_DIR / target_name

    if old_path != new_path:
        try:
            old_path.rename(new_path)
        except Exception as exc:
            logger.error(f"Error renaming file: {exc}")
            new_path = old_path

    try:
        new_path.write_text(content, encoding="utf-8")
    except Exception as exc:
        logger.error(f"Error writing file: {exc}")

    dropdown = gr.Dropdown(choices=list_files(), value=new_path.name)
    download = gr.File(value=str(new_path), visible=True)
    return dropdown, download, new_path.name


def delete_file(fname: str) -> gr.Dropdown:
    """Delete the selected file and refresh dropdown."""
    if not fname:
        return refresh_choices()

    path = FILES_DIR / fname
    try:
        path.unlink()
    except Exception as exc:
        logger.error(f"Error deleting file {fname}: {exc}")

    return refresh_choices()


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

                file_name = gr.Textbox(label="Název souboru", interactive=True)

                content = gr.Textbox(
                    label="Náhled obsahu",
                    lines=14,
                    interactive=True,
                    show_copy_button=True,
                    elem_classes=["file-preview"],
                )

                download_file = gr.File(
                    label="📥 Stáhnout vybraný soubor", visible=False
                )

                with gr.Row():
                    refresh_btn = gr.Button("🔄 Obnovit seznam", variant="secondary")
                    save_btn = gr.Button("💾 Uložit", variant="primary")
                    delete_btn = gr.Button("🗑️ Smazat", variant="stop")
                    download_btn = gr.Button("⬇️ Stáhnout", variant="primary")

        # Chat interactions
        chat_cb = functools.partial(chat_fn, live_log_markdown=live_log_markdown)
        msg_submit = msg.submit(
            chat_cb,
            inputs=[msg, chatbot, reveal, steps_state],
            outputs=[chatbot, chatbot, steps_state],
        ).then(lambda: "", outputs=[msg])

        submit_btn.click(
            chat_cb,
            inputs=[msg, chatbot, reveal, steps_state],
            outputs=[chatbot, chatbot, steps_state],
        ).then(lambda: "", outputs=[msg])

        clear_btn.click(clear_chat, outputs=[chatbot, msg]).then(
            lambda: [], outputs=[steps_state]
        ).then(lambda: gr.update(value="", visible=False), outputs=[live_log_markdown])

        # File interactions
        files.change(
            file_selected, inputs=[files], outputs=[content, download_file, file_name]
        )

        refresh_btn.click(refresh_choices, outputs=[files]).then(
            file_selected, inputs=[files], outputs=[content, download_file, file_name]
        )

        save_btn.click(
            save_file,
            inputs=[content, file_name, files],
            outputs=[files, download_file, file_name],
        )

        delete_btn.click(delete_file, inputs=[files], outputs=[files]).then(
            file_selected, inputs=[files], outputs=[content, download_file, file_name]
        )

        download_btn.click(trigger_download, inputs=[files], outputs=[download_file])

        # Load initial file if available
        if list_files():
            demo.load(
                file_selected,
                inputs=[files],
                outputs=[content, download_file, file_name],
            )

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
