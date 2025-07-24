"""Factory for building the Gradio interface."""

from __future__ import annotations

import functools

import gradio as gr

from . import audio_utils, chat_agent, file_utils


def create_interface() -> gr.Blocks:
    """Assemble the UI and wire all callbacks."""
    custom_css = """
    .gradio-container {
        max-width: 100% !important;
        margin: 0 auto;
    }
    .chat-message {
        margin: 10px 0;
    }
    .file-preview {
        font-family: monospace;
        font-size: 12px;
    }
    .small-button {
        padding: 0.25rem 0.5rem;
        font-size: 0.85rem;
    }
    .input-row textarea {
        width: 100%;
    }
    """

    with gr.Blocks(
        title="Universal AI Agent", theme=gr.themes.Soft(), css=custom_css
    ) as demo:
        gr.Markdown(
            "## 🤖 Universal AI Agent\nInteligentní asistent pro výzkum a analýzu dat"
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
                with gr.Row(elem_classes=["input-row"]):
                    msg = gr.Textbox(
                        lines=2,
                        scale=6,
                        placeholder="Zadejte svůj dotaz zde... (Shift+Enter pro odeslání)",
                        label="Váš dotaz",
                        show_copy_button=True,
                    )
                    mic_btn = gr.Button(
                        "🎤",
                        variant="secondary",
                        elem_classes=["small-button"],
                        scale=1,
                    )
                    with gr.Column(scale=1):
                        submit_btn = gr.Button(
                            "📤 Odeslat",
                            variant="primary",
                            elem_classes=["small-button"],
                        )
                        clear_btn = gr.Button(
                            "🗑️ Vyčistit",
                            variant="secondary",
                            elem_classes=["small-button"],
                        )
                audio_input = gr.Audio(
                    sources=["microphone"],
                    type="filepath",
                    visible=False,
                    show_label=False,
                    elem_id="mic_recorder",
                )
                audio_status = gr.Markdown("", visible=False)
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
                    choices=file_utils.list_files(),
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

        chat_cb = functools.partial(
            chat_agent.chat_fn, live_log_markdown=live_log_markdown
        )
        msg.submit(
            chat_cb,
            inputs=[msg, chatbot, reveal, steps_state],
            outputs=[chatbot, chatbot, steps_state],
        ).then(lambda: "", outputs=[msg])
        submit_btn.click(
            chat_cb,
            inputs=[msg, chatbot, reveal, steps_state],
            outputs=[chatbot, chatbot, steps_state],
        ).then(lambda: "", outputs=[msg])
        mic_btn.click(audio_utils.show_mic, outputs=[audio_input, audio_status])
        audio_input.start_recording(fn=None, js=audio_utils.AUTO_STOP_START_JS)
        audio_input.stop_recording(
            lambda: gr.update(value="⏳ Přepis…", visible=True),
            outputs=[audio_status],
            js=audio_utils.AUTO_STOP_STOP_JS,
        ).then(
            audio_utils.handle_recording,
            inputs=[audio_input],
            outputs=[audio_input, msg, audio_status],
        )
        audio_input.clear(
            audio_utils.cancel_recording,
            outputs=[audio_input, audio_status],
            js=audio_utils.AUTO_STOP_STOP_JS,
        )
        clear_btn.click(chat_agent.clear_chat, outputs=[chatbot, msg]).then(
            lambda: [], outputs=[steps_state]
        ).then(lambda: gr.update(value="", visible=False), outputs=[live_log_markdown])
        files.change(
            file_utils.file_selected,
            inputs=[files],
            outputs=[content, download_file, file_name],
        )
        refresh_btn.click(file_utils.refresh_choices, outputs=[files]).then(
            file_utils.file_selected,
            inputs=[files],
            outputs=[content, download_file, file_name],
        )
        save_btn.click(
            file_utils.save_file,
            inputs=[content, file_name, files],
            outputs=[files, download_file, file_name],
        ).then(
            file_utils.file_selected,
            inputs=[files],
            outputs=[content, download_file, file_name],
        )
        delete_btn.click(file_utils.delete_file, inputs=[files], outputs=[files]).then(
            file_utils.file_selected,
            inputs=[files],
            outputs=[content, download_file, file_name],
        )
        download_btn.click(
            file_utils.trigger_download, inputs=[files], outputs=[download_file]
        )
        if file_utils.list_files():
            demo.load(
                file_utils.file_selected,
                inputs=[files],
                outputs=[content, download_file, file_name],
            )
    return demo
