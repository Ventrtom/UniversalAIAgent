from __future__ import annotations

"""Factory for building the improved Gradio interface.

Layout-only changes: All functions, callbacks, and variable names remain unchanged to preserve full compatibility with
other modules (audio_utils, chat_agent, file_utils, etc.).  The visual structure has been modernised for better
responsiveness, readability and user‑focus.  The file‑manager panel is now collapsible (Accordion), and the chat input
row is sticky at the bottom of the viewport on tall screens.  Custom CSS keeps everything responsive down to mobile
sizes without breaking existing IDs / class hooks.
"""

import functools

import gradio as gr

from . import audio_utils, chat_agent, file_utils


# ---------------------------------------------------------------------------
#  Main factory
# ---------------------------------------------------------------------------

def create_interface() -> gr.Blocks:
    """Assemble the UI and wire all callbacks.

    Only layout / styling was updated – all original component instances, labels and
    callbacks are preserved so that existing tests and integrations work unchanged.
    """

    # --- Custom CSS ---------------------------------------------------------
    custom_css = """
    /* Container and general layout tweaks */
    .gradio-container {
        max-width: 100% !important;   /* tighten maximum width for better readability */
        width: 100% !important;
        margin: 0 auto;
        padding: 0 1rem;
    }

    /* Chatbot area */
    .chatbot-box {
        height: 65vh !important;        /* allow more breathing room, grows with viewport */
        width: 90%;
        overflow-y: auto;
    }

    .chat-message {
        margin: 10px 0;
    }

    /* Sticky input bar on tall screens */
    @media (min-height: 600px) {
        .input-toolbar {
            position: sticky;
            bottom: 0;
            background: var(--background-fill-primary);
            padding-bottom: 0.75rem;
            border-top: 1px solid var(--border-color-primary);
        }
    }

    /* File preview monospace */
    .file-preview {
        font-family: monospace;
        font-size: 12px;
    }

    /* Compact buttons */
    .small-button {
        padding: 0.25rem 0.5rem;
        font-size: 0.85rem;
    }

    /* Allow the chat textbox to fill the row */
    .input-toolbar textarea {
        width: 100% !important;
    }

    /* Better mobile handling */
    @media (max-width: 768px) {
        .chatbot-box { height: 55vh !important; }
    }
    """

    # --- Placeholder pro graf; skutečný HTML s Cytoscape dodá Python při loadu ---
    INITIAL_GRAPH_HTML = """
    <div style="width:100%;height:70vh;border:1px solid #ddd;border-radius:8px;
                background:#fff;display:flex;align-items:center;justify-content:center">
      Načítám graf…
    </div>
    """

    # --- JS renderer: načte Cytoscape a vykreslí graf z dodaného schématu -----
    CYTO_RENDER_JS = """
    async (schema) => {
      const mount = document.getElementById('graph_canvas');
      if (!mount) return "";
      // Připravíme kontejner pro Cytoscape
      mount.innerHTML = "<div id='cy' style=\\"width:100%;height:70vh;border:1px solid #ddd;border-radius:8px;background:#fff\\"></div>";
      // Dynamicky načti Cytoscape, pokud ještě není k dispozici
      const ensureScript = (src) => new Promise((resolve, reject) => {
        if (window.cytoscape) return resolve();
        const s = document.createElement('script');
        s.src = src; s.async = true;
        s.onload = () => resolve();
        s.onerror = (e) => reject(e);
        document.head.appendChild(s);
      });
      await ensureScript("https://unpkg.com/cytoscape@3/dist/cytoscape.min.js");
      const elements = [
        ...(schema?.nodes || []).map(n => ({ data: { id: n.id, label: n.label } })),
        ...(schema?.edges || []).map(e => ({ data: { source: e.source, target: e.target } })),
      ];
      const cy = window.cytoscape({
        container: document.getElementById('cy'),
        elements,
        layout: { name: 'breadthfirst', directed: true, padding: 10 },
        style: [
          { selector:'node', style:{ 'label':'data(label)','text-valign':'center','background-color':'#eee','border-width':1,'border-color':'#888' } },
          { selector:'node.active', style:{ 'background-color':'#cde','border-color':'#39f','border-width':2 } },
          { selector:'edge', style:{ 'curve-style':'bezier','target-arrow-shape':'triangle','width':1,'line-color':'#bbb','target-arrow-color':'#bbb' } },
        ]
      });
      // Uložíme instanci a zavedeme updateGraph pro budoucí "živé" zvýrazňování
      window._cy = cy;
      if (!window.updateGraph) {
        window.updateGraph = (evt) => {
          if (!window._cy || !evt) return;
          // 1) Uzlové události z jádra: {phase:'start'|'end', name:'act'|...}
          if (evt.phase && evt.name) {
            const id = String(evt.name);
            const sel = window._cy.$id(id);
            window._cy.$('node').removeClass('active');
            if (sel && sel.length) {
              sel.addClass('active');
              window._cy.center(sel);
            }
            return;
          }
          // 2) Volání nástrojů: {type:'tool', name:'jira_issue_detail' }
          if (evt.type === 'tool' && evt.name) {
            // placeholder – zde lze později doplnit např. badge/flash
          }
        };
      }
      // vrátíme aktuální HTML pro synchronizaci hodnoty komponenty
      return mount.innerHTML;
    }
    """

    # --- UI Construction ----------------------------------------------------
    with gr.Blocks(
        title="Universal AI Agent",
        theme=gr.themes.Soft(),
        css=custom_css,
    ) as demo:
        # Header -----------------------------------------------------------------
        gr.Markdown("""# 🤖 Universal AI Agent\n*Inteligentní asistent pro výzkum a analýzu dat*""")

        # Dvousloupcové rozdělení 60/40 (větší vlevo = 60 %, menší vpravo = 40 %)
        with gr.Row(equal_height=True):
            # Levý sloupec (60 %) – chat & ovládací prvky ------------------------
            with gr.Column(scale=3, min_width=640):
                # Chat area -------------------------------------------------------
                chatbot = gr.Chatbot(
                    type="messages",
                    height=500,  # CSS ho na větších viewportecht stejně roztáhne
                    label="Konverzace",
                    show_copy_button=True,
                    elem_classes=["chatbot-box", "chat-message"],
                    line_breaks=True,
                    sanitize_html=False,
                )

            # Input row (sticky toolbar) ------------------------------------
                with gr.Row(elem_classes=["input-toolbar"]):
                    with gr.Column(scale=0.8):
                        msg = gr.Textbox(
                            lines=4,
                            scale=6,
                            placeholder="Zadejte svůj dotaz zde... (Shift+Enter pro odeslání)",
                            label="Váš dotaz",
                            show_copy_button=True,
                            )
                    
                    # Column to stack submit / clear vertically on taller screens
                    with gr.Column(scale=0.2):
                        mic_btn = gr.Button(
                            "🎤 Audio input",
                            variant="secondary",
                            elem_classes=["small-button"],
                            scale=1,
                        )
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

            # Hidden audio components (unchanged) ----------------------------
                audio_input = gr.Audio(
                    sources=["microphone"],
                    type="filepath",
                    visible=False,
                    show_label=False,
                    elem_id="mic_recorder",
                )
                audio_status = gr.Markdown("", visible=False)

            # Reveal processing steps (kept outside accordion) ---------------
                with gr.Row():
                    reveal = gr.Checkbox(
                        label="🔍 Zobrazit kroky zpracování",
                        info="Ukáže detaily o tom, jak agent zpracovával váš dotaz",
                    )

            # ---------------------------------------------------------------
                # Collapsible file‑manager panel to declutter primary chat view
                # ---------------------------------------------------------------
                with gr.Accordion("📁 Výstupní soubory", open=False):
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
                    # File‑action buttons ---------------------------------------
                    with gr.Row():
                        refresh_btn = gr.Button("🔄 Obnovit seznam", variant="secondary")
                        save_btn = gr.Button("💾 Uložit", variant="primary")
                        delete_btn = gr.Button("🗑️ Smazat", variant="stop")
                        download_btn = gr.Button("⬇️ Stáhnout", variant="primary")

            # Pravý sloupec (40 %) – runtime vizualizace ------------------------
            with gr.Column(scale=2, min_width=420):
                gr.Markdown("### 🗺️ Runtime vizualizace agenta")
                graph_canvas = gr.HTML(
                    value=INITIAL_GRAPH_HTML,
                    label="Agent Runtime Map",
                    elem_id="graph_canvas"
                    )
                graph_events = gr.HTML(value="<div style='height:0;overflow:hidden'></div>",
                       show_label=False)
                graph_schema_state = gr.State({})
 
        # Po načtení stránky: 1) získat schéma z Pythonu; 2) vykreslit přes JS
        demo.load(chat_agent.graph_schema, outputs=[graph_schema_state]).then(
            None, inputs=[graph_schema_state], outputs=[graph_canvas], js=CYTO_RENDER_JS
        )

        #  CALLBACK WIRING – identical to original implementation
        chat_cb = functools.partial(
            chat_agent.chat_fn, live_log_markdown=live_log_markdown
        )

        msg.submit(
            chat_cb,
            inputs=[msg, chatbot, reveal, steps_state],
            outputs=[chatbot, chatbot, steps_state, live_log_markdown, graph_events],
                ).then(lambda: "", outputs=[msg])

        submit_btn.click(
            chat_cb,
            inputs=[msg, chatbot, reveal, steps_state],
            outputs=[chatbot, chatbot, steps_state, live_log_markdown, graph_events],
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
        ).then(lambda: gr.update(value="", visible=False), outputs=[live_log_markdown]
        ).then(lambda: INITIAL_GRAPH_HTML, outputs=[graph_canvas]
        ).then(lambda: "", outputs=[graph_events]
        ).then(chat_agent.graph_schema, outputs=[graph_schema_state]
        ).then(None, inputs=[graph_schema_state], outputs=[graph_canvas], js=CYTO_RENDER_JS)

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

        # Auto‑load first file preview if any files are present -----------------
        if file_utils.list_files():
            demo.load(
                file_utils.file_selected,
                inputs=[files],
                outputs=[content, download_file, file_name],
            )

    return demo
