# ui.py
"""
Lehký lokální front-end k Universal AI Agentovi.
Spuštění:  pip install -U gradio fastapi uvicorn
           python ui.py
"""

import asyncio, json, os, pathlib, re, warnings
import gradio as gr
from main import agent_executor

# — kosmetika —
warnings.filterwarnings("ignore", message=".*LangChainDeprecationWarning.*")
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "false")

# — složka output/ —
OUTPUT_DIR = pathlib.Path("output"); OUTPUT_DIR.mkdir(exist_ok=True)

def list_files():                     # -> list[str]
    return sorted(p.name for p in OUTPUT_DIR.iterdir() if p.is_file())

def read_file(fname):                 # -> str
    path = OUTPUT_DIR / fname
    try:
        return path.read_text("utf-8")
    except (UnicodeDecodeError, FileNotFoundError):
        return f"[Nelze zobrazit: binární nebo neexistuje] {fname}"

def file_path(fname):                 # -> str|None
    p = OUTPUT_DIR / fname
    return str(p) if p.exists() else None

def refresh_choices():
    """Kompatibilní update pro všechna vydání Gradia."""
    try:
        return gr.Dropdown.update(choices=list_files())
    except AttributeError:
        return gr.update(choices=list_files())

# — z JSONu uděláme hezký Markdown —
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

# — chat handler —
async def chat_fn(msg, history):
    history = history or []
    history.append({"role": "user", "content": msg})

    loop = asyncio.get_event_loop()
    raw = await loop.run_in_executor(None, lambda: agent_executor.invoke({"query": msg})["output"])
    history.append({"role": "assistant", "content": pretty(raw)})
    return history, history

# — callback pro výběr souboru —
def file_selected(fname):
    """Vrátí (náhled, update pro File komponentu) kompatibilně pro 3.x i 4.x."""
    path = file_path(fname)
    # 1) text do textboxu
    preview = read_file(fname)
    # 2) update File komponenty
    try:
        file_update = gr.File.update(value=path, visible=bool(path))
    except AttributeError:           # starší Gradio
        file_update = gr.update(value=path, visible=bool(path))
    return preview, file_update

# — callback pro refresh tlačítka —
def trigger_download(fname):
    path = file_path(fname)
    try:
        return gr.File.update(value=path, visible=bool(path))
    except AttributeError:
        return gr.update(value=path, visible=bool(path))

# — UI —
with gr.Blocks(title="Universal AI Agent") as demo:
    gr.Markdown("## 💬 AI Agent • 🗂 Output soubory")

    with gr.Row():
        # CHAT
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(type="messages", height=420)
            msg = gr.Textbox(lines=2, placeholder="Zadej dotaz… (Ctrl+Enter)")
            msg.submit(chat_fn, [msg, chatbot], [chatbot, chatbot]).then(lambda: "", None, msg)

        # FILE EXPLORER
        with gr.Column():
            files = gr.Dropdown(choices=list_files(), label="Output soubory", interactive=True)
            content = gr.Textbox(label="Náhled obsahu", lines=14, interactive=False, show_copy_button=True)
            download_file = gr.File(label="Klikni pro stažení", visible=False)

            # když uživatel zvolí soubor
            files.change(file_selected, files, [content, download_file])

            with gr.Row():
                gr.Button("↻ Refresh").click(refresh_choices, None, files)\
                                        .then(file_selected, files, [content, download_file])
                gr.Button("⬇ Stáhnout").click(trigger_download, files, download_file)

    # Autonáhled prvního souboru hned po startu (pokud nějaký existuje)
    if list_files():
        demo.load(file_selected, inputs=files, outputs=[content, download_file])

    demo.queue()
    demo.launch()
