# cli/ui2.py
from __future__ import annotations
import asyncio, gradio as gr
from agent.core2 import handle_query   # ← důležité

async def chat_fn(msg, history):
    history = history or []
    history.append({"role": "user", "content": msg})
    loop = asyncio.get_event_loop()
    answer = await loop.run_in_executor(None, lambda: handle_query(msg))
    history.append({"role": "assistant", "content": answer})
    return history, history

def launch():
    with gr.Blocks(title="Universal AI Agent • core2") as demo:
        chatbot = gr.Chatbot(type="messages", height=420)
        msg = gr.Textbox(lines=2, placeholder="Zadej dotaz… (Ctrl+Enter)")
        msg.submit(chat_fn, [msg, chatbot], [chatbot, chatbot]).then(lambda: "", None, msg)
        demo.queue().launch()

if __name__ == "__main__":
    launch()
