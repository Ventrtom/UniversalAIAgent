"""Wrapper functions around the agent for the web UI."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

import gradio as gr

from agent import handle_query_stream, ResearchResponse
from .config import MAX_HISTORY_LENGTH

logger = logging.getLogger(__name__)


def pretty_format_response(raw: str) -> str:
    """Format raw agent response for nicer display."""
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
    """Trim conversation history to ``max_length`` messages."""
    if len(history) <= max_length:
        return history
    system_messages = [msg for msg in history if msg.get("role") == "system"]
    recent_messages = history[-(max_length - len(system_messages)) :]
    return system_messages + recent_messages


def format_intermediate_steps(steps: List[Any]) -> str:
    """Human readable formatting of agent steps."""
    if not steps:
        return ""
    formatted_steps = []
    for i, step in enumerate(steps, 1):
        if hasattr(step, "tool") and hasattr(step, "tool_input"):
            tool_name = step.tool
            tool_input = step.tool_input
            if isinstance(tool_input, dict):
                input_str = ", ".join(f"{k}: {v}" for k, v in tool_input.items())
            else:
                input_str = str(tool_input)
            formatted_steps.append(f"**{i}.** 🔧 **{tool_name}**({input_str})")
        elif isinstance(step, tuple) and len(step) == 2:
            action, result = step
            if hasattr(action, "tool"):
                tool_name = action.tool
                tool_input = action.tool_input
                if isinstance(tool_input, dict):
                    input_str = ", ".join(f"{k}: {v}" for k, v in tool_input.items())
                else:
                    input_str = str(tool_input)
                formatted_steps.append(f"**{i}.** 🔧 **{tool_name}**({input_str})")
                if result and isinstance(result, str) and len(result) < 200:
                    formatted_steps.append(f"   📋 Result: {result}")
            else:
                formatted_steps.append(f"**{i}.** {str(step)}")
        else:
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
    viz_html: gr.HTML | None = None,
) -> AsyncGenerator[Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[str]], None]:
    """Handle chat interaction with the agent and yield streaming updates."""
    if not msg.strip():
        yield history or [], history or [], steps or [], gr.update()
        yield history or [], history or [], steps or [], gr.update(), gr.update()
        return
    history = history or []
    steps = steps or []
    steps.clear()
    viz_chunks: List[str] = []
    if live_log_markdown is not None:
        live_log_markdown.value = ""
        live_log_markdown.visible = True
    history.append({"role": "user", "content": msg})
    bot = {"role": "assistant", "content": ""}
    history.append(bot)
    try:
        async for tok in handle_query_stream(msg):
            if tok.startswith("§NODE§"):
                # runtime viz: node lifecycle
                try:
                    evt_json = tok[7:]
                    viz_chunks.append(f"<script>window.updateGraph({evt_json});</script>")
                except Exception:
                    pass
                # nezasahujeme do textu v chatu – jen viz panel
                current_history = limit_history(history)
                yield (current_history, current_history, steps,
                    gr.update(value="\n".join(f"**{i}.** {s}" for i, s in enumerate(steps, 1)), visible=True),
                    gr.update(value=viz_chunks[-1]))
                continue
            if tok.startswith("§STEP§"):
                steps.append(tok[7:])
                if live_log_markdown is not None:
                    live_log_markdown.value = "\n".join(
                        f"**{i}.** {s}" for i, s in enumerate(steps, 1)
                    )
                tool_name = None
                try:
                    _payload = tok[7:]
                    _w = _payload.split("🛠️", 1)[-1].strip()
                    tool_name = _w.split("(", 1)[0].strip()
                except Exception:
                    tool_name = None
                if tool_name:
                    viz_chunks.append(f"<script>window.updateGraph({{'type':'tool','name':{json.dumps(tool_name)}}});</script>")
                await asyncio.sleep(0)
                current_history = limit_history(history)
                yield (current_history, current_history, steps,
                    gr.update(value="\n".join(f"**{i}.** {s}" for i, s in enumerate(steps, 1)), visible=True),
                    gr.update(value=viz_chunks[-1]))
                continue
            bot["content"] += tok
            current_history = limit_history(history)
            yield current_history, current_history, steps, gr.update(), gr.update()
        try:
            raw_json = bot["content"].splitlines()[-1]
            parsed = ResearchResponse.parse_raw(raw_json)
            bot["content"] = parsed.answer
            if reveal and parsed.intermediate_steps:
                formatted_steps = format_intermediate_steps(parsed.intermediate_steps)
                bot["content"] += f"\n\n**🔍 Intermediate steps:**\n{formatted_steps}"
        except Exception as exc:  # noqa: BLE001
            logger.error("Error parsing response: %s", exc)
    except Exception as exc:  # noqa: BLE001
        logger.error("Error in chat function: %s", exc)
        bot["content"] = f"Omlouváme se, došlo k chybě: {exc}"
    final_history = limit_history(history)
    if not reveal and live_log_markdown is not None:
        live_log_markdown.value = ""
        live_log_markdown.visible = False
        steps.clear()
    yield (final_history, final_history, steps,
        gr.update(value="", visible=False),
        gr.update()) 


def clear_chat() -> Tuple[List, str]:
    """Utility used by the 'clear' button."""
    return [], ""

# --- graf init po načtení stránky ------------------------------------------
def graph_init() -> str:
    """
    Vrátí malý <script>, který předá do vizualizace seznam dostupných tools.
    """
    # import uvnitř funkce kvůli rychlejšímu startu a aby nebyla pevná vazba při testech
    tools = []
    try:
        from agent import get_tool_names  # preferovaný export z balíčku
        tools = get_tool_names()
    except Exception:
        try:
            from agent.core import get_tool_names  # fallback pro případ, že není re-export
            tools = get_tool_names()
        except Exception:
            tools = []
    # voláme globální window.updateGraph z druhého HTML komponentu
    return f"<script>window.updateGraph({{'type':'init','tools':{json.dumps(tools)}}});</script>"

def graph_schema() -> dict:
    """
    Vrátí aktuální topologii grafu (nodes, edges, tools) jako Python dict
    – bez jakéhokoli JS/HTML. Použije se jako vstup pro JS renderer v UI.
    """
    try:
        from agent import get_graph_schema  # preferovaný export z balíčku
        return get_graph_schema()
    except Exception:
        try:
            from agent.core import get_graph_schema  # fallback, pokud není re-export
            return get_graph_schema()
        except Exception:
            return {"nodes": [], "edges": [], "tools": []}

def graph_snapshot() -> str:
    """
    Vygeneruje samostatný HTML blok s Cytoscape, naplněný aktuální topologií
    (nodes/edges/tools) přímo z agentu.
    """
    try:
        from agent import get_graph_schema
        schema = get_graph_schema()
    except Exception:
        schema = {"nodes": [], "edges": [], "tools": []}

    # Postavíme Cytoscape z aktuálních dat; bez runtime efektů
    return f"""
    <div id='cy' style="width:100%;height:70vh;border:1px solid #ddd;border-radius:8px;background:#fff"></div>
    <script src="https://unpkg.com/cytoscape@3/dist/cytoscape.min.js"></script>
    <script>
    (function(){{
      const data = {json.dumps(schema)};
      const elements = [
        ...data.nodes.map(n => ({{ data: {{ id: n.id, label: n.label }} }})),
        ...data.edges.map(e => ({{ data: {{ source: e.source, target: e.target }} }})),
      ];
      const cy = cytoscape({{
        container: document.getElementById('cy'),
        elements,
        layout: {{ name: 'breadthfirst', directed: true, padding: 10 }},
        style: [
          {{ selector:'node', style:{{ 'label':'data(label)','text-valign':'center','background-color':'#eee','border-width':1,'border-color':'#888' }} }},
          {{ selector:'edge', style:{{ 'curve-style':'bezier','target-arrow-shape':'triangle','width':1,'line-color':'#bbb','target-arrow-color':'#bbb' }} }},
        ]
      }});
      // necháváme window.updateGraph volné pro budoucí zvýrazňování, ale teď je to čistý snapshot
      window.updateGraph = window.updateGraph || function(_){{}};
    }})();
    </script>
    """

__all__ = [
    "pretty_format_response",
    "limit_history",
    "format_intermediate_steps",
    "chat_fn",
    "clear_chat",
]
