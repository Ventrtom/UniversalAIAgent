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
) -> AsyncGenerator[Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[str]], None]:
    """Handle chat interaction with the agent and yield streaming updates."""
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
            current_history = limit_history(history)
            yield current_history, current_history, steps
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
    yield final_history, final_history, steps


def clear_chat() -> Tuple[List, str]:
    """Utility used by the 'clear' button."""
    return [], ""


__all__ = [
    "pretty_format_response",
    "limit_history",
    "format_intermediate_steps",
    "chat_fn",
    "clear_chat",
]
