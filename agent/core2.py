# agent/core2.py
"""
LangGraph‑based AI core for Productoo P4 agent.
Keeps the same public API as core.py (handle_query) so both can coexist.

Key features
------------
• Multi‑tier memory (short‑term window, persistent file log, long‑term vector store)
• Continuous learning (= automatic RAG enrichment after each exchange)
• Full tool‑calling capability identical to core.py
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import TypedDict

from dotenv import load_dotenv
import openai

# LangChain & LangGraph
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.chat_message_histories import FileChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import PydanticOutputParser
from langchain.schema import Document
from langchain.agents import create_tool_calling_agent, AgentExecutor

from langgraph.graph import StateGraph, END

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")   # rozpoznáno 0.4.2+

try:
    # 1) starší cesty
    from chromadb.telemetry import TelemetryClient
    TelemetryClient.capture = lambda self, *a, **k: None
except ImportError:
    pass

# Vector store
from langchain_chroma import Chroma
from chromadb.config import Settings

# Pydantic model
from pydantic import BaseModel
import json, asyncio

# Project‑specific tools (identické s core.py)
from tools import (
    search_tool,
    wiki_tool,
    save_tool,
    rag_tool,
    tavily_tool,
    jira_ideas,
    jira_issue_detail,
    jira_duplicates,
    jira_content_tools,
    process_input_tool,
    jira_update_description,
    jira_child_issues,
    jira_issue_links,
)

# ---------------------------------------------------------------------------
# Environment & telemetry (be consistent with core.py)
# ---------------------------------------------------------------------------
load_dotenv()

os.environ.setdefault("OPENAI_TELEMETRY", "0")
os.environ.setdefault("CHROMA_TELEMETRY",  "0")
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
if hasattr(openai, "telemetry") and hasattr(openai.telemetry, "TelemetryClient"):
    try:
        import chromadb.telemetry as _ct
        _ct.TelemetryClient.capture = staticmethod(lambda *a, **k: None)
    except ImportError:
        pass

# ---------------------------------------------------------------------------
# Vectorstore (shared long‑term memory)
# ---------------------------------------------------------------------------
CHROMA_DIR = os.getenv("CHROMA_DIR_V2", "rag_chroma_db")
_embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
_vectorstore = Chroma(
    persist_directory=CHROMA_DIR,
    embedding_function=_embeddings,
    client_settings=Settings(anonymized_telemetry=False),
)

# ---------------------------------------------------------------------------
# Multi‑tier memory configuration
# ---------------------------------------------------------------------------
SHORT_WINDOW = int(os.getenv("AGENT_SHORT_WINDOW", 10))

# 1) Short‑term window (keeps last N turns in RAM)
_short_term_memory = ConversationBufferWindowMemory(
    k=SHORT_WINDOW,
    memory_key="chat_history",
    return_messages=True,
    output_key="output"
)

# 2) Persistent chat log (restored across restarts → feeds the window buffer)
_persistent_history_file = os.getenv(
    "PERSISTENT_HISTORY_FILE",
    "persistent_chat_history.json"
    )
_short_term_memory.chat_memory = FileChatMessageHistory(file_path=_persistent_history_file)

# --- Node 4: Response (structured response in JSON) ------------------------------------
class ResearchResponse(BaseModel):
    answer: str
    intermediate_steps: list[str] | None = None

parser = PydanticOutputParser(pydantic_object=ResearchResponse)

# ---------------------------------------------------------------------------
# Prompt, LLM a agent stejně jako dříve
# ---------------------------------------------------------------------------
system_prompt = """
You are **Productoo’s senior AI assistant** specialised in manufacturing software P4.
Your mission:
  1. Analyse market trends, the current P4 roadmap and system state.
  2. Propose concrete next steps maximising customer value and minimising tech‑debt.
  3. Seamlessly use the available tools (`searchWeb`, `rag_retriever`,
     `tavily_search`, `jira_ideas_retriever`, `save_text_to_file`) when relevant
     and clearly cite your sources.

Guidelines:
- Think step‑by‑step, reason explicitly yet concisely.
- Ask clarifying questions whenever the user’s request is ambiguous.
- Make answers information‑dense—avoid filler.
- Prefer actionable recommendations backed by evidence and quantified impact.
- **After every answer, end with exactly:** _"Ještě něco, s čím mohu pomoci?"_
  (keeps the dialogue open).

Return your answer strictly as valid JSON conforming to the schema below.
{format_instructions}
""".strip()

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{query}"),
        MessagesPlaceholder("agent_scratchpad"),
    ]
).partial(format_instructions=parser.get_format_instructions())

_llm = ChatOpenAI(model="gpt-4o", temperature=0.3)

def _safe(t):
    t.handle_tool_error = True
    return t

_TOOLS = [
    search_tool,
    wiki_tool,
    save_tool,
    rag_tool,
    tavily_tool,
    jira_ideas,
    jira_issue_detail,
    jira_duplicates,
    *jira_content_tools,
    process_input_tool,
    jira_update_description,
    jira_child_issues,
    jira_issue_links,
]
_TOOLS = list(map(_safe, _TOOLS))

_agent = create_tool_calling_agent(llm=_llm, prompt=prompt, tools=_TOOLS)
_agent_executor = AgentExecutor(
    agent=_agent,
    tools=_TOOLS,
    memory=_short_term_memory,
    verbose=True,
    return_intermediate_steps=True,
)

# ---------------------------------------------------------------------------
# LangGraph: state model & nodes
# ---------------------------------------------------------------------------
class AgentState(TypedDict):
    """Datový balíček přenášený mezi uzly grafu."""
    query: str
    answer: str
    intermediate_steps: list
    retrieved_context: str

# --- Node 1: Retrieve relevant long‑term memory --------------------------------
def recall(state: AgentState) -> AgentState:
    """Vyhledá vektorově relevantní minulou konverzaci / znalosti."""
    docs = _vectorstore.similarity_search(state["query"], k=4)
    state["retrieved_context"] = "\n".join(d.page_content for d in docs)
    return state

# --- Node 2: Call the langchain agent -----------------------------------------
def act(state: AgentState) -> AgentState:
    """Spustí nástroj‑volajícího agenta s krátkodobou pamětí + kontextem."""
    q = state["query"]
    if ctx := state.get("retrieved_context"):
        q += f"\n\nRelevant past context:\n{ctx}"
    try:
        result = _agent_executor.invoke({"query": q})
        raw = result["output"].strip()
        if raw.startswith("```"):
            raw = raw.strip("`")                 # ořež zahajovací + ukončovací ```
            # po ořezu může zůstat "json\n{", smažeme jazykový prefix
            if raw.lstrip().lower().startswith("json"):
                raw = raw.lstrip()[4:].lstrip()

        # b) prefix bez fence:  json\n{ ...
        if raw.lower().startswith("json"):
            raw = raw[4:].lstrip()

        try: 
            parsed = ResearchResponse.parse_raw(raw) 
            state["answer"] = parsed.answer
            state["intermediate_steps"] = parsed.intermediate_steps
        except Exception as e: 
            state["answer"] = f"⚠️ LLM nevrátil validní JSON: {e}\\n{result['output']}"


        state["intermediate_steps"] = result.get("intermediate_steps", [])
    except Exception as e:   # ← pojistka, kdyby přece jen něco propadlo
        err = f"⚠️ Nástroj selhal: {e}"
        state["answer"] = err
        state["intermediate_steps"] = [err]
    return state

# --- Node 3: Learn (append to vectorstore) ------------------------------------
def learn(state: AgentState) -> AgentState:
    """Zapíše dialog do dlouhodobé paměti Po každém běhu."""
    ts = datetime.utcnow().isoformat()
    _vectorstore.add_documents(
        [
            Document(page_content=f"USER: {state['query']}", metadata={"role": "user", "ts": ts}),
            Document(page_content=f"ASSISTANT: {state['answer']}", metadata={"role": "assistant", "ts": ts}),
        ]
    )
    return state

# --- Public API -------------------------------------------------------------
def _final_to_json(final_state: AgentState) -> str:
    steps = final_state.get("intermediate_steps") or []
    rr = ResearchResponse(
        answer=final_state.get("answer", ""),
        intermediate_steps=[str(s) for s in steps] if steps else None,
    )
    return rr.json()

# ---------------------------------------------------------------------------
# Graf a workflow
# ---------------------------------------------------------------------------
graph = StateGraph(state_schema=AgentState)
graph.add_node("recall", recall)
graph.add_node("act", act)
graph.add_node("learn", learn)

graph.set_entry_point("recall")
graph.add_edge("recall", "act")
graph.add_edge("act", "learn")
graph.add_edge("learn", END)

# Kompilovaný workflow (lazy‑initialised, aby import nezdržoval start)
workflow = graph.compile()

# ---------------------------------------------------------------------------
# Veřejné API – zůstává stejné jako v core.py
# ---------------------------------------------------------------------------
def handle_query(query: str) -> str:
    """Jediný veřejný vstup: zpracuje dotaz a vrátí odpověď."""
    final = workflow.invoke({"query": query, "answer": "", "intermediate_steps": [], "retrieved_context": ""})
    return _final_to_json(final)

# -------- STREAMING (yielduje JSON lines) ------------------------------------
async def handle_query_stream(query: str):
    """Streamuje výstup *celého* grafu na jeden běh (žádné zdvojení tokenů)."""

    final_state = None

    async for ev in workflow.astream_events(
        {"query": query, "answer": "", "intermediate_steps": [], "retrieved_context": ""},
        version="v1",
    ):
        event_type = ev.get("event") or ev.get("event_name")
        node_name  = ev.get("name")  or ev.get("node_name")
        # detekce tokenů LLM
        # -- průběžné streamování tokenů do UI
        if event_type == "on_llm_new_token":
            yield ev["data"]["token"]

        # -- zachycení finálního stavu po uzlu „learn“
        if event_type == "on_node_end" and node_name == "learn":
            ds = ev.get("data", {})
            final_state = ds.get("output") or ds.get("state")

    if final_state is None:
        final_state = workflow.invoke(
            {
                "query": query,
                "answer": "",
                "intermediate_steps": [],
                "retrieved_context": "",
            }
        )

    yield "\n" + _final_to_json(final_state)

# Convenience alias pro případné externí diagnostiky
agent_workflow = workflow

__all__ = ["handle_query", "handle_query_stream", "agent_workflow", "ResearchResponse"]

