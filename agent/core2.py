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
from typing import TypedDict, Any, List

from dotenv import load_dotenv
import openai

# LangChain & LangGraph
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.chat_message_histories import FileChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import Document
from langchain.agents import create_tool_calling_agent, AgentExecutor

from langgraph.graph import StateGraph, END

# Vector store
from langchain_chroma import Chroma
from chromadb.config import Settings

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
if hasattr(openai, "telemetry") and hasattr(openai.telemetry, "TelemetryClient"):
    openai.telemetry.TelemetryClient.capture = staticmethod(lambda *a, **k: None)

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
)

# 2) Persistent chat log (restored across restarts → feeds the window buffer)
_persistent_history_file = os.getenv(
    "PERSISTENT_HISTORY_FILE",
    "persistent_chat_history.json"
    )
_short_term_memory.chat_memory = FileChatMessageHistory(file_path=_persistent_history_file)

# ---------------------------------------------------------------------------
# Prompt, LLM a agent stejně jako dříve
# ---------------------------------------------------------------------------
system_prompt = """
You are **Productoo’s senior AI assistant** specialised in manufacturing software P4.
Follow the guidelines from the original core while using available tools.
Always finish with: *Ještě něco, s čím mohu pomoci?*
""".strip()

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{query}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

_llm = ChatOpenAI(model="gpt-4o", temperature=0.3)

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
def call_agent(state: AgentState) -> AgentState:
    """Spustí nástroj‑volajícího agenta s krátkodobou pamětí + kontextem."""
    q = state["query"]
    ctx = state.get("retrieved_context", "")
    if ctx:
        q = f"{q}\n\nRelevant past context:\n{ctx}"
    result = _agent_executor.invoke({"query": q})
    state["answer"] = result["output"]
    state["intermediate_steps"] = result["intermediate_steps"]
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

# ---------------------------------------------------------------------------
# Graf a workflow
# ---------------------------------------------------------------------------
graph = StateGraph(state_schema=AgentState)
graph.add_node("recall", recall)
graph.add_node("agent", call_agent)
graph.add_node("learn", learn)

graph.set_entry_point("recall")
graph.add_edge("recall", "agent")
graph.add_edge("agent", "learn")
graph.add_edge("learn", END)

# Kompilovaný workflow (lazy‑initialised, aby import nezdržoval start)
workflow = graph.compile()

# ---------------------------------------------------------------------------
# Veřejné API – zůstává stejné jako v core.py
# ---------------------------------------------------------------------------
def handle_query(query: str) -> str:
    """Jediný veřejný vstup: zpracuje dotaz a vrátí odpověď."""
    init_state: AgentState = {"query": query, "answer": "", "intermediate_steps": [], "retrieved_context": ""}
    final_state = workflow.invoke(init_state)
    return final_state["answer"]  # + krátké 'něco dalšího' je už v promptu

# Convenience alias pro případné externí diagnostiky
agent_workflow = workflow

__all__ = ["handle_query", "agent_workflow"]
