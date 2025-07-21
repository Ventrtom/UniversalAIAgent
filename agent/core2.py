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
import threading
import math

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
from collections import deque
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
    kb_loader_tool,
    clear_rag_memory_tool,
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
CHROMA_DIR = os.getenv("CHROMA_DIR_V2", "data")
_embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Two separate collections: external knowledge base and chat memory
_kb_store = Chroma(
    collection_name="kb_docs",
    persist_directory=CHROMA_DIR,
    embedding_function=_embeddings,
    client_settings=Settings(anonymized_telemetry=False),
)

# Ensure all knowledge documents include metadata source="doc"
_kb_add_documents_orig = _kb_store.add_documents

def _kb_add_documents_with_source(docs, **kwargs):
    wrapped = []
    for d in docs:
        meta = dict(d.metadata or {})
        meta.setdefault("source", "doc")
        wrapped.append(Document(page_content=d.page_content, metadata=meta))
    return _kb_add_documents_orig(wrapped, **kwargs)

_kb_store.add_documents = _kb_add_documents_with_source



class ChatMemoryStore(Chroma):
    """Lightweight wrapper exposing a stable count API."""

    def get_total_records(self) -> int:
        """Return number of records in this collection."""
        try:
            return self._collection.count()
        except AttributeError:
            try:
                return self._collection._collection.count()  # type: ignore[attr-defined]
            except Exception:
                return 0


_chat_store = ChatMemoryStore(
    collection_name="chat_memory",
    persist_directory=CHROMA_DIR,
    embedding_function=_embeddings,
    client_settings=Settings(anonymized_telemetry=False),
)

# Backwards compatibility – old name points to the KB collection
_vectorstore = _kb_store

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

QUERY_TIME_THRESHOLD = int(os.getenv("QUERY_TIME_THRESHOLD", 120))
MAX_CHAT_RECORDS = int(os.getenv("MAX_CHAT_RECORDS", 10_000))
DUPLICATE_THRESHOLD = float(os.getenv("DUPLICATE_THRESHOLD", 0.95))

RERANK_ALPHA = float(os.getenv("RERANK_ALPHA", 0.7))
RERANK_BETA = float(os.getenv("RERANK_BETA", 0.25))
RERANK_GAMMA = float(os.getenv("RERANK_GAMMA", 0.05))
TYPE_BOOST_DOC = float(os.getenv("TYPE_BOOST_DOC", 0.1))
TYPE_BOOST_CHAT = float(os.getenv("TYPE_BOOST_CHAT", 0.0))
RECENCY_TAU = 30 * 24 * 3600

# Time of the last user query (used for heuristic routing)
_last_user_ts = datetime.utcnow().timestamp()
_ts_lock = threading.Lock()

# Recent user embeddings for duplicate detection
_embedding_cache: deque[list[float]] = deque(maxlen=500)
_cache_ready = False


def _update_last_user_ts() -> None:
    with _ts_lock:
        global _last_user_ts
        _last_user_ts = datetime.utcnow().timestamp()


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


def _ensure_cache() -> None:
    """Populate embedding cache with the latest user embeddings."""
    global _cache_ready
    if _cache_ready:
        return
    try:
        data = _chat_store.get(include=["embeddings", "metadatas"])
        pairs = [
            (e, m.get("ts"))
            for e, m in zip(data.get("embeddings", []), data.get("metadatas", []))
            if (m or {}).get("role") == "user"
        ]
        pairs.sort(key=lambda p: p[1] or "")
        for emb, _ in pairs[-500:]:
            _embedding_cache.append(emb)
    except Exception:
        pass
    _cache_ready = True

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
    kb_loader_tool,
    clear_rag_memory_tool,
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
def _is_information_query(query: str) -> bool:
    """Heuristic check whether the query is a standalone information request."""
    q = query.lower().strip()
    info_words = ("what", "why", "how", "where", "when", "who")
    return any(q.startswith(w) for w in info_words) or "?" in q


def recall(state: AgentState) -> AgentState:
    """Vyhledá vektorově relevantní kontext z odpovídající kolekce."""
    query = state["query"]
    with _ts_lock:
        since_last = datetime.utcnow().timestamp() - _last_user_ts

    use_kb = _is_information_query(query) or since_last > QUERY_TIME_THRESHOLD
    store = _kb_store if use_kb else _chat_store

    results = store.similarity_search_with_relevance_scores(query, k=4)
    reranked: list[tuple[float, Document]] = []
    now_ts = datetime.utcnow().timestamp()
    for doc, cos in results:
        ts_str = doc.metadata.get("ts")
        delta = float("inf")
        if ts_str:
            try:
                delta = now_ts - datetime.fromisoformat(ts_str).timestamp()
            except Exception:
                pass
        recency = math.exp(-delta / RECENCY_TAU) if delta != float("inf") else 0.0
        type_boost = TYPE_BOOST_DOC if doc.metadata.get("source") == "doc" else TYPE_BOOST_CHAT
        score = (
            RERANK_ALPHA * cos
            + RERANK_BETA * recency
            + RERANK_GAMMA * type_boost
        )
        reranked.append((score, doc))

    reranked.sort(key=lambda x: x[0], reverse=True)
    docs = [d for _, d in reranked]
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
    """Zapíše dialog do dlouhodobé paměti po každém běhu."""
    ts = datetime.utcnow().isoformat()
    docs = [
        Document(
            page_content=f"USER: {state['query']}",
            metadata={"role": "user", "ts": ts, "source": "chat"},
        ),
        Document(
            page_content=f"ASSISTANT: {state['answer']}",
            metadata={"role": "assistant", "ts": ts, "source": "chat"},
        ),
    ]

    try:
        _ensure_cache()
        new_embs = _embeddings.embed_documents([d.page_content for d in docs])
        user_emb = new_embs[0]
        if any(_cosine(user_emb, ex) > DUPLICATE_THRESHOLD for ex in _embedding_cache):
            docs = [docs[1]]  # store assistant answer only
        else:
            _embedding_cache.append(user_emb)
    except Exception:
        pass

    _chat_store.add_documents(docs)

    try:
        total = _chat_store.get_total_records()
        if total > MAX_CHAT_RECORDS:
            over = total - MAX_CHAT_RECORDS
            res = _chat_store.get(include=["metadatas"], limit=None)
            records = list(zip(res["ids"], res["metadatas"]))
            records.sort(key=lambda r: r[1].get("ts", ""))
            ids_to_del = [rid for rid, _ in records[:over]]
            if ids_to_del:
                _chat_store.delete(ids_to_del)
                for _ in range(over):
                    if _embedding_cache:
                        _embedding_cache.popleft()
    except Exception:
        pass
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
    result = _final_to_json(final)
    _update_last_user_ts()
    return result

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
    _update_last_user_ts()

# Convenience alias pro případné externí diagnostiky
agent_workflow = workflow

__all__ = ["handle_query", "handle_query_stream", "agent_workflow", "ResearchResponse"]

