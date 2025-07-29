# agent/core.py
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
import time
import functools
import inspect
import json
from pathlib import Path
from datetime import datetime
from typing import TypedDict, Any
from dataclasses import dataclass
import threading
import math
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv
import openai

# LangChain & LangGraph
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.chat_message_histories import FileChatMessageHistory
from utils.rotating_history import RotatingFileChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import PydanticOutputParser
from langchain.schema import Document
from langchain.agents import create_tool_calling_agent, AgentExecutor

from langgraph.graph import StateGraph, END

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")  # rozpoznáno 0.4.2+

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
import pandas as pd
from apscheduler.schedulers.background import BackgroundScheduler
import atexit
from utils.shutdown import register_task

# ---------------------------------------------------------------------------
# Utility: timing decorator for measuring production phase durations
# ---------------------------------------------------------------------------


def timed(name):
    """Measure and log execution time of the wrapped function."""

    def deco(fn):
        if inspect.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def awrap(*a, **k):
                t0 = time.perf_counter()
                try:
                    return await fn(*a, **k)
                finally:
                    print(f"[{name}] {(time.perf_counter() - t0):.2f}s")

            return awrap
        else:

            @functools.wraps(fn)
            def wrap(*a, **k):
                t0 = time.perf_counter()
                try:
                    return fn(*a, **k)
                finally:
                    print(f"[{name}] {(time.perf_counter() - t0):.2f}s")

            return wrap

    return deco


# Project‑specific tools (identické s core.py)
from tools import (
    search_tool,
    wiki_tool,
    save_tool,
    list_output_files_tool,
    read_output_file_tool,
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
    self_inspection_tool,
)

# ---------------------------------------------------------------------------
# Environment & telemetry (be consistent with core.py)
# ---------------------------------------------------------------------------
load_dotenv()

os.environ.setdefault("OPENAI_TELEMETRY", "0")
os.environ.setdefault("CHROMA_TELEMETRY", "0")
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
MAX_TOOL_ITERATIONS = int(os.getenv("MAX_TOOL_ITERATIONS", 6))

# 1) Short‑term window (keeps last N turns in RAM)
_short_term_memory = ConversationBufferWindowMemory(
    k=SHORT_WINDOW,
    memory_key="chat_history",
    return_messages=True,
    output_key="output",
    input_key="query",
)

# 2) Persistent chat log (restored across restarts → feeds the window buffer)
_persistent_history_file = Path(
    os.getenv("PERSISTENT_HISTORY_FILE", "data/persistent_chat_history.json")
)
_persistent_history_file.parent.mkdir(parents=True, exist_ok=True)


_short_term_memory.chat_memory = RotatingFileChatMessageHistory(
    file_path=_persistent_history_file
)

QUERY_TIME_THRESHOLD = int(os.getenv("QUERY_TIME_THRESHOLD", 120))
MAX_CHAT_RECORDS = int(os.getenv("MAX_CHAT_RECORDS", 10_000))
DUPLICATE_THRESHOLD = float(os.getenv("DUPLICATE_THRESHOLD", 0.90))

# Weights for duplicate detection (embedding vs. text similarity)
EMBED_WEIGHT = 0.7
TEXT_WEIGHT = 0.3

RERANK_ALPHA = float(os.getenv("RERANK_ALPHA", 0.7))
RERANK_BETA = float(os.getenv("RERANK_BETA", 0.25))
RERANK_GAMMA = float(os.getenv("RERANK_GAMMA", 0.05))
TYPE_BOOST_DOC = float(os.getenv("TYPE_BOOST_DOC", 0.1))
TYPE_BOOST_CHAT = float(os.getenv("TYPE_BOOST_CHAT", 0.0))
RECENCY_TAU = 30 * 24 * 3600

# Time of the last user query (used for heuristic routing)
_last_user_ts = datetime.utcnow().timestamp()
_ts_lock = threading.Lock()
_bg_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="bg")

# Recent user embeddings for duplicate detection
# Stores tuples of (embedding, original text)
_embedding_cache: deque[tuple[list[float], str]] = deque(maxlen=500)
_cache_ready = False
_cache_lock = threading.Lock()
_summary_lock = asyncio.Lock()


def _update_last_user_ts() -> None:
    with _ts_lock:
        global _last_user_ts
        _last_user_ts = datetime.utcnow().timestamp()


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


def _shingles(text: str, n: int = 3) -> set[str]:
    text = text.lower()
    return {text[i : i + n] for i in range(len(text) - n + 1)} if text else set()


def _jaccard(a: str, b: str) -> float:
    sa = _shingles(a)
    sb = _shingles(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _combined_similarity(u_emb: list[float], u_text: str, v_emb: list[float], v_text: str) -> float:
    emb = _cosine(u_emb, v_emb)
    txt = _jaccard(u_text, v_text)
    return EMBED_WEIGHT * emb + TEXT_WEIGHT * txt


def _rebuild_embedding_cache() -> None:
    """Recreate duplicate‑detection cache from the vector store."""
    with _cache_lock:
        _embedding_cache.clear()
        try:
            data = _chat_store.get(include=["embeddings", "metadatas", "documents"], limit=2000)
            pairs = [
                (e, d, m.get("ts"))
                for e, d, m in zip(
                    data.get("embeddings", []),
                    data.get("documents", []),
                    data.get("metadatas", []),
                )
                if (m or {}).get("role") == "user"
            ]
            pairs.sort(key=lambda p: p[2] or "")
            for emb, doc, _ in pairs[-500:]:
                _embedding_cache.append((emb, doc))
        except Exception:
            pass
        global _cache_ready
        _cache_ready = True


def _ensure_cache() -> None:
    """Populate embedding cache with the latest user embeddings."""
    global _cache_ready
    if _cache_ready:
        return
    _rebuild_embedding_cache()


import threading

threading.Thread(target=_ensure_cache, daemon=True).start()

# ---------------------------------------------------------------------------
# Periodic snapshotting of chat memory
# ---------------------------------------------------------------------------
SNAPSHOT_PATH = os.getenv("CHAT_SNAPSHOT_FILE", "data/chat_snapshot.jsonl")
SNAPSHOT_INTERVAL = int(os.getenv("CHAT_SNAPSHOT_INTERVAL", 600))


def _snapshot_chat(path: str = SNAPSHOT_PATH, mask_pii: bool = False) -> None:
    """Append new chat records to a JSONL snapshot."""
    try:
        last_ts = None
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    pass
                if line:
                    try:
                        last_ts = json.loads(line).get("ts")
                    except Exception:
                        last_ts = None

        data = _chat_store.get(include=["documents", "metadatas"], limit=None)
        rows = []
        for i, doc, meta in zip(
            data.get("ids", []),
            data.get("documents", []),
            data.get("metadatas", []),
        ):
            ts = (meta or {}).get("ts")
            if last_ts and ts and ts <= last_ts:
                continue
            if mask_pii:
                from utils.redaction import redact_pii

                doc = redact_pii(doc)
            rows.append({"id": i, "document": doc, **(meta or {})})

        if rows:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "a", encoding="utf-8") as f:
                for row in rows:
                    json.dump(row, f, ensure_ascii=False)
                    f.write("\n")
    except Exception:
        pass


_scheduler: BackgroundScheduler | None = None


def start_background_jobs(config: dict | None = None) -> BackgroundScheduler:
    """Start background tasks like periodic snapshotting."""
    global _scheduler
    if _scheduler is not None:
        return _scheduler

    cfg = config or {}
    interval = int(cfg.get("snapshot_interval", SNAPSHOT_INTERVAL))
    path = cfg.get("snapshot_path", SNAPSHOT_PATH)
    mask = bool(cfg.get("mask_pii", False))

    _scheduler = BackgroundScheduler(daemon=True)
    _scheduler.add_job(_snapshot_chat, "interval", seconds=interval, args=(path, mask))
    _scheduler.start()
    atexit.register(lambda: _scheduler.shutdown(wait=False))
    atexit.register(lambda: _snapshot_chat(path, mask))
    return _scheduler


def _ev_attr(ev, attr: str, default=None):
    """Vrať položku z dictu, nebo atribut objektu, případně default."""
    if isinstance(ev, dict):
        return ev.get(attr, default)
    return getattr(ev, attr, default)


# --- Node 4: Response (structured response in JSON) ------------------------------------
class ResearchResponse(BaseModel):
    answer: str
    intermediate_steps: list[str] | None = None


parser = PydanticOutputParser(pydantic_object=ResearchResponse)

# --- helper: extract first top-level JSON object from a string ---
def _extract_json_object(s: str) -> str | None:
    """Return the first top-level JSON object found in s, or None."""
    start = s.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    for i, ch in enumerate(s[start:], start):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
        else:
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return s[start:i+1]
    return None

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

Return your answer strictly as valid JSON conforming to the schema below. Do not use code fences and do not prefix with a language tag.
{format_instructions}
""".strip()

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("assistant", "{retrieved_context}"),
        ("human", "{query}"),
        MessagesPlaceholder("agent_scratchpad"),
    ]
).partial(format_instructions=parser.get_format_instructions())


_llm = ChatOpenAI(model="gpt-4o", temperature=0.3)


@dataclass
class ToolResult:
    """Uniform wrapper for tool outputs."""

    content: Any
    content_type: str  # "text" or "json"
    tool_name: str

    def __str__(self) -> str:  # noqa: D401 – simple wrapper
        """Return a string representation for the LLM context."""
        if self.content_type == "json":
            try:
                return json.dumps(self.content, ensure_ascii=False)
            except Exception:
                return str(self.content)
        return str(self.content)

def _coerce_tool_output_to_str(x):
    """
    Zajistí, že výstup nástroje bude string:
    - str vrací beze změny
    - dict/list serializuje do JSON
    - cokoli jiného převede přes str(x)
    """
    if isinstance(x, str):
        return x
    try:
        return json.dumps(x, ensure_ascii=False)
    except Exception:
        return str(x)

def _wrap_output(tool_name: str, out: Any) -> ToolResult:
    """Convert raw tool output into :class:`ToolResult`."""
    if isinstance(out, (dict, list)):
        return ToolResult(content=out, content_type="json", tool_name=tool_name)
    return ToolResult(
        content=_coerce_tool_output_to_str(out),
        content_type="text",
        tool_name=tool_name,
    )


def _safe(t):
    """Wrap tool to return :class:`ToolResult` and enforce error handling."""
    t.handle_tool_error = True

    func = getattr(t, "func", None)
    coro = getattr(t, "coroutine", None)

    if func is not None:
        if inspect.iscoroutinefunction(func):
            async def _wrapped_async(*a, **k):
                out = await func(*a, **k)
                return _wrap_output(t.name, out)
            t.coroutine = _wrapped_async
        else:
            def _wrapped_sync(*a, **k):
                out = func(*a, **k)
                return _wrap_output(t.name, out)
            t.func = _wrapped_sync
    elif coro is not None:
        async def _wrapped_coro(*a, **k):
            out = await coro(*a, **k)
            return _wrap_output(t.name, out)
        t.coroutine = _wrapped_coro
    return t


_TOOLS = [
    search_tool,
    wiki_tool,
    save_tool,
    list_output_files_tool,
    read_output_file_tool,
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
    self_inspection_tool,
]
_TOOLS = list(map(_safe, _TOOLS))

_agent = create_tool_calling_agent(llm=_llm, prompt=prompt, tools=_TOOLS)
_agent_executor = AgentExecutor(
    agent=_agent,
    tools=_TOOLS,
    memory=_short_term_memory,
    verbose=False,
    return_intermediate_steps=True,
    max_iterations=MAX_TOOL_ITERATIONS,
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
_INFO_KEYWORDS = (
    "co",
    "jak",
    "proč",
    "kde",
    "kdy",
    "kdo",
    "který",
    "jaký",
    "jaká",
    "jaké",
    "kolik",
    "what",
    "why",
    "how",
    "where",
    "when",
    "who",
)

_INFO_TEMPLATES = [
    "What is the capital of France?",
    "How do I reset my password?",
    "Why is the sky blue?",
    "Where can I find the manual?",
    "When does the event start?",
    "Who wrote this book?",
    "Co je to láska?",
    "Jak funguje wifi?",
    "Proč se to děje?",
    "Kde najdu manuál?",
    "Kdy to začne?",
    "Kdo to napsal?",
]

try:
    _INFO_EMBEDDINGS = [_embeddings.embed_query(t) for t in _INFO_TEMPLATES]
except Exception:
    _INFO_EMBEDDINGS = []


def _is_information_query(query: str) -> tuple[bool, str]:
    """Return (True, reason) if query looks like an information request."""
    q = query.lower().strip()
    if any(q.startswith(w) for w in _INFO_KEYWORDS) or "?" in q:
        return True, "keywords"

    if _INFO_EMBEDDINGS:
        try:
            emb = _embeddings.embed_query(q)
            if max(_cosine(emb, t) for t in _INFO_EMBEDDINGS) > 0.8:
                return True, "embedding"
        except Exception:
            pass

    return False, ""


def recall(state: AgentState) -> AgentState:
    """Vyhledá vektorově relevantní kontext z odpovídající kolekce."""
    query = state["query"]
    with _ts_lock:
        since_last = datetime.utcnow().timestamp() - _last_user_ts

    is_info, _ = _is_information_query(query)
    use_kb = is_info or since_last > QUERY_TIME_THRESHOLD
    store = _kb_store if use_kb else _chat_store

    words = len(query.split())
    k = 2 if words < 10 else 4
    results = store.similarity_search_with_relevance_scores(query, k=k)
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
        type_boost = (
            TYPE_BOOST_DOC if doc.metadata.get("source") == "doc" else TYPE_BOOST_CHAT
        )
        score = RERANK_ALPHA * cos + RERANK_BETA * recency + RERANK_GAMMA * type_boost
        reranked.append((score, doc))

    reranked.sort(key=lambda x: x[0], reverse=True)
    docs = [d for _, d in reranked]
    state["retrieved_context"] = "\n".join(d.page_content for d in docs)
    return state


# --- Node 2: Call the langchain agent -----------------------------------------
async def act(state: AgentState) -> AgentState:
    """Spustí nástroj‑volajícího agenta s krátkodobou pamětí + kontextem."""
    try:
        result = await _agent_executor.ainvoke(
            {
                "query": state["query"],
                "retrieved_context": state.get("retrieved_context", ""),
            }
        )
        raw = result["output"].strip()
        if raw.startswith("```"):
            raw = raw.strip("`")  # ořež zahajovací + ukončovací ```
            # po ořezu může zůstat "json\n{", smažeme jazykový prefix
            if raw.lstrip().lower().startswith("json"):
                raw = raw.lstrip()[4:].lstrip()

        # b) prefix bez fence:  json\n{ ...
        if raw.lower().startswith("json"):
            raw = raw[4:].lstrip()

        candidate = _extract_json_object(raw) or raw

        try:
            parsed = ResearchResponse.parse_raw(candidate)
            state["answer"] = parsed.answer
            state["intermediate_steps"] = parsed.intermediate_steps
        except Exception as e:
            state["answer"] = f"⚠️ LLM nevrátil validní JSON: {e}\\n{result['output']}"

        state["intermediate_steps"] = result.get("intermediate_steps", [])
    except Exception as e:  # ← pojistka, kdyby přece jen něco propadlo
        err = f"⚠️ Nástroj selhal: {e}"
        state["answer"] = err
        state["intermediate_steps"] = [err]
    t = asyncio.create_task(learn(state.copy()))
    register_task(t)
    return state


# --- Node 3: Learn (append to vectorstore) ------------------------------------
async def learn(state: AgentState) -> AgentState:
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
        new_embs = await _embeddings.aembed_documents([d.page_content for d in docs])
        user_emb = new_embs[0]
        user_text = docs[0].page_content
        with _cache_lock:
            if any(
                _combined_similarity(user_emb, user_text, ex[0], ex[1]) > DUPLICATE_THRESHOLD
                for ex in _embedding_cache
            ):
                docs = [docs[1]]  # store assistant answer only
            else:
                _embedding_cache.append((user_emb, user_text))
    except Exception:
        pass

    _chat_store.add_documents(docs)

    try:
        total = _chat_store.get_total_records()
        if total % 500 == 0:
            async with _summary_lock:
                before = _chat_store.get_total_records()
                t0 = time.perf_counter()
                res = _chat_store.get(include=["documents", "metadatas"], limit=None)
                records = list(zip(res["ids"], res["documents"], res["metadatas"]))
                records.sort(key=lambda r: r[2].get("ts", ""))
                first = records[:500]
                text = "\n".join(doc for _, doc, _ in first)
                summary = await _llm.ainvoke(
                    f"Summarise following 500 lines of chat history:\n{text}"
                )
                _chat_store.add_documents(
                    [
                        Document(
                            page_content=summary,
                            metadata={"role": "summary", "ts": ts, "source": "chat"},
                        )
                    ]
                )
                ids_to_del = [rid for rid, _, _ in first]
                if ids_to_del:
                    _chat_store.delete(ids_to_del)
                    with _cache_lock:
                        for _ in range(len(ids_to_del)):
                            if _embedding_cache:
                                _embedding_cache.popleft()
                after = _chat_store.get_total_records()
                latency = time.perf_counter() - t0
                print(
                    f"[learn/summarise] kept={after}, removed={before - after}, latency={latency:.2f}s"
                )
                _rebuild_embedding_cache()

        # Re-evaluate collection size after potential summarisation
        total = _chat_store.get_total_records()
        # Defer costly sort/delete until we exceed the limit by a margin
        if total > MAX_CHAT_RECORDS + 500:
            over = total - MAX_CHAT_RECORDS
            res = _chat_store.get(include=["metadatas"], limit=None)
            records = list(zip(res["ids"], res["metadatas"]))
            records.sort(key=lambda r: r[1].get("ts", ""))
            ids_to_del = [rid for rid, _ in records[:over]]
            if ids_to_del:
                _chat_store.delete(ids_to_del)
                with _cache_lock:
                    for _ in range(over):
                        if _embedding_cache:
                            _embedding_cache.popleft()
    except Exception:
        pass
    return state


# Apply timing decorators to key phases
recall = timed("recall")(recall)
act = timed("act")(act)
learn = timed("learn")(learn)


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

graph.set_entry_point("recall")
graph.add_edge("recall", "act")
graph.add_edge("act", END)

# Kompilovaný workflow (lazy‑initialised, aby import nezdržoval start)
workflow = graph.compile()


# ---------------------------------------------------------------------------
# Veřejné API – zůstává stejné jako v core.py
# ---------------------------------------------------------------------------
def handle_query(query: str) -> str:
    """Jediný veřejný vstup: zpracuje dotaz a vrátí odpověď."""
    final = asyncio.run(
        workflow.ainvoke(
            {
                "query": query,
                "answer": "",
                "intermediate_steps": [],
                "retrieved_context": "",
            }
        )
    )
    result = _final_to_json(final)
    _update_last_user_ts()
    return result


# -------- STREAMING (yielduje JSON lines) ------------------------------------
async def handle_query_stream(query: str):
    """Streamuje výstup *celého* grafu na jeden běh (žádné zdvojení tokenů)."""

    final_state = None

    async for ev in workflow.astream_events(
        {
            "query": query,
            "answer": "",
            "intermediate_steps": [],
            "retrieved_context": "",
        },
        version="v1",
    ):
        event_type = _ev_attr(ev, "event") or _ev_attr(ev, "event_name")
        node_name = _ev_attr(ev, "name") or _ev_attr(ev, "node_name")

        # --- Průběžné streamování tokenů LLM ---
        if event_type == "on_llm_new_token":
            data = _ev_attr(ev, "data", {})
            token = (
                data["token"] if isinstance(data, dict) else getattr(data, "token", "")
            )
            yield token
            continue

        # --- Streamování informací o běhu nástrojů ---
        if event_type in {"on_agent_action", "on_tool_start", "on_tool_end"}:
            data = _ev_attr(ev, "data", {})

            action = None
            if event_type == "on_agent_action":
                action = (
                    data.get("action")
                    if isinstance(data, dict)
                    else _ev_attr(data, "action")
                )
            tool = (
                _ev_attr(action or data, "tool")
                or _ev_attr(action or data, "name")
                or node_name
                or ""
            )
            tool_input = (
                _ev_attr(action or data, "tool_input")
                or _ev_attr(data, "input")
                or _ev_attr(data, "output")
                or ""
            )

            _short = (
                json.dumps(tool_input)
                if isinstance(tool_input, dict)
                else str(tool_input)
            )
            _short = _short.replace("\n", " ")
            if len(_short) > 60:
                _short = _short[:60] + "…"
            payload = f"🛠️ {tool}({_short})"
            yield f"§STEP§{payload}"
            continue

        # --- Zachycení finálního stavu po uzlu "act" ---
        if event_type == "on_node_end" and node_name == "act":
            ds = _ev_attr(ev, "data", {})
            if isinstance(ds, dict):
                final_state = ds.get("output") or ds.get("state")
            else:
                final_state = getattr(ds, "output", None) or getattr(ds, "state", None)

    if final_state is None:
        final_state = await workflow.ainvoke(
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

__all__ = [
    "handle_query",
    "handle_query_stream",
    "agent_workflow",
    "ResearchResponse",
    "start_background_jobs",
]
