# 🧠 Project Snapshot

Tento soubor obsahuje strukturu projektu a obsah jednotlivých souborů pro použití s AI asistenty.

## 📂 Struktura projektu

```
UniversalAIAgent/
├── .gitignore
├── README.md
├── agent
│   ├── __init__.py
│   ├── core.py
│   └── core2.py
├── clear_rag_memory.py
├── cli
│   ├── main.py
│   ├── main2.py
│   ├── ui.py
│   └── ui2.py
├── config.json
├── persistent_chat_history.json
├── prompts
│   ├── __init__.py
│   ├── epic.jinja2
│   ├── idea.jinja2
│   └── user_stories.jinja2
├── rag_chroma_db_v2
│   └── chroma.sqlite3
├── rag_confluence_loader.py
├── requirements.txt
├── services
│   ├── __init__.py
│   ├── input_loader.py
│   ├── jira_content_service.py
│   ├── jira_service.py
│   ├── rag_service.py
│   └── web_service.py
├── tools
│   ├── __init__.py
│   ├── input_loader.py
│   ├── jira_content_tools.py
│   ├── jira_tools.py
│   ├── rag_tools.py
│   └── web_tools.py
├── ui.py
├── ui2.py
└── utils
    └── io_utils.py
```

## 📄 Obsahy souborů


---

### `.gitignore`

```python
# Python
__pycache__/
*.py[cod]
*.egg-info/
.env
.vscode/
.idea/
.DS_Store

# Virtual environments
venv/
.env/

# Logs
*.log

# Vector db
rag_chroma_db/

output/
```


---

### `clear_rag_memory.py`

```python
#!/usr/bin/env python3
"""
clear_rag_memory.py
===================

Jednorázové promazání a/nebo validace Chroma DB ('rag_chroma_db/') používané jako RAG paměť.

• Smaže nebo zazálohuje DB a ověří, že po operaci nezůstaly žádné vektory.
• Závislost na 'chromadb' je volitelná – bez ní proběhne fallback validace přes velikost souborů.

Mazání + automatická validace	
    python clear_rag_memory.py -y	
    Po smazání ověří, jestli je DB opravdu prázdná; pokud ne, skript skončí s Neo (1).

Záloha + validace	
    python clear_rag_memory.py --backup -y	
    Přesune DB do zálohy, vytvoří čistý adresář, zkontroluje.

Pouhá kontrola (např. v CI)	
    python clear_rag_memory.py --check	
    Nemění data, jen vrátí 0/1 podle stavu (lze použít v bash if …).

"""

from __future__ import annotations

import argparse
import datetime as _dt
import pathlib as _pl
import shutil as _sh
import sys as _sys
from typing import Optional

try:
    import chromadb
except ImportError:  # pragma: no cover
    chromadb = None  # type: ignore

ROOT = _pl.Path(__file__).resolve().parent
DB_DIR = ROOT / "rag_chroma_db"


# ------------------------------------------------------------------------------
# Utility: počet vektorů v DB
# ------------------------------------------------------------------------------

def _count_vectors() -> int:
    """
    Sečte celkový počet embeddingů ve všech kolekcích.

    Vrací 0, pokud:
      • DB adresář neexistuje
      • Není dostupná knihovna chromadb a současně je adresář prázdný
    """
    if not DB_DIR.exists():
        return 0

    if chromadb is None:
        # Fallback: spočítáme soubory větší než 1 KiB (pravděp. data)
        return sum(1 for p in DB_DIR.rglob("*") if p.is_file() and p.stat().st_size > 1024)

    client = chromadb.PersistentClient(path=str(DB_DIR))
    total = 0
    for col_meta in client.list_collections():
        col = client.get_collection(col_meta.name)
        total += col.count()
    return total


def _validate_empty() -> bool:
    """Vrátí True, když vektorů == 0."""
    leftover = _count_vectors()
    if leftover == 0:
        print("✅ RAG paměť je prázdná, hotovo.")
        return True

    print(f"❌ V paměti zůstalo ještě {leftover} vektorů!", file=_sys.stderr)
    return False


# ------------------------------------------------------------------------------
# Hlavní operace (wipe / backup)
# ------------------------------------------------------------------------------

def _wipe_rag_db(force: bool, backup: bool) -> None:
    """Smaže nebo zálohuje adresář rag_chroma_db/ a pak jej znovu vytvoří."""
    if not DB_DIR.exists():
        print("Adresář 'rag_chroma_db' neexistuje – RAG paměť je už prázdná.")
        return

    if not force:
        prompt = f"Toto {'zálohuje a poté ' if backup else ''}SMAŽE '{DB_DIR}'. Pokračovat? [y/N] "
        if input(prompt).strip().lower() not in {"y", "yes"}:
            print("Operace zrušena.")
            return

    if backup:
        stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        dest = DB_DIR.with_name(f"rag_chroma_db_backup_{stamp}")
        _sh.move(str(DB_DIR), dest)
        print(f"Vektorová DB byla přesunuta do: {dest}")
    else:
        _sh.rmtree(DB_DIR)
        print("Vektorová DB byla nenávratně smazána.")

    DB_DIR.mkdir(exist_ok=True)
    print("Vytvořen nový prázdný 'rag_chroma_db'.")


# ------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------

def main() -> None:  # noqa: D401 – krátký název je OK
    parser = argparse.ArgumentParser(
        description="Vymaže (nebo zazálohuje) perzistentní RAG paměť a ověří, že je prázdná."
    )
    parser.add_argument("-y", "--yes", action="store_true", help="Provede akci bez interaktivního potvrzení.")
    parser.add_argument("--backup", action="store_true", help="Před smazáním vytvoří timestampovanou zálohu.")
    parser.add_argument("--check", action="store_true", help="Pouze zkontroluje, zda je paměť prázdná (nic nemaže).")
    args = parser.parse_args()

    if args.check:
        ok = _validate_empty()
        _sys.exit(0 if ok else 1)

    _wipe_rag_db(force=args.yes, backup=args.backup)
    ok = _validate_empty()
    _sys.exit(0 if ok else 1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        _sys.exit("\nPřerušeno uživatelem.")

```


---

### `config.json`

```python
{
  "jira": {
    "url":  "https://productoo.atlassian.net/",
    "user": "tomas.ventruba@productoo.com",
    "jql":   "project = P4 ORDER BY created DESC",
    "jira_project_key": "P4",
    "maxResults": 50,
    "roadmap_url": "https://roadmap.productoo.com/p4/Working-version/"
  },
    "confluence": {
        "base_url": "https://productoo.atlassian.net",
        "user": "tomas.ventruba@productoo.com",
        "spaces": {
            "documentation": "PD",
            "roadmap": "PD"
        },
        "ancestor_ids": [
            "677183489",
            "99745795"
        ]
  }
}
```


---

### `persistent_chat_history.json`

```python
[]
```


---

### `rag_confluence_loader.py`

```python
#!/usr/bin/env python3
"""
rag_confluence_loader.py

Load Confluence pages (and their descendants) into a Chroma vectorstore for RAG.
Skips pages already present in the vectorstore by page_id metadata.
"""

import os
import json
from pathlib import Path

from dotenv import load_dotenv

# ── Suppress Chroma telemetry errors ──────────────────────────────────────────
# Must be set before importing Chroma
os.environ["CHROMA_TELEMETRY"] = "0"

from langchain_community.document_loaders import ConfluenceLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter


def load_config(config_path: Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_existing_page_ids(vectorstore: Chroma) -> set:
    """Retrieve page_id metadata from all documents currently in the vectorstore."""
    col = vectorstore._collection
    metadatas = col.get(include=["metadatas"])["metadatas"]
    return { str(m.get("page_id") or m.get("id")) for m in metadatas if m.get("page_id") or m.get("id") }


def main():
    # ── Load environment and config ─────────────────────────────────────────────
    load_dotenv()
    token = os.getenv("JIRA_AUTH_TOKEN")
    if not token:
        raise RuntimeError("Missing JIRA_AUTH_TOKEN in environment")

    config_path = Path(__file__).parent / "config.json"
    cfg = load_config(config_path)
    conf_cfg = cfg.get("confluence", {})
    base_url = conf_cfg["base_url"]
    user = conf_cfg["user"]
    ancestor_ids = conf_cfg.get("ancestor_ids", [])
    if not ancestor_ids:
        print("No ancestor_ids configured. Nothing to load.")
        return

    # ── Prepare embeddings & vectorstore ────────────────────────────────────────
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma(
        persist_directory="rag_chroma_db",
        embedding_function=embeddings
    )

    existing_ids = get_existing_page_ids(vectorstore)
    print(f"{len(existing_ids)} pages already in vectorstore.")

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    new_chunks = []

    # ── Load & filter Confluence pages ──────────────────────────────────────────
    for anc in ancestor_ids:
        print(f"→ Fetching descendants of Confluence page {anc} …")
        loader = ConfluenceLoader(
            url=base_url,
            username=user,
            api_key=token,
            cql=f"(id={anc} OR ancestor={anc})",
            include_archived_content=False,
            include_restricted_content=False
        )
        docs = loader.load()
        print(f"   • Retrieved {len(docs)} pages under ancestor {anc}")

        fresh = []
        for doc in docs:
            pid = str(doc.metadata.get("id") or doc.metadata.get("page_id"))
            if pid in existing_ids:
                print(f"     – Skipping already indexed page {pid}")
                continue
            doc.metadata["page_id"] = pid
            fresh.append(doc)

        if not fresh:
            continue

        chunks = splitter.split_documents(fresh)
        print(f"Chunked into {len(chunks)} pieces")
        new_chunks.extend(chunks)

    # ── Add only new chunks to Chroma ───────────────────────────────────────────
    if new_chunks:
        vectorstore.add_documents(new_chunks)
        print(f"Added {len(new_chunks)} new document chunks.")
    else:
        print("No new documents to add.")

if __name__ == "__main__":
    main()

```


---

### `README.md`

```python
# 🧠 Universal AI Agent (Productoo P4)

*Conversational assistant for product managers, analysts and engineers working on Productoo’s P4 manufacturing suite.*

---

## ✨ What it does now

| Category                      | Status          | Details                                                                                                                 |
| ----------------------------- | --------------- | ----------------------------------------------------------------------------------------------------------------------- |
| **Conversational interface**  | **✅**           | CLI (`main.py`) **and** lightweight Gradio UI (`ui.py`).                                                                |
| **Knowledge retrieval (RAG)** | **✅**           | Chroma vector‑store (`rag_chroma_db/`) continuously enriched with every Q\&A turn.                                      |
| **Web search**                | **✅**           | DuckDuckGo (`searchWeb`) & Wikipedia snippet tool.                                                                      |
| **Semantic web search**       | **β**           | Tavily semantic search if `TAVILY_API_KEY` is present.                                                                  |
| **Jira integration**          | **✅**           | `jira_ideas_retriever` – lists *Idea* issues matching an optional keyword.                                              |
| **File output**               | **✅**           | `save_text_to_file` stores each answer in a *new* timestamped file under `./output/`. Visible & downloadable in the UI. |
| **Confluence loader**         | **✅ (offline)** | `rag_confluence_loader.py` indexes Confluence pages into RAG (manual run).                                              |
| **Continuous learning**       | **↺**           | Every chat exchange is appended to the vector store for long‑term memory.                                               |

---

## 🔎 Available tools

| Tool name              | Purpose                                                                                   |
| ---------------------- | ----------------------------------------------------------------------------------------- |
| `searchWeb`            | Quick DuckDuckGo search (definitions, news, blogs).                                       |
| `wikipedia_query`      | Short summary from Wikipedia.                                                             |
| `rag_retriever`        | Fetch up to 4 most relevant chunks from the internal vector store (docs, roadmap, chats). |
| `jira_ideas_retriever` | List *Ideas* from Jira project **P4**; optional `keyword` filter.                         |
| `tavily_search`        | LLM‑powered semantic web search (requires `TAVILY_API_KEY`).                              |
| `save_text_to_file`    | Persist any text to `output/…` (timestamped).                                             |

> **Planned tool** – `jira_issue_detail`: fetch a **single** Jira issue by key (e.g. `P4‑1234`) with full description, acceptance criteria, subtasks & comments.
> *Benefit:* quick deep‑dives, faster duplicate detection.

---

## 🏗 Architecture overview

```
┌───────────────┐     user query / feedback
│   Gradio UI   │  ◀──────────────────────┐
└──────┬────────┘                          │
       │ HTTP                             │
┌──────▼────────┐     internal call        │
│  LangChain    │  ──▶  Tool Router  ──▶───┘
│   Agent       │          │
└───────────────┘          ▼
   │     │    │    structured/tool calls
   │     │    │
   │     │    └─▶ Jira API (ideas / issue‑detail)
   │     └────────▶ Web search (DuckDuckGo / Tavily)
   └──────────────▶ RAG vector store (Chroma)
```

---

## 🚀 Quick start

```bash
# 1. Install deps (create venv beforehand)
pip install -r requirements.txt

# 2. Configure API keys (.env)
cp .env.example .env  # fill in OPENAI_API_KEY, JIRA_AUTH_TOKEN …

# 3a. Run conversational CLI
python main.py

# 3b. Or launch local web UI
python ui.py  # opens http://127.0.0.1:7860
```

### Required environment variables

```dotenv
OPENAI_API_KEY="sk‑…"
JIRA_AUTH_TOKEN="atlassian‑…"
TAVILY_API_KEY=""           # optional
```

---

## 🗂 Project layout

```
📁 universalagent/
├── main.py                # conversational loop
├── tools.py               # LangChain Tools (search, Jira, save, …)
├── ui.py                  # Gradio front‑end
├── jira_retriever.py      # low‑level Jira REST helper
├── rag_confluence_loader.py  # import Confluence pages into RAG
├── rag_vectorstore.py     # bulk‑import local docs into RAG
├── output/                # timestamped txt exports (git‑ignored)
└── rag_chroma_db/         # vector DB (git‑ignored)
```

---

## 🛣 Next steps (roadmap)

| Priority | Item                                     | Rationale                                                                               |
| -------- | ---------------------------------------- | --------------------------------------------------------------------------------------- |
| **⬆**    | **`jira_issue_detail` Tool**             | Fetch full Jira issue by key; enable deep context for duplicates & acceptance criteria. |                                            |
|  —       | Confluence incremental sync              | Schedule nightly run; mark removed pages as archived in RAG.                            |
|  —       | Auto‑summarise fresh Jira tickets to RAG | “Chronicle” new issues daily for fast retrieval.                                        |
|  —       | Duplicate‑idea detector                  | Hash & embedding similarity across Jira Ideas.                                          |
|  —       | KPI dashboard                            | Track solved tickets, average cycle‑time, top requested features.                       |                                |
|  —       | Unit & integration tests                 | pytest + Playwright for UI workflows.                                                   |
|  —       | Create Jira Epics, Stories & Release notes  | Write content of jira issues                                                  |

Contributions & ideas welcome – open an issue or ping **@tomas.ventruba**.

```


---

### `requirements.txt`

```python
langchain>=0.2.9,<1.0
wikipedia
langchain-community
langchain-openai
langchain-anthropic
langchain-chroma
langchain-unstructured
python-dotenv
pydantic
duckduckgo-search
tavily-python
atlassian-python-api
gradio
pytest
chromadb
unstructured
jinja2
langgraph
```


---

### `ui.py`

```python
"""Compatibility wrapper for the Gradio UI entry point."""
from cli.ui import launch

if __name__ == "__main__":
    launch()

```


---

### `ui2.py`

```python
"""Entry‑point pro Gradio UI nad core2."""
from cli.ui2 import launch

if __name__ == "__main__":
    launch()

```


---

### `agent\core.py`

```python
# agent/core.py
from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
import openai
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.memory.buffer import ConversationBufferMemory
from langchain.schema import Document
from langchain_chroma import Chroma
from chromadb.config import Settings
from pydantic import BaseModel

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
# Environment & telemetry
# ---------------------------------------------------------------------------
load_dotenv()

os.environ.setdefault("OPENAI_TELEMETRY", "0")
if hasattr(openai, "telemetry") and hasattr(openai.telemetry, "TelemetryClient"):
    openai.telemetry.TelemetryClient.capture = lambda *_, **__: None



# ---------------------------------------------------------------------------
# Vectorstore (RAG) setup
# ---------------------------------------------------------------------------
CHROMA_DIR = "rag_chroma_db"
_embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
_vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=_embeddings)


def _store_exchange(question: str, answer: str) -> None:
    """Save the dialogue pair into the RAG vector store for future retrieval."""
    ts = datetime.utcnow().isoformat()
    docs = [
        Document(page_content=f"USER: {question}", metadata={"role": "user", "ts": ts}),
        Document(page_content=f"ASSISTANT: {answer}", metadata={"role": "assistant", "ts": ts}),
    ]
    _vectorstore.add_documents(docs)


# ---------------------------------------------------------------------------
# Structured output schema
# ---------------------------------------------------------------------------
class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]


parser = PydanticOutputParser(pydantic_object=ResearchResponse)

# ---------------------------------------------------------------------------
# Core prompt template
# ---------------------------------------------------------------------------
system_prompt = (
    """
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
"""
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions())

# ---------------------------------------------------------------------------
# LLM & conversation memory
# ---------------------------------------------------------------------------
_llm = ChatOpenAI(model="gpt-4o", temperature=0.3)
_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# ---------------------------------------------------------------------------
# Agent assembly
# ---------------------------------------------------------------------------
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
agent_executor = AgentExecutor(
    agent=_agent,
    tools=_TOOLS,
    memory=_memory,
    verbose=True,
    return_intermediate_steps=True,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def handle_query(query: str) -> str:
    """Vrátí plný návrh (Markdown), pokud je k dispozici, jinak fallback na summary."""
    raw = agent_executor.invoke({"query": query})
    full = next(
        (obs for _act, obs in reversed(raw.get("intermediate_steps", []))
         if isinstance(obs, str) and obs.strip()),
        "",
    )
    answer = full or raw.get("output", "")
    _store_exchange(query, answer)
    return answer


__all__ = ["handle_query", "agent_executor", "ResearchResponse", "parser"]

```


---

### `agent\core2.py`

```python
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
graph = StateGraph(input_type=AgentState)
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

```


---

### `agent\__init__.py`

```python
# agent/__init__.py
from importlib import import_module
import os

_selected = os.getenv("AGENT_VERSION", "v1").lower()   # v1 | v2
if _selected == "v2":
    core2 = import_module(".core2", __name__)
    handle_query = core2.handle_query
    agent_workflow = core2.agent_workflow
else:
    core = import_module(".core", __name__)
    handle_query = core.handle_query
    agent_executor = core.agent_executor
    ResearchResponse = core.ResearchResponse
    parser = core.parser

__all__ = [
    "handle_query",
    "agent_executor",
    "agent_workflow",
    "ResearchResponse",
    "parser",
]

```


---

### `cli\main.py`

```python
# cli/main.py
from __future__ import annotations

from agent import handle_query, parser, ResearchResponse


def main() -> None:
    """Launch interactive CLI until user exits."""
    print("\nUniversal AI Agent (Productoo) — napište 'exit' pro ukončení.\n")
    while True:
        try:
            user_query = input("Vy: ").strip()
            if user_query.lower() in {"exit", "quit", "bye"}:
                print("Asistent: Rád jsem pomohl! Mějte se.")
                break

            answer = handle_query(user_query)

            try:
                structured: ResearchResponse = parser.parse(answer)
                print(f"\nAsistent: {structured.summary}\n")
                if structured.sources:
                    print("Zdroj(e):", ", ".join(structured.sources))
            except Exception:
                print("\nAsistent:", answer)

        except KeyboardInterrupt:
            print("\nAsistent: Končím. Mějte se.")
            break


if __name__ == "__main__":
    main()

```


---

### `cli\main2.py`

```python
# cli/main2.py
from __future__ import annotations
from agent.core2 import handle_query   # ← důležité

BANNER = "\nUniversal AI Agent • *core2* (LangGraph) — napište 'exit' pro ukončení.\n"

def main() -> None:
    print(BANNER)
    while True:
        try:
            q = input("Vy: ").strip()
            if q.lower() in {"exit", "quit", "bye"}:
                print("Asistent: Rád jsem pomohl! Mějte se.")
                break
            print("\nAsistent:", handle_query(q), "\n")
        except KeyboardInterrupt:
            print("\nAsistent: Končím. Mějte se.")
            break

if __name__ == "__main__":
    main()

```


---

### `cli\ui.py`

```python
# cli/ui.py
"""Lightweight local front-end for the Universal AI Agent."""
from __future__ import annotations

import asyncio
import json
import os
import pathlib
import re
import warnings

import gradio as gr

from agent import handle_query

warnings.filterwarnings("ignore", message=".*LangChainDeprecationWarning.*")
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "false")

OUTPUT_DIR = pathlib.Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)


def list_files() -> list[str]:
    return sorted(p.name for p in OUTPUT_DIR.iterdir() if p.is_file())


def read_file(fname: str) -> str:
    path = OUTPUT_DIR / fname
    try:
        return path.read_text("utf-8")
    except (UnicodeDecodeError, FileNotFoundError):
        return f"[Nelze zobrazit: binární nebo neexistuje] {fname}"


def file_path(fname: str) -> str | None:
    p = OUTPUT_DIR / fname
    return str(p) if p.exists() else None


def refresh_choices():
    try:
        return gr.Dropdown.update(choices=list_files())
    except AttributeError:
        return gr.update(choices=list_files())


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


async def chat_fn(msg, history):
    history = history or []
    history.append({"role": "user", "content": msg})

    loop = asyncio.get_event_loop()
    raw = await loop.run_in_executor(None, lambda: handle_query(msg))
    history.append({"role": "assistant", "content": pretty(raw)})
    return history, history


def file_selected(fname):
    path = file_path(fname)
    preview = read_file(fname)
    try:
        file_update = gr.File.update(value=path, visible=bool(path))
    except AttributeError:
        file_update = gr.update(value=path, visible=bool(path))
    return preview, file_update


def trigger_download(fname):
    path = file_path(fname)
    try:
        return gr.File.update(value=path, visible=bool(path))
    except AttributeError:
        return gr.update(value=path, visible=bool(path))


def launch() -> None:
    with gr.Blocks(title="Universal AI Agent") as demo:
        gr.Markdown("## 💬 AI Agent • 🗂 Output soubory")

        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(type="messages", height=420)
                msg = gr.Textbox(lines=2, placeholder="Zadej dotaz… (Ctrl+Enter)")
                msg.submit(chat_fn, [msg, chatbot], [chatbot, chatbot]).then(lambda: "", None, msg)

            with gr.Column():
                files = gr.Dropdown(choices=list_files(), label="Output soubory", interactive=True)
                content = gr.Textbox(label="Náhled obsahu", lines=14, interactive=False, show_copy_button=True)
                download_file = gr.File(label="Klikni pro stažení", visible=False)

                files.change(file_selected, files, [content, download_file])

                with gr.Row():
                    gr.Button("↻ Refresh").click(refresh_choices, None, files).then(
                        file_selected, files, [content, download_file]
                    )
                    gr.Button("⬇ Stáhnout").click(trigger_download, files, download_file)

        if list_files():
            demo.load(file_selected, inputs=files, outputs=[content, download_file])

        demo.queue()
        demo.launch()


if __name__ == "__main__":
    launch()

```


---

### `cli\ui2.py`

```python
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

```


---

### `prompts\epic.jinja2`

```python
{# Jira Epic – Czech #}
Jsi produktový vlastník a připravuješ **Jira Epic** v češtině.
Text je věcný, bez marketingových superlativů.

Vrať Markdown v následujícím pořadí:

**Epic Goal** – jedna věta  
**Context** – 1‑2 odstavce, propojuje problém a řešení  
**Definition of Done** – odrážky  
**Acceptance criteria** – Given / When / Then (≤ 7)  
**Out of scope** – odrážky

Idea
----
Název: {{ summary }}
Popis: {{ description or '(none)' }}

```


---

### `prompts\idea.jinja2`

```python
{# Jira Idea – English, business‑ready #}
{%- set audience_phrase = {"business":"executives and sales",
                           "technical":"engineering teams",
                           "mixed":"cross‑functional stakeholders"}[audience] %}
You are a senior product manager writing **concise, board‑ready Jira Ideas**.
Audience: {{ audience_phrase }}.
Tone: plain British English.
Limit total length to ≤ {{ max_words }} words.
Highlight the core insight in the first sentence.
Use exactly the headings you see in the example.

## Problem
Our maintenance costs grew 15 % YoY due to frequent unplanned packaging‑line stops.

## Proposed solution
Monitor vibration and temperature, predict failures 48 h ahead and auto‑schedule maintenance.

## Business value
- Reduce downtime by 10 h / month  
- Save €30 k per quarter  

## Acceptance criteria
- **Given** line sensors are online  
- **When** failure probability > 70 % is detected  
- **Then** a work order is created and a maintenance slot reserved  

### Raw Idea
SUMMARY: {{ summary }}
DESCRIPTION: {{ description or '(none)' }}

### Task
Rewrite the Raw Idea into a polished Jira Idea description using the **same four headings** (Problem, Proposed solution, Business value, Acceptance criteria).  
Use active voice, quantify benefits, avoid jargon, keep total ≤ {{ max_words }} words.

```


---

### `prompts\user_stories.jinja2`

```python
{# INVEST User Stories – Czech #}
Jsi agile coach. Vytvoř **{{ count }} nezávislých user stories** z tohoto Epicu
v češtině a dodrž INVEST.

EPIC: {{ epic_name }}

{{ epic_description }}

Formát každé story:

### <Název>
**User story**: Jako <persona> chci … aby …  
**Acceptance criteria**
- Given / When / Then odrážky  
**Estimate**: <S / M / L>

```


---

### `prompts\__init__.py`

```python

```


---

### `rag_chroma_db_v2\chroma.sqlite3`

```python


---

### `rag_chroma_db_v2\chroma.sqlite3`

```python
# Nelze načíst soubor: 'utf-8' codec can't decode byte 0xfb in position 106: invalid start byte
```


---

### `services\input_loader.py`

```python
# services/input_loader.py

"""
Input Loader – import files from ./input do Chroma vektorové DB
---------------------------------------------------------------
➤ Přetahuje soubory z lokální složky `./input` (případně podmnožinu podle
  argumentu) a indexuje je pro RAG dotazy.

Argumenty (string):
    ""            → plný sken (přeskočí už zaindexované soubory)
    "force"       → znovu zaindexuje všechno (ignoruje fingerprinty)
    "pdf" / ".pdf"→ jen danou příponu
    "subdir/foo"  → jen daný (pod)adresář
    "file.ext"    → jen konkrétní soubor

Vrací:
    Markdown report – jedna řádka na každý nově naimportovaný soubor.
"""
from __future__ import annotations

import hashlib
import inspect
from pathlib import Path
from typing import List

from langchain_chroma import Chroma
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores.utils import filter_complex_metadata

# ---- konfigurace -----------------------------------------------------------
CHROMA_DIR = "rag_chroma_db"
INPUT_DIR = Path("input")
INPUT_DIR.mkdir(exist_ok=True)

_embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
_vectorstore = Chroma(
    collection_name="rag_store",
    embedding_function=_embeddings,
    persist_directory=CHROMA_DIR,
)

# ---- utilitky --------------------------------------------------------------
def _file_id(path: Path) -> str:
    """Fingerprint souboru = cesta::md5(content)."""
    h = hashlib.md5(path.read_bytes()).hexdigest()  # noqa: S324 – lokální hash
    return f"{path.as_posix()}::{h}"


def _already_indexed(file_id: str) -> bool:
    col = _vectorstore._collection
    return file_id in (m.get("file_id") for m in col.get(include=["metadatas"])["metadatas"])


def _summarise(text: str, llm: ChatOpenAI) -> str:
    prompt = "Shrň následující dokument do max 1 věty (čeština):\n" + text[:4000]
    return llm.invoke(prompt).content.strip()


# ---- robustní sanitizace metadat ------------------------------------------
# API detekujeme dynamicky – pokud selže volání s Documentem, zkusíme dict
def _sanitize_metadata(doc: Document) -> None:
    """Zaručí, že v metadata zůstanou jen skaláry a zavolá korektní variantu
    filter_complex_metadata bez ohledu na verzi knihovny."""
    try:
        # 🅰️ novější API (bere Document, mutuje in-place)
        filter_complex_metadata(doc)
    except Exception:
        try:
            # 🅱️ starší API (bere dict, vrací dict)
            doc.metadata = filter_complex_metadata(doc.metadata)  # type: ignore[arg-type]
        except Exception:
            # 🆘 poslední záchrana – ručně převést neskalární hodnoty na str
            doc.metadata = {
                k: (str(v) if isinstance(v, (list, dict, set, tuple)) else v)
                for k, v in doc.metadata.items()
            }


# ----------------------------------------------------------------------------
def process_input_files(arg: str = "") -> str:
    """Hlavní API volané LangChain toolem."""
    arg = arg.strip()
    force_reindex = arg.lower() == "force"

    # -- vyhodnocení filtru ---------------------------------------------------
    ext_filter = None
    single_target: Path | None = None
    dir_filter: Path | None = None

    if arg and not force_reindex:
        if arg.lower().lstrip(".") in {"pdf", "docx", "pptx", "xlsx", "txt", "md", "html", "csv"}:
            ext_filter = arg.lower().lstrip(".")
        else:
            maybe_path = INPUT_DIR / arg
            if maybe_path.exists():
                single_target = maybe_path if maybe_path.is_file() else None
                dir_filter = maybe_path if maybe_path.is_dir() else None

    paths: List[Path] = (
        [single_target]
        if single_target
        else [p for p in (dir_filter or INPUT_DIR).rglob("*") if p.is_file()]
    )
    if ext_filter:
        paths = [p for p in paths if p.suffix.lower().lstrip(".") == ext_filter]
    if not paths:
        return "Nenalezeny žádné soubory odpovídající zadání."

    # -- načítání & chunkování ------------------------------------------------
    from langchain_unstructured import UnstructuredLoader  # lazy import

    new_docs: List[Document] = []
    reports: List[str] = []
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    for path in paths:
        fid = _file_id(path)
        if not force_reindex and _already_indexed(fid):
            continue

        # 1️⃣ načti dokument(y)
        try:
            raw_docs = UnstructuredLoader(str(path)).load()
        except Exception as exc:  # pragma: no cover
            reports.append(f"- {path.name}: ❌ Nepodařilo se načíst ({exc})")
            continue

        # 2️⃣ normalizuj na Document
        docs: List[Document] = []
        for item in raw_docs:
            if isinstance(item, Document):
                docs.append(item)
            elif isinstance(item, tuple) and len(item) == 2:
                content, meta = item
                docs.append(Document(page_content=str(content), metadata=dict(meta or {})))
            else:
                docs.append(Document(page_content=str(item), metadata={}))

        # 3️⃣ metadata & sanitizace
        base_meta = {"source": "input_folder", "file_path": str(path), "file_id": fid}
        for d in docs:
            d.metadata.update(base_meta)
            _sanitize_metadata(d)

        # 4️⃣ split + přidej do sbírky
        new_docs.extend(splitter.split_documents(docs))

        # 5️⃣ shrnutí pro report
        summary = _summarise(" ".join(d.page_content for d in docs)[:8000], llm)
        reports.append(f"- {path.name}: {summary}")

    if new_docs:
        _vectorstore.add_documents(new_docs)  # Chroma se uloží automaticky

    if not reports:
        return "Žádné nové soubory nebyly naimportovány."
    return f"✅ Naimportováno {len(reports)} souborů:\n" + "\n".join(reports)

```


---

### `services\jira_content_service.py`

```python
# services/jira_content_service.py
# ════════════════════════════════
"""OpenAI-powered helpers for drafting high-quality Jira issue content.

Public API
----------
enhance_idea(...)           -> Markdown for an *Idea* ticket
epic_from_idea(...)         -> Markdown for an *Epic* ticket
user_stories_for_epic(...)  -> Markdown list of INVEST-ready User Stories

"""

from __future__ import annotations

import os
import openai
from pathlib import Path
from typing import List
from jinja2 import Environment, FileSystemLoader, select_autoescape

from langchain_openai import ChatOpenAI

# ── Disable OpenAI telemetry ───────────────────────────────────────────────────
os.environ.setdefault("OPENAI_TELEMETRY", "0")
if hasattr(openai, "telemetry") and hasattr(openai.telemetry, "TelemetryClient"):
    openai.telemetry.TelemetryClient.capture = lambda *_, **__: None

# ── LLM configuration ─────────────────────────────────────────────────────────
_MODEL = os.getenv("JIRA_CONTENT_MODEL", "gpt-4o")
_TEMPERATURE = float(os.getenv("JIRA_CONTENT_TEMPERATURE", "0.2"))

_llm = ChatOpenAI(model=_MODEL, temperature=_TEMPERATURE)

# ── Jinja2 prostředí pro šablony ────────────────────────────────────────────
_PROMPT_ENV = Environment(
    loader=FileSystemLoader(Path(__file__).resolve().parent.parent / "prompts"),
    autoescape=select_autoescape(disabled_extensions=("jinja2",)),
    trim_blocks=True,
    lstrip_blocks=True,
)

def _render(name: str, **kwargs) -> str:
    """Vyrenderuj zadanou .jinja2 šablonu s parametry."""
    return _PROMPT_ENV.get_template(name).render(**kwargs)

# ── Internal helper ───────────────────────────────────────────────────────────
def _run(prompt: str) -> str:
    """Call the LLM synchronously and return trimmed text."""
    return _llm.invoke(prompt).content.strip()


# ── High-level generators ─────────────────────────────────────────────────────
def enhance_idea(
    summary: str,
    description: str | None = None,
    audience: str = "mixed",
    max_words: int = 360,
    ) -> str:

    prompt = _render(
        "idea.jinja2",
        summary=summary,
        description=description,
        audience=audience,
        max_words=max_words,
    )
    return _run(prompt)


def epic_from_idea(
        summary: str,
        description: str | None = None
        ) -> str:
    """
    Scale an Idea into a **Jira Epic**.  Outputs Markdown containing:
      • Epic Goal
      • Context
      • Definition of Done
      • Acceptance criteria
      • Out of scope
    """
    prompt = _render(
        "epic.jinja2",
        summary=summary,
        description=description,
    )
    return _run(prompt)


def user_stories_for_epic(
    epic_name: str,
    epic_description: str,
    count: int = 5,
) -> str:
    """
    Generate *count* INVEST-compliant User Stories for the given Epic.
    Each story includes title, user-story sentence, acceptance criteria
    and a T-shirt-size estimate.
    """
    prompt = _render(
        "user_stories.jinja2",
        epic_name=epic_name,
        epic_description=epic_description,
        count=count,
    )
    return _run(prompt)


__all__: List[str] = [
    "enhance_idea",
    "epic_from_idea",
    "user_stories_for_epic",
]




```


---

### `services\jira_service.py`

```python
"""Jira related service layer extracted from legacy jira_client module."""
from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple
from datetime import datetime
from pathlib import Path
import json
import os
import time
import math
import inspect

from atlassian import Jira
from langchain_openai import OpenAIEmbeddings

__all__ = [
    "JiraClient",
    "find_duplicate_ideas",
    "JiraClientError",
    "_extract_text_from_adf",
]

_CFG_PATH = Path("config.json")
_RETRY_SLEEP = (0.5, 1.5, 3.0)  # exponential back-off in seconds


class JiraClientError(RuntimeError):
    """Raised when JiraClient cannot complete a request."""


# ──────────────────────────────────────────────────────────────────────────────
# Utility helpers
# ──────────────────────────────────────────────────────────────────────────────

def _load_cfg() -> Dict[str, Any]:
    if _CFG_PATH.exists():
        with _CFG_PATH.open(encoding="utf-8") as fh:
            return json.load(fh).get("jira", {})
    return {}


def _fields_to_string(fields: Sequence[str] | str | None) -> str:
    if fields is None:
        return "*all"
    if isinstance(fields, str):
        return fields
    return ",".join(fields)


def _extract_text_from_adf(adf: Any) -> str:
    """Very small ADF→plain-text converter (sufficient for descriptions)."""
    if adf is None:
        return ""
    stack: List[Any] = [adf]
    parts: List[str] = []
    while stack:
        n = stack.pop()
        if isinstance(n, dict):
            if n.get("type") == "text" and "text" in n:
                parts.append(n["text"])
            for child_key in ("content", "items"):
                if isinstance(n.get(child_key), list):
                    stack.extend(n[child_key])
        elif isinstance(n, list):
            stack.extend(n)
    return " ".join(parts).strip()


def _merge(issue: Dict[str, Any]) -> Dict[str, Any]:
    merged = {**issue.get("fields", {})}
    for k, v in issue.items():
        if k not in {"fields", "expand", "id", "key", "self"}:
            merged.setdefault(k, v)
    return merged


# ──────────────────────────────────────────────────────────────────────────────
# Main client – public API only search_issues & get_issue
# ──────────────────────────────────────────────────────────────────────────────


class JiraClient:  # pylint: disable=too-few-public-methods
    """Typed, minimal wrapper around *atlassian-python-api*'s ``Jira``."""

    def __init__(
        self,
        url: str | None = None,
        email: str | None = None,
        token: str | None = None,
        *,
        max_retries: int = 3,
        timeout: int = 10,
    ) -> None:
        cfg = _load_cfg()
        url = url or cfg.get("url")
        email = email or cfg.get("user") or cfg.get("email")
        token = token or os.getenv("JIRA_AUTH_TOKEN")
        if not (url and email and token):
            raise ValueError(
                "Missing Jira credentials – provide via arguments, config.json or env vars."
            )
        self._jira = Jira(url=url, username=email, password=token, cloud=True, timeout=timeout)
        self._retries = max_retries

    # ------------------------------------------------------------------ helpers
    def _call(self, func, *args, **kwargs):  # noqa: ANN001 – 3rd-party sig
        for delay in (*_RETRY_SLEEP, None):
            try:
                return func(*args, **kwargs)
            except Exception as exc:  # noqa: BLE001 – surface last
                if delay is None:
                    raise JiraClientError(str(exc)) from exc
                time.sleep(delay)

    # ------------------------------------------------------------------ public
    def search_issues(
        self,
        jql: str,
        *,
        max_results: int = 50,
        fields: Sequence[str] | str | None = None,
    ) -> List[Dict[str, Any]]:
        wanted = _fields_to_string(
            fields or ["summary", "status", "issuetype", "labels", "description"]
        )
        payload: Dict[str, Any] = self._call(
            self._jira.jql, jql, limit=max_results, fields=wanted
        )
        issues: List[Dict[str, Any]] = payload.get("issues", [])
        for iss in issues:
            fld = _merge(iss)
            if isinstance(fld.get("description"), (dict, list)):
                plain = _extract_text_from_adf(fld["description"])
                fld["description_plain"] = plain
                fld["description"] = plain
            iss["fields"] = fld
        return issues

    def get_issue(
        self,
        key: str,
        *,
        fields: Sequence[str] | str | None = None,
    ) -> Dict[str, Any]:
        wanted = _fields_to_string(fields or "*all")
        raw: Dict[str, Any] = self._call(self._jira.issue, key, fields=wanted)
        fld = _merge(raw)
        if isinstance(fld.get("description"), (dict, list)):
            plain = _extract_text_from_adf(fld["description"])
            fld["description_plain"] = plain
            fld["description"] = plain
        return {"key": key, "fields": fld}
    
        # ──────────────────────────────────────────────────────────────────
    # JiraClient.update_issue – jediný veřejný mutační entry-point
    # ──────────────────────────────────────────────────────────────────
    def update_issue(
        self,
        key: str,
        data: Dict[str, Any] | None = None,
    ) -> None:
        """
        Update arbitrary fields of a Jira issue (např. description, labels…).

        data = { "fields": {...}, "update": {...} }  # podle REST API
        """
        import inspect   # <- už je naimportován nahoře, přidejte jen pokud chybí

        data = data or {}
        fields_param = data.get("fields")
        update_param = data.get("update")

        # 1️⃣ Vybereme metodu (update_issue › edit_issue › Issue.update)
        if hasattr(self._jira, "update_issue"):
            func = self._jira.update_issue
        elif hasattr(self._jira, "edit_issue"):
            func = self._jira.edit_issue
        else:
            issue = self._call(self._jira.issue, key)
            func = issue.update

        # 2️⃣ Připravíme payload pro „starší“ signaturu (jediný dict)
        payload: Dict[str, Any] = {}
        if fields_param is not None:
            payload["fields"] = fields_param
        if update_param is not None:
            payload["update"] = update_param

        # Pokud posíláme ADF (tj. description=dict), obejdeme SDK
        if (
            fields_param
            and isinstance(fields_param.get("description"), dict)  # ADF poznáme podle dict
        ):
            url = f"{self._jira.url}/rest/api/3/issue/{key}"
            #   self._jira.session je requests.Session() už přihlášený BasicAuth-Tokenem
            resp = self._jira.session.put(url, json={"fields": fields_param})
            if not resp.ok:
                raise JiraClientError(
                    f"Jira update failed: {resp.status_code} {resp.text}"
                )
            return

        # 3️⃣ Zavoláme správnou signaturu podle toho, co metoda opravdu umí
        sig = inspect.signature(func)
        try:
            if "fields" in sig.parameters:
                # Novější atlassian-python-api (>=3.41) – podporuje pojmenované parametry
                self._call(func, key, fields=fields_param, update=update_param)
            else:
                # Starší verze – očekává jediný slovník payload
                self._call(func, key, payload)
        except Exception as exc:                 # noqa: BLE001
            raise JiraClientError(f"Jira update failed: {exc}") from exc





# ────────────────────────── Embed-cache + normalizace ─────────────────────────
_ABBREV = {
    "2fa": "two factor authentication",
    "mfa": "multi factor authentication",
    "sso": "single sign on",
    # přidej další zkratky podle potřeby…
}


def _normalize(text: str) -> str:
    """Lower-case + rozbalí běžné zkratky před embedováním."""
    t = text.lower()
    for short, full in _ABBREV.items():
        t = t.replace(short, full)
    return t


_CACHE_TTL = 300  # sekund
_EMBED_CACHE: dict[str, tuple[list[float], float]] = {}
_MODEL: OpenAIEmbeddings | None = None


def _get_model() -> OpenAIEmbeddings:
    global _MODEL
    if _MODEL is None:
        _MODEL = OpenAIEmbeddings(model="text-embedding-3-small")
    return _MODEL


def _cached_embedding(text: str) -> list[float]:
    """Embed normalizovaný text a výsledek podrž v paměti max. 5 minut."""
    text = _normalize(text)
    now = time.time()
    if (cache := _EMBED_CACHE.get(text)) and now - cache[1] < _CACHE_TTL:
        return cache[0]

    emb = _get_model().embed_query(text)
    _EMBED_CACHE[text] = (emb, now)
    return emb


def _cosine(u: Sequence[float], v: Sequence[float]) -> float:
    dot = sum(a * b for a, b in zip(u, v))
    nu = math.sqrt(sum(a * a for a in u))
    nv = math.sqrt(sum(b * b for b in v))
    return 0.0 if not nu or not nv else dot / (nu * nv)


# ─────────────────────── Public helper – DUPLICATE CHECK ─────────────────────-

def find_duplicate_ideas(
    summary: str,
    description: str | None = None,
    threshold: float = 0.75,
) -> List[str]:
    """Return keys of Jira Ideas similar to the provided summary."""
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")

    query_text = summary if description is None else f"{summary} {description}"
    query_vec = _cached_embedding(query_text)

    client = JiraClient()
    jql = "issuetype = Idea AND resolution = Unresolved"
    issues = client.search_issues(
        jql, max_results=200, fields=["summary", "description"]
    )

    scored: List[Tuple[str, float]] = []
    for issue in issues:
        key = issue["key"]
        fld = issue.get("fields", {})
        idea_text = f"{fld.get('summary', '')} {fld.get('description', '')}".strip()
        if not idea_text:
            continue
        sim = _cosine(query_vec, _cached_embedding(idea_text))
        if sim >= threshold:
            scored.append((key, sim))

    scored.sort(key=lambda x: x[1], reverse=True)
    return [k for k, _ in scored]

```


---

### `services\rag_service.py`

```python
"""Utilities for working with the RAG vector store."""
from __future__ import annotations

import os
import json
from pathlib import Path
from datetime import datetime
from typing import List

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import DirectoryLoader, ConfluenceLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document


# ───────────────────────── Helper: Robust TXT Saver ───────────────────────────
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)


def save_text_to_file(data: str, prefix: str = "research") -> str:
    """Save *data* into a new TXT file and report the path."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{prefix}_{timestamp}.txt"
    path = OUTPUT_DIR / filename
    path.write_text(data, encoding="utf-8")
    return f"Data saved to {path.as_posix()}"


# ───────────────────────── Vector store builder ─────────────────────────────--

def build_vectorstore(docs_path: str = "./data", persist_directory: str = "rag_chroma_db") -> None:
    """Create a Chroma vector store from text documents."""
    load_dotenv()
    loader = DirectoryLoader(docs_path, glob="**/*.txt")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    Chroma.from_documents(documents=chunks, embedding=embedding_model, persist_directory=persist_directory)


# ───────────────────────── RAG Retriever over Chroma ─────────────────────────-
CHROMA_DIR = "rag_chroma_db"
_embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
_retriever = Chroma(persist_directory=CHROMA_DIR, embedding_function=_embeddings).as_retriever(search_kwargs={"k": 4})


def rag_lookup(query: str) -> str:
    """Return up to 4 relevant documents from the vector store."""
    docs: List[Document] = _retriever.invoke(query)
    if not docs:
        return "Žádné interní dokumenty se k dotazu nenašly."
    return "\n\n".join(d.page_content for d in docs)


# ───────────────────────── Confluence loader ----------------------------------

def load_confluence_pages(config_path: Path | str = "config.json") -> None:
    """Load Confluence pages referenced in *config_path* into the vector store."""
    load_dotenv()
    token = os.getenv("JIRA_AUTH_TOKEN")
    if not token:
        raise RuntimeError("Missing JIRA_AUTH_TOKEN in environment")

    cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
    conf_cfg = cfg.get("confluence", {})
    base_url = conf_cfg.get("base_url")
    user = conf_cfg.get("user")
    ancestor_ids = conf_cfg.get("ancestor_ids", [])
    if not ancestor_ids:
        return

    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
    existing_ids = {
        str(m.get("page_id") or m.get("id"))
        for m in vectorstore._collection.get(include=["metadatas"])["metadatas"]
        if m.get("page_id") or m.get("id")
    }

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    new_chunks: List[Document] = []

    for anc in ancestor_ids:
        loader = ConfluenceLoader(
            url=base_url,
            username=user,
            api_key=token,
            cql=f"ancestor={anc}",
            include_archived_content=False,
            include_restricted_content=False,
        )
        docs = loader.load()
        fresh = []
        for doc in docs:
            pid = str(doc.metadata.get("id") or doc.metadata.get("page_id"))
            if pid in existing_ids:
                continue
            doc.metadata["page_id"] = pid
            fresh.append(doc)
        if not fresh:
            continue
        new_chunks.extend(splitter.split_documents(fresh))

    if new_chunks:
        vectorstore.add_documents(new_chunks)


```


---

### `services\web_service.py`

```python
"""Web search helpers split from the legacy tools module."""
from __future__ import annotations

import os
import openai
from tavily import TavilyClient
from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

# Opt‑out from OpenAI telemetry
os.environ.setdefault("OPENAI_TELEMETRY", "0")
if hasattr(openai, "telemetry") and hasattr(openai.telemetry, "TelemetryClient"):
    openai.telemetry.TelemetryClient.capture = lambda *_, **__: None


# ───────────────────────── General Web Search ────────────────────────────────
_duck = DuckDuckGoSearchRun()


def search_web(query: str) -> str:
    """Return DuckDuckGo search snippets for *query*."""
    return _duck.run(query)


# ───────────────────────── Wikipedia Snippet ----------------------------------
_api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=400)
_wiki_runner = WikipediaQueryRun(api_wrapper=_api_wrapper)


def wiki_snippet(query: str) -> str:
    """Return a short Wikipedia summary for *query*."""
    return _wiki_runner.run(query)


# ───────────────────────── Tavily Semantic Search ────────────────────────────
_TAVILY_KEY = os.getenv("TAVILY_API_KEY")
_client: TavilyClient | None = TavilyClient(api_key=_TAVILY_KEY) if _TAVILY_KEY else None


def tavily_search(query: str) -> str:
    """Perform semantic search via Tavily if configured."""
    if _client is None:
        return "Tavily není nakonfigurováno (chybí TAVILY_API_KEY)."
    try:
        raw = _client.search(query=query, max_results=6)
    except Exception as exc:  # pragma: no cover - external call
        return f"Tavily search selhalo: {exc}"

    results = raw.get("results", [])
    if not results:
        return "Tavily nic nenašlo."
    snippets = [f"- {r['url']}\n  {r['content'][:400].strip()}…" for r in results]
    return "\n\n".join(snippets)

```


---

### `services\__init__.py`

```python
# services/__init__.py
"""Service layer consolidating Jira, RAG and web helpers."""
from .jira_service import JiraClient, JiraClientError, find_duplicate_ideas, _extract_text_from_adf
from .rag_service import (
    save_text_to_file,
    build_vectorstore,
    rag_lookup,
    load_confluence_pages,
)
from .web_service import search_web, wiki_snippet, tavily_search
from .jira_content_service import (
    enhance_idea,
    epic_from_idea,
    user_stories_for_epic,
)
from .input_loader import process_input_files

__all__ = [
    "JiraClient",
    "JiraClientError",
    "find_duplicate_ideas",
    "save_text_to_file",
    "build_vectorstore",
    "rag_lookup",
    "load_confluence_pages",
    "search_web",
    "wiki_snippet",
    "tavily_search",
    "_extract_text_from_adf",
    "jira_content_tools",
    "process_input_files",
]

```


---

### `tools\input_loader.py`

```python
# tools/input_loader.py
"""
LangChain Tool: process_input_files
-----------------------------------
Wrapper, který zpracuje volitelný `arg` od agenta a přepošle ho do služební
funkce.  Díky tomu může agent jemně řídit import (viz docstring v services).
"""
from langchain.tools import Tool
from services.input_loader import process_input_files


def _process_input_files(arg: str = "") -> str:
    """Proxy, která předá argument dál do business-logiky."""
    return process_input_files(arg)


process_input_tool = Tool(
    name="process_input_files",
    func=_process_input_files,
    description=(
        """
        Purpose
        -------
        Scan the local **`./input`** drop-folder, import every file that matches the
        given filter into the Chroma vector database so RAG queries can reference
        them, and return a one-sentence Markdown summary per file.

        Parameters
        ----------
        arg : str, optional — controls behaviour

            • ""            → default full scan (skip already-indexed files)  
            • "force"       → re-index **everything** (ignore fingerprints)  
            • "pdf" / ".pdf"→ import *only* files with this extension  
            • "subdir/foo"  → limit scan to the given (sub)directory  
            • "file.ext"    → import exactly the specified file  

        Returns
        -------
        Markdown report, e.g.:

            ✅ Imported 2 files:
            - report_Q3.pdf: Overview of KPI results for Q3 2024 …
            - shifts.xlsx  : Daily OEE and breakdown statistics …

        Use this tool whenever the user wants to make newly dropped local files
        available to the retrieval pipeline or asks “what did I just upload?”.
        """
    ),
)

__all__ = ["process_input_tool"]

```


---

### `tools\jira_content_tools.py`

```python
# tools/jira_content_tools.py
# ═══════════════════════════
"""
Tool-layer wrapper for the OpenAI-powered Jira content helpers.

These tools turn fuzzy inputs into production-ready Jira tickets in
professional English, following agile / software-engineering best-practices.

Public attribute
----------------
ALL_TOOLS : list[StructuredTool]
    Import and add to your agent's tool list, e.g.:

    from tools.jira_content_tools import ALL_TOOLS
    agent = initialize_agent(
        tools=ALL_TOOLS + other_tools,
        ...
    )
"""
from __future__ import annotations

from typing import List, Literal

from langchain.tools import StructuredTool
from pydantic import BaseModel, Field

from services.jira_content_service import (
    enhance_idea as _enhance_idea,
    epic_from_idea as _epic_from_idea,
    user_stories_for_epic as _user_stories_for_epic,
)

# ──────────────────────────────────────────────────────────────────────────────
# 1. Enhance Idea → Jira Idea
# ──────────────────────────────────────────────────────────────────────────────
class EnhanceIdeaInput(BaseModel):
    """Arguments for the *enhance_idea* tool."""
    summary: str = Field(
        ...,
        description="One-line product idea summary. Keep it short and action-oriented.",
        examples=["Predictive maintenance for packaging line"],
    )
    description: str | None = Field(
        default=None,
        description="Optional free-form description copied from stakeholder notes.",
        examples=[
            "Operators complain about frequent unplanned stops. "
            "We should investigate sensor trends, forecast failures and schedule maintenance."
        ],
    )
    audience: Literal["business", "technical", "mixed"] = Field(
        default="mixed",
        description="Target audience that will read the Idea (affects tone).",
        examples=["business"],
    )
    max_words: int | None = Field(
        default=360,
        description="Hard limit for total word count of the generated description.",
        examples=[100],
    )


def _enhance_idea_tool(
    *,
    summary: str,
    description: str | None = None,
    audience: str = "mixed",
    max_words: int | None = 120,
    ) -> str:
    """Convert a rough product idea into a polished **Jira Idea** ticket body.

    Output format (Markdown):
    - ## Problem
    - ## Proposed solution
    - ## Business value (bullets)
    - ## Acceptance criteria (G/W/T bullets)

    Always writes in professional British English, avoids fluff, and
    adheres to company style (second-level headings, max 5 ACs)."""

    return _enhance_idea(
        summary=summary,
        description=description,
        audience=audience,
        max_words=max_words or 120,
    )


enhance_idea_tool = StructuredTool.from_function(
    name="enhance_idea",
    description=(
        "Transform a **raw product idea** (summary + optional notes) into a concise, "
        "board-ready *Jira Idea* body in Markdown.\n\n"
        "**When to call**  • Any time stakeholder wording is informal, incomplete, "
        "in Czech, or otherwise unfit for an executive audience.\n\n"
        "**Output**  • Four second-level headings — *Problem*, *Proposed solution*, "
        "*Business value*, *Acceptance criteria* — in professional British English, "
        "capped at `max_words` (default 360).\n\n"
        "**Args**\n"
        " • `summary` (STR, required) – one-line tagline.\n"
        " • `description` (STR, optional) – stakeholder context.\n"
        " • `audience` (ENUM) – \"business\", \"technical\", or \"mixed\"; "
        "controls tone and terminology.\n"
        " • `max_words` (INT) – hard length limit.\n\n"
        "Avoid jargon, keep headings exactly as above, respect the word limit."
    ),
    func=_enhance_idea_tool,
    args_schema=EnhanceIdeaInput,
)

# ──────────────────────────────────────────────────────────────────────────────
# 2. Idea → Epic
# ──────────────────────────────────────────────────────────────────────────────
class EpicFromIdeaInput(BaseModel):
    """Arguments for the *epic_from_idea* tool."""
    summary: str = Field(
        ...,
        description="Idea summary that will become the epic goal.",
        examples=["Predictive maintenance for packaging line"],
    )
    description: str | None = Field(
        default=None,
        description="Optional idea description giving context/problem statement.",
    )


def _epic_from_idea_tool(*, summary: str, description: str | None = None) -> str:
    """Produce a complete **Jira Epic** template in Markdown.

    Sections:
    - Epic Goal
    - Context
    - Definition of Done (bullets)
    - Acceptance criteria (G/W/T, ≤7)
    - Out of scope

    The text is direct, neutral, no marketing adjectives."""
    return _epic_from_idea(summary, description)


epic_from_idea_tool = StructuredTool.from_function(
    name="epic_from_idea",
    description=(
        "Expand a validated Idea into a comprehensive *Epic* draft "
        "ready for backlog refinement, with DoD & ACs."
    ),
    func=_epic_from_idea_tool,
    args_schema=EpicFromIdeaInput,
)

# ──────────────────────────────────────────────────────────────────────────────
# 3. Epic → User Stories
# ──────────────────────────────────────────────────────────────────────────────
class UserStoriesForEpicInput(BaseModel):
    """Arguments for the *user_stories_for_epic* tool."""
    epic_name: str = Field(
        ...,
        description="Exact Jira Epic name / goal.",
        examples=["Predictive maintenance platform (Phase I)"],
    )
    epic_description: str = Field(
        ...,
        description="Full epic description (context, DoD, ACs).",
    )
    count: int = Field(
        default=5,
        ge=1,
        le=15,
        description="How many user stories to generate. Default 5.",
    )


def _user_stories_for_epic_tool(
    *, epic_name: str, epic_description: str, count: int = 5
) -> str:
    """Generate INVEST-compliant **User Stories** for the given epic in czech language.

    For each story:
    - Title
    - One-paragraf abstract
    - ‘As a … I want … so that …’
    - Acceptance criteria (G/W/T)
    - T-shirt estimate"""
    return _user_stories_for_epic(epic_name, epic_description, count)


user_stories_for_epic_tool = StructuredTool.from_function(
    name="user_stories_for_epic",
    description=(
        """Generate INVEST-compliant **User Stories** for the given epic in czech language.

        For each story:
        - Title
        - One-paragraf abstract
        - ‘As a … I want … so that …’
        - Acceptance criteria (G/W/T)
        - T-shirt estimate"""
    ),
    func=_user_stories_for_epic_tool,
    args_schema=UserStoriesForEpicInput,
)

# ──────────────────────────────────────────────────────────────────────────────
# Convenience export
# ──────────────────────────────────────────────────────────────────────────────
ALL_TOOLS: List[StructuredTool] = [
    enhance_idea_tool,
    epic_from_idea_tool,
    user_stories_for_epic_tool,
]

__all__ = [
    "enhance_idea_tool",
    "epic_from_idea_tool",
    "user_stories_for_epic_tool",
    "ALL_TOOLS",
]

```


---

### `tools\jira_tools.py`

```python
from __future__ import annotations

import os
import re
import openai
from typing import List, Optional

from langchain.tools import StructuredTool
from pydantic import BaseModel, Field
from requests.exceptions import HTTPError
import difflib

from services import JiraClient, find_duplicate_ideas, _extract_text_from_adf

# Opt‑out from OpenAI telemetry
os.environ.setdefault("OPENAI_TELEMETRY", "0")
if hasattr(openai, "telemetry") and hasattr(openai.telemetry, "TelemetryClient"):
    openai.telemetry.TelemetryClient.capture = lambda *_, **__: None

# Single Jira client instance ---------------------------------------------------
_JIRA = JiraClient()


# ───────────────────────── JIRA Ideas Retriever ─────────────────────────────--
class JiraIdeasInput(BaseModel):
    """Optional keyword filter for JIRA Ideas."""

    keyword: Optional[str] = Field(
        default=None,
        description=(
            "Klíčové slovo pro filtrování Ideas podle summary/description. Pokud None, vrátí vše."
        ),
    )


def _jira_ideas_struct(keyword: Optional[str] = None) -> str:
    try:
        issues = _JIRA.search_issues("project = P4 ORDER BY created DESC", max_results=100)
    except Exception as exc:  # pragma: no cover – user-facing path only
        return f"Chyba při načítání JIRA: {exc}"

    if not issues:
        return "Nenalezeny žádné JIRA Ideas."

    def _plain(issue):
        f = {**issue.get("fields", {}), **{k: v for k, v in issue.items() if k != "fields"}}
        return {
            "key": issue["key"],
            "summary": f.get("summary", ""),
            "description": f.get("description_plain") or f.get("description", ""),
            "status": f.get("status", {}).get("name", ""),
        }

    ideas = [_plain(i) for i in issues]

    if keyword:
        kw = keyword.lower()
        ideas = [i for i in ideas if kw in (i["summary"] + i["description"]).lower()]
        if not ideas:
            return f"Žádné Ideas neobsahují klíčové slovo '{keyword}'."

    lines = [
        f"{i['key']} | {i['status']} | {i['summary']}\n" f"{i['description'] or '- žádný popis -'}"
        for i in ideas
    ]
    return "\n\n".join(lines)


jira_ideas = StructuredTool.from_function(
    func=_jira_ideas_struct,
    name="jira_ideas_retriever",
    description=(
        """
        Purpose
        -------
        List items from the P4 Jira “Ideas” backlog in a readable table-like text form.
        Useful for spotting duplicates manually, expanding an idea into deeper detail,
        or extracting candidate acceptance criteria.

        Parameters
        ----------
        keyword : str | None – optional free-text filter applied to summary + description
                (case-insensitive).  If None, returns the entire backlog (max 100).

        Returns
        -------
        For each match: "KEY | Status | Summary" on one line, followed by the description
        on the next line.
        """
    ),
    args_schema=JiraIdeasInput,
)

# ───────────────────────── Jira Issue Detail ─────────────────────────────────--
class JiraIssueDetailInput(BaseModel):
    """Schema for ``jira_issue_detail`` tool."""

    key: str = Field(..., description="Jira key, e.g. P4-123")


def _format_acceptance_criteria(text: str) -> str:
    pattern = re.compile(r"^\s*(?:\*|-)?\s*(Given|When|Then)\b.*", re.IGNORECASE)
    items = [ln.strip() for ln in text.splitlines() if pattern.match(ln)]
    return "\n".join(f"- {ln.lstrip('*- ').strip()}" for ln in items)


def _jira_issue_detail(key: str) -> str:
    try:
        issue = _JIRA.get_issue(
            key,
            fields=[
                "summary",
                "status",
                "issuetype",
                "labels",
                "description",
                "subtasks",
                "comment",
            ],
        )
    except HTTPError as http_exc:
        code = getattr(http_exc.response, "status_code", None)
        if code in (403, 404) or "does not exist" in str(http_exc).lower():
            return f"Issue {key} neexistuje nebo k němu nemáte přístup."
        return f"HTTP chyba při načítání {key}: {http_exc}"
    except Exception as exc:  # pragma: no cover – other unexpected error
        return f"Chyba při načítání issue {key}: {exc}"

    f = {**issue.get("fields", {}), **{k: v for k, v in issue.items() if k != "fields"}}

    title = f.get("summary", "—")
    status = f.get("status", {}).get("name", "—")
    itype = f.get("issuetype", {}).get("name", "—")
    labels = ", ".join(f.get("labels") or []) or "—"

    from services import _extract_text_from_adf

    raw_desc = f.get("description")
    description_src = f.get("description_plain") or f.get("description")
    description = (
        _extract_text_from_adf(raw_desc)
        if isinstance(description_src, (dict, list))
        else (raw_desc or "—")
    ).strip()

    ac_block = _format_acceptance_criteria(description)

    subtasks = f.get("subtasks") or []
    sub_lines = [
        f"- {st['key']} – {st.get('fields', {}).get('summary', '')}".rstrip()
        for st in subtasks
    ]

    comments = sorted(
        (f.get("comment", {}).get("comments") or []),
        key=lambda c: c.get("created", ""),
        reverse=True,
    )[:3]

    def _fmt_date(ts: str) -> str:
        return ts.split("T")[0] if ts else "—"

    com_lines = [
        f"- **{c.get('author', {}).get('displayName', 'Unknown')}** "
        f"({_fmt_date(c.get('created'))}): {c.get('body', '').strip()}"
        for c in comments
    ]

    parts: list[str] = [
        f"**{key} – {title}**",
        f"Status: {status} | Type: {itype} | Labels: {labels}",
        "",
        "### Description",
        description or "—",
    ]
    if ac_block:
        parts += ["", "### Acceptance Criteria", ac_block]
    if sub_lines:
        parts += ["", "### Sub-tasks", *sub_lines]
    if com_lines:
        parts += ["", "### Latest Comments", *com_lines]

    return "\n".join(parts).strip()


jira_issue_detail = StructuredTool.from_function(
    func=_jira_issue_detail,
    name="jira_issue_detail",
    description=(
        """
        Purpose
        -------
        Fetch a single Jira issue and format it as rich Markdown, giving a 360° snapshot
        for rapid context-building.

        Parameters
        ----------
        key : str – Jira key, e.g. "P4-24".

        Output sections
        ---------------
        • Summary line
        • Status | Type | Labels
        • Full Description (ADF converted to text)
        • Acceptance Criteria (auto-extracted Given/When/Then, if present)
        • Sub-tasks list
        • Up to three latest comments with authors and dates
        """
    ),
    args_schema=JiraIssueDetailInput,
)

# ─────────────── Jira Duplicate-Idea Checker (Structured) --------------------
class DuplicateIdeasInput(BaseModel):
    summary: str = Field(..., description="Krátký popis/summary nové Idea.")
    description: str | None = Field(
        default=None,
        description="(Volitelné) Delší popis nápadu – zahrne se do kontroly duplicity.",
    )
    threshold: float = Field(
        default=0.8,
        ge=0,
        le=1,
        description="Práh kosinové podobnosti 0-1 (výchozí 0.8).",
    )


def _duplicate_ideas(
    summary: str, description: str | None = None, threshold: float = 0.8
) -> str:
    try:
        matches = find_duplicate_ideas(summary, description, threshold)
    except ValueError as exc:  # invalid threshold
        return f"Neplatný parametr `threshold`: {exc}"

    if not matches:
        return "Žádné potenciální duplicity nad daným prahem nenalezeny."
    return "Možné duplicitní nápady: " + ", ".join(matches)


jira_duplicates = StructuredTool.from_function(
    func=_duplicate_ideas,
    name="jira_duplicate_idea_checker",
    description=(
        """
        Purpose
        -------
        Detect whether a *new* idea is likely a **duplicate** of an existing P4 Jira Idea
        using cosine similarity of OpenAI embeddings.

        Parameters
        ----------
        summary     : str   (required) – one-line headline of the proposed idea.
        description : str | None (optional) – longer text; concatenated with summary for
                    embedding.
        threshold   : float       (optional, 0-1, default 0.8) – similarity cutoff;
                    lower for fuzzier matches.

        Returns
        -------
        Either "No potential duplicates above threshold"
        OR a comma-separated list of Jira keys ordered by similarity (highest first).

        Implementation notes
        --------------------
        • Uses text-embedding-3-small.
        • Performs acronym expansion (e.g. 2FA → "two factor authentication").
        • Considers *summary + description* on both sides.
        """
    ),
    args_schema=DuplicateIdeasInput,
)

# ──────────────────────────────────────────────────────────────────────────────
# Jira Update-Description Tool  (human-in-the-loop commit)
# ──────────────────────────────────────────────────────────────────────────────


class UpdateDescriptionInput(BaseModel):
    """Arguments for *jira_update_description* tool."""
    key: str = Field(..., description="Issue key, e.g. 'P4-123'.")
    new_description: str = Field(
        ...,
        description="Entire new description (plain text nebo Markdown – "
                    "Jira ho automaticky převede).",
    )
    confirm: bool = Field(
        default=False,
        description=(
            "⚠️  SECURITY SWITCH – musí být **True**, aby se změna zapsala do Jira. "
            "Pokud False (default), nástroj pouze zobrazí diff a požádá o potvrzení."
        ),
    )


# ──────────────────────────────────────────────────────────────────────────────
def _jira_update_description(
    *, key: str, new_description: str, confirm: bool = False
) -> str:
    """
    Two-step safe update of *description* field:

    1. confirm=False → diff preview
    2. confirm=True  → actual write (now with correct ADF!)
    """
    try:
        old_issue = _JIRA.get_issue(key, fields=["description"])
    except Exception as exc:
        return f"❌ Nelze načíst issue {key}: {exc}"

    old_raw = old_issue.get("fields", {}).get("description") or ""
    old_desc = (
        _extract_text_from_adf(old_raw)
        if isinstance(old_raw, (dict, list))
        else str(old_raw)
    ).strip()

    new_desc = new_description.strip()

    # Step 1 – náhled diffu
    if not confirm:
        diff = "\n".join(
            difflib.unified_diff(
                old_desc.splitlines(),
                new_desc.splitlines(),
                fromfile="aktuální",
                tofile="navrhované",
                lineterm="",
            )
        ) or "*Žádný rozdíl*"
        return (
            f"### Náhled změny popisu pro **{key}**\n"
            f"```diff\n{diff}\n```\n"
            "Toto je **pouze náhled** – nic nebylo uloženo.\n"
            "Chcete-li změnu potvrdit, znovu spusťte `jira_update_description` "
            "se stejnými parametry a `confirm=True`."
        )

    # Step 2 – skutečný zápis přes ADF
    if old_desc == new_desc:
        return "ℹ️ Nový popis je identický se stávajícím – nic se nezměnilo."

    # → zde zabalíme plain text do ADF
    adf_doc = {
        "version": 1,
        "type": "doc",
        "content": [
            {
                "type": "paragraph",
                "content": [
                    {"type": "text", "text": new_desc}
                ],
            }
        ],
    }

    try:
        _JIRA.update_issue(key, {"fields": {"description": adf_doc}})
    except Exception as exc:
        return f"❌ Aktualizace selhala: {exc}"

    return f"✅ Popis issue **{key}** byl úspěšně aktualizován."
# ──────────────────────────────────────────────────────────────────────────────

jira_update_description = StructuredTool.from_function(
    func=_jira_update_description,
    name="jira_update_description",
    description=(
        """
        Safe, human-confirmed update of a Jira issue’s **Description** field.

        Typical workflow
        ----------------
        1. Call tool **without** `confirm` → diff preview is returned.  
        2. If the preview is correct, call again with `confirm=True` to commit.

        Parameters
        ----------
        key            : str   – issue key.  
        new_description: str   – full replacement body (plain/Markdown).  
        confirm        : bool  – must be True to actually save.

        Returns
        -------
        • Preview (`confirm=False`) – unified diff in ```diff``` block.  
        • Commit   (`confirm=True`) – success / error message.
        """
    ),
    args_schema=UpdateDescriptionInput,
)


# ───────────────────────── Jira Child-Issues Retriever ──────────────────────
class ChildIssuesInput(BaseModel):
    """Vrátí přímé child issues (Stories, Tasks…) pod zadaným issue/Epicem."""
    key: str = Field(..., description="Jira key nadřazeného issue, např. 'P4-123'.")

def _jira_child_issues(key: str) -> str:
    """
    Najde všechny *přímé* potomky (parent/„Epic Link“) zadaného issue.

    • Pro Company-managed projekty platí JQL `parent = KEY`
    • Pro Classic projekty (starší Epic Link) `\"Epic Link\" = KEY`
      → kombinujeme obě podmínky, abychom pokryli obě varianty.
    """
    try:
        jql = f'parent = "{key}" OR "Epic Link" = "{key}"'
        issues = _JIRA.search_issues(
            jql,
            max_results=100,
            fields=["summary", "status", "issuetype"],
        )
    except Exception as exc:                        # pragma: no cover
        return f"Chyba při načítání JIRA: {exc}"

    if not issues:
        return f"Issue {key} nemá žádné přímé child issues."

    def _row(i):
        f = i.get("fields", {})
        t = f.get("issuetype", {}).get("name", "—")
        s = f.get("status", {}).get("name", "—")
        return f"{i['key']} | {t} | {s} | {f.get('summary', '')}"

    return "\n".join(_row(i) for i in issues)

jira_child_issues = StructuredTool.from_function(
    func=_jira_child_issues,
    name="jira_child_issues",
    description=(
        """
        Purpose
        -------
        Zobrazí seznam všech přímých child issues (Stories, Tasks, Bugs…)
        pod zadaným nadřazeným issue (typicky Epic).

        Parameters
        ----------
        key : str – Jira key, např. "P4-42".

        Returns
        -------
        Každý řádek: "KEY | IssueType | Status | Summary".
        """
    ),
    args_schema=ChildIssuesInput,
)

# ───────────────────── JIRA Issue-Links Explorer ────────────────────────
from typing import List               # pokud už máte, tento import můžete smazat
from pydantic import BaseModel, Field
from langchain.tools import StructuredTool

class IssueLinksInput(BaseModel):
    """Vypíše všechny vazby (issue links) k danému Jira issue."""
    key: str = Field(..., description="Jira key, např. 'P4-42'.")

def _jira_issue_links(key: str) -> str:
    """
    Vrátí kompletní seznam všech propojených issues a typ vazby.

    • Pro každou linku rozliší směr (inward/outward) podle Jiry.
    • Formát: "KEY | Relation | Type | Status | Summary"
    """
    try:
        issue = _JIRA.get_issue(key, fields=["issuelinks"])
    except Exception as exc:                         # pragma: no cover
        return f"Chyba při načítání JIRA: {exc}"

    links = issue.get("fields", {}).get("issuelinks", [])
    if not links:
        return f"Issue {key} nemá žádné vazby."

    rows: List[str] = []
    for ln in links:
        ltype = ln.get("type", {})
        relation = ltype.get("name", "—")
        # rozlišíme směr
        if "outwardIssue" in ln:
            other = ln["outwardIssue"]
            relation = ltype.get("outward") or relation
        elif "inwardIssue" in ln:
            other = ln["inwardIssue"]
            relation = ltype.get("inward") or relation
        else:
            continue

        okey = other.get("key", "???")
        f = other.get("fields", {})
        otype = f.get("issuetype", {}).get("name", "—")
        ostatus = f.get("status", {}).get("name", "—")
        osummary = f.get("summary", "")

        rows.append(f"{okey} | {relation} | {otype} | {ostatus} | {osummary}")

    return "\n".join(rows)


jira_issue_links = StructuredTool.from_function(
    func=_jira_issue_links,
    name="jira_issue_links",
    description=(
        """
        Purpose
        -------
        Returns all relations (links) of selected Jira issue – duplicates, relates-to,
        blocks, etc. Every relation has information direction (inward/outward).

        Parameters
        ----------
        key : str – Jira key (for example "P4-42").

        Returns
        -------
        One line per relation:
        "KEY | Relation | IssueType | Status | Summary"
        """
    ),
    args_schema=IssueLinksInput,
)


__all__ = [
    "jira_ideas",
    "jira_issue_detail",
    "jira_duplicates",
    "_JIRA",
    "_jira_issue_detail",
    "jira_update_description",
    "jira_child_issues",
    "jira_issue_links",
]

```


---

### `tools\rag_tools.py`

```python
from __future__ import annotations

import os
import openai
from datetime import datetime
from pathlib import Path
from typing import List

from langchain.tools import Tool
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document

# Opt‑out from OpenAI telemetry
os.environ.setdefault("OPENAI_TELEMETRY", "0")
if hasattr(openai, "telemetry") and hasattr(openai.telemetry, "TelemetryClient"):
    openai.telemetry.TelemetryClient.capture = lambda *_, **__: None

# ───────────────────────── Helper: Robust TXT Saver ──────────────────────────
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)


def _save_to_txt(data: str, prefix: str = "research") -> str:
    """Save *data* into a new TXT file and report the path."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{prefix}_{timestamp}.txt"
    path = OUTPUT_DIR / filename
    path.write_text(data, encoding="utf-8")
    return f"Data saved to {path.as_posix()}"


save_tool = Tool(
    name="save_text_to_file",
    func=_save_to_txt,
    description=(
        """
        Purpose
        -------
        Persist any piece of plain text to a timestamped *.txt* file under ./output/.
        Ideal for jotting down idea drafts, SWOT analyses, user-stories, or scratch notes
        that you might want to share later.

        Parameters
        ----------
        data   : str   (required) – the full body of text to save.
        prefix : str   (optional, default "research") – filename prefix; the final name
                is <prefix>_YYYY-MM-DD_HH-MM-SS.txt.

        Returns
        -------
        Confirmation string with the relative file path.  No file object is returned.

        Caveats
        -------
        • Simply writes to disk – it does NOT version existing files or push anything
        back to Jira/Drive.
        """
    ),
)

# ───────────────────────── RAG Retriever over Chroma ─────────────────────────
CHROMA_DIR = "rag_chroma_db"
_embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
_retriever = Chroma(persist_directory=CHROMA_DIR, embedding_function=_embeddings).as_retriever(search_kwargs={"k": 4})


def _rag_lookup(query: str) -> str:
    docs: List[Document] = _retriever.invoke(query)
    if not docs:
        return "Žádné interní dokumenty se k dotazu nenašly."
    return "\n\n".join(d.page_content for d in docs)


rag_tool = Tool(
    name="rag_retriever",
    func=_rag_lookup,
    description=(
        """
        Purpose
        -------
        Semantic Retrieval-Augmented Generation over Productoo’s **internal knowledge**
        base (embedded in a Chroma DB).  Sources include P4 user documentation,
        roadmap pages, and archived conversation snippets.

        Parameters
        ----------
        query : str – a natural-language question or keyword string.

        Returns
        -------
        Up to 4 highly relevant passages concatenated together.

        Typical use cases
        -----------------
        P4 application feature details, implementation scenarios, or roadmap justifications
        to ground an answer before calling an LLM.
        """
    ),
)

__all__ = ["save_tool", "rag_tool"]

```


---

### `tools\web_tools.py`

```python
from __future__ import annotations

import os
import openai
from langchain.tools import Tool
from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from tavily import TavilyClient

# Opt‑out from OpenAI telemetry
os.environ.setdefault("OPENAI_TELEMETRY", "0")
if hasattr(openai, "telemetry") and hasattr(openai.telemetry, "TelemetryClient"):
    openai.telemetry.TelemetryClient.capture = lambda *_, **__: None

# ───────────────────────── General Web Search ────────────────────────────────
_duck = DuckDuckGoSearchRun()
search_tool = Tool(
    name="searchWeb",
    func=_duck.run,
    description=(
        """
        Purpose
        -------
        Fast, general-purpose web lookup powered by DuckDuckGo.  Best for definitions,
        quick facts, fresh news articles, or blog posts.

        Parameters
        ----------
        query : str – the search phrase.

        Returns
        -------
        DDG’s textual snippet(s); no rich metadata or full RSS feed.

        When *not* to use
        -----------------
        For in-depth competitor or market research prefer tavily_search, which does
        semantic ranking.
        """
    ),
)

# ───────────────────────── Wikipedia Snippet ─────────────────────────────────
api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=400)
wiki_tool = Tool(
    name="wikipedia_query",
    func=WikipediaQueryRun(api_wrapper=api_wrapper).run,
    description=(
        """
        Purpose
        -------
        Pull a concise (< 400 chars) summary paragraph from Wikipedia for quick
        background, historical context, industry standards, or terminology.

        Parameters
        ----------
        query : str – topic name or phrase.

        Returns
        -------
        Plain-text summary (no infobox tables or references).
        """
    ),
)

# ───────────────────────── Tavily Semantic Search ────────────────────────────
_TAVILY_KEY = os.getenv("TAVILY_API_KEY")
_client: TavilyClient | None = None
if _TAVILY_KEY:
    _client = TavilyClient(api_key=_TAVILY_KEY)


def _tavily_search(query: str) -> str:
    if _client is None:
        return "Tavily není nakonfigurováno (chybí TAVILY_API_KEY)."

    try:
        raw = _client.search(query=query, max_results=6)
    except Exception as exc:  # pragma: no cover - external call
        return f"Tavily search selhalo: {exc}"

    results = raw.get("results", [])
    if not results:
        return "Tavily nic nenašlo."

    snippets = [f"- {r['url']}\n  {r['content'][:400].strip()}…" for r in results]
    return "\n\n".join(snippets)


tavily_tool = Tool(
    name="tavily_search",
    func=_tavily_search,
    description=(
        """
        Purpose
        -------
        LLM-backed **semantic** web search via the Tavily API – tailored for competitive
        intelligence, white-papers, press releases, or discovering market trends that
        keyword search might miss.

        Parameters
        ----------
        query : str – question or topic.

        Returns
        -------
        Up to 6 items: each entry shows the URL plus a ~400-character snippet.

        Prerequisite
        ------------
        Environment variable TAVILY_API_KEY must be set; otherwise the tool politely
        reports it is unavailable.
        """
    ),
)

__all__ = ["search_tool", "wiki_tool", "tavily_tool"]

```


---

### `tools\__init__.py`

```python
# tools/__init__.py
"""Collection of LangChain tools split into dedicated modules."""
from __future__ import annotations

from .web_tools import search_tool, wiki_tool, tavily_tool
from .rag_tools import save_tool, rag_tool
from .jira_tools import (
    jira_ideas,
    jira_issue_detail,
    jira_duplicates,
    _JIRA,
    _jira_issue_detail,
    jira_update_description,
    jira_child_issues,
    jira_issue_links,
)
from .jira_content_tools import ALL_TOOLS as jira_content_tools
from .input_loader import process_input_tool

__all__ = [
    "search_tool",
    "wiki_tool",
    "rag_tool",
    "jira_ideas",
    "jira_issue_detail",
    "tavily_tool",
    "save_tool",
    "jira_duplicates",
    "_JIRA",
    "_jira_issue_detail",
    "jira_content_tools",
    "process_input_tool",
    "jira_update_description",
    "jira_child_issues",
    "jira_issue_links",
]

```


---

### `utils\io_utils.py`

```python

```
