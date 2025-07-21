README.md file that needsy to be updated accordingly to current project state and structure.

# 🧠 Universal AI Agent (Productoo P4)

*Conversational assistant for product managers, analysts and engineers working on Productoo’s P4 manufacturing suite.*

---

## ✨ What it does now

| Category                      | Status          | Details                                                                                                                 |
| ----------------------------- | --------------- | ----------------------------------------------------------------------------------------------------------------------- |
| **Conversational interface**  | **✅**           | CLI (`main.py`) **and** lightweight Gradio UI (`ui.py`).                                                                |
| **Knowledge retrieval (RAG)** | **✅**           | Chroma vector‑store (default `./data/`, configurable via `CHROMA_DIR_V2`) continuously enriched with every Q\&A turn.                                      |
| **Web search**                | **✅**           | DuckDuckGo (`searchWeb`) & Wikipedia snippet tool.                                                                      |
| **Semantic web search**       | **β**           | Tavily semantic search if `TAVILY_API_KEY` is present.                                                                  |
| **Jira integration**          | **✅**           | `jira_ideas_retriever` – lists *Idea* issues matching an optional keyword.                                              |
| **File output**               | **✅**           | `save_text_to_file` stores each answer in a *new* timestamped file under `./output/`. Visible & downloadable in the UI. |
| **Knowledge base loader**     | **✅**           | `kb_loader` tool syncs Confluence pages and local files into the knowledge base. |
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
| `kb_loader`            | Import Confluence pages and new files from `input/` into the long‑term knowledge base. |

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
└── data/                # vector DB (path via $CHROMA_DIR_V2)
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
