# 🧠 Universal AI Agent

Conversational assistant tailored for **Productoo's P4** platform. It leverages OpenAI models, a local knowledge base and several Jira utilities to help product teams research, plan and document features.

---

## Key Features

| Area | Description |
| --- | --- |
| **Multi-modal interface** | Command line client (`cli/main.py`) and Gradio web UI (`cli/ui.py` or `run.py`). |
| **Retrieval-Augmented Generation** | Chroma vector store fed from Confluence, local files and previous chats (`CHROMA_DIR_V2` path). |
| **Web search** | DuckDuckGo and Wikipedia snippets with optional semantic search via Tavily. |
| **Jira integration** | Tools for listing Ideas, fetching issue detail, checking duplicates and updating descriptions. |
| **Content generators** | Create Jira Ideas, Epics and User Stories from short prompts. |
| **File ingestion** | Drop files into `./input` and import them with `process_input_files` or `kb_loader`. |
| **Persistent memory** | Chat history saved to `persistent_chat_history.json` and stored in the vector DB. |

---

## Available Tools

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

## Quick Start

```bash
# Install dependencies (preferably inside a virtual environment)
pip install -r requirements.txt

# Set environment variables (.env file or shell)
export OPENAI_API_KEY="sk-..."
export JIRA_AUTH_TOKEN="atlassian-..."
# optional
export TAVILY_API_KEY="..."

# Launch web interface
python run.py

# Or run the simple CLI
python cli/main.py
```

Key environment options:

- `CHROMA_DIR_V2` – path to the Chroma persistence directory (default `data/`).
- `AGENT_VERSION` – set to `v1` to use the older LangChain core; default `v2` uses LangGraph.
- `PERSISTENT_HISTORY_FILE` – JSON file storing chat history (default `persistent_chat_history.json`).

---

## Project Layout

```
agent/                # LangChain/LangGraph cores and public API
cli/                  # CLI and Gradio UI entry points
services/             # Business logic for Jira, RAG and web search
tools/                # LangChain tool wrappers
prompts/              # Jinja2 templates for Jira content generators
input/                # Drop-box for user files
output/               # Saved answers (git‑ignored)
```

---

## Roadmap

- Incremental Confluence synchronisation
- Automatic summarisation of fresh Jira tickets
- Duplicate idea detection improvements
- KPI dashboard and basic tests

Contributions are welcome!
