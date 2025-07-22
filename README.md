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

| Tool | Purpose |
| --- | --- |
| `searchWeb` | Quick DuckDuckGo lookup. |
| `wikipedia_query` | Short Wikipedia summary. |
| `tavily_search` | Semantic web search (requires `TAVILY_API_KEY`). |
| `rag_retriever` | Retrieve relevant chunks from the internal knowledge base. |
| `jira_ideas_retriever` | List Jira Ideas in project P4. |
| `jira_issue_detail` | Display full Jira issue context (description, AC, subtasks, comments). |
| `jira_duplicate_idea_checker` | Find possible duplicates of a new idea. |
| `jira_update_description` | Two-step safe update of the Description field. |
| `jira_child_issues` | List direct children of an Epic or parent issue. |
| `jira_issue_links` | Show all issue links and their direction. |
| `enhance_idea` | Turn a rough idea into a polished Jira Idea ticket. |
| `epic_from_idea` | Generate a Jira Epic template from an idea. |
| `user_stories_for_epic` | Produce INVEST-style user stories for a given Epic. |
| `process_input_files` | Import files from `./input` into the vector store. |
| `kb_loader` | Synchronise Confluence pages and local files into `kb_docs`. |
| `save_text_to_file` | Persist arbitrary text to a timestamped file in `./output`. |

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
