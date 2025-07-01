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
