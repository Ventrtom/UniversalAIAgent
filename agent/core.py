# agent/core.py
from __future__ import annotations

import os
from datetime import datetime
from typing import List

from dotenv import load_dotenv
import openai
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor

try:  # LangChain >= 0.1.12
    from langchain.memory import (
        ConversationSummaryBufferMemory,
        VectorStoreRetrieverMemory,
        CombinedMemory,
    )
except ImportError:  # starší build
    from langchain.memory.buffer import ConversationSummaryBufferMemory
    from langchain.memory import VectorStoreRetrieverMemory, CombinedMemory

from langchain.schema import Document
from langchain_chroma import Chroma
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

# --------------------------------------------------------------------------- #
# Prostředí & telemetrie
# --------------------------------------------------------------------------- #
load_dotenv()
os.environ.setdefault("OPENAI_TELEMETRY", "0")
if hasattr(openai, "telemetry") and hasattr(openai.telemetry, "TelemetryClient"):
    openai.telemetry.TelemetryClient.capture = lambda *_, **__: None

# --------------------------------------------------------------------------- #
# Modely
# --------------------------------------------------------------------------- #
_MAIN_MODEL = os.getenv("MAIN_LLM", "gpt-4o")
_SUM_MODEL = os.getenv("SUMMARY_LLM", "gpt-3.5-turbo")
_MAIN_TEMP = float(os.getenv("MAIN_TEMPERATURE", 0.3))
_llm = ChatOpenAI(model=_MAIN_MODEL, temperature=_MAIN_TEMP)
_summary_llm = ChatOpenAI(model=_SUM_MODEL, temperature=0.0)

_embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# --------------------------------------------------------------------------- #
# Úložiště
# --------------------------------------------------------------------------- #
RAG_DIR = "rag_chroma_db"
_vectorstore_rag = Chroma(persist_directory=RAG_DIR, embedding_function=_embeddings)

MEM_DIR = "agent_memory_db"
_memory_store = Chroma(persist_directory=MEM_DIR, embedding_function=_embeddings)

# --------------------------------------------------------------------------- #
# Strukturovaný výstup
# --------------------------------------------------------------------------- #
class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]


parser = PydanticOutputParser(pydantic_object=ResearchResponse)

system_prompt = """
You are **Productoo’s senior AI assistant** specialised in manufacturing software P4.
Your mission:
  1. Analyse market trends, the current P4 roadmap and system state.
  2. Propose concrete next steps maximising customer value and minimising tech-debt.
  3. Seamlessly use the available tools when relevant and clearly cite sources.

Guidelines:
- Think step-by-step, reason explicitly yet concisely.
- Ask clarifying questions whenever the user’s request is ambiguous.
- Make answers information-dense—avoid filler.
- Prefer actionable recommendations backed by evidence and quantified impact.
- **After every answer, end with exactly:** _"Ještě něco, s čím mohu pomoci?"_
{format_instructions}
"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),                 # obecná role + instrukce
        ("placeholder", "{chat_history}"),         # Tier-1 souhrnná paměť (list BaseMessage)
        ("system", "{long_term_memory}"),          # Tier-2 semantická paměť (string)
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions())

# --------------------------------------------------------------------------- #
# Paměti
# --------------------------------------------------------------------------- #
episodic_memory = ConversationSummaryBufferMemory(
    llm=_summary_llm,
    memory_key="chat_history",
    return_messages=True,
    max_token_limit=4000,
    input_key="query",
)

long_term_memory = VectorStoreRetrieverMemory(
    retriever=_memory_store.as_retriever(search_kwargs={"k": 5}),
    memory_key="long_term_memory",
)

memory = CombinedMemory(memories=[episodic_memory, long_term_memory])

# --------------------------------------------------------------------------- #
# Agent a nástroje
# --------------------------------------------------------------------------- #
_TOOLS: List = [
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
    memory=memory,
    verbose=True,
    return_intermediate_steps=True,
)

# --------------------------------------------------------------------------- #
# Zápis výměn do dlouhodobé paměti
# --------------------------------------------------------------------------- #
from datetime import datetime  # (opět import – kvůli přehlednosti nad bloky)

def _store_exchange(question: str, answer: str) -> None:
    ts = datetime.utcnow().isoformat()
    docs = [
        Document(page_content=f"USER: {question}", metadata={"role": "user", "ts": ts}),
        Document(page_content=f"ASSISTANT: {answer}", metadata={"role": "assistant", "ts": ts}),
    ]
    _memory_store.add_documents(docs)

# --------------------------------------------------------------------------- #
# Veřejné API
# --------------------------------------------------------------------------- #
def handle_query(query: str) -> str:
    raw = agent_executor.invoke({"query": query})
    full = next(
        (
            obs
            for _act, obs in reversed(raw.get("intermediate_steps", []))
            if isinstance(obs, str) and obs.strip()
        ),
        "",
    )
    answer = full or raw.get("output", "")
    _store_exchange(query, answer)
    return answer


__all__ = ["handle_query", "agent_executor", "ResearchResponse", "parser"]
