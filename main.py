"""main.py — Interactive Universal AI Agent with Self‑Learning RAG
==============================================================
▪ Keeps a running conversation until the user types exit/quit/bye or hits Ctrl‑C.
▪ Remembers dialogue context through ConversationBufferMemory.
▪ After every turn, automatically stores the user ⇄ assistant exchange into the
  Chroma vector‑store (RAG) so future sessions can pick up historical context.

Run with:  $ python main.py
Prereqs:   python‑dotenv, langchain>=0.2, langchain‑openai, langchain‑chroma,
           pydantic>=2, the tools.py module in project root.
"""
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
from langchain.memory import ConversationBufferMemory
from langchain.schema import Document
from langchain_chroma import Chroma
from pydantic import BaseModel

# ── Local project imports ──────────────────────────────────────────────────────
from tools import (
    search_tool,
    wiki_tool,
    save_tool,
    rag_tool,
    tavily_tool,
    jira_ideas,
    jira_issue_detail,
    jira_duplicates
)

# ───────────────────────────── Environment & Telemetry ─────────────────────────
load_dotenv()

os.environ.setdefault("OPENAI_TELEMETRY", "0")
if hasattr(openai, "telemetry") and hasattr(openai.telemetry, "TelemetryClient"):
    openai.telemetry.TelemetryClient.capture = lambda *_, **__: None

# ───────────────────────────── Vectorstore (RAG) setup ─────────────────────────
CHROMA_DIR = "rag_chroma_db"
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)


def _store_exchange(question: str, answer: str) -> None:
    """Save the dialogue pair into the RAG vector store for future retrieval."""
    ts = datetime.utcnow().isoformat()
    docs = [
        Document(page_content=f"USER: {question}", metadata={"role": "user", "ts": ts}),
        Document(page_content=f"ASSISTANT: {answer}", metadata={"role": "assistant", "ts": ts}),
    ]
    vectorstore.add_documents(docs)


# ────────────────────────── Structured output schema ───────────────────────────
class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]


parser = PydanticOutputParser(pydantic_object=ResearchResponse)

# ────────────────────────────── Core prompt template ───────────────────────────
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

# ───────────────────────────── LLM & conversation memory ───────────────────────
llm = ChatOpenAI(model="gpt-4o", temperature=0.3)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# ──────────────────────────────── Agent assembly ───────────────────────────────
TOOLS = [search_tool, wiki_tool, save_tool, rag_tool, tavily_tool, jira_ideas, jira_issue_detail, jira_duplicates]
agent = create_tool_calling_agent(llm=llm, prompt=prompt, tools=TOOLS)
agent_executor = AgentExecutor(agent=agent, tools=TOOLS, memory=memory, verbose=True)

# ─────────────────────────────── CLI interaction loop ──────────────────────────

def main() -> None:  # noqa: D401 – simple function name is fine
    """Launch interactive CLI until user exits."""

    print("\nUniversal AI Agent (Productoo) — napište 'exit' pro ukončení.\n")
    while True:
        try:
            user_query = input("Vy: ").strip()
            if user_query.lower() in {"exit", "quit", "bye"}:
                print("Asistent: Rád jsem pomohl! Mějte se.")
                break

            raw = agent_executor.invoke({"query": user_query})
            answer = raw.get("output", "")

            # Try structured parsing; fall back to raw
            try:
                structured: ResearchResponse = parser.parse(answer)
                print(f"\nAsistent: {structured.summary}\n")
                if structured.sources:
                    print("Zdroj(e):", ", ".join(structured.sources))
            except Exception:
                print("\nAsistent:", answer)

            # Persist exchange into RAG for future retrieval
            _store_exchange(user_query, answer)

        except KeyboardInterrupt:
            print("\nAsistent: Končím. Mějte se.")
            break


if __name__ == "__main__":
    main()
