from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.tools import Tool
from datetime import datetime
import openai
import os

os.environ["OPENAI_TELEMETRY"] = "0"
if hasattr(openai, "telemetry") and hasattr(openai.telemetry, "TelemetryClient"):
    openai.telemetry.TelemetryClient.capture = lambda *args, **kwargs: None


def save_to_txt(data: str, filename: str = "research_output.txt"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_text = f"--- Research Output ---\nTimestamp: {timestamp}\n\n{data}\n\n"

    with open(filename, "a", encoding="utf-8") as f:
        f.write(formatted_text)

    return f"Data succesfully saved to {filename}"

save_tool = Tool(
    name = "save_text_to_file",
    func = save_to_txt,
    description = "Saves structured data to a text file.",
)

search = DuckDuckGoSearchRun()
search_tool = Tool(
    name = "searchWeb",
    func = search.run,
    description = "Search the web for information with DuckDuckGo.",
)

api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=100)
wiki_tool = WikipediaQueryRun(api_wrapper=api_wrapper)

CHROMA_DIR = "rag_chroma_db"

retriever = Chroma(
    persist_directory=CHROMA_DIR,
    embedding_function=OpenAIEmbeddings(model="text-embedding-3-small")
).as_retriever(search_kwargs={"k": 3})

rag_tool = Tool(
    name="rag_retriever",
    func=lambda q: "\n\n".join([doc.page_content for doc in retriever.invoke(q)]),
    description="Získá relevantní informace z interní dokumentace pomocí RAG. Použij, pokud dotaz souvisí s tématy jako predictive maintenance, CMMS, atd."
)
