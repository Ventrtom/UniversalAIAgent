from langchain.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv

os.environ["OPENAI_TELEMETRY"] = "0"
import openai
if hasattr(openai, "telemetry") and hasattr(openai.telemetry, "TelemetryClient"):
    openai.telemetry.TelemetryClient.capture = lambda *args, **kwargs: None

load_dotenv()

def main():
    docs_path = os.getenv("DOCS_PATH", "./data")
    persist_directory = os.getenv("PERSIST_DIRECTORY", "rag_chroma_db")

    loader = DirectoryLoader(docs_path, glob="**/*.txt")
    documents = loader.load()
    print(f"Načteno {len(documents)} dokumentů z `{docs_path}`.")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Rozděleno na {len(chunks)} chunků.")

    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory,
    )

    print(f"{len(chunks)} chunků bylo uloženo do vektorové DB: `{persist_directory}`")


if __name__ == "__main__":
    main()
