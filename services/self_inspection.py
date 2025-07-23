# services/self_inspection.py

from pathlib import Path
from io import StringIO
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

def generate_project_snapshot(root_dir: Path) -> str:
    # Inspirace merge_project.py
    result = StringIO()
    result.write("## 📁 Struktura\n\n```\n")
    for path in sorted(root_dir.rglob("*")):
        if path.is_file() and not path.name.startswith("."):
            rel_path = path.relative_to(root_dir)
            result.write(f"{rel_path}\n")
    result.write("```\n\n## Obsahy\n")
    for path in sorted(root_dir.rglob("*.py")):
        rel_path = path.relative_to(root_dir)
        try:
            content = path.read_text("utf-8")
        except:
            continue
        result.write(f"\n---\n### `{rel_path}`\n```python\n{content}\n```\n")
    return result.getvalue()

def summarize_agent(snapshot: str) -> str:
    llm = ChatOpenAI(model="gpt-4o", temperature=0.2)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a software architecture analyst. Summarize the architecture, capabilities and tools of the assistant based on this code snapshot."),
        ("human", "{snapshot}")
    ])
    chain = prompt | llm
    return chain.invoke({"snapshot": snapshot}).content

def agent_introspect_tool() -> str:
    root = Path(__file__).parent.parent  # root of project
    snapshot = generate_project_snapshot(root)
    summary = summarize_agent(snapshot)
    timestamp = datetime.now().isoformat()
    return f"#Agent Introspection ({timestamp})\n\n{summary}"
