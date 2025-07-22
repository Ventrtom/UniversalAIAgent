# agent/__init__.py
"""
Instantiates the Universal AI Agent.
• default: LangGraph core (v2)
• set AGENT_VERSION=v1 to use legacy LangChain core
"""
from importlib import import_module
from typing import TYPE_CHECKING, TypeAlias, Union
import os
from dotenv import load_dotenv

load_dotenv()

_sel = os.getenv("AGENT_VERSION", "v2").lower()

# ──────────────────────────────────────────────────────────────
# 1)  Statická deklarace aliasu – jen při type‑checku
# ──────────────────────────────────────────────────────────────
if TYPE_CHECKING:
    # Importy jsou jen pro mypy/pyright, za běhu se neprovedou.
    from .core import ResearchResponse as _RR1
    from .core2 import ResearchResponse as _RR2

    # Jeden sjednocený alias, který IDE rozpozná
    ResearchResponse: TypeAlias = Union[_RR1, _RR2]

# ──────────────────────────────────────────────────────────────
# 2)  Runtime přiřazení (ignored by type‑checker)
# ──────────────────────────────────────────────────────────────
if not TYPE_CHECKING:
    if _sel == "v1":
        core = import_module(".core", __name__)
        handle_query = core.handle_query
        agent_executor = core.agent_executor
        ResearchResponse = core.ResearchResponse          # type: ignore[assignment]
        parser = core.parser
    else:                                # default = v2
        core2 = import_module(".core2", __name__)
        handle_query = core2.handle_query
        handle_query_stream = core2.handle_query_stream
        agent_workflow = core2.agent_workflow
        ResearchResponse = core2.ResearchResponse         # type: ignore[assignment]
 
__all__ = [
     "handle_query",
     "handle_query_stream",
     "agent_workflow",
     "agent_executor",
     "ResearchResponse",
    ]
