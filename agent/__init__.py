# agent/__init__.py

from importlib import import_module

core = import_module(".core", __name__)

handle_query = core.handle_query
handle_query_stream = core.handle_query_stream
agent_workflow = core.agent_workflow
ResearchResponse = core.ResearchResponse
 
__all__ = [
     "handle_query",
     "handle_query_stream",
     "agent_workflow",
     "ResearchResponse",
    ]
