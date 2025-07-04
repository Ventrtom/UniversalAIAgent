from __future__ import annotations

from datetime import datetime
from typing import Any, List
from uuid import uuid4

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, get_buffer_string
from langchain.memory.chat_memory import BaseChatMemory
from langgraph.checkpoint.memory import MemorySaver

class LangGraphMemory(BaseChatMemory):
    """Simple conversation memory backed by LangGraph's MemorySaver."""

    def __init__(
        self,
        *,
        memory_key: str = "chat_history",
        input_key: str = "query",
        return_messages: bool = True,
        thread_id: str = "default",
    ) -> None:
        super().__init__()
        self.memory_key = memory_key
        self.input_key = input_key
        self.return_messages = return_messages
        self.thread_id = thread_id
        self._saver = MemorySaver()
        self._messages: List[dict] = []

    @property
    def memory_variables(self) -> List[str]:
        return [self.memory_key]

    def _load(self) -> None:
        cfg = {"configurable": {"thread_id": self.thread_id}}
        ck = self._saver.get_tuple(cfg)
        if ck:
            self._messages = ck.checkpoint["channel_values"].get("messages", [])

    def load_memory_variables(self, inputs: dict[str, Any]) -> dict[str, Any]:
        self._load()
        msgs: List[BaseMessage] = [
            HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"])
            for m in self._messages
        ]
        if self.return_messages:
            return {self.memory_key: msgs}
        return {self.memory_key: get_buffer_string(msgs)}

    def save_context(self, inputs: dict[str, Any], outputs: dict[str, str]) -> None:
        self._load()
        self._messages.extend([
            {"role": "user", "content": inputs[self.input_key]},
            {"role": "assistant", "content": outputs.get("output", "")},
        ])
        ck = {
            "v": 1,
            "id": str(uuid4()),
            "ts": datetime.utcnow().isoformat(),
            "channel_values": {"messages": self._messages},
            "channel_versions": {"messages": self._saver.get_next_version(None, None)},
            "versions_seen": {},
        }
        self._saver.put(
            {"configurable": {"thread_id": self.thread_id, "checkpoint_ns": "", "checkpoint_id": ck["id"]}},
            ck,
            {},
            ck["channel_versions"],
        )

    def clear(self) -> None:
        self._messages = []
        self._saver.delete_thread(self.thread_id)
