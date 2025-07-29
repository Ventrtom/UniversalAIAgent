from __future__ import annotations

import json
import os
from pathlib import Path
from datetime import datetime
from typing import Optional, List

from langchain_community.chat_message_histories import FileChatMessageHistory
from langchain.schema import BaseMessage
from langchain.schema.messages import messages_from_dict, messages_to_dict


class RotatingFileChatMessageHistory(FileChatMessageHistory):
    """File-based chat history with simple size-based rotation and indexes."""

    def __init__(
        self,
        file_path: str | Path,
        *,
        encoding: Optional[str] = None,
        ensure_ascii: bool = True,
        max_mb: int = 5,
    ) -> None:
        self.max_bytes = max_mb * 1024 * 1024
        self.file_path = Path(file_path)
        self.encoding = encoding
        self.ensure_ascii = ensure_ascii
        self.index_path = self.file_path.with_suffix(".idx")

        if not self.file_path.exists():
            self.file_path.parent.mkdir(parents=True, exist_ok=True)
            self.file_path.write_text(
                json.dumps([], ensure_ascii=self.ensure_ascii), encoding=self.encoding
            )
        if not self.index_path.exists():
            self.index_path.write_text(
                json.dumps({"timestamp": {}, "session_id": {}}, ensure_ascii=self.ensure_ascii),
                encoding=self.encoding,
            )
        self._load_index()

    def _load_index(self) -> None:
        try:
            self._index = json.loads(self.index_path.read_text(encoding=self.encoding))
        except Exception:
            self._index = {"timestamp": {}, "session_id": {}}

    def _save_index(self) -> None:
        self.index_path.write_text(
            json.dumps(self._index, ensure_ascii=self.ensure_ascii),
            encoding=self.encoding,
        )

    def _rotate(self) -> None:
        ts = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        rotated = self.file_path.with_name(f"{self.file_path.stem}_{ts}{self.file_path.suffix}")
        rotated_idx = rotated.with_suffix(".idx")
        os.rename(self.file_path, rotated)
        os.rename(self.index_path, rotated_idx)
        self.file_path.write_text(
            json.dumps([], ensure_ascii=self.ensure_ascii), encoding=self.encoding
        )
        self.index_path.write_text(
            json.dumps({"timestamp": {}, "session_id": {}}, ensure_ascii=self.ensure_ascii),
            encoding=self.encoding,
        )
        self._index = {"timestamp": {}, "session_id": {}}

    @property
    def messages(self) -> List[BaseMessage]:  # type: ignore[override]
        items = json.loads(self.file_path.read_text(encoding=self.encoding))
        return messages_from_dict(items)

    def add_message(self, message: BaseMessage) -> None:
        messages = messages_to_dict(self.messages)
        record = messages_to_dict([message])[0]
        messages.append(record)
        self.file_path.write_text(
            json.dumps(messages, ensure_ascii=self.ensure_ascii), encoding=self.encoding
        )
        meta = record.get("data", {}).get("additional_kwargs", {})
        ts = meta.get("timestamp")
        sid = meta.get("session_id")
        pos = len(messages) - 1
        if ts is not None:
            self._index["timestamp"][ts] = pos
        if sid is not None:
            self._index["session_id"].setdefault(sid, []).append(pos)
        self._save_index()
        if self.file_path.stat().st_size > self.max_bytes:
            self._rotate()

    def clear(self) -> None:  # type: ignore[override]
        self.file_path.write_text(
            json.dumps([], ensure_ascii=self.ensure_ascii), encoding=self.encoding
        )
        self._index = {"timestamp": {}, "session_id": {}}
        self._save_index()
