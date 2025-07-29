import pathlib
import sys
import json

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from utils.rotating_history import RotatingFileChatMessageHistory
from langchain.schema import HumanMessage


def test_rotation(tmp_path):
    path = tmp_path / "history.json"
    hist = RotatingFileChatMessageHistory(path, max_mb=0.0001)
    big_msg = "x" * 2048
    for _ in range(3):
        hist.add_message(HumanMessage(content=big_msg))
    rotated = [p for p in tmp_path.glob("history_*.json") if p.name != "history.json"]
    assert len(rotated) >= 1
    assert path.exists()


def test_indexing(tmp_path):
    path = tmp_path / "history.json"
    hist = RotatingFileChatMessageHistory(path, max_mb=1)
    hist.add_message(HumanMessage(content="hi", additional_kwargs={"timestamp": "t1", "session_id": "s"}))
    idx = path.with_suffix('.idx')
    data = json.loads(idx.read_text())
    assert data["timestamp"]["t1"] == 0
    assert "s" in data["session_id"]
