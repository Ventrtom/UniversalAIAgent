import pytest
import types

import jira_client


# ---------- Test doubles ----------

class DummyEmbeddings:
    """Deterministické embeddingy pro testy (dvě dimenze)."""
    calls: int = 0
    _vectors = {
        "duplicate idea": [1.0, 0.0],
        "duplicate 1":    [1.0, 0.0],
        "different":      [0.0, 1.0],
    }

    def embed_query(self, text: str):
        self.__class__.calls += 1
        return self._vectors.get(text, [0.0, 0.0])


class DummyJira:
    """Vrací dvě otevřené Ideas."""
    def search_issues(self, *_, **__):
        return [
            {"key": "P4-1", "fields": {"summary": "duplicate 1"}},
            {"key": "P4-2", "fields": {"summary": "different"}},
        ]


# ---------- Fixtures ----------

@pytest.fixture(autouse=True)
def _patch_dependencies(monkeypatch):
    monkeypatch.setattr(jira_client, "OpenAIEmbeddings", lambda **_: DummyEmbeddings())
    monkeypatch.setattr(jira_client, "JiraClient", lambda **_: DummyJira())


# ---------- Tests ----------

def test_find_duplicate_ideas_happy():
    res = jira_client.find_duplicate_ideas("duplicate idea", threshold=0.95)
    assert res == ["P4-1"]


def test_find_duplicate_ideas_invalid_threshold():
    with pytest.raises(ValueError):
        jira_client.find_duplicate_ideas("x", threshold=1.1)
