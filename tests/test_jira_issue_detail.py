import types

import pytest

import tools


@pytest.fixture
def fake_issue():
    return {
        "key": "P4-123",
        "fields": {
            "summary": "Implement feature X",
            "status": {"name": "In Progress"},
            "issuetype": {"name": "Story"},
            "labels": ["backend", "urgent"],
            "description": "Given user is logged in\nWhen clicking button\nThen modal is shown",
            "subtasks": [
                {"key": "P4-124", "fields": {"summary": "Implement API"}},
                {"key": "P4-125", "fields": {"summary": "Write tests"}},
            ],
            "comment": {
                "comments": [
                    {
                        "author": {"displayName": "Bob"},
                        "created": "2025-06-29T08:00:00.0000000",
                        "body": "Please add unit tests",
                    },
                    {
                        "author": {"displayName": "Alice"},
                        "created": "2025-06-28T12:00:00.0000000",
                        "body": "Looks good",
                    },
                ]
            },
        },
    }


def test_happy_path(monkeypatch, fake_issue):
    monkeypatch.setattr(tools, "_JIRA", types.SimpleNamespace(get_issue=lambda k, fields: fake_issue))
    out = tools._jira_issue_detail("P4-123")
    assert "Implement feature X" in out
    assert "In Progress" in out
    assert "### Sub-tasks" in out
    assert "- **Bob**" in out


def test_issue_not_found(monkeypatch):
    from jira_client import JiraClientError

    def _raise(*_, **__):
        raise JiraClientError("404 Not Found")

    monkeypatch.setattr(tools, "_JIRA", types.SimpleNamespace(get_issue=_raise))
    out = tools._jira_issue_detail("P4-999")
    assert out == "Issue P4-999 not found"

def test_flat_issue(monkeypatch):
    """API may return a flattened dict; tool must still work."""
    flat = {
        "summary": "Do something",
        "status": {"name": "Done"},
        "issuetype": {"name": "Task"},
        "labels": [],
        "description": "Given user…",
        "subtasks": [],
        "comment": {"comments": []},
    }
    monkeypatch.setattr(tools, "_JIRA",
                        types.SimpleNamespace(get_issue=lambda *_, **__: flat))
    out = tools._jira_issue_detail("P4-000")
    assert "**P4-000 – Do something**" in out

