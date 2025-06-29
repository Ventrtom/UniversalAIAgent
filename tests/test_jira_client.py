import os
from typing import Any, Dict, List

import pytest
import responses

from jira_client import JiraClient, JiraClientError, _extract_text_from_adf

BASE_URL = "https://example.atlassian.net"


@pytest.fixture(autouse=True)
def _set_env(monkeypatch):  # noqa: D401 – fixture name is conventional
    """Ensure the auth token is always present during tests."""
    monkeypatch.setenv("JIRA_AUTH_TOKEN", "dummy")
    yield


@pytest.fixture
def client() -> JiraClient:
    """Return a JiraClient instance configured for *BASE_URL*."""
    return JiraClient(url=BASE_URL, email="test@example.com", token="dummy")


# Helper for registering mocked endpoints -------------------------------------------------

def _add_response(method: str, endpoint: str, *, status: int, json_body: Dict[str, Any] | List[Dict[str, Any]] | None = None):
    responses.add(
        getattr(responses, method),
        f"{BASE_URL}{endpoint}",
        json=json_body or {},
        status=status,
        content_type="application/json",
    )


# ---------------------------------------------------------------------------
# Happy‑path tests
# ---------------------------------------------------------------------------


@responses.activate
def test_search_issues_success(client: JiraClient):
    """search_issues returns the parsed issue list and plaintext description."""
    issues_payload: Dict[str, Any] = {
        "issues": [
            {
                "id": "1",
                "key": "P4-1",
                "fields": {
                    "summary": "Hello",
                    "description": {
                        "type": "doc",
                        "content": [
                            {
                                "type": "paragraph",
                                "content": [
                                    {"type": "text", "text": "Hi"},
                                ],
                            }
                        ],
                    },
                },
            }
        ]
    }
    _add_response("GET", "/rest/api/3/search", status=200, json_body=issues_payload)

    result = client.search_issues("project = P4")
    assert len(result) == 1
    assert result[0]["fields"]["description_plain"] == "Hi"


@responses.activate
def test_create_issue_success(client: JiraClient):
    """create_issue forwards the payload and returns the server response."""
    data: Dict[str, Any] = {
        "fields": {
            "summary": "X",
            "project": {"key": "P4"},
            "issuetype": {"name": "Task"},
        }
    }
    _add_response("POST", "/rest/api/3/issue", status=201, json_body={"id": "10001"})

    resp = client.create_issue(data)
    assert resp["id"] == "10001"


@responses.activate
@pytest.mark.parametrize(
    "method,endpoint,call,args",
    [
        ("PUT", "/rest/api/3/issue/P4-1", "update_issue", ("P4-1", {"fields": {"summary": "New"}})),
        ("POST", "/rest/api/3/issue/P4-1/transitions", "transition_issue", ("P4-1", 31)),
        ("POST", "/rest/api/3/issue/P4-1/comment", "add_comment", ("P4-1", "Done!")),
    ],
)
def test_mutating_methods_success(client: JiraClient, method: str, endpoint: str, call: str, args: tuple):
    """update/transition/comment helpers succeed on 2xx status codes."""
    _add_response(method, endpoint, status=204 if method == "PUT" else 201, json_body={})
    getattr(client, call)(*args)


# ---------------------------------------------------------------------------
# Error‑path tests
# ---------------------------------------------------------------------------


@responses.activate
def test_retry_and_fail(client: JiraClient):
    """After 3× 5xx the client raises *JiraClientError*."""
    for _ in range(3):
        _add_response("GET", "/rest/api/3/search", status=500)

    with pytest.raises(JiraClientError):
        client.search_issues("project = P4")


def test_extract_text_from_adf():
    """ADF helper concatenates text nodes."""
    adf = {
        "type": "doc",
        "content": [
            {"type": "paragraph", "content": [{"type": "text", "text": "Hello"}]},
            {"type": "paragraph", "content": [{"type": "text", "text": " world"}]},
        ],
    }
    assert _extract_text_from_adf(adf) == "Hello world"


def test_get_issue_handles_flat_structure(monkeypatch):
    client = JiraClient(url="https://dummy", email="u@e", token="x")
    flat = {"summary": "X", "description": {"type": "doc", "content": []}}
    monkeypatch.setattr(client, "_call_with_retry", lambda *a, **k: flat)
    out = client.get_issue("P4-1")
    assert out["fields"]["summary"] == "X"