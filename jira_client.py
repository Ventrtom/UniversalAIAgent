
"""jira_client.py – single authoritative client (fixed __init__ defaults).

This version resolves *TypeError: JiraClient.__init__() missing …* by making
all constructor parameters optional and loading missing values from
``config.json`` and the ``JIRA_AUTH_TOKEN`` environment variable.

It also keeps the rich, uniform payload for ``search_issues`` / ``get_issue``
so higher‑level tools display *description*, *status* … correctly.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from atlassian import Jira  # type: ignore

__all__ = ["JiraClient"]

_CFG_PATH = Path("config.json")
_RETRY_SLEEP = (0.5, 1.5, 3.0)  # exponential back‑off in seconds


# ──────────────────────────────────────────────────────────────────────────────
# Utility helpers
# ──────────────────────────────────────────────────────────────────────────────


def _load_cfg() -> Dict[str, Any]:
    if _CFG_PATH.exists():
        with _CFG_PATH.open(encoding="utf-8") as fh:
            return json.load(fh).get("jira", {})
    return {}


def _fields_to_string(fields: Sequence[str] | str | None) -> str:
    if fields is None:
        return "*all"
    if isinstance(fields, str):
        return fields
    return ",".join(fields)


def _extract_text_from_adf(adf: Any) -> str:
    """Very small ADF→plain‑text converter (sufficient for descriptions)."""
    if adf is None:
        return ""
    stack: List[Any] = [adf]
    parts: List[str] = []
    while stack:
        n = stack.pop()
        if isinstance(n, dict):
            if n.get("type") == "text" and "text" in n:
                parts.append(n["text"])
            for child_key in ("content", "items"):
                if isinstance(n.get(child_key), list):
                    stack.extend(n[child_key])
        elif isinstance(n, list):
            stack.extend(n)
    return " ".join(parts).strip()


def _merge(issue: Dict[str, Any]) -> Dict[str, Any]:
    merged = {**issue.get("fields", {})}
    for k, v in issue.items():
        if k not in {"fields", "expand", "id", "key", "self"}:
            merged.setdefault(k, v)
    return merged


# ──────────────────────────────────────────────────────────────────────────────
# Main client – public API only search_issues & get_issue
# ──────────────────────────────────────────────────────────────────────────────


class JiraClient:  # pylint: disable=too-few-public-methods
    """Typed, minimal wrapper around *atlassian‑python‑api*'s ``Jira``."""

    def __init__(
        self,
        url: str | None = None,
        email: str | None = None,
        token: str | None = None,
        *,
        max_retries: int = 3,
        timeout: int = 10,
    ) -> None:
        cfg = _load_cfg()
        url = url or cfg.get("url")
        email = email or cfg.get("user") or cfg.get("email")
        token = token or os.getenv("JIRA_AUTH_TOKEN")
        if not (url and email and token):
            raise ValueError("Missing Jira credentials – provide via arguments, config.json or env vars.")
        self._jira = Jira(url=url, username=email, password=token, cloud=True, timeout=timeout)
        self._retries = max_retries

    # ------------------------------------------------------------------ helpers
    def _call(self, func, *args, **kwargs):  # noqa: ANN001 – 3rd‑party sig
        for delay in (*_RETRY_SLEEP, None):
            try:
                return func(*args, **kwargs)
            except Exception as exc:  # noqa: BLE001 – surface last
                if delay is None:
                    raise
                time.sleep(delay)

    # ------------------------------------------------------------------ public
    def search_issues(
        self,
        jql: str,
        *,
        max_results: int = 50,
        fields: Sequence[str] | str | None = None,
    ) -> List[Dict[str, Any]]:
        wanted = _fields_to_string(fields or [
            "summary",
            "status",
            "issuetype",
            "labels",
            "description",
        ])
        payload: Dict[str, Any] = self._call(self._jira.jql, jql, limit=max_results, fields=wanted)
        issues: List[Dict[str, Any]] = payload.get("issues", [])
        for iss in issues:
            fld = _merge(iss)
            if isinstance(fld.get("description"), (dict, list)):
                plain = _extract_text_from_adf(fld["description"])
                fld["description_plain"] = plain
                fld["description"] = plain
            iss["fields"] = fld
        return issues

    def get_issue(
        self,
        key: str,
        *,
        fields: Sequence[str] | str | None = None,
    ) -> Dict[str, Any]:
        wanted = _fields_to_string(fields or "*all")
        raw: Dict[str, Any] = self._call(self._jira.issue, key, fields=wanted)
        fld = _merge(raw)
        if isinstance(fld.get("description"), (dict, list)):
            plain = _extract_text_from_adf(fld["description"])
            fld["description_plain"] = plain
            fld["description"] = plain
        return {"key": key, "fields": fld}
