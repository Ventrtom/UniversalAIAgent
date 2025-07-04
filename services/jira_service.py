"""Jira related service layer extracted from legacy jira_client module."""
from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple
from datetime import datetime
from pathlib import Path
import json
import os
import time
import math
import inspect

from atlassian import Jira
from langchain_openai import OpenAIEmbeddings

__all__ = [
    "JiraClient",
    "find_duplicate_ideas",
    "JiraClientError",
    "_extract_text_from_adf",
]

_CFG_PATH = Path("config.json")
_RETRY_SLEEP = (0.5, 1.5, 3.0)  # exponential back-off in seconds


class JiraClientError(RuntimeError):
    """Raised when JiraClient cannot complete a request."""


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
    """Very small ADF→plain-text converter (sufficient for descriptions)."""
    if adf is None:
        return ""
    stack: List[Any] = [adf]
    parts: List[str] = []
    while stack:
        n = stack.pop(0)
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
    """Typed, minimal wrapper around *atlassian-python-api*'s ``Jira``."""

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
            raise ValueError(
                "Missing Jira credentials – provide via arguments, config.json or env vars."
            )
        self._jira = Jira(url=url, username=email, password=token, cloud=True, timeout=timeout)
        self._retries = max_retries

    # ------------------------------------------------------------------ helpers
    def _call(self, func, *args, **kwargs):  # noqa: ANN001 – 3rd-party sig
        for delay in (*_RETRY_SLEEP, None):
            try:
                return func(*args, **kwargs)
            except Exception as exc:  # noqa: BLE001 – surface last
                if delay is None:
                    raise JiraClientError(str(exc)) from exc
                time.sleep(delay)

    # Backwards-compat shim for tests
    _call_with_retry = _call

    # ------------------------------------------------------------------ public
    def search_issues(
        self,
        jql: str,
        *,
        max_results: int = 50,
        fields: Sequence[str] | str | None = None,
    ) -> List[Dict[str, Any]]:
        wanted = _fields_to_string(
            fields or ["summary", "status", "issuetype", "labels", "description"]
        )
        payload: Dict[str, Any] = self._call(
            self._jira.jql, jql, limit=max_results, fields=wanted
        )
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
    
        # ──────────────────────────────────────────────────────────────────
    # JiraClient.update_issue – jediný veřejný mutační entry-point
    # ──────────────────────────────────────────────────────────────────
    def update_issue(
        self,
        key: str,
        data: Dict[str, Any] | None = None,
    ) -> None:
        """
        Update arbitrary fields of a Jira issue (např. description, labels…).

        data = { "fields": {...}, "update": {...} }  # podle REST API
        """
        import inspect   # <- už je naimportován nahoře, přidejte jen pokud chybí

        data = data or {}
        fields_param = data.get("fields")
        update_param = data.get("update")

        # 1️⃣ Vybereme metodu (update_issue › edit_issue › Issue.update)
        if hasattr(self._jira, "update_issue"):
            func = self._jira.update_issue
        elif hasattr(self._jira, "edit_issue"):
            func = self._jira.edit_issue
        else:
            issue = self._call(self._jira.issue, key)
            func = issue.update

        # 2️⃣ Připravíme payload pro „starší“ signaturu (jediný dict)
        payload: Dict[str, Any] = {}
        if fields_param is not None:
            payload["fields"] = fields_param
        if update_param is not None:
            payload["update"] = update_param

        # Pokud posíláme ADF (tj. description=dict), obejdeme SDK
        if (
            fields_param
            and isinstance(fields_param.get("description"), dict)  # ADF poznáme podle dict
        ):
            url = f"{self._jira.url}/rest/api/3/issue/{key}"
            #   self._jira.session je requests.Session() už přihlášený BasicAuth-Tokenem
            resp = self._jira.session.put(url, json={"fields": fields_param})
            if not resp.ok:
                raise JiraClientError(
                    f"Jira update failed: {resp.status_code} {resp.text}"
                )
            return

        # 3️⃣ Zavoláme správnou signaturu podle toho, co metoda opravdu umí
        sig = inspect.signature(func)
        try:
            if "fields" in sig.parameters:
                # Novější atlassian-python-api (>=3.41) – podporuje pojmenované parametry
                self._call(func, key, fields=fields_param, update=update_param)
            else:
                # Starší verze – očekává jediný slovník payload
                self._call(func, key, payload)
        except Exception as exc:                 # noqa: BLE001
            raise JiraClientError(f"Jira update failed: {exc}") from exc





# ────────────────────────── Embed-cache + normalizace ─────────────────────────
_ABBREV = {
    "2fa": "two factor authentication",
    "mfa": "multi factor authentication",
    "sso": "single sign on",
    # přidej další zkratky podle potřeby…
}


def _normalize(text: str) -> str:
    """Lower-case + rozbalí běžné zkratky před embedováním."""
    t = text.lower()
    for short, full in _ABBREV.items():
        t = t.replace(short, full)
    return t


_CACHE_TTL = 300  # sekund
_EMBED_CACHE: dict[str, tuple[list[float], float]] = {}
_MODEL: OpenAIEmbeddings | None = None


def _get_model() -> OpenAIEmbeddings:
    global _MODEL
    if _MODEL is None:
        _MODEL = OpenAIEmbeddings(model="text-embedding-3-small")
    return _MODEL


def _cached_embedding(text: str) -> list[float]:
    """Embed normalizovaný text a výsledek podrž v paměti max. 5 minut."""
    text = _normalize(text)
    now = time.time()
    if (cache := _EMBED_CACHE.get(text)) and now - cache[1] < _CACHE_TTL:
        return cache[0]

    emb = _get_model().embed_query(text)
    _EMBED_CACHE[text] = (emb, now)
    return emb


def _cosine(u: Sequence[float], v: Sequence[float]) -> float:
    dot = sum(a * b for a, b in zip(u, v))
    nu = math.sqrt(sum(a * a for a in u))
    nv = math.sqrt(sum(b * b for b in v))
    return 0.0 if not nu or not nv else dot / (nu * nv)


# ─────────────────────── Public helper – DUPLICATE CHECK ─────────────────────-

def find_duplicate_ideas(
    summary: str,
    description: str | None = None,
    threshold: float = 0.75,
) -> List[str]:
    """Return keys of Jira Ideas similar to the provided summary."""
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")

    query_text = summary if description is None else f"{summary} {description}"
    query_vec = _cached_embedding(query_text)

    client = JiraClient()
    jql = "issuetype = Idea AND resolution = Unresolved"
    issues = client.search_issues(
        jql, max_results=200, fields=["summary", "description"]
    )

    scored: List[Tuple[str, float]] = []
    for issue in issues:
        key = issue["key"]
        fld = issue.get("fields", {})
        idea_text = f"{fld.get('summary', '')} {fld.get('description', '')}".strip()
        if not idea_text:
            continue
        sim = _cosine(query_vec, _cached_embedding(idea_text))
        if sim >= threshold:
            scored.append((key, sim))

    scored.sort(key=lambda x: x[1], reverse=True)
    return [k for k, _ in scored]
