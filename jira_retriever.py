#!/usr/bin/env python3
import os
import logging
import requests
from typing import List, Dict, Any
from dotenv import load_dotenv

# 1) Načtení .env proměnných
load_dotenv()
JIRA_URL         = os.getenv("JIRA_URL")
JIRA_USER        = os.getenv("JIRA_USER")
JIRA_AUTH_TOKEN  = os.getenv("JIRA_AUTH_TOKEN")
JIRA_JQL         = os.getenv("JIRA_JQL", "project = P4 ORDER BY created DESC")
JIRA_MAX_RESULTS = int(os.getenv("JIRA_MAX_RESULTS", "50"))

# 2) Logger
logger = logging.getLogger("jira_fetcher")
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
logger.addHandler(handler)

def _extract_adf_text(node: Dict[str, Any]) -> str:
    """
    Rekurzivně projde Atlassian Document Format (ADF) a vrátí čistý text.
    """
    text = ""
    if isinstance(node, dict):
        if node.get("text"):
            text += node["text"]
        for child in node.get("content", []):
            text += _extract_adf_text(child)
    elif isinstance(node, list):
        for item in node:
            text += _extract_adf_text(item)
    return text

def fetch_jira_issues(
    jql: str = JIRA_JQL,
    max_results: int = JIRA_MAX_RESULTS
) -> List[Dict[str, Any]]:
    """
    Zavolá Jira REST API /search a vrátí seznam issue dictů:
      { key, summary, description, status, labels }
    """
    url = f"{JIRA_URL}/rest/api/3/search"
    params = {
        "jql": jql,
        "maxResults": max_results,
        "fields": "summary,description,status,labels"
    }
    auth = (JIRA_USER, JIRA_AUTH_TOKEN)
    headers = {"Accept": "application/json"}

    logger.debug(f"GET {url} params={params}")
    resp = requests.get(url, headers=headers, params=params, auth=auth)
    try:
        resp.raise_for_status()
    except requests.HTTPError:
        logger.error(f"HTTP {resp.status_code} – {resp.text}")
        raise

    data = resp.json()
    issues = []
    for issue in data.get("issues", []):
        f = issue.get("fields", {})
        # description může být None nebo ADF dict
        raw_desc = f.get("description")
        if raw_desc:
            description = _extract_adf_text(raw_desc)
        else:
            description = ""
        issues.append({
            "key":         issue.get("key"),
            "summary":     f.get("summary", ""),
            "description": description,
            "status":      f.get("status", {}).get("name", ""),
            "labels":      f.get("labels", []),
        })

    logger.info(f"Načteno {len(issues)} issue(s).")
    return issues

if __name__ == "__main__":
    all_issues = fetch_jira_issues()
    for iss in all_issues:
        print(f"{iss['key']:10} | {iss['status']:15} | {iss['summary']}")
        print(f"  Labels: {', '.join(iss['labels']) or '-'}")
        print(f"  Description:\n{iss['description'] or '- žádný popis -'}")
        print("-" * 80)
