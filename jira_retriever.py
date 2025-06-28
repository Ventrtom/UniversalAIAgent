import os, json
from pathlib import Path
import requests
from dotenv import load_dotenv

# Načtení konfigurace z .env
load_dotenv()
cfg = json.loads(Path("config.json").read_text())

JIRA_URL       = cfg["jira"]["url"]
JIRA_USER      = cfg["jira"]["user"]
JIRA_TOKEN     = os.getenv("JIRA_AUTH_TOKEN")
JQL            = cfg["jira"]["jql"]
MAX_RESULTS    = cfg["jira"]["maxResults"]


def _extract_text(node):
    """
    Rekurzivně projde ADF uzel nebo seznam a vrátí čistý text.
    """
    text = ""
    if isinstance(node, dict):
        text += node.get("text", "")
        for child in node.get("content", []):
            text += _extract_text(child)
    elif isinstance(node, list):
        for item in node:
            text += _extract_text(item)
    return text


def fetch_jira_issues(jql: str = JQL, max_results: int = MAX_RESULTS):
    """
    Zavolá Jira API a vrátí seznam issue dictů.
    """
    if not JIRA_URL:
        raise RuntimeError("JIRA_URL není nastavená v .env")

    url = f"{JIRA_URL.rstrip('/')}/rest/api/3/search"
    params = {
        "jql": jql,
        "maxResults": max_results,
        "fields": "summary,description,status,labels"
    }
    resp = requests.get(url, params=params, auth=(JIRA_USER, JIRA_TOKEN))
    resp.raise_for_status()

    data = resp.json().get("issues", [])
    issues = []
    for issue in data:
        fields = issue.get("fields", {})
        desc_node = fields.get("description") or {}
        description = _extract_text(desc_node)
        issues.append({
            "key": issue.get("key"),
            "summary": fields.get("summary", ""),
            "description": description,
            "status": fields.get("status", {}).get("name", ""),
            "labels": fields.get("labels", []),
        })
    return issues


if __name__ == "__main__":
    for i in fetch_jira_issues():
        print(f"{i['key']} | {i['status']} | {i['summary']}")
