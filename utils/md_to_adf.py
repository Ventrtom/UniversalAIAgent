# utils/md_to_adf.py
from typing import Dict, List

def wrap_markdown(text: str) -> Dict:
    lines = text.strip().splitlines()
    content: List[Dict] = []
    buffer = []

    def flush_paragraph():
        if buffer:
            content.append({
                "type": "paragraph",
                "content": [{"type": "text", "text": " ".join(buffer)}]
            })
            buffer.clear()

    for line in lines:
        line = line.strip()
        if not line:
            flush_paragraph()
            continue

        if line.startswith("#"):
            flush_paragraph()
            level = min(line.count("#"), 6)
            content.append({
                "type": "heading",
                "attrs": {"level": level},
                "content": [{"type": "text", "text": line[level:].strip()}]
            })
        elif line.startswith(("-", "*")):
            flush_paragraph()
            item_text = line[1:].strip()
            # Pokud předchozí blok nebyl seznam, přidej nový
            if not content or content[-1]["type"] != "bulletList":
                content.append({"type": "bulletList", "content": []})
            content[-1]["content"].append({
                "type": "listItem",
                "content": [{
                    "type": "paragraph",
                    "content": [{"type": "text", "text": item_text}]
                }]
            })
        else:
            buffer.append(line)

    flush_paragraph()

    return {
        "version": 1,
        "type": "doc",
        "content": content or [{
            "type": "paragraph",
            "content": [{"type": "text", "text": text.strip()}]
        }]
    }
