# utils/md_to_adf.py
from typing import Dict, List
import re

def _inline(text: str) -> List[Dict]:
    """
    Convert **bold**, *italic* and `code` to ADF text nodes with marks.
    Minimal, neřeší vnořování/okrajové případy.
    """
    tokens: List[Dict] = []
    i = 0
    pattern = re.compile(r"(\*\*[^*]+?\*\*|\*[^*]+?\*|`[^`]+?`)")
    for m in pattern.finditer(text):
        if m.start() > i:
            tokens.append({"type": "text", "text": text[i:m.start()]})
        seg = m.group(0)
        if seg.startswith("**"):
            tokens.append({"type": "text", "text": seg[2:-2], "marks": [{"type": "strong"}]})
        elif seg.startswith("*"):
            tokens.append({"type": "text", "text": seg[1:-1], "marks": [{"type": "em"}]})
        else:  # `code`
            tokens.append({"type": "text", "text": seg[1:-1], "marks": [{"type": "code"}]})
        i = m.end()
    if i < len(text):
        tokens.append({"type": "text", "text": text[i:]})
    return tokens


def wrap_markdown(text: str) -> Dict:
    lines = text.strip().splitlines()
    content: List[Dict] = []
    buffer = []

    def flush_paragraph():
        if buffer:
            content.append({
                "type": "paragraph",
                "content": _inline(" ".join(buffer))
            })
            buffer.clear()

    # Heuristika pro sekční hlavičky bez mřížky
    HEAD_RE = re.compile(r"^(Problem|Proposed Solution|Business Value|Acceptance Criteria)\s*:?\s*$", re.I)
    in_business_value = False

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
                "content": _inline(line[level:].strip())
            })
            in_business_value = False
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
                    "content": _inline(item_text)
                }]
            })
            in_business_value = False
        elif HEAD_RE.match(line):
            flush_paragraph()
            # Sekční heading bez '#'
            title = HEAD_RE.match(line).group(1)
            content.append({
                "type": "heading",
                "attrs": {"level": 3},
                "content": _inline(title)
            })
            in_business_value = title.lower().startswith("business value")
        else:
            # Heuristika: po hlavičce "Business Value" ber jednotlivé řádky jako odrážky,
            # i když nemají '-' nebo '*'. Umožní to váš vstup bez pomlček.
            if in_business_value:
                # založ seznam, pokud ještě není
                if not content or content[-1]["type"] != "bulletList":
                    content.append({"type": "bulletList", "content": []})
                content[-1]["content"].append({
                    "type": "listItem",
                    "content": [{
                        "type": "paragraph",
                        "content": _inline(line)
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
            "content": _inline(text.strip())
        }]
    }
