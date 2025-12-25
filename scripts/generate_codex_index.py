import datetime
import json
import re
from pathlib import Path
from typing import Any

# Paths
DOCS_DIR = Path("docs")
SOURCE_FILE = DOCS_DIR / "sagacodex_python_fastapi.md"
OUTPUT_FILE = "sagacodex_index.json"

def parse_codex_markdown(file_path: Path) -> list[dict[str, Any]]:
    content = file_path.read_text(encoding="utf-8")
    rules = []

    # Regex to find rule blocks
    # Looking for "### N. Title" patterns
    rule_headers = re.finditer(r"^### (\d+)\. (.+?)$", content, re.MULTILINE)

    for match in rule_headers:
        rule_num = match.group(1)
        title = match.group(2)

        # Extract the section for this rule
        start_idx = match.end()
        # Find next header or end of file
        next_header_search = re.search(r"^### \d+\.", content[start_idx:], re.MULTILINE)
        if next_header_search:
            end_idx = start_idx + next_header_search.start()
        else:
            end_idx = len(content)

        rule_content = content[start_idx:end_idx].strip()

        # Parse fields from the content
        severity_match = re.search(r"\*\*Severity\*\*: (\w+)", rule_content)
        category_match = re.search(r"\*\*Category\*\*: (\w+)", rule_content)
        tags_match = re.search(r"\*\*Tags\*\*: (.+)", rule_content)
        checklist_match = re.search(r"\*\*Rule\*\*: (.+)", rule_content)

        # Default values if not found (some rules might just be text)
        severity = severity_match.group(1) if severity_match else "CRITICAL" # Assume strict
        category = category_match.group(1) if category_match else "General"
        tags = [t.strip() for t in tags_match.group(1).split(",")] if tags_match else []
        checklist = checklist_match.group(1) if checklist_match else title

        rule = {
            "id": rule_num,
            "title": title,
            "severity": severity,
            "category": category,
            "tags": tags,
            "affected_artifacts": ["all"], # Default for now
            "enforcement_phase": "pre-merge", # Default
            "description": title, # Simplified
            "checklist_item": checklist
        }
        rules.append(rule)

    return rules

def generate_index(rules: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "version": "1.0.0",
        "language": "Python",
        "framework": "FastAPI",
        "generated_at": datetime.datetime.utcnow().isoformat() + "Z",
        "rules": rules
    }

def main():
    if not SOURCE_FILE.exists():
        print(f"Error: Source file {SOURCE_FILE} not found.")
        return

    print(f"Parsing {SOURCE_FILE}...")
    rules = parse_codex_markdown(SOURCE_FILE)
    print(f"Found {len(rules)} rules.")

    index_data = generate_index(rules)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(index_data, f, indent=2)

    print(f"Generated {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
