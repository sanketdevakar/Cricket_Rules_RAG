import re
import json
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_TEXT_PATH = os.path.join(PROJECT_ROOT, "data/extracted/combined.txt")
OUTPUT_JSON_PATH = os.path.join(PROJECT_ROOT, "data/extracted/chunks.json")
os.makedirs(os.path.dirname(OUTPUT_JSON_PATH), exist_ok=True)

# Regex patterns
LAW_FULL_LINE_REGEX = r"^LAW\s*(\d+)\s+(.+)$"   # LAW number + title on same line
LAW_NUMBER_ONLY_REGEX = r"^LAW\s*(\d+)$"        # LAW number only (title on next line)

def load_raw_text(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def chunk_laws(text):
    lines = re.split(r'\n+', text)
    chunks = []
    current_law = None
    law_title_next_line = False

    for line in lines:
        line = line.strip()
        if not line:
            continue

        

        # Case 2: LAW number only (title on next line)
        match_number = re.match(LAW_NUMBER_ONLY_REGEX, line)
        if match_number:
            # Only create a new chunk if current law has no title yet
            if current_law and current_law["law_title"] == "":
                current_law["law_number"] = int(match_number.group(1))
                law_title_next_line = True
            else:
                if current_law:
                    chunks.append(current_law)
                law_number = int(match_number.group(1))
                current_law = {"law_number": law_number, "law_title": "", "text": ""}
                law_title_next_line = True
            continue

        # Next line is law title (from separate line case)
        if law_title_next_line and current_law["law_title"] == "":
            current_law["law_title"] = line
            law_title_next_line = False
            continue

        # Otherwise, append line to law text
        if current_law:
            current_law["text"] += line + "\n"

    # Append last law
    if current_law:
        chunks.append(current_law)

    return chunks

def save_chunks(chunks, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=4, ensure_ascii=False)
    print(f"[INFO] {len(chunks)} law chunks saved to {path}")

def main():
    raw_text = load_raw_text(RAW_TEXT_PATH)
    chunks = chunk_laws(raw_text)
    save_chunks(chunks, OUTPUT_JSON_PATH)

if __name__ == "__main__":
    main()
