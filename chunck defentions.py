"""
chunk_definitions.py
────────────────────
Parses Definitions.txt and writes one .txt chunk file per definition
into splitted_chunks/splitted_chunks/ using the same metadata header
format the RAGEngine already expects:

    [المستند N]
    العنوان: <term name>
    التصنيف: تعريفات التأمين
    الرابط: definitions
    الوصف: <first 80 chars of definition>

    <full definition text>

Run once:
    python chunk_definitions.py
"""

import re
import os

DEFINITIONS_FILE = "Definitions.txt"
OUTPUT_DIR = os.path.join("splitted_chunks", "splitted_chunks")
CATEGORY = "تعريفات التأمين"
START_DOC_ID = 5000   # high number so IDs don't clash with existing chunks


def parse_definitions(filepath: str):
    """
    Returns a list of (number, term, definition_text) tuples.
    Handles multi-line definitions that end when the next numbered
    item begins.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        raw = f.read()

    # Match patterns like:  25. Primary care: ...  or  25. Term name\n...
    pattern = re.compile(
        r'^(\d+)\.\s+(.+?)(?=^\d+\.\s|\Z)',
        re.MULTILINE | re.DOTALL
    )

    results = []
    for m in pattern.finditer(raw):
        number = int(m.group(1))
        body = m.group(2).strip()

        # Try to split "Term name: definition body"
        colon_split = body.split(":", 1)
        if len(colon_split) == 2 and len(colon_split[0]) < 80:
            term = colon_split[0].strip()
            definition = colon_split[1].strip()
        else:
            # First line is the term, rest is definition
            lines = body.splitlines()
            term = lines[0].strip().rstrip(":")
            definition = " ".join(l.strip() for l in lines[1:]).strip() or body

        results.append((number, term, definition))

    return results


def write_chunk(output_dir: str, doc_id: int, number: int, term: str, definition: str):
    os.makedirs(output_dir, exist_ok=True)
    filename = f"def_{number:03d}.txt"
    filepath = os.path.join(output_dir, filename)

    short_desc = definition[:80].replace("\n", " ")
    full_text = (
        f"[المستند {doc_id}]\n"
        f"العنوان: {term}\n"
        f"التصنيف: {CATEGORY}\n"
        f"الرابط: definitions\n"
        f"الوصف: {short_desc}\n\n"
        f"{number}. {term}: {definition}"
    )

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(full_text)

    return filename


def main():
    definitions = parse_definitions(DEFINITIONS_FILE)
    print(f"Parsed {len(definitions)} definitions.")

    written = 0
    for number, term, definition in definitions:
        doc_id = START_DOC_ID + number
        fname = write_chunk(OUTPUT_DIR, doc_id, number, term, definition)
        print(f"  ✓ [{number:02d}] {term[:50]:<50} → {fname}")
        written += 1

    print(f"\nDone. {written} chunk files written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()