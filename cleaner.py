import re

INPUT_FILE = "firecrawl.txt"
OUTPUT_FILE = "cleaned_text.txt"


def parse_markdown_links_and_images(text: str) -> str:
    """
    Character-level parser that:
    - Removes all images: ![alt](url)
    - Replaces links with their text: [text](url) -> text
    Handles nested parentheses correctly (e.g., data URIs with literal parens).
    """
    result = []
    i = 0
    n = len(text)

    def skip_bracket_content(pos: int) -> int:
        """Starting after '[', find and return position after matching ']'."""
        depth = 1
        while pos < n and depth > 0:
            if text[pos] == "[":
                depth += 1
            elif text[pos] == "]":
                depth -= 1
            pos += 1
        return pos

    def skip_paren_content(pos: int) -> int:
        """Starting after '(', find and return position after matching ')'."""
        depth = 1
        while pos < n and depth > 0:
            if text[pos] == "(":
                depth += 1
            elif text[pos] == ")":
                depth -= 1
            pos += 1
        return pos

    def collect_link_inner(pos: int):
        """
        Starting after the opening '[' of a link/image, collect the visible text
        (stripping nested images) and return (text, end_pos) where end_pos is
        after the closing ']'.
        """
        parts = []
        depth = 1
        while pos < n and depth > 0:
            ch = text[pos]
            if ch == "[":
                depth += 1
                parts.append(ch)
                pos += 1
            elif ch == "]":
                depth -= 1
                if depth > 0:
                    parts.append(ch)
                pos += 1
            elif ch == "!" and pos + 1 < n and text[pos + 1] == "[":
                # Nested image inside link text — skip it entirely
                pos += 2  # skip '!['
                pos = skip_bracket_content(pos)  # skip alt text + ']'
                if pos < n and text[pos] == "(":
                    pos += 1
                    pos = skip_paren_content(pos)  # skip URL + ')'
            else:
                parts.append(ch)
                pos += 1
        return "".join(parts).strip(), pos

    while i < n:
        ch = text[i]

        if ch == "!" and i + 1 < n and text[i + 1] == "[":
            # Image: ![alt](url) — drop entirely
            i += 2  # skip '!['
            i = skip_bracket_content(i)  # skip alt + ']'
            if i < n and text[i] == "(":
                i += 1
                i = skip_paren_content(i)  # skip url + ')'
            # else: malformed, just continue

        elif ch == "[":
            # Possible link: [text](url)
            inner_text, after_bracket = collect_link_inner(i + 1)
            if after_bracket < n and text[after_bracket] == "(":
                # It's a link — output just the text, skip the URL
                url_start = after_bracket + 1
                url_end = skip_paren_content(url_start)
                result.append(inner_text)
                i = url_end
            else:
                # Not a link — output as-is
                result.append(ch)
                i += 1

        else:
            result.append(ch)
            i += 1

    return "".join(result)


def clean_text(text: str) -> str:
    # Remove === URL: ... === section headers
    text = re.sub(r"=== URL: .+ ===\n?", "", text)

    # Use the character-level parser to strip images and extract link text
    text = parse_markdown_links_and_images(text)

    # Remove any leftover bare URLs
    text = re.sub(r"https?://\S+", "", text)

    # Remove markdown heading symbols but keep the heading text
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)

    # Remove horizontal rules
    text = re.sub(r"^[-*_]{3,}\s*$", "", text, flags=re.MULTILINE)

    # Remove lines that are clearly URL-encoded SVG/data fragments
    # (lines with dense percent-encoded sequences like %20, %3c etc.)
    lines = []
    for line in text.splitlines():
        pct = line.count("%2") + line.count("%3") + line.count("%20")
        if pct > 3:
            continue
        lines.append(line)
    text = "\n".join(lines)

    # Strip leading/trailing whitespace per line, collapse multiple blank lines
    cleaned_lines = []
    prev_blank = False
    for line in text.splitlines():
        line = line.strip()
        if line == "":
            if not prev_blank:
                cleaned_lines.append("")
            prev_blank = True
        else:
            cleaned_lines.append(line)
            prev_blank = False

    return "\n".join(cleaned_lines).strip()


def main():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        raw = f.read()

    cleaned = clean_text(raw)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(cleaned)

    print(f"Done. Cleaned text written to '{OUTPUT_FILE}'.")
    print(f"  Original length : {len(raw):,} chars")
    print(f"  Cleaned length  : {len(cleaned):,} chars")
    print(f"  Reduction       : {(1 - len(cleaned)/len(raw))*100:.1f}%")


if __name__ == "__main__":
    main()
