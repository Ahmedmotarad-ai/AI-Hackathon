"""
ESC HF 2023 PDF Parser

Extracts text from data/raw/ESC_HF_2023_Guideline.pdf and produces
data/parsed/ESC_parsed_documents.json in the same schema as the existing
NICE parsed JSON.

Uses PyMuPDF (fitz) which is already installed in the environment.
"""

import json
import re
import statistics
from pathlib import Path

import fitz


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PDF_PATH = PROJECT_ROOT / "data" / "raw" / "ESC_HF_2023_Guideline.pdf"
OUTPUT_PATH = PROJECT_ROOT / "data" / "parsed" / "ESC_parsed_documents.json"
DOCUMENT_NAME = "ESC_HF_2023_Guideline.pdf"

# Pages with no extractable text or only whitespace are kept as empty records
# with text="" to preserve page numbering, same as the NICE pipeline.


def clean(text):
    """Minimal text cleaning — preserve content, fix encoding artifacts."""
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def detect_section(text, page_number):
    """
    Detect section headings from ESC page text.

    ESC pages in this summary document have clear structural headings
    like "Document Overview", "Document Scope, Criteria, and Use Cases", etc.

    Strategy:
    - Look for standalone heading lines that appear early in the page
    - A heading is a short line (<= 80 chars) that is followed by structured
      content (key-value pairs, bullet points, or body text)
    - Skip the recurring header line "Design and created by Guideline Central..."

    Returns the detected section name or "Unknown".
    """
    lines = text.split("\n")

    # Skip the recurring header
    header_prefixes = [
        "design and created by",
        "design and created",
    ]

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        # Skip the recurring header line
        if any(stripped.lower().startswith(p) for p in header_prefixes):
            continue

        # A section heading is typically:
        # - Short (<= 80 chars)
        # - Not ending with typical sentence punctuation
        # - Not starting with a bullet or number
        # - Contains alphabetic characters
        if (
            len(stripped) <= 80
            and not stripped.endswith((".", ";", ","))
            and not stripped.startswith(("•", "-", "1", "2", "3", "4", "5",
                                         "6", "7", "8", "9"))
            and re.search(r"[A-Za-z]{3,}", stripped)
            and not stripped.startswith("http")
            and not stripped.startswith("(")
        ):
            # Additional check: the line should look like a heading,
            # not a data value. Headings tend to have few or no special
            # URL characters and no parentheses with long content.
            if "(" not in stripped or stripped.count("(") <= 1:
                # Check it's not a key-value pair (contains ":" with short value)
                if ":" not in stripped or len(stripped.split(":")[0]) > len(stripped) * 0.5:
                    return stripped

    return "Unknown"


def parse_pdf(pdf_path):
    """Parse the ESC PDF and return a list of page records."""
    doc = fitz.open(str(pdf_path))
    page_count = doc.page_count
    records = []

    for page_idx in range(page_count):
        page = doc[page_idx]
        raw_text = page.get_text()
        text = clean(raw_text)
        page_number = page_idx + 1

        section = detect_section(text, page_number)

        records.append({
            "document": DOCUMENT_NAME,
            "page_number": page_number,
            "section": section,
            "text": text,
        })

    doc.close()
    return records, page_count


def validate(records, page_count):
    """Validate the parsed output and print diagnostics."""
    print(f"\n{'=' * 60}")
    print("VALIDATION")
    print(f"{'=' * 60}")

    # 1. Record count vs PDF page count
    print(f"\nPDF page count:       {page_count}")
    print(f"Parsed record count:  {len(records)}")
    if len(records) == page_count:
        print("  PASS: Record count matches PDF page count")
    else:
        print("  FAIL: Record count does not match PDF page count")

    # 2. Required fields
    required_fields = {"document", "page_number", "section", "text"}
    missing_fields = []
    for i, rec in enumerate(records):
        fields = set(rec.keys())
        if not required_fields.issubset(fields):
            missing = required_fields - fields
            missing_fields.append((i + 1, missing))
    if missing_fields:
        print(f"\n  FAIL: {len(missing_fields)} records missing fields:")
        for pg, mf in missing_fields[:5]:
            print(f"    Page {pg}: missing {mf}")
    else:
        print("  PASS: All records have all 4 required fields")

    # 3. Document field validation
    wrong_doc = [r["page_number"] for r in records if r["document"] != DOCUMENT_NAME]
    if wrong_doc:
        print(f"  FAIL: {len(wrong_doc)} records have wrong document field")
    else:
        print(f"  PASS: All records have document = '{DOCUMENT_NAME}'")

    # 4. Page number sequence
    pages = [r["page_number"] for r in records]
    expected = list(range(1, page_count + 1))
    if pages == expected:
        print("  PASS: Page numbers are sequential 1..N")
    else:
        print(f"  FAIL: Page numbers not sequential. Got: {pages}")

    # 5. Empty pages
    empty_pages = [r["page_number"] for r in records if not r["text"].strip()]
    if empty_pages:
        print(f"  INFO: Empty pages: {empty_pages}")
    else:
        print("  INFO: No empty pages")

    # 6. Characters per page
    char_counts = [len(r["text"]) for r in records]
    non_empty = [c for c in char_counts if c > 0]
    if non_empty:
        print(f"\nCharacters per page:")
        print(f"  Mean:   {statistics.mean(char_counts):.0f}")
        print(f"  Median: {statistics.median(char_counts):.0f}")
        print(f"  Min:    {min(char_counts)}")
        print(f"  Max:    {max(char_counts)}")
        print(f"  Total:  {sum(char_counts)}")
    else:
        print("  All pages are empty")

    # 7. Section distribution
    sections = {}
    for r in records:
        s = r["section"]
        sections[s] = sections.get(s, 0) + 1
    print(f"\nSection distribution:")
    for s, count in sorted(sections.items()):
        print(f"  {s}: {count} page(s)")

    return empty_pages


def print_previews(records):
    """Print first, middle, and last page previews."""
    print(f"\n{'=' * 60}")
    print("PAGE PREVIEWS")
    print(f"{'=' * 60}")

    indices = [0]
    if len(records) > 2:
        indices.append(len(records) // 2)
    if len(records) > 1:
        indices.append(len(records) - 1)

    for idx in indices:
        r = records[idx]
        label = {0: "FIRST", len(records) // 2: "MIDDLE", len(records) - 1: "LAST"}
        tag = label.get(idx, "")
        print(f"\n--- {tag} Page (page {r['page_number']}) ---")
        print(f"Section: {r['section']}")
        print(f"Chars:   {len(r['text'])}")
        preview = r["text"][:500]
        if len(r["text"]) > 500:
            preview += "..."
        print(f"Text:\n{preview}")


def main():
    print("=" * 60)
    print("ESC HF 2023 PDF Parser")
    print("=" * 60)
    print(f"\nInput:  {PDF_PATH}")
    print(f"Output: {OUTPUT_PATH}")

    if not PDF_PATH.exists():
        print(f"\nFAIL: PDF not found: {PDF_PATH}")
        return

    # Parse
    records, page_count = parse_pdf(PDF_PATH)
    print(f"\nParsed {len(records)} pages from PDF ({page_count} expected)")

    # Validate
    empty_pages = validate(records, page_count)

    # Previews
    print_previews(records)

    # Write output
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 60}")
    print("OUTPUT")
    print(f"{'=' * 60}")
    print(f"Written {len(records)} records to {OUTPUT_PATH}")
    print(f"File size: {OUTPUT_PATH.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
