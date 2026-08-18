"""
Generic multi-document PDF parser.

Processes the three valid guideline PDFs and produces a single combined
parsed JSON at data/parsed/03_all_documents_parsed.json.

Uses PyMuPDF (fitz) which is already installed in the environment.

Usage:
    python src/parse_documents.py
"""

import json
import re
import statistics
from pathlib import Path

import fitz


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_PATH = PROJECT_ROOT / "data" / "parsed" / "03_all_documents_parsed.json"

# Whitelist of PDFs to process. Order matters for final output ordering.
TARGET_PDFS = [
    "NICE_HF_2018_Guideline.pdf",
    "ESC_HF_2021_Guideline.pdf",
    "ESC_HF_2023_Focused_Update.pdf",
]

# PDFs that must never be processed
BLACKLIST_PDFS = {
    "ESC_HF_2023_Guideline.pdf",  # old 5-page Guideline Central summary
}


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

def clean(text):
    """Minimal text cleaning — preserve clinical content, fix encoding artifacts."""
    if not text:
        return ""
    # Replace non-breaking space with normal space
    text = text.replace("\u00a0", " ")
    text = text.replace("\xa0", " ")
    # Normalize tabs to spaces
    text = text.replace("\t", " ")
    # Collapse runs of spaces (but not newlines)
    text = re.sub(r"[ ]{2,}", " ", text)
    # Collapse runs of blank lines to at most 2 newlines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Generic section heading detection
# ---------------------------------------------------------------------------

def detect_section(text):
    """
    Conservative generic heading detector.

    Looks for a standalone short line in the first ~30 lines of the page that
    resembles a section heading. Returns "Unknown" when uncertain.

    Rules (all must pass):
      - 3-80 chars after stripping
      - All alphabetic characters (with common punctuation: ampersand, colon,
        slash, parentheses, en/em-dash, roman numerals)
      - Does NOT start with a digit (rules out numbered recommendations,
        table entries, page numbers)
      - Does NOT end with sentence-ending punctuation (. ; , :)
      - Does NOT contain long URL fragments or excessive parentheses
      - Appears in the first 30 lines (headings are near top of page)
    """
    if not text:
        return "Unknown"

    lines = text.split("\n")
    candidate_limit = min(len(lines), 30)

    for line in lines[:candidate_limit]:
        stripped = line.strip()
        if not stripped:
            continue

        # Must be short-ish
        if len(stripped) > 80:
            continue

        # Must start with a letter (rules out numbered items, page nums)
        if not stripped[0].isalpha():
            continue

        # Must NOT end with sentence punctuation
        if stripped[-1] in ".;,:!?)":
            continue

        # Must contain at least one word of 3+ letters
        if not re.search(r"[A-Za-z]{3,}", stripped):
            continue

        # Must NOT look like a URL or contain URL fragments
        if "http" in stripped.lower() or "www." in stripped.lower():
            continue

        # Must NOT have too many parentheses (likely a reference or URL)
        if stripped.count("(") > 2 or stripped.count(")") > 2:
            continue

        # Must NOT be a pure roman numeral (I, II, III, IV, V, etc.)
        if re.match(r"^[IVXLC]+\.?$", stripped):
            continue

        # Must NOT contain mostly non-alpha characters (tables, symbols)
        alpha_ratio = sum(c.isalpha() for c in stripped) / max(len(stripped), 1)
        if alpha_ratio < 0.5:
            continue

        # Looks like a heading
        return stripped

    return "Unknown"


# ---------------------------------------------------------------------------
# Per-document parsing
# ---------------------------------------------------------------------------

def parse_pdf(pdf_path):
    """Parse a single PDF and return (records, page_count)."""
    doc = fitz.open(str(pdf_path))
    page_count = doc.page_count
    records = []

    for page_idx in range(page_count):
        page = doc[page_idx]
        raw_text = page.get_text()
        text = clean(raw_text)
        page_number = page_idx + 1

        section = detect_section(text)

        records.append({
            "document": pdf_path.name,
            "page_number": page_number,
            "section": section,
            "text": text,
        })

    doc.close()
    return records, page_count


# ---------------------------------------------------------------------------
# Per-document validation & preview
# ---------------------------------------------------------------------------

def validate_document(name, records, page_count):
    """Print per-document validation and return stats."""
    print(f"\n{'-' * 60}")
    print(f"  {name}")
    print(f"{'-' * 60}")

    print(f"  PDF page count:      {page_count}")
    print(f"  Parsed record count: {len(records)}")

    if len(records) == page_count:
        print("  PASS: Record count matches PDF page count")
    else:
        print("  FAIL: Record count does NOT match PDF page count")

    # Required fields
    required = {"document", "page_number", "section", "text"}
    missing = []
    for i, rec in enumerate(records):
        if not required.issubset(rec.keys()):
            missing.append((i + 1, required - set(rec.keys())))
    if missing:
        print(f"  FAIL: {len(missing)} records missing fields")
        for pg, mf in missing[:5]:
            print(f"    Page {pg}: missing {mf}")
    else:
        print("  PASS: All records have document/page_number/section/text")

    # Document field
    wrong_doc = [r["page_number"] for r in records if r["document"] != name]
    if wrong_doc:
        print(f"  FAIL: {len(wrong_doc)} records have wrong document field")
    else:
        print(f"  PASS: All records have document = '{name}'")

    # Page number sequence
    pages = [r["page_number"] for r in records]
    expected = list(range(1, page_count + 1))
    if pages == expected:
        print("  PASS: Page numbers sequential 1..N")
    else:
        print(f"  FAIL: Page numbers not sequential. Got: {pages[:10]}...")

    # Empty pages
    empty = [r["page_number"] for r in records if not r["text"].strip()]
    if empty:
        print(f"  INFO: Empty pages: {empty}")
    else:
        print("  INFO: No empty pages")

    # Char stats
    chars = [len(r["text"]) for r in records]
    non_empty = [c for c in chars if c > 0]
    if non_empty:
        print(f"\n  Characters per page:")
        print(f"    Total:  {sum(chars):,}")
        print(f"    Min:    {min(chars):,}")
        print(f"    Median: {statistics.median(chars):,.0f}")
        print(f"    Max:    {max(chars):,}")
    else:
        print("  All pages are empty")

    # Section distribution
    sections = {}
    for r in records:
        sections[r["section"]] = sections.get(r["section"], 0) + 1
    print(f"\n  Sections ({len(sections)} unique):")
    for s, count in sorted(sections.items(), key=lambda x: -x[1]):
        print(f"    {s}: {count} page(s)")

    return {"empty": empty, "chars": chars, "sections": sections}


def print_preview(label, records):
    """Print first, middle, and last page preview for a document."""
    if not records:
        return
    indices = [0]
    if len(records) > 2:
        indices.append(len(records) // 2)
    if len(records) > 1:
        indices.append(len(records) - 1)

    tags = {0: "FIRST"}
    if len(records) > 2:
        tags[len(records) // 2] = "MIDDLE"
    if len(records) > 1:
        tags[len(records) - 1] = "LAST"

    print(f"\n  --- {label} Page Previews ---")
    for idx in indices:
        r = records[idx]
        tag = tags.get(idx, "")
        text_preview = r["text"][:400].replace("\n", " ")
        if len(r["text"]) > 400:
            text_preview += "..."
        print(f"\n  [{tag}] Page {r['page_number']} | Section: {r['section']} | Chars: {len(r['text']):,}")
        print(f"  {text_preview}")


# ---------------------------------------------------------------------------
# Corpus-level validation
# ---------------------------------------------------------------------------

def validate_corpus(all_records, pdf_records):
    """Run corpus-level checks across all documents."""
    print(f"\n{'=' * 60}")
    print("CORPUS VALIDATION")
    print(f"{'=' * 60}")

    # 1. Exactly 3 expected documents
    doc_names = sorted(pdf_records.keys())
    expected = sorted(TARGET_PDFS)
    if doc_names == expected:
        print(f"PASS: Exactly 3 expected documents found: {doc_names}")
    else:
        print(f"FAIL: Expected {expected}, got {doc_names}")

    # 2. Each doc record count matches PDF page count
    all_match = True
    for name, (_, page_count) in pdf_records.items():
        count = sum(1 for r in all_records if r["document"] == name)
        if count != page_count:
            print(f"FAIL: {name} has {count} records but PDF has {page_count} pages")
            all_match = False
    if all_match:
        print("PASS: Every document's record count matches its PDF page count")

    # 3. Page numbers sequential 1..N within each document
    all_seq = True
    for name in pdf_records:
        pages = [r["page_number"] for r in all_records if r["document"] == name]
        expected_pages = list(range(1, len(pages) + 1))
        if pages != expected_pages:
            print(f"FAIL: {name} page numbers not sequential")
            all_seq = False
    if all_seq:
        print("PASS: Page numbers sequential within each document")

    # 4. All records have required fields
    required = {"document", "page_number", "section", "text"}
    missing_count = sum(1 for r in all_records if not required.issubset(r.keys()))
    if missing_count == 0:
        print("PASS: All records have document/page_number/section/text")
    else:
        print(f"FAIL: {missing_count} records missing required fields")

    # 5. No record from old ESC_HF_2023_Guideline.pdf
    bad_doc = [r for r in all_records if r["document"] == "ESC_HF_2023_Guideline.pdf"]
    if not bad_doc:
        print("PASS: No records from blacklisted ESC_HF_2023_Guideline.pdf")
    else:
        print(f"FAIL: {len(bad_doc)} records from blacklisted ESC_HF_2023_Guideline.pdf")

    # 6. No duplicate (document, page_number) pairs
    seen = set()
    dupes = []
    for r in all_records:
        key = (r["document"], r["page_number"])
        if key in seen:
            dupes.append(key)
        seen.add(key)
    if not dupes:
        print("PASS: No duplicate (document, page_number) pairs")
    else:
        print(f"FAIL: {len(dupes)} duplicate pairs: {dupes[:10]}")

    # 7. Total stats
    total_chars = sum(len(r["text"]) for r in all_records)
    total_empty = sum(1 for r in all_records if not r["text"].strip())
    print(f"\nCorpus totals:")
    print(f"  Total records: {len(all_records):,}")
    print(f"  Total chars:   {total_chars:,}")
    print(f"  Empty pages:   {total_empty}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Multi-Document PDF Parser")
    print("=" * 60)
    print(f"\nOutput: {OUTPUT_PATH}")
    print(f"Target PDFs: {TARGET_PDFS}")

    all_records = []
    pdf_records = {}  # name -> (records, page_count)

    for pdf_name in TARGET_PDFS:
        pdf_path = RAW_DIR / pdf_name

        if pdf_name in BLACKLIST_PDFS:
            print(f"\nSKIP (blacklisted): {pdf_name}")
            continue

        if not pdf_path.exists():
            print(f"\nFAIL: PDF not found: {pdf_path}")
            continue

        print(f"\nParsing: {pdf_path}")
        records, page_count = parse_pdf(pdf_path)
        print(f"  Extracted {len(records)} pages ({page_count} expected)")

        pdf_records[pdf_name] = (records, page_count)
        all_records.extend(records)

    # Per-document validation
    print(f"\n{'=' * 60}")
    print("PER-DOCUMENT VALIDATION")
    print(f"{'=' * 60}")

    for pdf_name in TARGET_PDFS:
        if pdf_name in pdf_records:
            records, page_count = pdf_records[pdf_name]
            validate_document(pdf_name, records, page_count)

    # Previews
    print(f"\n{'=' * 60}")
    print("PAGE PREVIEWS")
    print(f"{'=' * 60}")

    for pdf_name in TARGET_PDFS:
        if pdf_name in pdf_records:
            records, _ = pdf_records[pdf_name]
            print_preview(pdf_name, records)

    # Corpus validation
    validate_corpus(all_records, pdf_records)

    # Write output
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_records, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 60}")
    print("OUTPUT")
    print(f"{'=' * 60}")
    print(f"Written {len(all_records):,} records to {OUTPUT_PATH}")
    print(f"File size: {OUTPUT_PATH.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
