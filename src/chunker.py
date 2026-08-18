import json
import re
import sys
from pathlib import Path

from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT = PROJECT_ROOT / "data" / "parsed" / "03_all_documents_parsed.json"
OUTPUT = PROJECT_ROOT / "data" / "chunks" / "chunks.jsonl"

# Chunks are kept below this size whenever the source page/section allows it.
# Based on BGE tokenizer analysis: avg 4.43 chars/token, targeting ~677 tokens avg.
TARGET_CHARS = 3000

# Absolute token ceiling — no chunk may exceed this after tokenization.
MAX_TOKENS = 450

_tokenizer = None


def _get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-small-en-v1.5")
    return _tokenizer


def token_count(text):
    return len(_get_tokenizer().encode(text, add_special_tokens=True))


# ---------------------------------------------------------------------------
# Safe print (handles Windows cp1252 console)
# ---------------------------------------------------------------------------

def _safe_print(*args, **kwargs):
    """Print with ASCII fallback for non-encodable characters."""
    try:
        print(*args, **kwargs)
    except UnicodeEncodeError:
        safe_args = []
        for a in args:
            if isinstance(a, str):
                safe_args.append(a.encode("ascii", "replace").decode("ascii"))
            else:
                safe_args.append(str(a))
        print(*safe_args, **kwargs)


# ---------------------------------------------------------------------------
# Document metadata
# ---------------------------------------------------------------------------

def _derive_prefix(filename):
    """Derive a chunk-ID prefix from the PDF filename.

    NICE_HF_2018_Guideline.pdf      -> nice_hf_2018
    ESC_HF_2021_Guideline.pdf        -> esc_hf_2021
    ESC_HF_2023_Focused_Update.pdf   -> esc_hf_2023_focused_update
    """
    stem = Path(filename).stem
    parts = stem.split("_")
    # Remove trailing "Guideline" if present
    if parts and parts[-1].lower() == "guideline":
        parts = parts[:-1]
    if len(parts) >= 3:
        prefix_parts = [parts[0].lower(), parts[1].lower(), parts[2].lower()]
        if len(parts) > 3:
            prefix_parts.extend(p.lower() for p in parts[3:])
        return "_".join(prefix_parts)
    return stem.lower().replace(" ", "_").replace("-", "_")


_DOC_META = {
    "NICE_HF_2018_Guideline.pdf": {
        "guideline_family": "NICE_HF",
        "guideline_year": 2018,
        "guideline_type": "full_guideline",
    },
    "ESC_HF_2021_Guideline.pdf": {
        "guideline_family": "ESC_HF",
        "guideline_year": 2021,
        "guideline_type": "full_guideline",
        "superseded_by": "ESC_HF_2023_Focused_Update.pdf",
    },
    "ESC_HF_2023_Focused_Update.pdf": {
        "guideline_family": "ESC_HF",
        "guideline_year": 2023,
        "guideline_type": "focused_update",
        "parent_guideline": "ESC_HF_2021_Guideline.pdf",
    },
}


# ---------------------------------------------------------------------------
# NICE-specific section detection (preserved from original)
# ---------------------------------------------------------------------------

MAIN_SECTIONS = {
    "1.1": "1.1 Team working in the management of heart failure",
    "1.2": "1.2 Diagnosing heart failure",
    "1.3": "1.3 Giving information to people with heart failure",
    "1.4": "1.4 Treating people with newly diagnosed and pre-existing heart failure with reduced ejection fraction",
    "1.5": "1.5 Treating people with newly diagnosed and pre-existing heart failure with mildly reduced or preserved ejection fraction",
    "1.6": "1.6 Treating heart failure in people with chronic kidney disease",
    "1.7": "1.7 Starting and monitoring medication use",
    "1.8": "1.8 Clinical review",
    "1.9": "1.9 Other treatments and advice for all types of heart failure",
    "1.10": "1.10 Interventional procedures",
    "1.11": "1.11 Cardiac rehabilitation",
    "1.12": "1.12 Palliative care",
}

OTHER_SECTIONS = [
    "Terms used in this guideline",
    "Recommendations for research",
    "Key recommendations for research",
    "Rationale and impact",
    "Context",
    "Finding more information and committee details",
    "Update information",
]

SUBSECTIONS = [
    "Symptoms, signs and investigations",
    "Heart failure caused by valve disease",
    "Reviewing existing diagnoses",
    "First consultations for people with newly diagnosed heart failure",
    "Treatment combinations",
    "Alternative treatment combinations if certain medicines are not tolerated",
    "Intravenous iron therapy",
    "Specialist treatment",
    "Ivabradine",
    "Hydralazine in combination with nitrate",
    "Digoxin",
    "Calcium-channel blockers",
    "Mildly reduced ejection fraction",
    "Preserved ejection fraction",
    "Tailoring treatment",
    "ACE inhibitors, ARNIs, ARBs and MRAs",
    "Beta-blockers",
    "People under 75 with normal renal function",
    "Diuretics",
    "Amiodarone",
    "Anticoagulants",
    "Vaccinations",
    "Salt and fluid restriction",
    "Smoking and alcohol",
    "Air travel",
    "Driving",
    "Contraception and pregnancy",
    "Depression",
    "Resynchronisation therapy",
    "Coronary revascularisation",
    "Cardiac transplantation",
    "Implantable cardioverter defibrillators and cardiac resynchronisation therapy",
    "Care after an acute event",
    "Writing a care plan",
    "Key facts and figures",
    "Current practice",
]

_NICE_HEADING_PATTERNS = [
    (r"^Care after an acute event\b", "Care after an acute event"),
    (r"^Writing a care plan\b", "Writing a care plan"),
    (r"^Symptoms, signs and investigations\b", "Symptoms, signs and investigations"),
    (r"^Preserved ejection fraction\b", "Preserved ejection fraction"),
    (r"^Tailoring treatment\b", "Tailoring treatment"),
    (r"^Beta-blockers\b", "Beta-blockers"),
    (r"^Vaccinations\b", "Vaccinations"),
    (r"^Smoking and alcohol\b", "Smoking and alcohol"),
    (r"^Specialist treatment\b", "Specialist treatment"),
]


def _detect_sections_nice(raw_text, page):
    """NICE-specific section detection (original logic preserved)."""
    if page <= 5:
        return [("Front matter", raw_text, page)]

    lines = raw_text.splitlines()
    starts = []

    for i, line in enumerate(lines):
        s = line.strip()

        match = re.match(r"^(1\.(?:10|11|12|[1-9]))\s+", s)
        if match:
            starts.append((i, MAIN_SECTIONS[match.group(1)]))
            continue

        if re.match(r"^Finding more information and committee\b", s, re.I):
            starts.append((i, "Finding more information and committee details"))
            continue

        for heading in OTHER_SECTIONS:
            if s.lower() == heading.lower():
                starts.append((i, heading))
                break

        for pattern, heading in _NICE_HEADING_PATTERNS:
            if re.match(pattern, s, re.I):
                starts.append((i, heading))
                break

    starts = sorted(set(starts))

    if not starts:
        return [(None, raw_text, page)]

    parts = []
    if starts[0][0] > 0:
        prefix = clean("\n".join(lines[:starts[0][0]]))
        if prefix:
            parts.append((None, prefix, page))

    for j, (start, section) in enumerate(starts):
        end = starts[j + 1][0] if j + 1 < len(starts) else len(lines)
        part = clean("\n".join(lines[start:end]))
        if part:
            parts.append((section, part, page))

    return parts


# ---------------------------------------------------------------------------
# ESC section detection (conservative, structure-aware)
# ---------------------------------------------------------------------------

# Known ESC section headings from the 2021 and 2023 guidelines.
# Using a whitelist approach for reliability.
_ESC_KNOWN_SECTIONS = [
    "List of figures",
    "List of tables",
    "Table of contents",
    "Abbreviations and acronyms",
    "Definitions",
    "Heart failure with reduced ejection fraction",
    "Heart failure with mildly reduced ejection fraction",
    "Heart failure with preserved ejection fraction",
    "Epidemiology and diagnosis",
    "Diagnosis",
    "Treatment",
    "Recommendations",
    "Recommendations for the prevention of chronic HF",
    "Recommendations for the management of patients with HFrEF",
    "Recommendations for the treatment of patients with HFmrEF",
    "Recommendations for the treatment of patients with HFpEF",
    "Recommendations for the treatment of patients with advanced HF",
    "Recommendations for management of patients with AHF",
    "Recommendations for the management of patients with HF and comorbidities",
    "Recommendations for the treatment of transthyretin amyloid cardiomyopathy",
    "Recommendations for the treatment of patients with HF and aortic stenosis",
    "Recommendations for the management of patients receiving potential cardiotoxic treatments",
    "General aspects",
    "Heart transplantation",
    "Left ventricular assist devices",
    "Palliative care",
    "Endomyocardial biopsy",
    "Suspected heart failure",
    "Management of patients with HFrEF",
    "Management of patients with HFmrEF",
    "Management of patients with HFpEF",
    "Management of patients with advanced heart failure",
    "Management of patients with acute heart failure",
    "Management of patients with isolated right ventricular failure",
    "Management of atrial fibrillation in patients with HFrEF",
    "Management of patients with heart failure and diabetes",
    "Management of patients with heart failure and anaemia/iron deficiency",
    "Management of patients with renal dysfunction",
    "Management of patients with concurrent valvular heart disease",
    "Management of patients with heart failure and cancer therapies",
    "Management of patients with heart failure and cardiac amyloidosis",
    "Management of patients with heart failure and inherited cardiomyopathies",
    "Management of patients with heart failure and myocarditis",
    "Management of patients with concomitant diseases",
    "Pregnancy and heart failure",
    "Coronary artery disease",
    "Atrial fibrillation",
    "Aortic stenosis",
    "Mitral regurgitation",
    "Tricuspid regurgitation",
    "Pulmonary embolism",
    "Lung ultrasound",
    "Left ventricular hypertrophy",
    "Secondary prevention of sudden cardiac death",
    "Cardiac rehabilitation",
    "Vaccinations",
    "Invasive and non-invasive management",
    "Acute heart failure",
    "End-of-life care",
    "Patient assessment",
    "Initial assessment",
    "Diagnostic workup",
    "Risk stratification",
    "Disposition and planning",
    "Monitoring and follow-up",
    "Symptom control and end-of-life care",
    "Recommendation Table 1",
    "Recommendation Table 2",
    "Recommendation Table 3",
    "Recommendation Table 4",
    "Recommendation Table 5",
    "Recommendation Table 6",
    "Recommendation Table 7",
    "Recommendation Table 8",
    "Recommendation Table 9",
    "Recommendation Table 10",
    "Recommendation Table 11",
    "Recommendation Table 12",
    "Recommendation Table 13",
    "Recommendation Table 14",
    "Recommendation Table 15",
    "Rate control",
    "Rhythm control",
    "Sinus rhythm",
    "Loop diuretics",
    "Mineralocorticoid receptor antagonists",
    "Angiotensin receptor-neprysilin inhibitors",
    "Angiotensin receptor-neprilysin inhibitor",
    "Beta-blockers",
    "Sodium-glucose co-transporter 2 inhibitors",
    "Ivabradine",
    "Hydralazine and nitrate",
    "Digoxin",
    "Cardiac resynchronisation therapy",
    "Implantable cardioverter-defibrillators",
    "Short-term mechanical circulatory support",
    "Education topic",
    "Goal for the patient and caregiver",
    "Professional behaviour and educational tools",
    "Admission, during",
    "Pre-discharge",
    "After discharge",
    "Recommended for prognosis",
    "Renal replacement",
    "Pulmonary hypertension",
    "Iron deficiency",
    "Obesity",
    "Sleep-disordered breathing",
    "Diabetes",
    "Recommendations for management of patients with HF and",
]


def _detect_sections_esc(raw_text, page, document):
    """Conservative section detection for ESC documents.

    Uses a whitelist of known section headings plus strict structural
    heuristics. Returns 'Unknown' when uncertain.
    """
    if not raw_text.strip():
        return [("Unknown", raw_text, page)]

    lower_text = raw_text.lower()

    # Front matter detection for ESC docs
    if page <= 2:
        if any(kw in lower_text for kw in [
            "task force", "disclosure", "writing committee",
            "table of contents", "contents", "author",
            "document overview", "guideline central",
            "supplemental implementation", "conflict of interest",
        ]):
            return [("Front matter", raw_text, page)]

    lines = raw_text.splitlines()
    starts = []

    for i, line in enumerate(lines):
        s = line.strip()
        if not s:
            continue

        # Check against known section whitelist (exact or prefix match)
        for known in _ESC_KNOWN_SECTIONS:
            if s.lower().startswith(known.lower()):
                starts.append((i, known))
                break
        else:
            # "Recommendation Table N" pattern
            m = re.match(r"^(Recommendation Table \d+)", s, re.I)
            if m:
                starts.append((i, m.group(1)))
                continue

            # "Table N:" pattern for actual data tables used as sections
            m = re.match(r"^(Table \d+)\s*[:\-]", s)
            if m and len(s) < 60:
                starts.append((i, m.group(1)))
                continue

    starts = sorted(set(starts))

    if not starts:
        return [(None, raw_text, page)]

    parts = []
    if starts[0][0] > 0:
        prefix = clean("\n".join(lines[:starts[0][0]]))
        if prefix:
            parts.append((None, prefix, page))

    for j, (start, section) in enumerate(starts):
        end = starts[j + 1][0] if j + 1 < len(starts) else len(lines)
        part = clean("\n".join(lines[start:end]))
        if part:
            parts.append((section, part, page))

    return parts


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

def clean(text):
    text = text.replace("\u00a0", " ")
    text = re.sub(r"conditions#notice-of-rights\)\.?(?:\s*39)?", "", text, flags=re.I)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Per-document page splitting
# ---------------------------------------------------------------------------

def split_page(item):
    """Split a page into section-labeled parts."""
    document = item["document"]
    page = int(item["page_number"])
    raw_text = item["text"]

    if document == "NICE_HF_2018_Guideline.pdf":
        return _detect_sections_nice(raw_text, page)
    else:
        return _detect_sections_esc(raw_text, page, document)


def infer_subsection(text, section):
    for heading in SUBSECTIONS:
        if heading != section and re.search(
            rf"(?m)^{re.escape(heading)}\s*$", text
        ):
            return heading
    return None


# ---------------------------------------------------------------------------
# Main chunking pipeline
# ---------------------------------------------------------------------------

def main():
    raw = json.loads(INPUT.read_text(encoding="utf-8"))

    # Group by document, preserving input order
    doc_order = []
    doc_pages = {}
    for item in raw:
        doc = item["document"]
        if doc not in doc_pages:
            doc_order.append(doc)
            doc_pages[doc] = []
        doc_pages[doc].append(item)

    all_chunks = []

    for doc_name in doc_order:
        pages = doc_pages[doc_name]
        prefix = _derive_prefix(doc_name)
        meta = _DOC_META.get(doc_name, {})

        # Reset section state for each document
        current_section = "Front matter"
        segments = []

        for item in pages:
            for section, text, page in split_page(item):
                section = section or current_section
                current_section = section
                if text:
                    segments.append((section, text, page, doc_name))

        # Buffer segments by section
        chunks = []
        buffer_text = ""
        buffer_pages = []
        buffer_section = None

        def emit(text, pages, section, document):
            text = text.strip()
            if not text:
                return

            subsection = infer_subsection(text, section)
            section_path = [section]
            if subsection:
                section_path.append(subsection)

            chunk = {
                "chunk_id": f"{prefix}_chunk_{len(chunks) + 1:04d}",
                "text": text,
                "document": document,
                "section": section,
                "section_path": section_path,
                "page": min(pages),
                "page_start": min(pages),
                "page_end": max(pages),
            }
            chunk.update(meta)
            chunks.append(chunk)

        for section, text, page, doc_name_seg in segments:
            if buffer_section is None:
                buffer_section = section

            if section != buffer_section:
                emit(buffer_text, buffer_pages, buffer_section, doc_name)
                buffer_text = ""
                buffer_pages = []
                buffer_section = section

            candidate = (
                f"{buffer_text}\n\n{text}".strip()
                if buffer_text else text
            )

            if len(candidate) <= TARGET_CHARS:
                buffer_text = candidate
                buffer_pages.append(page)
            else:
                emit(buffer_text, buffer_pages, buffer_section, doc_name)
                buffer_text = text
                buffer_pages = [page]

        emit(buffer_text, buffer_pages, buffer_section, doc_name)

        # Phase 1: character-based split
        final_chunks = []
        for chunk in chunks:
            text = chunk["text"]

            while len(text) > TARGET_CHARS:
                cut = text.rfind("\n", 0, TARGET_CHARS)
                if cut < 500:
                    cut = text.rfind(". ", 0, TARGET_CHARS)

                if cut < 500:
                    cut = TARGET_CHARS

                part = text[:cut].strip()
                text = text[cut:].strip()

                new_chunk = dict(chunk)
                new_chunk["text"] = part
                final_chunks.append(new_chunk)

            if text:
                new_chunk = dict(chunk)
                new_chunk["text"] = text
                final_chunks.append(new_chunk)

        # Phase 2: token-based split for dense-formatting pages
        token_chunks = []
        for chunk in final_chunks:
            text = chunk["text"]
            if token_count(text) <= MAX_TOKENS:
                token_chunks.append(chunk)
                continue

            lines = text.split("\n")
            current = ""
            for line in lines:
                candidate = f"{current}\n{line}".strip() if current else line
                if token_count(candidate) > MAX_TOKENS and current:
                    new_chunk = dict(chunk)
                    new_chunk["text"] = current.strip()
                    token_chunks.append(new_chunk)
                    current = line
                else:
                    current = candidate
            if current.strip():
                new_chunk = dict(chunk)
                new_chunk["text"] = current.strip()
                token_chunks.append(new_chunk)

        # Renumber chunk IDs within this document
        for i, chunk in enumerate(token_chunks, start=1):
            chunk["chunk_id"] = f"{prefix}_chunk_{i:04d}"

        all_chunks.extend(token_chunks)

    # Write output
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    _validate(all_chunks, doc_order)


def _validate(chunks, doc_order):
    """Print validation diagnostics."""
    _safe_print(f"\n{'=' * 60}")
    _safe_print("CHUNKING COMPLETE")
    _safe_print(f"{'=' * 60}")
    _safe_print(f"Total chunks: {len(chunks)}")
    _safe_print(f"Output: {OUTPUT}")

    # Chunks per document
    _safe_print(f"\nChunks per document:")
    doc_counts = {}
    for c in chunks:
        doc_counts[c["document"]] = doc_counts.get(c["document"], 0) + 1
    for doc in doc_order:
        _safe_print(f"  {doc}: {doc_counts.get(doc, 0)}")

    # Chunks per guideline year/type
    _safe_print(f"\nChunks per guideline year/type:")
    yt_counts = {}
    for c in chunks:
        key = (c.get("guideline_year"), c.get("guideline_type"))
        yt_counts[key] = yt_counts.get(key, 0) + 1
    for (year, gtype), count in sorted(yt_counts.items()):
        _safe_print(f"  {year} ({gtype}): {count}")

    # Unique chunk IDs
    ids = [c["chunk_id"] for c in chunks]
    unique_ids = set(ids)
    _safe_print(f"\nUnique chunk IDs: {len(unique_ids)}")
    if len(ids) != len(unique_ids):
        dupes = [x for x in ids if ids.count(x) > 1]
        _safe_print(f"  DUPLICATE IDs: {set(dupes)}")
    else:
        _safe_print(f"  No duplicates (PASS)")

    # Missing metadata
    required_meta = {"guideline_family", "guideline_year", "guideline_type"}
    missing = [c["chunk_id"] for c in chunks if not required_meta.issubset(c.keys())]
    if missing:
        _safe_print(f"  Missing metadata in: {missing[:5]}")
    else:
        _safe_print(f"  All chunks have guideline metadata (PASS)")

    # Verify superseded_by / parent_guideline
    esc_2021 = [c for c in chunks if c["document"] == "ESC_HF_2021_Guideline.pdf"]
    esc_2023 = [c for c in chunks if c["document"] == "ESC_HF_2023_Focused_Update.pdf"]
    nice = [c for c in chunks if c["document"] == "NICE_HF_2018_Guideline.pdf"]

    if esc_2021:
        has_superseded = all("superseded_by" in c for c in esc_2021)
        _safe_print(f"  ESC 2021 superseded_by field: {'PASS' if has_superseded else 'FAIL'}")
    if esc_2023:
        has_parent = all("parent_guideline" in c for c in esc_2023)
        _safe_print(f"  ESC 2023 parent_guideline field: {'PASS' if has_parent else 'FAIL'}")
    if nice:
        no_parent = all("parent_guideline" not in c for c in nice)
        no_superseded = all("superseded_by" not in c for c in nice)
        _safe_print(f"  NICE no spurious parent/superseded: {'PASS' if no_parent and no_superseded else 'FAIL'}")

    # Section distribution per document
    _safe_print(f"\nSection distribution per document:")
    for doc in doc_order:
        doc_chunks = [c for c in chunks if c["document"] == doc]
        sections = {}
        for c in doc_chunks:
            sections[c["section"]] = sections.get(c["section"], 0) + 1
        _safe_print(f"\n  {doc} ({len(doc_chunks)} chunks):")
        for s, count in sorted(sections.items(), key=lambda x: -x[1])[:15]:
            _safe_print(f"    {s}: {count}")
        if len(sections) > 15:
            _safe_print(f"    ... and {len(sections) - 15} more sections")

    # page_start / page_end validity
    bad_pages = []
    for c in chunks:
        if c["page_start"] > c["page_end"]:
            bad_pages.append(c["chunk_id"])
        if c["page"] != c["page_start"]:
            bad_pages.append(c["chunk_id"])
    if bad_pages:
        _safe_print(f"\n  Invalid page ranges: {bad_pages[:5]}")
    else:
        _safe_print(f"\n  All page_start <= page_end (PASS)")

    # Verify chunk IDs match document prefix
    prefix_mismatches = []
    for c in chunks:
        expected_prefix = _derive_prefix(c["document"])
        if not c["chunk_id"].startswith(expected_prefix):
            prefix_mismatches.append(c["chunk_id"])
    if prefix_mismatches:
        _safe_print(f"  Prefix mismatches: {prefix_mismatches[:5]}")
    else:
        _safe_print(f"  All chunk IDs match document prefix (PASS)")

    # Section count per document
    doc_sections = {}
    for c in chunks:
        doc_sections.setdefault(c["document"], set()).add(c["section"])

    _safe_print(f"\n  Sections per document (for manual review):")
    for doc, secs in doc_sections.items():
        _safe_print(f"    {doc}: {len(secs)} unique sections")


if __name__ == "__main__":
    main()
