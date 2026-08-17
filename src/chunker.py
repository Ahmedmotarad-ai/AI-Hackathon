import json
import re
from pathlib import Path

from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT = PROJECT_ROOT / "data" / "parsed" / "02_parsed_documents.json"
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

def clean(text):
    text = text.replace("\u00a0", " ")
    text = re.sub(r"conditions#notice-of-rights\)\.?(?:\s*39)?", "", text, flags=re.I)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def split_page(item):
    """Split a page if it contains more than one main section."""
    page = int(item["page_number"])
    text = clean(item["text"])

    if page <= 5:
        return [("Front matter", text, page)]

    lines = text.splitlines()
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

    starts = sorted(set(starts))

    if not starts:
        return [(None, text, page)]

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

def infer_subsection(text, section):
    for heading in SUBSECTIONS:
        if heading != section and re.search(
            rf"(?m)^{re.escape(heading)}\s*$", text
        ):
            return heading
    return None

def main():
    raw = json.loads(INPUT.read_text(encoding="utf-8"))

    segments = []
    current_section = "Front matter"

    for item in raw:
        for section, text, page in split_page(item):
            section = section or current_section
            current_section = section
            if text:
                segments.append((section, text, page))

    chunks = []
    buffer_text = ""
    buffer_pages = []
    buffer_section = None

    def emit(text, pages, section):
        text = text.strip()
        if not text:
            return

        subsection = infer_subsection(text, section)
        section_path = [section]
        if subsection:
            section_path.append(subsection)

        chunks.append({
            "chunk_id": f"nice_hf_2018_chunk_{len(chunks) + 1:04d}",
            "text": text,
            "document": "NICE_HF_2018_Guideline.pdf",
            "section": section,
            "section_path": section_path,
            "page": min(pages),
            "page_start": min(pages),
            "page_end": max(pages),
        })

    for section, text, page in segments:
        if buffer_section is None:
            buffer_section = section

        if section != buffer_section:
            emit(buffer_text, buffer_pages, buffer_section)
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
            emit(buffer_text, buffer_pages, buffer_section)
            buffer_text = text
            buffer_pages = [page]

    emit(buffer_text, buffer_pages, buffer_section)

    # A single source page can occasionally exceed TARGET_CHARS.
    # Split only on a natural newline boundary, never in the middle of a word.
    final_chunks = []
    for chunk in chunks:
        text = chunk["text"]

        # Phase 1: character-based split (fast, handles most cases)
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

    # Phase 2: token-based split for dense-formatting pages (TOC, etc.)
    # Character-based split can't catch pages where chars/token ratio is low.
    token_chunks = []
    for chunk in final_chunks:
        text = chunk["text"]
        if token_count(text) <= MAX_TOKENS:
            token_chunks.append(chunk)
            continue

        # Split on newline boundaries, checking token count
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

    final_chunks = token_chunks

    for i, chunk in enumerate(final_chunks, start=1):
        chunk["chunk_id"] = f"nice_hf_2018_chunk_{i:04d}"

    with OUTPUT.open("w", encoding="utf-8") as f:
        for chunk in final_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print(f"Created {len(final_chunks)} chunks -> {OUTPUT}")

if __name__ == "__main__":
    main()
