"""
Post-chunking quality audit for data/chunks/chunks.jsonl.
Read-only: does not modify chunker.py, embeddings, ChromaDB, or eval datasets.
"""

import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHUNKS_PATH = PROJECT_ROOT / "data" / "chunks" / "chunks.jsonl"
PARSED_PATH = PROJECT_ROOT / "data" / "parsed" / "03_all_documents_parsed.json"
REPORT_PATH = PROJECT_ROOT / "data" / "evaluation" / "results" / "chunk_quality_audit.json"

# Load tokenizer (same as chunker.py)
tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-small-en-v1.5")


def load_chunks():
    chunks = []
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def load_parsed():
    with open(PARSED_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def token_count(text):
    return len(tokenizer.encode(text, add_special_tokens=True))


def percentile(data, p):
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * p / 100
    f = int(k)
    c = f + 1
    if c >= len(sorted_data):
        return sorted_data[-1]
    return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])


def audit(chunks, parsed):
    report = {}

    # =========================================================================
    # 1. Basic Statistics
    # =========================================================================
    doc_counts = Counter(c["document"] for c in chunks)
    year_counts = Counter(c.get("guideline_year") for c in chunks)
    type_counts = Counter(c.get("guideline_type") for c in chunks)

    report["basic_statistics"] = {
        "total_chunks": len(chunks),
        "chunks_per_document": dict(doc_counts),
        "chunks_per_guideline_year": {str(k): v for k, v in year_counts.items()},
        "chunks_per_guideline_type": dict(type_counts),
    }

    # =========================================================================
    # 2. Token Statistics
    # =========================================================================
    token_counts = []
    for c in chunks:
        tc = token_count(c["text"])
        token_counts.append(tc)

    report["token_statistics"] = {
        "min": min(token_counts),
        "median": round(statistics.median(token_counts), 1),
        "mean": round(statistics.mean(token_counts), 1),
        "p90": round(percentile(token_counts, 90), 1),
        "p95": round(percentile(token_counts, 95), 1),
        "max": max(token_counts),
        "chunks_gt_450": sum(1 for t in token_counts if t > 450),
        "chunks_gt_512": sum(1 for t in token_counts if t > 512),
    }

    # =========================================================================
    # 3. Character Statistics
    # =========================================================================
    char_counts = [len(c["text"]) for c in chunks]
    report["character_statistics"] = {
        "min": min(char_counts),
        "median": round(statistics.median(char_counts), 1),
        "mean": round(statistics.mean(char_counts), 1),
        "p95": round(percentile(char_counts, 95), 1),
        "max": max(char_counts),
    }

    # =========================================================================
    # 4. Very-Small Chunks
    # =========================================================================
    small_100 = []
    small_50 = []
    for i, c in enumerate(chunks):
        tc = token_counts[i]
        if tc < 100:
            small_100.append({
                "index": i,
                "chunk_id": c["chunk_id"],
                "document": c["document"],
                "tokens": tc,
                "chars": len(c["text"]),
                "text_preview": c["text"][:120],
            })
        if tc < 50:
            small_50.append({
                "index": i,
                "chunk_id": c["chunk_id"],
                "document": c["document"],
                "tokens": tc,
                "chars": len(c["text"]),
                "text_preview": c["text"][:120],
            })

    report["very_small_chunks"] = {
        "lt_100_tokens": {"count": len(small_100), "examples": small_100[:10]},
        "lt_50_tokens": {"count": len(small_50), "examples": small_50[:10]},
    }

    # =========================================================================
    # 5. Section Quality
    # =========================================================================
    doc_sections = defaultdict(set)
    unknown_chunks = []
    empty_section = []
    for c in chunks:
        doc_sections[c["document"]].add(c["section"])
        if c["section"] == "Unknown":
            unknown_chunks.append(c["chunk_id"])
        if not c.get("section"):
            empty_section.append(c["chunk_id"])

    # Top sections by chunk count
    section_counts = Counter(c["section"] for c in chunks)
    top_sections = section_counts.most_common(20)

    report["section_quality"] = {
        "unique_sections_per_document": {k: len(v) for k, v in doc_sections.items()},
        "top_sections_by_chunk_count": [{"section": s, "count": n} for s, n in top_sections],
        "chunks_with_unknown_section": {"count": len(unknown_chunks), "chunk_ids": unknown_chunks[:20]},
        "chunks_with_empty_section": {"count": len(empty_section), "chunk_ids": empty_section[:20]},
    }

    # =========================================================================
    # 6. Page Continuity
    # =========================================================================
    bad_page_range = []
    page_doc_mismatch = []
    page_jumps = []
    multi_doc_chunks = []

    for c in chunks:
        if c["page_start"] > c["page_end"]:
            bad_page_range.append(c["chunk_id"])
        if c["page"] != c["page_start"]:
            bad_page_range.append(c["chunk_id"])

    # Group by document and check page continuity
    doc_pages = defaultdict(list)
    for c in chunks:
        doc_pages[c["document"]].append(c)

    for doc, doc_chunks in doc_pages.items():
        sorted_chunks = sorted(doc_chunks, key=lambda x: x["page_start"])
        prev_end = 0
        for c in sorted_chunks:
            if c["page_start"] < prev_end:
                page_jumps.append({
                    "chunk_id": c["chunk_id"],
                    "page_start": c["page_start"],
                    "prev_end": prev_end,
                })
            prev_end = max(prev_end, c["page_end"])

    report["page_continuity"] = {
        "bad_page_ranges": bad_page_range,
        "page_jumps": page_jumps[:20],
        "multi_doc_chunks": multi_doc_chunks,
    }

    # =========================================================================
    # 7. Document Integrity
    # =========================================================================
    # Check that NICE chunks don't contain ESC content markers and vice versa
    nice_markers = ["NICE", "www.nice.org.uk", "NG106"]
    esc_markers = ["ESC", "European Society of Cardiology", "European Heart Journal"]

    nice_bad = []
    esc_bad = []

    for c in chunks:
        text_lower = c["text"].lower()
        if c["document"] == "NICE_HF_2018_Guideline.pdf":
            # Check it doesn't have strong ESC markers
            if any(m.lower() in text_lower for m in ["European Society of Cardiology", "European Heart Journal"]):
                nice_bad.append(c["chunk_id"])
        elif "ESC" in c["document"]:
            # Check it doesn't have strong NICE markers
            if any(m.lower() in text_lower for m in ["www.nice.org.uk", "NG106"]):
                esc_bad.append(c["chunk_id"])

    report["document_integrity"] = {
        "nice_chunks_with_esc_markers": {"count": len(nice_bad), "chunk_ids": nice_bad[:10]},
        "esc_chunks_with_nice_markers": {"count": len(esc_bad), "chunk_ids": esc_bad[:10]},
    }

    # =========================================================================
    # 8. Metadata Integrity
    # =========================================================================
    meta_issues = []
    for c in chunks:
        if not c.get("guideline_family"):
            meta_issues.append({"chunk_id": c["chunk_id"], "issue": "missing guideline_family"})
        if not c.get("guideline_year"):
            meta_issues.append({"chunk_id": c["chunk_id"], "issue": "missing guideline_year"})
        if not c.get("guideline_type"):
            meta_issues.append({"chunk_id": c["chunk_id"], "issue": "missing guideline_type"})

    # Check superseded_by for ESC 2021
    esc_2021_missing_superseded = []
    for c in chunks:
        if c["document"] == "ESC_HF_2021_Guideline.pdf":
            if not c.get("superseded_by"):
                esc_2021_missing_superseded.append(c["chunk_id"])

    # Check parent_guideline for ESC 2023
    esc_2023_missing_parent = []
    for c in chunks:
        if c["document"] == "ESC_HF_2023_Focused_Update.pdf":
            if not c.get("parent_guideline"):
                esc_2023_missing_parent.append(c["chunk_id"])

    # Check NICE doesn't have spurious parent/superseded
    nice_spurious = []
    for c in chunks:
        if c["document"] == "NICE_HF_2018_Guideline.pdf":
            if c.get("parent_guideline") or c.get("superseded_by"):
                nice_spurious.append(c["chunk_id"])

    report["metadata_integrity"] = {
        "missing_metadata": {"count": len(meta_issues), "issues": meta_issues[:20]},
        "esc_2021_missing_superseded_by": {"count": len(esc_2021_missing_superseded), "chunk_ids": esc_2021_missing_superseded[:10]},
        "esc_2023_missing_parent_guideline": {"count": len(esc_2023_missing_parent), "chunk_ids": esc_2023_missing_parent[:10]},
        "nice_spurious_parent_superseded": {"count": len(nice_spurious), "chunk_ids": nice_spurious[:10]},
    }

    # =========================================================================
    # 9. Duplicate / Near-Duplicate Analysis
    # =========================================================================
    text_map = defaultdict(list)
    for i, c in enumerate(chunks):
        text_map[c["text"]].append({"index": i, "chunk_id": c["chunk_id"], "document": c["document"]})

    exact_dupes = {text: locs for text, locs in text_map.items() if len(locs) > 1}

    # Repeated headers/footers: texts that appear in multiple chunks
    repeated_short = {}
    for text, locs in text_map.items():
        if len(text) < 200 and len(locs) > 2:
            repeated_short[text] = locs

    report["duplicate_analysis"] = {
        "exact_duplicate_texts": {
            "count": len(exact_dupes),
            "examples": [
                {"text_preview": text[:100], "locations": locs}
                for text, locs in list(exact_dupes.items())[:10]
            ],
        },
        "repeated_short_texts": {
            "count": len(repeated_short),
            "examples": [
                {"text_preview": text[:100], "occurrences": len(locs)}
                for text, locs in list(repeated_short.items())[:10]
            ],
        },
    }

    # =========================================================================
    # 10. Clinical Content Sanity Check
    # =========================================================================
    def count_terms(texts, terms):
        combined = " ".join(texts).lower()
        return {term: combined.count(term.lower()) for term in terms}

    nice_texts = [c["text"] for c in chunks if c["document"] == "NICE_HF_2018_Guideline.pdf"]
    esc2021_texts = [c["text"] for c in chunks if c["document"] == "ESC_HF_2021_Guideline.pdf"]
    esc2023_texts = [c["text"] for c in chunks if c["document"] == "ESC_HF_2023_Focused_Update.pdf"]

    report["clinical_content"] = {
        "NICE_HF_2018": count_terms(nice_texts, [
            "heart failure", "diagnosis", "symptoms", "ace inhibitor",
            "beta-blocker", "diuretic", "nice",
        ]),
        "ESC_HF_2021": count_terms(esc2021_texts, [
            "HFrEF", "HFmrEF", "HFpEF", "SGLT2", "ARNI",
            "beta-blocker", "MRA", "diuretic", "ACE inhibitor", "recommendation",
        ]),
        "ESC_HF_2023": count_terms(esc2023_texts, [
            "SGLT2", "HFmrEF", "HFpEF", "acute heart failure",
            "recommendation", "dapagliflozin", "empagliflozin",
        ]),
    }

    # =========================================================================
    # Final Assessment
    # =========================================================================
    issues = []
    if report["token_statistics"]["chunks_gt_450"] > 0:
        issues.append(f"{report['token_statistics']['chunks_gt_450']} chunks exceed 450 tokens")
    if report["token_statistics"]["chunks_gt_512"] > 0:
        issues.append(f"{report['token_statistics']['chunks_gt_512']} chunks exceed 512 tokens")
    if len(small_50) > 0:
        issues.append(f"{len(small_50)} chunks under 50 tokens")
    if len(unknown_chunks) > 0:
        issues.append(f"{len(unknown_chunks)} chunks with Unknown section")
    if bad_page_range:
        issues.append(f"{len(bad_page_range)} chunks with bad page ranges")
    if len(exact_dupes) > 0:
        issues.append(f"{len(exact_dupes)} exact duplicate text blocks")
    if len(nice_bad) > 0:
        issues.append(f"{len(nice_bad)} NICE chunks with ESC markers")
    if len(esc_bad) > 0:
        issues.append(f"{len(esc_bad)} ESC chunks with NICE markers")
    if len(meta_issues) > 0:
        issues.append(f"{len(meta_issues)} chunks with missing metadata")
    if len(esc_2021_missing_superseded) > 0:
        issues.append(f"{len(esc_2021_missing_superseded)} ESC 2021 chunks missing superseded_by")
    if len(esc_2023_missing_parent) > 0:
        issues.append(f"{len(esc_2023_missing_parent)} ESC 2023 chunks missing parent_guideline")
    if len(nice_spurious) > 0:
        issues.append(f"{len(nice_spurious)} NICE chunks with spurious parent/superseded")

    # Determine pass/fail
    # Hard failures: cross-document contamination, missing metadata, page range errors
    hard_fail = bool(nice_bad or esc_bad or meta_issues or bad_page_range
                     or esc_2021_missing_superseded or esc_2023_missing_parent)
    # Soft warnings: small chunks, unknown sections, duplicates
    soft_warnings = bool(small_50 or unknown_chunks or exact_dupes)

    if hard_fail:
        verdict = "FAIL"
    elif soft_warnings:
        verdict = "PASS_WITH_WARNINGS"
    else:
        verdict = "PASS"

    report["final_assessment"] = {
        "verdict": verdict,
        "hard_failures": [i for i in issues if any(kw in i for kw in
            ["missing", "spurious", "bad page", "NICE chunks with ESC", "ESC chunks with NICE"])],
        "soft_warnings": [i for i in issues if i not in report["final_assessment"].get("hard_failures", [])] if False else [i for i in issues if not any(kw in i for kw in
            ["missing", "spurious", "bad page", "NICE chunks with ESC", "ESC chunks with NICE"])],
        "summary": (
            f"Corpus: {len(chunks)} chunks across {len(doc_counts)} documents. "
            f"Token range: {report['token_statistics']['min']}-{report['token_statistics']['max']}, "
            f"median {report['token_statistics']['median']}. "
            f"{report['token_statistics']['chunks_gt_450']} chunks >450 tokens, "
            f"{len(small_50)} chunks <50 tokens. "
            f"{len(unknown_chunks)} Unknown sections. "
            f"{len(exact_dupes)} exact duplicate text blocks."
        ),
    }

    return report


def main():
    print("Loading chunks...")
    chunks = load_chunks()
    print(f"Loaded {len(chunks)} chunks")

    print("Loading parsed documents...")
    parsed = load_parsed()
    print(f"Loaded {len(parsed)} parsed pages")

    print("Running audit...")
    report = audit(chunks, parsed)

    # Save report
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nReport saved to {REPORT_PATH}")

    # Print summary
    print("\n" + "=" * 60)
    print("CHUNK QUALITY AUDIT SUMMARY")
    print("=" * 60)

    bs = report["basic_statistics"]
    ts = report["token_statistics"]
    cs = report["character_statistics"]
    ss = report["section_quality"]
    pc = report["page_continuity"]
    di = report["document_integrity"]
    mi = report["metadata_integrity"]
    da = report["duplicate_analysis"]
    cc = report["clinical_content"]
    fa = report["final_assessment"]

    print(f"\nVerdict: {fa['verdict']}")
    print(f"\n{fa['summary']}")

    print(f"\n--- Basic Statistics ---")
    print(f"Total chunks: {bs['total_chunks']}")
    for doc, count in bs['chunks_per_document'].items():
        print(f"  {doc}: {count}")

    print(f"\n--- Token Statistics ---")
    print(f"Min: {ts['min']}, Median: {ts['median']}, Mean: {ts['mean']}")
    print(f"P90: {ts['p90']}, P95: {ts['p95']}, Max: {ts['max']}")
    print(f"Chunks >450 tokens: {ts['chunks_gt_450']}")
    print(f"Chunks >512 tokens: {ts['chunks_gt_512']}")

    print(f"\n--- Character Statistics ---")
    print(f"Min: {cs['min']}, Median: {cs['median']}, Mean: {cs['mean']}")
    print(f"P95: {cs['p95']}, Max: {cs['max']}")

    print(f"\n--- Very Small Chunks ---")
    print(f"<100 tokens: {report['very_small_chunks']['lt_100_tokens']['count']}")
    print(f"<50 tokens: {report['very_small_chunks']['lt_50_tokens']['count']}")

    print(f"\n--- Section Quality ---")
    for doc, count in ss['unique_sections_per_document'].items():
        print(f"  {doc}: {count} unique sections")
    print(f"Unknown section chunks: {ss['chunks_with_unknown_section']['count']}")
    print(f"Empty section chunks: {ss['chunks_with_empty_section']['count']}")

    print(f"\n--- Page Continuity ---")
    print(f"Bad page ranges: {len(pc['bad_page_ranges'])}")
    print(f"Page jumps: {len(pc['page_jumps'])}")

    print(f"\n--- Document Integrity ---")
    print(f"NICE chunks with ESC markers: {di['nice_chunks_with_esc_markers']['count']}")
    print(f"ESC chunks with NICE markers: {di['esc_chunks_with_nice_markers']['count']}")

    print(f"\n--- Metadata Integrity ---")
    print(f"Missing metadata: {mi['missing_metadata']['count']}")
    print(f"ESC 2021 missing superseded_by: {mi['esc_2021_missing_superseded_by']['count']}")
    print(f"ESC 2023 missing parent_guideline: {mi['esc_2023_missing_parent_guideline']['count']}")
    print(f"NICE spurious parent/superseded: {mi['nice_spurious_parent_superseded']['count']}")

    print(f"\n--- Duplicate Analysis ---")
    print(f"Exact duplicate texts: {da['exact_duplicate_texts']['count']}")
    print(f"Repeated short texts: {da['repeated_short_texts']['count']}")

    print(f"\n--- Clinical Content ---")
    for doc_name, terms in cc.items():
        print(f"\n  {doc_name}:")
        for term, count in terms.items():
            print(f"    {term}: {count}")

    if fa['hard_failures']:
        print(f"\n--- HARD FAILURES ---")
        for h in fa['hard_failures']:
            print(f"  FAIL: {h}")

    if fa['soft_warnings']:
        print(f"\n--- Soft Warnings ---")
        for w in fa['soft_warnings']:
            print(f"  WARN: {w}")


if __name__ == "__main__":
    main()
