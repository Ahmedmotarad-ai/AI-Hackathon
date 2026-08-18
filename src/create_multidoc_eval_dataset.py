"""
Multi-Document Retrieval Evaluation Dataset Creator

Creates data/evaluation/multidoc_eval_dataset.json with 30 queries
across NICE 2018, ESC 2021, ESC 2023, and cross-document categories.

Read-only: does not modify chunks, embeddings, ChromaDB, or existing eval datasets.
"""

import json
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHUNKS_PATH = PROJECT_ROOT / "data" / "chunks" / "chunks.jsonl"
OUTPUT_PATH = PROJECT_ROOT / "data" / "evaluation" / "multidoc_eval_dataset.json"
EXISTING_EVAL = PROJECT_ROOT / "data" / "evaluation" / "eval_dataset.json"

# Blacklisted document
BLACKLIST = {"ESC_HF_2023_Guideline.pdf"}


def load_chunks():
    chunks = []
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def build_index(chunks):
    """Build lookup structures from chunks."""
    by_id = {c["chunk_id"]: c for c in chunks}
    by_doc = defaultdict(list)
    by_section = defaultdict(list)
    by_year = defaultdict(list)
    for c in chunks:
        by_doc[c["document"]].append(c)
        by_section[c["section"]].append(c)
        by_year[c.get("guideline_year")].append(c)
    return by_id, by_doc, by_section, by_year


def _search_chunks(chunks, keywords, case_sensitive=False):
    """Return chunks whose text contains any of the keywords."""
    results = []
    for c in chunks:
        text = c["text"] if case_sensitive else c["text"].lower()
        for kw in keywords:
            k = kw if case_sensitive else kw.lower()
            if k in text:
                results.append(c)
                break
    return results


def _search_sections(chunks, section_keywords):
    """Return chunks whose section name contains any of the keywords."""
    results = []
    for c in chunks:
        sec = c["section"].lower()
        for kw in section_keywords:
            if kw.lower() in sec:
                results.append(c)
                break
    return results


def create_queries(chunks, by_id, by_doc, by_section, by_year):
    """Create 30 queries with relevance judgments."""
    queries = []

    nice_chunks = by_doc.get("NICE_HF_2018_Guideline.pdf", [])
    esc2021_chunks = by_doc.get("ESC_HF_2021_Guideline.pdf", [])
    esc2023_chunks = by_doc.get("ESC_HF_2023_Focused_Update.pdf", [])

    # =========================================================================
    # NICE 2018 queries (8)
    # =========================================================================

    # Q1: NICE first-line HFrEF pharmacotherapy
    nice_q1_relevant = {}
    for c in nice_chunks:
        score = 0
        sec = c["section"].lower()
        text = c["text"].lower()
        if "1.4 treating" in sec or "1.4 treating" in text[:100]:
            if any(kw in text for kw in ["ace inhibitor", "beta-blocker", "mra", "sglt2"]):
                score = 2
            else:
                score = 1
        elif "specialist treatment" in sec:
            if any(kw in text for kw in ["ivabradine", "hydralazine", "nitrate"]):
                score = 1
        elif "tailoring treatment" in sec:
            score = 1
        elif "preserved ejection fraction" in sec and "mildly reduced" in text:
            score = 1
        nice_q1_relevant[c["chunk_id"]] = score
    queries.append({
        "query_id": "mdq001",
        "query": "What are the recommended first-line pharmacological treatments for heart failure with reduced ejection fraction according to NICE guidelines?",
        "category": "nice",
        "relevant_chunks": nice_q1_relevant,
    })

    # Q2: NICE diagnosis approach
    nice_q2_relevant = {}
    for c in nice_chunks:
        score = 0
        sec = c["section"].lower()
        text = c["text"].lower()
        if "diagnosing heart failure" in sec or "1.2 diagnosing" in text[:100]:
            score = 2
        elif "symptoms, signs" in sec:
            if any(kw in text for kw in ["nt-probnp", "natriuretic", "echocardiograph", "ecg"]):
                score = 2
            else:
                score = 1
        elif "giving information" in sec:
            score = 1
        nice_q2_relevant[c["chunk_id"]] = score
    queries.append({
        "query_id": "mdq002",
        "query": "How does NICE recommend diagnosing chronic heart failure in adults, including the role of natriuretic peptides and echocardiography?",
        "category": "nice",
        "relevant_chunks": nice_q2_relevant,
    })

    # Q3: NICE palliative care
    nice_q3_relevant = {}
    for c in nice_chunks:
        score = 0
        sec = c["section"].lower()
        text = c["text"].lower()
        if "palliative care" in sec:
            score = 2
        elif "other treatments" in sec and "palliative" in text:
            score = 1
        nice_q3_relevant[c["chunk_id"]] = score
    queries.append({
        "query_id": "mdq003",
        "query": "What palliative care approach does NICE recommend for patients with advanced heart failure?",
        "category": "nice",
        "relevant_chunks": nice_q3_relevant,
    })

    # Q4: NICE preserved EF treatment
    nice_q4_relevant = {}
    for c in nice_chunks:
        score = 0
        sec = c["section"].lower()
        text = c["text"].lower()
        if "preserved ejection fraction" in sec:
            if any(kw in text for kw in ["mra", "sglt2", "empagliflozin", "dapagliflozin"]):
                score = 2
            elif any(kw in text for kw in ["ace inhibitor", "beta-blocker"]):
                score = 1
            else:
                score = 1
        elif "1.5 treating" in text[:100]:
            score = 2
        nice_q4_relevant[c["chunk_id"]] = score
    queries.append({
        "query_id": "mdq004",
        "query": "How should heart failure with preserved ejection fraction be treated according to NICE?",
        "category": "nice",
        "relevant_chunks": nice_q4_relevant,
    })

    # Q5: NICE cardiac rehabilitation
    nice_q5_relevant = {}
    for c in nice_chunks:
        score = 0
        sec = c["section"].lower()
        text = c["text"].lower()
        if "cardiac rehabilitation" in sec:
            score = 2
        elif "other treatments" in sec and "rehabilitation" in text:
            score = 1
        nice_q5_relevant[c["chunk_id"]] = score
    queries.append({
        "query_id": "mdq005",
        "query": "What cardiac rehabilitation does NICE recommend for heart failure patients?",
        "category": "nice",
        "relevant_chunks": nice_q5_relevant,
    })

    # Q6: NICE SGLT2 inhibitors
    nice_q6_relevant = {}
    for c in nice_chunks:
        score = 0
        text = c["text"].lower()
        if any(kw in text for kw in ["sglt2", "dapagliflozin", "empagliflozin"]):
            if "preserved ejection fraction" in c["section"].lower() or "1.4 treating" in c["section"].lower() or "1.5 treating" in text[:100]:
                score = 2
            else:
                score = 1
        nice_q6_relevant[c["chunk_id"]] = score
    queries.append({
        "query_id": "mdq006",
        "query": "What is the role of SGLT2 inhibitors in heart failure treatment according to NICE, and for which patient groups are they recommended?",
        "category": "nice",
        "relevant_chunks": nice_q6_relevant,
    })

    # Q7: NICE interventional procedures
    nice_q7_relevant = {}
    for c in nice_chunks:
        score = 0
        sec = c["section"].lower()
        text = c["text"].lower()
        if "interventional procedures" in sec:
            score = 2
        elif any(kw in text for kw in ["icd", "cardiac resynchronisation", "crt", "revascularisation"]):
            if "interventional" in sec or "1.10" in text[:50]:
                score = 2
            else:
                score = 1
        nice_q7_relevant[c["chunk_id"]] = score
    queries.append({
        "query_id": "mdq007",
        "query": "What interventional procedures does NICE recommend for heart failure patients, including ICDs and cardiac resynchronisation therapy?",
        "category": "nice",
        "relevant_chunks": nice_q7_relevant,
    })

    # Q8: NICE monitoring and review
    nice_q8_relevant = {}
    for c in nice_chunks:
        score = 0
        sec = c["section"].lower()
        text = c["text"].lower()
        if "clinical review" in sec:
            score = 2
        elif "starting and monitoring" in sec:
            score = 2
        elif "1.1 team working" in sec and "review" in text:
            score = 1
        nice_q8_relevant[c["chunk_id"]] = score
    queries.append({
        "query_id": "mdq008",
        "query": "What monitoring and clinical review schedule does NICE recommend for heart failure patients?",
        "category": "nice",
        "relevant_chunks": nice_q8_relevant,
    })

    # =========================================================================
    # ESC 2021 queries (8)
    # =========================================================================

    # Q9: ESC 2021 HFrEF cornerstone therapy
    esc_q9_relevant = {}
    for c in esc2021_chunks:
        score = 0
        sec = c["section"].lower()
        text = c["text"].lower()
        if "recommendations" in sec and any(kw in text for kw in ["hfrhf", "reduced ejection fraction"]):
            if any(kw in text for kw in ["ace", "arni", "beta-blocker", "mra", "sglt2"]):
                score = 2
            else:
                score = 1
        elif "treatment" in sec and "hfrhf" in text:
            if any(kw in text for kw in ["ace", "arni", "beta-blocker", "mra"]):
                score = 1
        elif "diagnosis" in sec and "hfrhf" in text and "treatment" in text:
            score = 1
        esc_q9_relevant[c["chunk_id"]] = score
    queries.append({
        "query_id": "mdq009",
        "query": "What are the cornerstone pharmacological therapies recommended by the 2021 ESC guidelines for heart failure with reduced ejection fraction?",
        "category": "esc_2021",
        "relevant_chunks": esc_q9_relevant,
    })

    # Q10: ESC 2021 HFpEF diagnosis and treatment
    esc_q10_relevant = {}
    for c in esc2021_chunks:
        score = 0
        sec = c["section"].lower()
        text = c["text"].lower()
        if "heart failure with preserved ejection fraction" in sec:
            if any(kw in text for kw in ["treatment", "sglt2", "diagnosis", "hfpef"]):
                score = 2
            else:
                score = 1
        elif "treatment" in sec and "hfpef" in text:
            if any(kw in text for kw in ["sglt2", "diuretic", "diagnosis"]):
                score = 1
        elif "diagnosis" in sec and "hfpef" in text:
            score = 1
        esc_q10_relevant[c["chunk_id"]] = score
    queries.append({
        "query_id": "mdq010",
        "query": "How do the 2021 ESC guidelines approach the diagnosis and treatment of heart failure with preserved ejection fraction?",
        "category": "esc_2021",
        "relevant_chunks": esc_q10_relevant,
    })

    # Q11: ESC 2021 AF management in HF
    esc_q11_relevant = {}
    for c in esc2021_chunks:
        score = 0
        sec = c["section"].lower()
        text = c["text"].lower()
        if "atrial fibrillation" in sec:
            if any(kw in text for kw in ["anticoagulation", "doac", "rate control", "rhythm", "ablation"]):
                score = 2
            else:
                score = 1
        elif "rate control" in sec or "rhythm control" in sec:
            if any(kw in text for kw in ["af", "atrial fibrillation", "heart failure"]):
                score = 1
        esc_q11_relevant[c["chunk_id"]] = score
    queries.append({
        "query_id": "mdq011",
        "query": "How should atrial fibrillation be managed in patients with heart failure according to the 2021 ESC guidelines?",
        "category": "esc_2021",
        "relevant_chunks": esc_q11_relevant,
    })

    # Q12: ESC 2021 acute HF management
    esc_q12_relevant = {}
    for c in esc2021_chunks:
        score = 0
        sec = c["section"].lower()
        text = c["text"].lower()
        if "acute heart failure" in sec:
            score = 2
        elif "treatment" in sec and any(kw in text for kw in ["acute", "decompensated", "ahf"]):
            if any(kw in text for kw in ["diuretic", "vasodilator", "inotrope", "mcs"]):
                score = 2
            else:
                score = 1
        elif "pre-discharge" in sec:
            score = 1
        elif "loop diuretics" in sec and "acute" in text:
            score = 1
        esc_q12_relevant[c["chunk_id"]] = score
    queries.append({
        "query_id": "mdq012",
        "query": "What is the recommended management approach for acute heart failure according to the 2021 ESC guidelines?",
        "category": "esc_2021",
        "relevant_chunks": esc_q12_relevant,
    })

    # Q13: ESC 2021 device therapy
    esc_q13_relevant = {}
    for c in esc2021_chunks:
        score = 0
        sec = c["section"].lower()
        text = c["text"].lower()
        if any(kw in sec for kw in ["secondary prevention", "sudden cardiac death"]):
            if any(kw in text for kw in ["icd", "defibrillator", "primary prevention"]):
                score = 2
            else:
                score = 1
        elif "recommendations" in sec and any(kw in text for kw in ["icd", "crt", "device"]):
            score = 1
        elif "treatment" in sec and any(kw in text for kw in ["icd", "crt", "cardiac resynchronisation"]):
            score = 1
        esc_q13_relevant[c["chunk_id"]] = score
    queries.append({
        "query_id": "mdq013",
        "query": "What are the 2021 ESC recommendations for device therapy including ICDs and cardiac resynchronisation in heart failure?",
        "category": "esc_2021",
        "relevant_chunks": esc_q13_relevant,
    })

    # Q14: ESC 2021 diabetes and HF
    esc_q14_relevant = {}
    for c in esc2021_chunks:
        score = 0
        sec = c["section"].lower()
        text = c["text"].lower()
        if "diabetes" in sec:
            if any(kw in text for kw in ["sglt2", "metformin", "glp-1", "insulin"]):
                score = 2
            else:
                score = 1
        elif "treatment" in sec and any(kw in text for kw in ["diabetes", "diabetic", "t2dm"]):
            if any(kw in text for kw in ["sglt2", "metformin"]):
                score = 1
        esc_q14_relevant[c["chunk_id"]] = score
    queries.append({
        "query_id": "mdq014",
        "query": "How should diabetes be managed in heart failure patients according to the 2021 ESC guidelines, particularly regarding SGLT2 inhibitors?",
        "category": "esc_2021",
        "relevant_chunks": esc_q14_relevant,
    })

    # Q15: ESC 2021 iron deficiency
    esc_q15_relevant = {}
    for c in esc2021_chunks:
        score = 0
        sec = c["section"].lower()
        text = c["text"].lower()
        if "iron deficiency" in sec:
            if any(kw in text for kw in ["ferric", "iron", "ferritin", "tsat"]):
                score = 2
            else:
                score = 1
        elif "treatment" in sec and any(kw in text for kw in ["iron deficiency", "ferric carboxymaltose", "iron therapy"]):
            score = 1
        elif "recommendations" in sec and "iron" in text:
            score = 1
        esc_q15_relevant[c["chunk_id"]] = score
    queries.append({
        "query_id": "mdq015",
        "query": "What are the 2021 ESC recommendations for managing iron deficiency in heart failure patients?",
        "category": "esc_2021",
        "relevant_chunks": esc_q15_relevant,
    })

    # Q16: ESC 2021 obesity and HF
    esc_q16_relevant = {}
    for c in esc2021_chunks:
        score = 0
        sec = c["section"].lower()
        text = c["text"].lower()
        if "obesity" in sec:
            if any(kw in text for kw in ["hfpef", "weight", "bmi", "caloric"]):
                score = 2
            else:
                score = 1
        elif "treatment" in sec and any(kw in text for kw in ["obesity", "obese", "bmi"]):
            score = 1
        esc_q16_relevant[c["chunk_id"]] = score
    queries.append({
        "query_id": "mdq016",
        "query": "How does obesity affect heart failure management according to the 2021 ESC guidelines?",
        "category": "esc_2021",
        "relevant_chunks": esc_q16_relevant,
    })

    # =========================================================================
    # ESC 2023 queries (6)
    # =========================================================================

    # Q17: ESC 2023 HFmrEF new evidence
    esc23_q17_relevant = {}
    for c in esc2023_chunks:
        score = 0
        sec = c["section"].lower()
        text = c["text"].lower()
        if "hfmref" in sec or "hfmrEF" in c["section"]:
            if any(kw in text for kw in ["dapagliflozin", "empagliflozin", "sglt2", "deliver", "class i"]):
                score = 2
            else:
                score = 1
        elif "treatment" in sec and "hfmrEF" in c["text"]:
            score = 1
        elif "recommendation" in sec and "hfmrEF" in c["text"]:
            score = 1
        esc23_q17_relevant[c["chunk_id"]] = score
    queries.append({
        "query_id": "mdq017",
        "query": "What new evidence regarding HFmrEF management is presented in the 2023 ESC focused update, particularly from the DELIVER and EMPEROR-Preserved trials?",
        "category": "esc_2023",
        "relevant_chunks": esc23_q17_relevant,
    })

    # Q18: ESC 2023 HFpEF updates
    esc23_q18_relevant = {}
    for c in esc2023_chunks:
        score = 0
        sec = c["section"].lower()
        text = c["text"].lower()
        if "hfpef" in sec or "hfpef" in c["section"].lower():
            score = 2
        elif "treatment" in sec and "hfpef" in text:
            score = 1
        elif "recommendation" in sec and "hfpef" in text:
            score = 1
        elif "diabetes" in sec and any(kw in text for kw in ["hfpef", "preserved"]):
            score = 1
        esc23_q18_relevant[c["chunk_id"]] = score
    queries.append({
        "query_id": "mdq018",
        "query": "What are the updated recommendations for heart failure with preserved ejection fraction in the 2023 ESC focused update?",
        "category": "esc_2023",
        "relevant_chunks": esc23_q18_relevant,
    })

    # Q19: ESC 2023 SGLT2 inhibitor upgrade
    esc23_q19_relevant = {}
    for c in esc2023_chunks:
        score = 0
        text = c["text"].lower()
        sec = c["section"].lower()
        if any(kw in text for kw in ["sglt2", "dapagliflozin", "empagliflozin"]):
            if any(kw in text for kw in ["class i", "recommended", "recommendation"]):
                score = 2
            elif any(kw in text for kw in ["class ii", "considered"]):
                score = 1
            else:
                score = 1
        elif "treatment" in sec and any(kw in text for kw in ["sglt2", "sodium-glucose"]):
            score = 1
        esc23_q19_relevant[c["chunk_id"]] = score
    queries.append({
        "query_id": "mdq019",
        "query": "How have SGLT2 inhibitor recommendations changed in the 2023 ESC focused update compared to 2021?",
        "category": "esc_2023",
        "relevant_chunks": esc23_q19_relevant,
    })

    # Q20: ESC 2023 acute HF
    esc23_q20_relevant = {}
    for c in esc2023_chunks:
        score = 0
        text = c["text"].lower()
        sec = c["section"].lower()
        if any(kw in text for kw in ["acute heart failure", "acute hf", "hospitalised", "hospitalization"]):
            if any(kw in text for kw in ["empagliflozin", "dapagliflozin", "empulse", "post-discharge"]):
                score = 2
            else:
                score = 1
        elif "pre-discharge" in sec:
            score = 1
        elif "treatment" in sec and "acute" in text:
            score = 1
        esc23_q20_relevant[c["chunk_id"]] = score
    queries.append({
        "query_id": "mdq020",
        "query": "What updates does the 2023 ESC focused update provide for the management of acute heart failure?",
        "category": "esc_2023",
        "relevant_chunks": esc23_q20_relevant,
    })

    # Q21: ESC 2023 diabetes management
    esc23_q21_relevant = {}
    for c in esc2023_chunks:
        score = 0
        sec = c["section"].lower()
        text = c["text"].lower()
        if "diabetes" in sec:
            if any(kw in text for kw in ["sglt2", "empagliflozin", "dapagliflozin", "hf", "heart failure"]):
                score = 2
            else:
                score = 1
        elif "treatment" in sec and any(kw in text for kw in ["diabetes", "diabetic", "t2dm"]):
            score = 1
        esc23_q21_relevant[c["chunk_id"]] = score
    queries.append({
        "query_id": "mdq021",
        "query": "What are the 2023 ESC recommendations for managing diabetes in heart failure patients?",
        "category": "esc_2023",
        "relevant_chunks": esc23_q21_relevant,
    })

    # Q22: ESC 2023 iron deficiency
    esc23_q22_relevant = {}
    for c in esc2023_chunks:
        score = 0
        sec = c["section"].lower()
        text = c["text"].lower()
        if "iron deficiency" in sec:
            if any(kw in text for kw in ["ferric", "iron", "ferritin", "recommendation"]):
                score = 2
            else:
                score = 1
        elif "treatment" in sec and "iron" in text:
            score = 1
        elif "recommendation" in sec and "iron" in text:
            score = 1
        esc23_q22_relevant[c["chunk_id"]] = score
    queries.append({
        "query_id": "mdq022",
        "query": "What are the updated 2023 ESC recommendations for iron deficiency management in heart failure?",
        "category": "esc_2023",
        "relevant_chunks": esc23_q22_relevant,
    })

    # =========================================================================
    # Cross-document queries (8)
    # =========================================================================

    # Q23: NICE vs ESC HFrEF treatment
    cross_q23_relevant = {}
    for c in chunks:
        score = 0
        doc = c["document"]
        sec = c["section"].lower()
        text = c["text"].lower()
        if doc == "NICE_HF_2018_Guideline.pdf":
            if ("1.4 treating" in sec or "1.4 treating" in text[:100]) and any(kw in text for kw in ["ace inhibitor", "beta-blocker", "mra"]):
                score = 2
            elif "tailoring treatment" in sec and any(kw in text for kw in ["ace", "arni", "beta-blocker"]):
                score = 1
        elif doc == "ESC_HF_2021_Guideline.pdf":
            if "recommendations" in sec and any(kw in text for kw in ["hfrhf", "reduced ejection"]) and any(kw in text for kw in ["ace", "arni", "beta-blocker", "mra"]):
                score = 2
            elif "treatment" in sec and "hfrhf" in text and any(kw in text for kw in ["ace", "arni"]):
                score = 1
        elif doc == "ESC_HF_2023_Focused_Update.pdf":
            if any(kw in text for kw in ["hfrhf", "reduced ejection fraction"]) and "treatment" in sec:
                score = 1
        cross_q23_relevant[c["chunk_id"]] = score
    queries.append({
        "query_id": "mdq023",
        "query": "How do the NICE 2018 and ESC 2021 guidelines differ in their first-line treatment recommendations for heart failure with reduced ejection fraction?",
        "category": "cross_document",
        "relevant_chunks": cross_q23_relevant,
    })

    # Q24: ARNI role across guidelines
    # Requires ARNI/sacubitril to appear in a substantive treatment context
    # specifically about HFrEF/HFmrEF, not amyloidosis or other comorbidities.
    cross_q24_relevant = {}
    for c in chunks:
        score = 0
        doc = c["document"]
        text = c["text"].lower()
        sec = c["section"].lower()
        # Skip abbreviation/reference sections
        if "abbreviation" in sec or "reference" in sec or "list of" in sec:
            cross_q24_relevant[c["chunk_id"]] = 0
            continue
        has_arni = "arni" in text or "sacubitril" in text
        if has_arni:
            if doc == "NICE_HF_2018_Guideline.pdf":
                # NICE: ARNI replacement of ACE-I for symptomatic patients is key
                if any(kw in text for kw in ["replace", "switch"]):
                    if any(kw in text for kw in ["ace inhibitor", "ace-i", "symptomatic", "36"]):
                        score = 2
                    else:
                        score = 1
                elif "tailoring" in sec and "arni" in text:
                    score = 1
            elif doc == "ESC_HF_2021_Guideline.pdf":
                # ESC: ARNI as cornerstone of HFrEF treatment
                is_hfrhf = "hfrhf" in text or "reduced ejection fraction" in text
                is_treatment_or_rec = "treatment" in sec or "recommendation" in sec
                is_amyloidosis = "amyloid" in text
                # Only score=2 if ARNI is discussed in HFrEF context (not amyloidosis)
                if is_hfrhf and is_treatment_or_rec and not is_amyloidosis:
                    score = 2
                elif is_treatment_or_rec and "arni" in text and not is_amyloidosis:
                    # ARNI in treatment section but not specifically HFrEF
                    score = 1
            elif doc == "ESC_HF_2023_Focused_Update.pdf":
                if "treatment" in sec and "arni" in text:
                    score = 1
        cross_q24_relevant[c["chunk_id"]] = score
    queries.append({
        "query_id": "mdq024",
        "query": "What is the role of ARNI (sacubitril/valsartan) in heart failure treatment across the NICE and ESC guidelines?",
        "category": "cross_document",
        "relevant_chunks": cross_q24_relevant,
    })

    # Q25: SGLT2 inhibitors across all guidelines
    # Requires SGLT2/dapa/empa to appear in a substantive treatment context,
    # not abbreviation lists, references, or passing mentions.
    cross_q25_relevant = {}
    for c in chunks:
        score = 0
        text = c["text"].lower()
        sec = c["section"].lower()
        # Skip abbreviation/reference sections
        if "abbreviation" in sec or "reference" in sec or "list of" in sec:
            cross_q25_relevant[c["chunk_id"]] = 0
            continue
        has_sglt2 = any(kw in text for kw in ["sglt2", "dapagliflozin", "empagliflozin"])
        if has_sglt2:
            if c["document"] == "NICE_HF_2018_Guideline.pdf":
                # NICE: SGLT2 for preserved ejection fraction specifically
                if "preserved ejection" in sec or "1.5 treating" in text[:200]:
                    score = 2
                elif "1.4 treating" in sec and "sglt2" in text:
                    score = 2
                elif "rationale" in sec and "sglt2" in text:
                    score = 1
            elif c["document"] == "ESC_HF_2021_Guideline.pdf":
                # ESC 2021: SGLT2 in diabetes section + HFrEF cornerstone
                is_treatment_or_rec = "treatment" in sec or "recommendation" in sec
                is_diabetes = "diabetes" in sec
                if is_treatment_or_rec and any(kw in text for kw in ["hfrhf", "reduced ejection"]) and "sglt2" in text:
                    score = 2
                elif is_diabetes and "sglt2" in text:
                    score = 2
                elif is_treatment_or_rec and "sglt2" in text:
                    score = 1
            elif c["document"] == "ESC_HF_2023_Focused_Update.pdf":
                # ESC 2023: SGLT2 upgraded to Class I for HFmrEF/HFpEF
                if any(kw in text for kw in ["class i"]) and "sglt2" in text:
                    score = 2
                elif "treatment" in sec and "sglt2" in text:
                    score = 1
        cross_q25_relevant[c["chunk_id"]] = score
    queries.append({
        "query_id": "mdq025",
        "query": "What is the evolving role of SGLT2 inhibitors across the NICE 2018, ESC 2021, and ESC 2023 guidelines for heart failure?",
        "category": "cross_document",
        "relevant_chunks": cross_q25_relevant,
    })

    # Q26: Diagnostic approach across guidelines
    cross_q26_relevant = {}
    for c in chunks:
        score = 0
        doc = c["document"]
        sec = c["section"].lower()
        text = c["text"].lower()
        if doc == "NICE_HF_2018_Guideline.pdf":
            if "diagnosing heart failure" in sec or "symptoms, signs" in sec:
                if any(kw in text for kw in ["nt-probnp", "natriuretic", "echocardiograph"]):
                    score = 2
                else:
                    score = 1
        elif doc == "ESC_HF_2021_Guideline.pdf":
            if "diagnosis" in sec and any(kw in text for kw in ["natriuretic", "np", "echocardiograph", "hfmrhf", "hfpef", "hfrhf"]):
                score = 2
            elif "suspected heart failure" in sec:
                score = 2
            elif "epidemiology and diagnosis" in sec:
                score = 1
        elif doc == "ESC_HF_2023_Focused_Update.pdf":
            if "diagnosis" in sec or "front matter" in sec:
                if any(kw in text for kw in ["diagnosis", "classification", "lvef"]):
                    score = 1
        cross_q26_relevant[c["chunk_id"]] = score
    queries.append({
        "query_id": "mdq026",
        "query": "How do the NICE and ESC guidelines differ in their diagnostic approach to heart failure, including the use of natriuretic peptides and echocardiography?",
        "category": "cross_document",
        "relevant_chunks": cross_q26_relevant,
    })

    # Q27: ESC 2021 vs 2023 treatment changes
    cross_q27_relevant = {}
    for c in chunks:
        score = 0
        doc = c["document"]
        text = c["text"].lower()
        sec = c["section"].lower()
        if doc == "ESC_HF_2021_Guideline.pdf":
            if any(kw in text for kw in ["hfmrhf", "hfpef", "mildly reduced"]):
                if any(kw in text for kw in ["treatment", "recommendation", "mra", "arni", "sglt2"]):
                    score = 1
        elif doc == "ESC_HF_2023_Focused_Update.pdf":
            if any(kw in text for kw in ["hfmrhf", "hfpef", "mildly reduced"]):
                if any(kw in text for kw in ["class i", "recommended", "new evidence", "deliver", "emperor"]):
                    score = 2
                elif any(kw in text for kw in ["treatment", "recommendation", "sglt2"]):
                    score = 1
            elif "recommendation" in sec and any(kw in text for kw in ["hfmrhf", "hfpef"]):
                score = 1
        cross_q27_relevant[c["chunk_id"]] = score
    queries.append({
        "query_id": "mdq027",
        "query": "What specific changes in HFmrEF and HFpEF treatment recommendations occurred between the 2021 ESC guideline and the 2023 focused update?",
        "category": "cross_document",
        "relevant_chunks": cross_q27_relevant,
    })

    # Q28: Beta-blocker evidence across guidelines
    # Requires beta-blockers to appear in a substantive HF treatment context,
    # not just any mention in the beta-blockers section (which covers comorbidities too).
    cross_q28_relevant = {}
    for c in chunks:
        score = 0
        doc = c["document"]
        text = c["text"].lower()
        sec = c["section"].lower()
        # Skip abbreviation/reference sections
        if "abbreviation" in sec or "reference" in sec or "list of" in sec:
            cross_q28_relevant[c["chunk_id"]] = 0
            continue
        has_bb = "beta-blocker" in text or "beta blocker" in text
        if has_bb:
            if doc == "NICE_HF_2018_Guideline.pdf":
                # NICE: dedicated beta-blockers section + treatment section
                if "beta-blockers" in sec:
                    # Only score=2 if about HF (not just ECG/monitoring)
                    if any(kw in text for kw in ["hfrhf", "heart failure", "hf", "1.7"]):
                        score = 2
                    else:
                        score = 1
                elif "1.4 treating" in sec and "beta-blocker" in text:
                    score = 2
                elif "tailoring" in sec and "beta-blocker" in text:
                    score = 1
            elif doc == "ESC_HF_2021_Guideline.pdf":
                if "beta-blockers" in sec:
                    # Only score=2 if about HF (not aortic regurgitation, COPD, etc.)
                    is_hf_context = any(kw in text for kw in [
                        "hfrhf", "hfmrhf", "reduced ejection", "mildly reduced",
                        "heart failure", "hf and af", "hf and cad", "rate control"
                    ])
                    is_comorbidity_only = any(kw in text for kw in [
                        "aortic regurgitation", "copd", "asthma", "cancer",
                        "hypertrophic cardiomyopathy", "peripheral arterial"
                    ])
                    if is_hf_context and not is_comorbidity_only:
                        score = 2
                    elif is_hf_context:
                        score = 1
                    else:
                        # Beta-blockers section but not clearly HF-related
                        score = 1
                elif "treatment" in sec and "beta-blocker" in text:
                    if any(kw in text for kw in ["hfrhf", "hfmrhf", "reduced ejection", "mildly reduced"]):
                        score = 1
                    elif "rate control" in text or "af" in text or "atrial" in text:
                        score = 1
            elif doc == "ESC_HF_2023_Focused_Update.pdf":
                if "beta-blockers" in sec:
                    score = 2
                elif "treatment" in sec and "beta-blocker" in text:
                    score = 1
        cross_q28_relevant[c["chunk_id"]] = score
    queries.append({
        "query_id": "mdq028",
        "query": "What evidence supports the use of beta-blockers in heart failure across the NICE and ESC guidelines?",
        "category": "cross_document",
        "relevant_chunks": cross_q28_relevant,
    })

    # Q29: ACE inhibitor vs ARNI switching
    # Requires the chunk to discuss ARNI as a replacement/substitute for ACE-I,
    # or discuss switching considerations (washout, tolerability, etc.).
    cross_q29_relevant = {}
    for c in chunks:
        score = 0
        doc = c["document"]
        text = c["text"].lower()
        sec = c["section"].lower()
        # Skip abbreviation/reference sections
        if "abbreviation" in sec or "reference" in sec or "list of" in sec:
            cross_q29_relevant[c["chunk_id"]] = 0
            continue
        mentions_ace = "ace inhibitor" in text or "ace-i" in text
        mentions_arni = "arni" in text or "sacubitril" in text
        mentions_switch = any(kw in text for kw in [
            "switch", "replace", "washout", "36 hour", "36-hour",
            "substitute", "intolerant", "remain symptomatic"
        ])
        if mentions_arni:
            if doc == "NICE_HF_2018_Guideline.pdf":
                # NICE: the 36-hour washout and symptomatic replacement is key
                if mentions_switch and mentions_ace:
                    score = 2
                elif mentions_switch or ("replace" in text and "ace" in text):
                    score = 2
                elif "tailoring" in sec and "arni" in text:
                    score = 1
            elif doc == "ESC_HF_2021_Guideline.pdf":
                # ESC: ARNI as substitute for ACE-I in HFrEF
                is_hfrhf = "hfrhf" in text or "reduced ejection" in text
                if is_hfrhf and mentions_arni:
                    if mentions_switch or mentions_ace:
                        score = 2
                    else:
                        score = 1
                elif mentions_arni and mentions_ace:
                    score = 1
            elif doc == "ESC_HF_2023_Focused_Update.pdf":
                if mentions_arni and mentions_switch:
                    score = 1
        cross_q29_relevant[c["chunk_id"]] = score
    queries.append({
        "query_id": "mdq029",
        "query": "How do the guidelines recommend switching from ACE inhibitors to ARNI, and what are the key considerations?",
        "category": "cross_document",
        "relevant_chunks": cross_q29_relevant,
    })

    # Q30: Comprehensive HF classification comparison
    cross_q30_relevant = {}
    for c in chunks:
        score = 0
        doc = c["document"]
        text = c["text"].lower()
        sec = c["section"].lower()
        if any(kw in text for kw in ["hfrhf", "hfmrhf", "hfpef", "reduced ejection", "mildly reduced", "preserved ejection"]):
            if doc == "NICE_HF_2018_Guideline.pdf":
                if "terms used in this guideline" in sec:
                    score = 2
                elif "symptoms, signs" in sec and any(kw in text for kw in ["reduced", "preserved", "mildly"]):
                    score = 1
                elif "1.5 treating" in text[:100]:
                    score = 1
            elif doc == "ESC_HF_2021_Guideline.pdf":
                if "heart failure with reduced ejection fraction" in sec or "heart failure with preserved ejection fraction" in sec:
                    score = 2
                elif "diagnosis" in sec and any(kw in text for kw in ["classification", "lvef", "41", "49", "50"]):
                    score = 2
                elif "epidemiology" in sec:
                    score = 1
            elif doc == "ESC_HF_2023_Focused_Update.pdf":
                if "front matter" in sec and any(kw in text for kw in ["classification", "lvef", "hfmrhf"]):
                    score = 1
                elif "hfmrhf" in sec or "hfpef" in sec:
                    score = 1
        cross_q30_relevant[c["chunk_id"]] = score
    queries.append({
        "query_id": "mdq030",
        "query": "How do the NICE and ESC guidelines classify heart failure by ejection fraction, and where do the definitions differ?",
        "category": "cross_document",
        "relevant_chunks": cross_q30_relevant,
    })

    return queries


def validate(queries, by_id):
    """Validate the dataset. Returns (pass_bool, issues_list)."""
    issues = []

    # 1. Exactly 30 queries
    if len(queries) != 30:
        issues.append(f"Expected 30 queries, got {len(queries)}")

    # 2. Unique query IDs
    qids = [q["query_id"] for q in queries]
    if len(qids) != len(set(qids)):
        dupes = [x for x in qids if qids.count(x) > 1]
        issues.append(f"Duplicate query IDs: {set(dupes)}")

    # 3. Unique query text
    texts = [q["query"] for q in queries]
    if len(texts) != len(set(texts)):
        issues.append("Duplicate query text found")

    # 4. Every referenced chunk_id exists
    all_chunk_ids = set()
    for q in queries:
        for cid in q["relevant_chunks"]:
            all_chunk_ids.add(cid)
    missing = all_chunk_ids - set(by_id.keys())
    if missing:
        issues.append(f"Missing chunk IDs: {missing}")

    # 5. No invalid relevance scores
    for q in queries:
        for cid, score in q["relevant_chunks"].items():
            if score not in (0, 1, 2):
                issues.append(f"Invalid score {score} in {q['query_id']} for {cid}")

    # 6. Every query has at least one score=2
    for q in queries:
        has_relevant = any(s == 2 for s in q["relevant_chunks"].values())
        if not has_relevant:
            issues.append(f"Query {q['query_id']} has no score=2 chunks")

    # 7. Category counts
    cats = Counter(q["category"] for q in queries)
    expected = {"nice": 8, "esc_2021": 8, "esc_2023": 6, "cross_document": 8}
    for cat, count in expected.items():
        if cats.get(cat) != count:
            issues.append(f"Category {cat}: expected {count}, got {cats.get(cat, 0)}")

    # 8. Cross-document queries reference at least 2 documents
    for q in queries:
        if q["category"] == "cross_document":
            docs = set()
            for cid in q["relevant_chunks"]:
                if q["relevant_chunks"][cid] >= 1:
                    if cid in by_id:
                        docs.add(by_id[cid]["document"])
            if len(docs) < 2:
                issues.append(f"Cross-doc query {q['query_id']} only references {docs}")

    # 9. No references to old 45-chunk dataset
    old_ids = {f"nice_hf_2018_chunk_{i:04d}" for i in range(1, 46)}
    for q in queries:
        for cid in q["relevant_chunks"]:
            if cid in old_ids and cid not in by_id:
                issues.append(f"Reference to old dataset chunk: {cid} in {q['query_id']}")

    # 10. No blacklisted document references
    for q in queries:
        for cid in q["relevant_chunks"]:
            if cid in by_id:
                if by_id[cid]["document"] in BLACKLIST:
                    issues.append(f"Blacklisted doc reference: {cid} in {q['query_id']}")

    return len(issues) == 0, issues


def print_summary(queries, by_id):
    """Print dataset summary."""
    cats = Counter(q["category"] for q in queries)
    total_judgments = sum(len(q["relevant_chunks"]) for q in queries)
    score_counts = Counter()
    doc_counts = Counter()
    for q in queries:
        for cid, score in q["relevant_chunks"].items():
            score_counts[score] += 1
            if cid in by_id:
                doc_counts[by_id[cid]["document"]] += 1

    cross_doc_queries = [q for q in queries if q["category"] == "cross_document"]
    cross_doc_with_multi = 0
    for q in cross_doc_queries:
        docs = set()
        for cid, score in q["relevant_chunks"].items():
            if score >= 1 and cid in by_id:
                docs.add(by_id[cid]["document"])
        if len(docs) >= 2:
            cross_doc_with_multi += 1

    print(f"\nTotal queries: {len(queries)}")
    print(f"Queries per category:")
    for cat in ["nice", "esc_2021", "esc_2023", "cross_document"]:
        print(f"  {cat}: {cats.get(cat, 0)}")
    print(f"\nTotal judgments: {total_judgments}")
    print(f"  Relevant (2): {score_counts.get(2, 0)}")
    print(f"  Partial (1): {score_counts.get(1, 0)}")
    print(f"  Not relevant (0): {score_counts.get(0, 0)}")
    print(f"\nQueries by source document (non-zero judgments):")
    for doc, count in doc_counts.most_common():
        print(f"  {doc}: {count}")
    print(f"\nCross-document queries referencing >=2 documents: {cross_doc_with_multi}/{len(cross_doc_queries)}")


def print_sample_queries(queries, by_id, n=5):
    """Print sample queries with details."""
    print(f"\n{'=' * 70}")
    print("SAMPLE QUERIES")
    print(f"{'=' * 70}")

    for q in queries[:n]:
        print(f"\n--- {q['query_id']} [{q['category']}] ---")
        print(f"Query: {q['query']}")

        relevant = [(cid, s) for cid, s in q["relevant_chunks"].items() if s >= 1]
        relevant.sort(key=lambda x: -x[1])

        print(f"Relevant chunks (score >= 1): {len(relevant)}")
        for cid, score in relevant[:8]:
            if cid in by_id:
                c = by_id[cid]
                print(f"  [{score}] {cid} | {c['document'][:20]}... | {c['section'][:40]} | p{c['page']}")
            else:
                print(f"  [{score}] {cid} | NOT FOUND")

    # Also print one cross-document query
    cross = [q for q in queries if q["category"] == "cross_document"]
    if cross:
        q = cross[0]
        print(f"\n--- {q['query_id']} [{q['category']}] ---")
        print(f"Query: {q['query']}")
        relevant = [(cid, s) for cid, s in q["relevant_chunks"].items() if s >= 1]
        relevant.sort(key=lambda x: -x[1])
        print(f"Relevant chunks (score >= 1): {len(relevant)}")
        for cid, score in relevant[:10]:
            if cid in by_id:
                c = by_id[cid]
                print(f"  [{score}] {cid} | {c['document'][:25]} | {c['section'][:40]} | p{c['page']}")


def main():
    print("Loading chunks...")
    chunks = load_chunks()
    print(f"Loaded {len(chunks)} chunks")

    by_id, by_doc, by_section, by_year = build_index(chunks)
    print(f"Documents: {list(by_doc.keys())}")
    print(f"Years: {list(by_year.keys())}")

    print("\nCreating queries...")
    queries = create_queries(chunks, by_id, by_doc, by_section, by_year)

    print("\nValidating...")
    passed, issues = validate(queries, by_id)

    if passed:
        print("VALIDATION: PASS")
    else:
        print("VALIDATION: FAIL")
        for issue in issues:
            print(f"  - {issue}")

    print_summary(queries, by_id)
    print_sample_queries(queries, by_id, n=5)

    # Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    dataset = {
        "dataset_version": "2.0",
        "created": "2026-08-18",
        "document_scope": [
            "NICE_HF_2018_Guideline.pdf",
            "ESC_HF_2021_Guideline.pdf",
            "ESC_HF_2023_Focused_Update.pdf",
        ],
        "chunk_count": len(chunks),
        "relevance_scale": {
            "2": "Relevant -- chunk contains direct information needed to answer the query",
            "1": "Partially Relevant -- chunk contains supporting or related information",
            "0": "Not Relevant -- chunk does not meaningfully contribute to answering the query",
        },
        "queries": queries,
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    print(f"\nSaved to {OUTPUT_PATH}")
    print(f"File size: {OUTPUT_PATH.stat().st_size:,} bytes")

    if not passed:
        print("\nFINAL: FAIL -- fix issues above before using this dataset")
    else:
        print("\nFINAL: PASS")


if __name__ == "__main__":
    main()
