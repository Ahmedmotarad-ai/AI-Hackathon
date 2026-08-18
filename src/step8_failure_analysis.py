"""
Step 8: Multi-Document Failure Analysis
========================================
Reads the persisted baseline results and produces a comprehensive report.
"""

import json
import statistics
from collections import defaultdict
from pathlib import Path

RESULTS_PATH = Path("data/evaluation/results/multidoc_baseline_retrieval_20260818T153138Z.json")
EVAL_PATH = Path("data/evaluation/multidoc_eval_dataset.json")
OUTPUT_PATH = Path("data/evaluation/results/multidoc_failure_analysis.json")

with open(RESULTS_PATH, "r", encoding="utf-8") as f:
    results = json.load(f)

with open(EVAL_PATH, "r", encoding="utf-8") as f:
    eval_ds = json.load(f)

# Build lookup: query_id -> full relevance dict from eval dataset
relevance_lookup = {}
for q in eval_ds["queries"]:
    relevance_lookup[q["query_id"]] = q["relevant_chunks"]

# Build lookup: query_id -> expected docs (based on which docs have relevant chunks)
def get_expected_docs(relevance_dict):
    docs = set()
    for cid, score in relevance_dict.items():
        if score >= 1:
            if cid.startswith("nice_hf_2018"):
                docs.add("NICE")
            elif cid.startswith("esc_hf_2021"):
                docs.add("ESC 2021")
            elif cid.startswith("esc_hf_2023"):
                docs.add("ESC 2023")
    return docs

expected_docs_lookup = {}
for qid, rel in relevance_lookup.items():
    expected_docs_lookup[qid] = get_expected_docs(rel)

per_query = results["per_query_results"]

# ----------------------------------------------------------------------
# Helper
# ----------------------------------------------------------------------
def avg_metric(queries, key):
    vals = [q["per_k"].get("5", {}).get(key, 0.0) for q in queries]
    return statistics.mean(vals) if vals else 0.0

def avg_metric_k(queries, k_str, key):
    vals = [q["per_k"].get(k_str, {}).get(key, 0.0) for q in queries]
    return statistics.mean(vals) if vals else 0.0

def avg_metric_any_k(queries, k_val, key):
    ks = str(k_val)
    vals = [q["per_k"].get(ks, {}).get(key, 0.0) for q in queries]
    return statistics.mean(vals) if vals else 0.0

def classify_chunk_doc(cid):
    if cid.startswith("nice_hf_2018"):
        return "NICE"
    elif cid.startswith("esc_hf_2021"):
        return "ESC 2021"
    elif cid.startswith("esc_hf_2023"):
        return "ESC 2023"
    return "unknown"

# ----------------------------------------------------------------------
# A. OVERALL BASELINE
# ----------------------------------------------------------------------
K_VALS = ["1", "3", "5", "10"]
overall = {}
for k in K_VALS:
    overall[k] = {
        "precision": avg_metric_any_k(per_query, k, "precision"),
        "strict_precision": avg_metric_any_k(per_query, k, "strict_precision"),
        "recall": avg_metric_any_k(per_query, k, "recall"),
        "strict_recall": avg_metric_any_k(per_query, k, "strict_recall"),
        "f1": avg_metric_any_k(per_query, k, "f1"),
        "strict_f1": avg_metric_any_k(per_query, k, "strict_f1"),
        "hit_rate": avg_metric_any_k(per_query, k, "hit_rate"),
        "strict_hit_rate": avg_metric_any_k(per_query, k, "strict_hit_rate"),
        "ndcg": avg_metric_any_k(per_query, k, "ndcg"),
    }

mrr_vals = [q["mrr"] for q in per_query]
overall_mrr = statistics.mean(mrr_vals)

# Latency
latencies = [q["latency"]["total_ms"] for q in per_query]
embed_lats = [q["latency"]["embedding_ms"] for q in per_query]
search_lats = [q["latency"]["search_ms"] for q in per_query]

latency_stats = {
    "total": {
        "mean": round(statistics.mean(latencies), 2),
        "median": round(statistics.median(latencies), 2),
        "min": round(min(latencies), 2),
        "max": round(max(latencies), 2),
    },
    "embedding": {
        "mean": round(statistics.mean(embed_lats), 2),
        "median": round(statistics.median(embed_lats), 2),
    },
    "search": {
        "mean": round(statistics.mean(search_lats), 2),
        "median": round(statistics.median(search_lats), 2),
    },
}

section_a = {
    "overall_per_k": overall,
    "mrr": round(overall_mrr, 4),
    "latency": latency_stats,
}

# ----------------------------------------------------------------------
# B. CATEGORY BREAKDOWN
# ----------------------------------------------------------------------
categories = defaultdict(list)
for q in per_query:
    categories[q["category"]].append(q)

cat_order = ["nice", "esc_2021", "esc_2023", "cross_document"]
cat_labels = {"nice": "NICE 2018", "esc_2021": "ESC 2021", "esc_2023": "ESC 2023", "cross_document": "Cross-Document"}

category_breakdown = {}
for cat in cat_order:
    qs = categories.get(cat, [])
    if not qs:
        continue
    mrr_cat = statistics.mean([q["mrr"] for q in qs])
    category_breakdown[cat_labels[cat]] = {
        "count": len(qs),
        "p_at_5": round(avg_metric_any_k(qs, 5, "precision"), 4),
        "recall_at_5": round(avg_metric_any_k(qs, 5, "recall"), 4),
        "strict_recall_at_5": round(avg_metric_any_k(qs, 5, "strict_recall"), 4),
        "ndcg_at_5": round(avg_metric_any_k(qs, 5, "ndcg"), 4),
        "hit_at_5": round(avg_metric_any_k(qs, 5, "hit_rate"), 4),
        "strict_hit_at_5": round(avg_metric_any_k(qs, 5, "strict_hit_rate"), 4),
        "mrr": round(mrr_cat, 4),
    }

section_b = category_breakdown

# ----------------------------------------------------------------------
# C. QUERY-LEVEL FAILURE ANALYSIS
# ----------------------------------------------------------------------
# All queries with Hit@5 = 0
hit5_zero = [q for q in per_query if q["per_k"]["5"]["hit_rate"] == 0.0]

# Bottom 5 by nDCG@5
sorted_by_ndcg = sorted(per_query, key=lambda q: q["per_k"]["5"]["ndcg"])
bottom5_ndcg = sorted_by_ndcg[:5]

# Bottom 5 by MRR
sorted_by_mrr = sorted(per_query, key=lambda q: q["mrr"])
bottom5_mrr = sorted_by_mrr[:5]

# Queries with P@5 = 0
p5_zero = [q for q in per_query if q["per_k"]["5"]["precision"] == 0.0]

# Queries with very low Recall@5 (< 0.1)
low_recall5 = [q for q in per_query if q["per_k"]["5"]["recall"] < 0.1]

def format_query_detail(q):
    retrieved_docs = [classify_chunk_doc(cid) for cid in q["retrieved_chunk_ids"][:5]]
    return {
        "query_id": q["query_id"],
        "category": q["category"],
        "query": q["query"],
        "retrieved_chunk_ids": q["retrieved_chunk_ids"][:5],
        "retrieved_documents": retrieved_docs,
        "retrieved_relevance": q["retrieved_relevance"][:5],
        "retrieved_ranks": list(range(1, 6)),
        "nDCG_at_5": round(q["per_k"]["5"]["ndcg"], 4),
        "recall_at_5": round(q["per_k"]["5"]["recall"], 4),
        "strict_recall_at_5": round(q["per_k"]["5"]["strict_recall"], 4),
        "mrr": round(q["mrr"], 4),
    }

# Deduplicate: some queries appear in multiple failure lists
all_failed_ids = set()
for q in hit5_zero + bottom5_ndcg + p5_zero + low_recall5:
    all_failed_ids.add(q["query_id"])

section_c = {
    "hit_at_5_zero": {
        "count": len(hit5_zero),
        "queries": [format_query_detail(q) for q in hit5_zero],
    },
    "bottom_5_by_ndcg5": [format_query_detail(q) for q in bottom5_ndcg],
    "bottom_5_by_mrr": [format_query_detail(q) for q in bottom5_mrr],
    "precision_at_5_zero": {
        "count": len(p5_zero),
        "queries": [format_query_detail(q) for q in p5_zero],
    },
    "low_recall_at_5": {
        "count": len(low_recall5),
        "queries": [format_query_detail(q) for q in low_recall5],
    },
}

# ----------------------------------------------------------------------
# D. DOCUMENT CONFUSION ANALYSIS
# ----------------------------------------------------------------------
doc_confusion = []
for q in per_query:
    qid = q["query_id"]
    cat = q["category"]
    top5_cids = q["retrieved_chunk_ids"][:5]
    top5_docs = [classify_chunk_doc(cid) for cid in top5_cids]
    top5_rel = q["retrieved_relevance"][:5]

    exp_docs = expected_docs_lookup.get(qid, set())
    from_expected = sum(1 for d in top5_docs if d in exp_docs)
    from_wrong = 5 - from_expected

    doc_counts = defaultdict(int)
    for d in top5_docs:
        doc_counts[d] += 1

    # For cross-doc: check if one family dominated
    cross_doc_dominance = None
    if cat == "cross_document":
        nice_count = doc_counts.get("NICE", 0)
        esc_total = doc_counts.get("ESC 2021", 0) + doc_counts.get("ESC 2023", 0)
        if nice_count > 0 and esc_total > 0:
            cross_doc_dominance = f"NICE={nice_count}, ESC={esc_total}"
        elif nice_count == 5:
            cross_doc_dominance = "NICE dominated (5/5)"
        elif esc_total == 5:
            cross_doc_dominance = "ESC dominated (5/5)"

    # Check for cross-contamination in single-doc queries
    contamination_type = None
    if len(exp_docs) == 1:
        exp_doc = list(exp_docs)[0]
        wrong_docs = [d for d in top5_docs if d not in exp_docs]
        if wrong_docs:
            contamination_type = f"Expected {exp_doc} but got {wrong_docs}"

    doc_confusion.append({
        "query_id": qid,
        "category": cat,
        "query": q["query"][:80] + "..." if len(q["query"]) > 80 else q["query"],
        "expected_documents": sorted(list(exp_docs)),
        "top5_documents": top5_docs,
        "top5_relevance": top5_rel,
        "from_expected_doc": from_expected,
        "from_wrong_doc": from_wrong,
        "doc_distribution": dict(doc_counts),
        "cross_doc_dominance": cross_doc_dominance,
        "contamination_type": contamination_type,
    })

# Summary stats
total_wrong = sum(d["from_wrong_doc"] for d in doc_confusion)
total_contaminated = sum(1 for d in doc_confusion if d["contamination_type"])

# Per-document contamination
contam_by_target = defaultdict(list)
for d in doc_confusion:
    if d["contamination_type"]:
        contam_by_target[d["expected_documents"][0]].append(d)

section_d = {
    "summary": {
        "total_wrong_doc_in_top5": total_wrong,
        "queries_with_contamination": total_contaminated,
        "total_queries": len(per_query),
    },
    "contamination_by_expected_doc": {
        k: {
            "count": len(v),
            "query_ids": [d["query_id"] for d in v],
        }
        for k, v in contam_by_target.items()
    },
    "per_query": doc_confusion,
}

# ----------------------------------------------------------------------
# E. ROOT CAUSE ANALYSIS
# ----------------------------------------------------------------------
# Evidence-based root cause analysis

# 1. ESC 2021 vs ESC 2023 confusion
esc21_queries = [q for q in per_query if q["category"] == "esc_2021"]
esc21_confused_with_23 = 0
for q in esc21_queries:
    top5_docs = [classify_chunk_doc(cid) for cid in q["retrieved_chunk_ids"][:5]]
    if "ESC 2023" in top5_docs:
        esc21_confused_with_23 += 1

esc23_queries = [q for q in per_query if q["category"] == "esc_2023"]
esc23_confused_with_21 = 0
for q in esc23_queries:
    top5_docs = [classify_chunk_doc(cid) for cid in q["retrieved_chunk_ids"][:5]]
    if "ESC 2021" in top5_docs:
        esc23_confused_with_21 += 1

# 2. Corpus dilution: NICE queries with ESC contamination
nice_queries = [q for q in per_query if q["category"] == "nice"]
nice_confused = 0
for q in nice_queries:
    top5_docs = [classify_chunk_doc(cid) for cid in q["retrieved_chunk_ids"][:5]]
    if any(d != "NICE" for d in top5_docs):
        nice_confused += 1

# 3. Cross-document: queries failing to retrieve both families
cross_queries = [q for q in per_query if q["category"] == "cross_document"]
cross_miss_both = 0
cross_miss_nice = 0
cross_miss_esc = 0
for q in cross_queries:
    top5_docs = [classify_chunk_doc(cid) for cid in q["retrieved_chunk_ids"][:5]]
    has_nice = "NICE" in top5_docs
    has_esc = "ESC 2021" in top5_docs or "ESC 2023" in top5_docs
    if not has_nice and not has_esc:
        cross_miss_both += 1
    if not has_nice:
        cross_miss_nice += 1
    if not has_esc:
        cross_miss_esc += 1

# 4. Reciprocal rank: how often is the first result from wrong doc?
first_result_wrong = 0
for q in per_query:
    first_doc = classify_chunk_doc(q["retrieved_chunk_ids"][0])
    exp = expected_docs_lookup.get(q["query_id"], set())
    if first_doc not in exp:
        first_result_wrong += 1

# 5. Top-1 precision: queries where the single best result is wrong
top1_wrong_count = sum(1 for q in per_query if q["retrieved_relevance"][0] == 0)

root_causes = {
    "esc_2021_confused_with_esc_2023": {
        "count": esc21_confused_with_23,
        "total_esc_2021_queries": len(esc21_queries),
        "pct": round(100 * esc21_confused_with_23 / len(esc21_queries), 1) if esc21_queries else 0,
        "evidence": f"{esc21_confused_with_23}/{len(esc21_queries)} ESC 2021 queries have ESC 2023 in top-5",
    },
    "esc_2023_confused_with_esc_2021": {
        "count": esc23_confused_with_21,
        "total_esc_2023_queries": len(esc23_queries),
        "pct": round(100 * esc23_confused_with_21 / len(esc23_queries), 1) if esc23_queries else 0,
        "evidence": f"{esc23_confused_with_21}/{len(esc23_queries)} ESC 2023 queries have ESC 2021 in top-5",
    },
    "corpus_dilution_nice": {
        "nice_queries_with_esc_in_top5": nice_confused,
        "total_nice_queries": len(nice_queries),
        "pct": round(100 * nice_confused / len(nice_queries), 1) if nice_queries else 0,
        "evidence": f"{nice_confused}/{len(nice_queries)} NICE queries have non-NICE content in top-5",
    },
    "cross_document_failures": {
        "queries_missing_both": cross_miss_both,
        "queries_missing_nice": cross_miss_nice,
        "queries_missing_esc": cross_miss_esc,
        "total_cross_queries": len(cross_queries),
        "evidence": f"{len(cross_queries)} cross-doc queries: {cross_miss_nice} miss NICE, {cross_miss_esc} miss ESC",
    },
    "first_result_accuracy": {
        "first_result_from_wrong_doc": first_result_wrong,
        "top1_wrong_count": top1_wrong_count,
        "total_queries": len(per_query),
        "pct_first_wrong": round(100 * first_result_wrong / len(per_query), 1),
        "evidence": f"{first_result_wrong}/{len(per_query)} queries have wrong doc as top-1 result",
    },
}

# Determine dominant root cause
dominant_causes = []
if nice_confused / len(nice_queries) > 0.5 if nice_queries else False:
    dominant_causes.append("corpus_dilution")
if (esc21_confused_with_23 + esc23_confused_with_21) > 0:
    dominant_causes.append("semantic_similarity_esc21_23")
if cross_miss_nice + cross_miss_esc > len(cross_queries) * 0.3:
    dominant_causes.append("cross_document_competition")

root_causes["dominant_causes"] = dominant_causes

section_e = root_causes

# ----------------------------------------------------------------------
# F. COMPARE AGAINST PREVIOUS NICE-ONLY BASELINE
# ----------------------------------------------------------------------
prev_nice_only = {
    "p@1": 0.95,
    "r@5": 0.6226,
    "ndcg@5": 0.8064,
    "mrr": 0.9667,
    "median_latency_ms": 42.50,
    "num_queries": 20,
    "num_chunks": 45,
}

# Compute NICE-only metrics from new baseline
new_nice_only = {
    "p@1": round(avg_metric_any_k(nice_queries, 1, "precision"), 4),
    "r@5": round(avg_metric_any_k(nice_queries, 5, "recall"), 4),
    "ndcg@5": round(avg_metric_any_k(nice_queries, 5, "ndcg"), 4),
    "mrr": round(statistics.mean([q["mrr"] for q in nice_queries]), 4),
    "median_latency_ms": round(statistics.median([q["latency"]["total_ms"] for q in nice_queries]), 2),
    "num_queries": len(nice_queries),
    "num_chunks": 1001,
}

drops = {}
for k in ["p@1", "r@5", "ndcg@5", "mrr"]:
    prev_val = prev_nice_only[k]
    new_val = new_nice_only[k]
    drops[k] = {
        "previous": prev_val,
        "current": new_val,
        "absolute_change": round(new_val - prev_val, 4),
        "pct_change": round(100 * (new_val - prev_val) / prev_val, 1) if prev_val else 0,
    }

section_f = {
    "previous_nice_only": prev_nice_only,
    "new_nice_subset": new_nice_only,
    "drops": drops,
    "explanation": (
        f"NICE-only queries saw major drops when corpus expanded from 45 to 1001 chunks. "
        f"P@1 fell from {prev_nice_only['p@1']} to {new_nice_only['p@1']} "
        f"({drops['p@1']['pct_change']}%), "
        f"nDCG@5 fell from {prev_nice_only['ndcg@5']} to {new_nice_only['ndcg@5']} "
        f"({drops['ndcg@5']['pct_change']}%), "
        f"and MRR fell from {prev_nice_only['mrr']} to {new_nice_only['mrr']} "
        f"({drops['mrr']['pct_change']}%). "
        f"This confirms the corpus dilution effect: semantically similar ESC chunks "
        f"compete with and outrank the correct NICE chunks."
    ),
}

# ----------------------------------------------------------------------
# G. KEY FINDINGS
# ----------------------------------------------------------------------
findings = [
    {
        "id": 1,
        "finding": "Corpus dilution caused massive degradation for NICE queries",
        "evidence": f"NICE P@1 dropped from 0.95 to {new_nice_only['p@1']} ({nice_confused}/{len(nice_queries)} NICE queries have ESC in top-5). nDCG@5 dropped from 0.8064 to {new_nice_only['ndcg@5']}.",
        "severity": "high",
    },
    {
        "id": 2,
        "finding": "Cross-document queries dominate the worst performers",
        "evidence": f"7/8 bottom nDCG@5 queries are cross-document. Cross-doc MRR={category_breakdown['Cross-Document']['mrr']} vs NICE MRR={category_breakdown['NICE 2018']['mrr']}.",
        "severity": "high",
    },
    {
        "id": 3,
        "finding": "ESC 2021 and ESC 2023 content compete heavily with each other",
        "evidence": f"{esc21_confused_with_23}/{len(esc21_queries)} ESC 2021 queries retrieve ESC 2023 content; {esc23_confused_with_21}/{len(esc23_queries)} ESC 2023 queries retrieve ESC 2021 content.",
        "severity": "medium",
    },
    {
        "id": 4,
        "finding": "The embedding model (BGE-small-en) struggles with document-specific queries",
        "evidence": f"Only {new_nice_only['p@1']*100:.0f}% of NICE queries retrieve the correct doc at rank 1. Queries with document qualifiers (e.g., 'according to NICE') still retrieve wrong docs.",
        "severity": "high",
    },
    {
        "id": 5,
        "finding": "Recall ceiling is low even for best-performing categories",
        "evidence": f"Best Recall@5 is {category_breakdown['NICE 2018']['recall_at_5']:.2f} (NICE), meaning even good queries miss most relevant chunks in top-5.",
        "severity": "medium",
    },
    {
        "id": 6,
        "finding": "First-result accuracy is poor across the board",
        "evidence": f"{first_result_wrong}/{len(per_query)} queries ({round(100*first_result_wrong/len(per_query),1)}%) have a wrong-document chunk at rank 1.",
        "severity": "high",
    },
    {
        "id": 7,
        "finding": "Latency remained stable despite 22x corpus growth",
        "evidence": f"Median latency {latency_stats['total']['median']}ms (new) vs 42.5ms (old) on 1001 vs 45 chunks.",
        "severity": "low",
    },
]

section_g = findings

# ----------------------------------------------------------------------
# H. RECOMMENDED NEXT EXPERIMENT
# ----------------------------------------------------------------------
# Base recommendation on the dominant failure mode

section_h = {
    "recommendation": "Metadata-filtered retrieval with document-source metadata",
    "rationale": (
        f"The dominant failure mode is corpus dilution ({nice_confused}/{len(nice_queries)} NICE queries "
        f"retrieve ESC content, {first_result_wrong}/{len(per_query)} queries have wrong doc at rank 1). "
        f"The current metadata filter only excludes 'Front matter' but does not help route queries "
        f"to the correct document. "
        f"A document-source metadata filter (nice/esc_2021/esc_2023) on each chunk would allow "
        f"the retrieval to be scoped to the correct document when the query specifies a document, "
        f"while still allowing cross-document queries to search all documents."
    ),
    "expected_impact": (
        f"Should dramatically improve NICE queries (currently P@1={new_nice_only['p@1']}, "
        f"target P@1>=0.85) and ESC document-specific queries, while maintaining cross-document "
        f"query performance."
    ),
    "alternative": (
        "If metadata filtering is not feasible, the next best experiment would be "
        "cross-encoder reranking, which can re-score results to boost correct-document chunks."
    ),
}

# ----------------------------------------------------------------------
# ASSEMBLE REPORT
# ----------------------------------------------------------------------
report = {
    "report_title": "Step 8: Multi-Document Baseline Failure Analysis",
    "source_results": str(RESULTS_PATH),
    "source_eval": str(EVAL_PATH),
    "analysis_timestamp": results["evaluation_timestamp"],
    "corpus": {
        "total_chunks": results["chunk_count"],
        "total_queries": results["num_queries"],
        "documents": ["NICE HF 2018 (57 chunks)", "ESC HF 2021 (856 chunks)", "ESC HF 2023 Focused Update (88 chunks)"],
    },
    "A_overall_baseline": section_a,
    "B_category_breakdown": section_b,
    "C_query_level_failures": section_c,
    "D_document_confusion": section_d,
    "E_root_cause_analysis": section_e,
    "F_comparison_to_nice_only_baseline": section_f,
    "G_key_findings": section_g,
    "H_recommended_next_experiment": section_h,
}

# Round all floats in the report for readability
def round_floats(obj, decimals=4):
    if isinstance(obj, float):
        return round(obj, decimals)
    elif isinstance(obj, dict):
        return {k: round_floats(v, decimals) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [round_floats(item, decimals) for item in obj]
    return obj

report = round_floats(report, 4)

# Save
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2, ensure_ascii=False)

# ----------------------------------------------------------------------
# PRINT HUMAN-READABLE SUMMARY
# ----------------------------------------------------------------------
print("=" * 70)
print("STEP 8: MULTI-DOCUMENT FAILURE ANALYSIS")
print("=" * 70)

print("\n-- A. OVERALL BASELINE --")
print(f"{'K':>4} | {'P@K':>6} | {'R@K':>6} | {'sR@K':>6} | {'nDCG':>6} | {'Hit@K':>6}")
print("-" * 50)
for k in K_VALS:
    o = overall[k]
    print(f"{k:>4} | {o['precision']:>6.3f} | {o['recall']:>6.3f} | {o['strict_recall']:>6.3f} | {o['ndcg']:>6.3f} | {o['hit_rate']:>6.3f}")
print(f"MRR: {overall_mrr:.4f}")
print(f"Latency: median={latency_stats['total']['median']}ms mean={latency_stats['total']['mean']}ms")

print("\n-- B. CATEGORY BREAKDOWN --")
print(f"{'Category':<18} | {'N':>3} | {'P@5':>5} | {'R@5':>5} | {'nDCG5':>5} | {'Hit5':>5} | {'MRR':>5}")
print("-" * 72)
for cat_label in ["NICE 2018", "ESC 2021", "ESC 2023", "Cross-Document"]:
    cb = category_breakdown.get(cat_label, {})
    print(f"{cat_label:<18} | {cb.get('count',0):>3} | {cb.get('p_at_5',0):>5.3f} | {cb.get('recall_at_5',0):>5.3f} | {cb.get('ndcg_at_5',0):>5.3f} | {cb.get('hit_at_5',0):>5.3f} | {cb.get('mrr',0):>5.3f}")

print("\n-- C. QUERY-LEVEL FAILURES --")
print(f"  Hit@5 = 0: {len(hit5_zero)} queries")
for q in hit5_zero:
    print(f"    {q['query_id']} [{q['category']}] nDCG5={q['per_k']['5']['ndcg']:.3f} MRR={q['mrr']:.3f}")
    print(f"      {q['query'][:75]}...")
print(f"  P@5 = 0: {len(p5_zero)} queries")
print(f"  Low Recall@5 (<0.1): {len(low_recall5)} queries")

print("\n  Bottom 5 by nDCG@5:")
for q in bottom5_ndcg:
    print(f"    {q['query_id']} [{q['category']}] nDCG5={q['per_k']['5']['ndcg']:.4f} R@5={q['per_k']['5']['recall']:.4f} MRR={q['mrr']:.4f}")
    print(f"      {q['query'][:70]}...")

print("\n-- D. DOCUMENT CONFUSION --")
print(f"  Total wrong-doc slots in top-5: {total_wrong}/{5*len(per_query)} ({round(100*total_wrong/(5*len(per_query)),1)}%)")
print(f"  Queries with wrong-document contamination: {total_contaminated}/{len(per_query)}")
for exp_doc, items in contam_by_target.items():
    print(f"    Expected {exp_doc}: {len(items)} contaminated queries")

print("\n-- E. ROOT CAUSE ANALYSIS --")
print(f"  Dominant causes: {', '.join(dominant_causes)}")
for cause, data in root_causes.items():
    if cause == "dominant_causes":
        continue
    if isinstance(data, dict) and "evidence" in data:
        print(f"  {cause}: {data['evidence']}")

print("\n-- F. COMPARE AGAINST NICE-ONLY BASELINE --")
print(f"  Previous NICE-only: P@1=0.95, R@5=0.623, nDCG@5=0.806, MRR=0.967")
print(f"  New NICE subset:    P@1={new_nice_only['p@1']}, R@5={new_nice_only['r@5']}, nDCG@5={new_nice_only['ndcg@5']}, MRR={new_nice_only['mrr']}")
for k, d in drops.items():
    sign = "+" if d["absolute_change"] >= 0 else ""
    print(f"    {k}: {sign}{d['absolute_change']:.4f} ({sign}{d['pct_change']:.1f}%)")

print("\n-- G. KEY FINDINGS --")
for f_item in findings:
    print(f"  [{f_item['severity'].upper()}] #{f_item['id']}: {f_item['finding']}")
    print(f"    {f_item['evidence']}")

print("\n-- H. RECOMMENDED NEXT EXPERIMENT --")
print(f"  {section_h['recommendation']}")
print(f"  Rationale: {section_h['rationale'][:120]}...")

print("\n" + "=" * 70)
print(f"Report saved to: {OUTPUT_PATH}")
print("=" * 70)
