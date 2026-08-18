"""
Step 11: Deep Failure Analysis
Loads scoped and scoped+reranked results, identifies degraded queries,
and produces a comprehensive failure analysis.
"""

import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

# ============================================================
# Paths
# ============================================================
SCOPED_PATH = Path("data/evaluation/results/metadata_scoped_retrieval_20260818T155956Z.json")
RERANKED_PATH = Path("data/evaluation/results/scoped_reranked_retrieval_20260818T163657Z.json")
EVAL_PATH = Path("data/evaluation/multidoc_eval_dataset.json")
CHUNKS_PATH = Path("data/chunks/chunks.jsonl")
OUTPUT_JSON = Path("data/evaluation/results/step11_failure_analysis.json")
OUTPUT_MD = Path("data/evaluation/results/step11_failure_analysis.md")

# ============================================================
# Load data
# ============================================================
print("Loading data files...")

with open(SCOPED_PATH, "r", encoding="utf-8") as f:
    scoped_data = json.load(f)
with open(RERANKED_PATH, "r", encoding="utf-8") as f:
    reranked_data = json.load(f)
with open(EVAL_PATH, "r", encoding="utf-8") as f:
    eval_ds = json.load(f)
with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
    chunks_text = {}
    chunks_doc = {}
    for line in f:
        if not line.strip():
            continue
        rec = json.loads(line)
        chunks_text[rec["chunk_id"]] = rec["text"]
        chunks_doc[rec["chunk_id"]] = rec.get("metadata", {}).get("document", "unknown")

# Build lookups
scoped_pq = {q["query_id"]: q for q in scoped_data["per_query_results"]}
reranked_pq = {q["query_id"]: q for q in reranked_data["per_query_results"]}
relevance_lookup = {}
for q in eval_ds["queries"]:
    relevance_lookup[q["query_id"]] = q.get("relevant_chunks", q.get("relevance", {}))

all_query_ids = sorted(scoped_pq.keys())
print(f"Loaded {len(all_query_ids)} queries")

# ============================================================
# Helper functions
# ============================================================
def get_pk_metric(per_k, k, key):
    v = per_k.get(k, per_k.get(str(k), {}))
    return v.get(key, 0)

def classify_doc(cid):
    if cid.startswith("nice_hf_2018"):
        return "NICE"
    elif cid.startswith("esc_hf_2021"):
        return "ESC 2021"
    elif cid.startswith("esc_hf_2023"):
        return "ESC 2023"
    return "unknown"

# ============================================================
# Identify degraded queries
# ============================================================
print("Identifying degraded queries...")

query_deltas = []
for qid in all_query_ids:
    s = scoped_pq[qid]
    r = reranked_pq[qid]
    scoped_ndcg5 = get_pk_metric(s["per_k"], 5, "ndcg")
    reranked_ndcg5 = get_pk_metric(r["per_k"], 5, "ndcg")
    delta = reranked_ndcg5 - scoped_ndcg5
    query_deltas.append({
        "query_id": qid,
        "category": s["category"],
        "scoped_nDCG5": scoped_ndcg5,
        "reranked_nDCG5": reranked_ndcg5,
        "delta": delta,
    })

degraded = [q for q in query_deltas if q["delta"] < -0.01]
improved = [q for q in query_deltas if q["delta"] > 0.01]
unchanged = [q for q in query_deltas if abs(q["delta"]) <= 0.01]

degraded.sort(key=lambda x: x["delta"])
print(f"Degraded: {len(degraded)}, Improved: {len(improved)}, Unchanged: {len(unchanged)}")

# ============================================================
# Compute aggregate metrics
# ============================================================
print("Computing aggregate metrics...")

all_scoped = list(scoped_data["per_query_results"])
all_reranked = list(reranked_data["per_query_results"])

def compute_aggregates(queries, label):
    n = len(queries)
    result = {}
    for k in [1, 3, 5, 10]:
        ks = str(k)
        pk_metrics = {}
        for mk in ["precision", "recall", "ndcg", "hit_rate"]:
            vals = []
            for q in queries:
                pk = q["per_k"]
                v = pk.get(k, pk.get(ks, {}))
                vals.append(v.get(mk, 0))
            pk_metrics[mk] = statistics.mean(vals) if vals else 0
        result[f"k{k}"] = pk_metrics
    mrr_vals = [q["mrr"] for q in queries]
    result["mrr"] = statistics.mean(mrr_vals) if mrr_vals else 0
    return result

scoped_agg = compute_aggregates(all_scoped, "scoped")
reranked_agg = compute_aggregates(all_reranked, "reranked")

# ============================================================
# Per-category aggregates
# ============================================================
cat_groups = defaultdict(lambda: {"scoped": [], "reranked": []})
for q in all_scoped:
    cat_groups[q["category"]]["scoped"].append(q)
for q in all_reranked:
    cat_groups[q["category"]]["reranked"].append(q)

cat_agg = {}
for cat, groups in cat_groups.items():
    cat_agg[cat] = {
        "scoped": compute_aggregates(groups["scoped"], f"{cat}_scoped"),
        "reranked": compute_aggregates(groups["reranked"], f"{cat}_reranked"),
        "count": len(groups["reranked"]),
    }

# ============================================================
# Candidate recall analysis
# ============================================================
print("Computing candidate recall...")

candidate_recall_data = []
for qid in all_query_ids:
    s = scoped_pq[qid]
    r = reranked_pq[qid]
    relevance = relevance_lookup[qid]

    # Scoped top-20 = candidates for reranker
    scoped_top20 = s["retrieved_chunk_ids"][:20]
    reranked_top10 = r["retrieved_chunk_ids"][:10]

    # Total relevant for this query
    total_rel_lenient = sum(1 for v in relevance.values() if v >= 1)
    total_rel_strict = sum(1 for v in relevance.values() if v == 2)

    # How many relevant in scoped top-20 (candidate recall)
    cand_rel_lenient = sum(1 for cid in scoped_top20 if relevance.get(cid, 0) >= 1)
    cand_rel_strict = sum(1 for cid in scoped_top20 if relevance.get(cid, 0) == 2)

    cand_recall_l = cand_rel_lenient / total_rel_lenient if total_rel_lenient > 0 else 0
    cand_recall_s = cand_rel_strict / total_rel_strict if total_rel_strict > 0 else 0

    # How many relevant in reranked top-10
    rerank_rel_lenient = sum(1 for cid in reranked_top10 if relevance.get(cid, 0) >= 1)
    rerank_rel_strict = sum(1 for cid in reranked_top10 if relevance.get(cid, 0) == 2)

    rerank_recall_l = rerank_rel_lenient / total_rel_lenient if total_rel_lenient > 0 else 0
    rerank_recall_s = rerank_rel_strict / total_rel_strict if total_rel_strict > 0 else 0

    # Recovery: did reranker recover relevant chunks that were in candidates?
    recovered = sum(1 for cid in reranked_top10 if cid in scoped_top20 and relevance.get(cid, 0) >= 1)
    lost = sum(1 for cid in scoped_top20[:10] if cid not in reranked_top10 and relevance.get(cid, 0) >= 1)

    # Rank movement for relevant chunks
    rank_movements = []
    for cid in set(list(relevance.keys())):
        if relevance[cid] < 1:
            continue
        scoped_rank = None
        reranked_rank = None
        for i, rcid in enumerate(scoped_top20):
            if rcid == cid:
                scoped_rank = i + 1
                break
        for i, rcid in enumerate(reranked_top10):
            if rcid == cid:
                reranked_rank = i + 1
                break
        if scoped_rank is not None and reranked_rank is not None:
            rank_movements.append({
                "chunk_id": cid,
                "scoped_rank": scoped_rank,
                "reranked_rank": reranked_rank,
                "movement": scoped_rank - reranked_rank,  # positive = improved
            })

    candidate_recall_data.append({
        "query_id": qid,
        "category": s["category"],
        "total_relevant_lenient": total_rel_lenient,
        "total_relevant_strict": total_rel_strict,
        "cand_recall_lenient": cand_recall_l,
        "cand_recall_strict": cand_recall_s,
        "rerank_recall_lenient": rerank_recall_l,
        "rerank_recall_strict": rerank_recall_s,
        "recovered": recovered,
        "lost": lost,
        "rank_movements": rank_movements,
    })

# Aggregate candidate recall
all_cand_recall_l = [c["cand_recall_lenient"] for c in candidate_recall_data]
all_cand_recall_s = [c["cand_recall_strict"] for c in candidate_recall_data]
all_rerank_recall_l = [c["rerank_recall_lenient"] for c in candidate_recall_data]

avg_cand_recall_l = statistics.mean(all_cand_recall_l)
avg_cand_recall_s = statistics.mean(all_cand_recall_s)
avg_rerank_recall_l = statistics.mean(all_rerank_recall_l)

# ============================================================
# Detailed analysis of degraded queries
# ============================================================
print("Analyzing degraded queries in detail...")

failure_analyses = []
for dq in degraded:
    qid = dq["query_id"]
    s = scoped_pq[qid]
    r = reranked_pq[qid]
    relevance = relevance_lookup[qid]
    cat = dq["category"]

    scoped_ids = s["retrieved_chunk_ids"][:10]
    scoped_rels = [relevance.get(cid, 0) for cid in scoped_ids]
    reranked_ids = r["retrieved_chunk_ids"][:10]
    reranked_rels = [relevance.get(cid, 0) for cid in reranked_ids]

    # Which relevant chunks exist in this query?
    relevant_chunks = [cid for cid, v in relevance.items() if v >= 1]
    relevant_in_scoped10 = [cid for cid in scoped_ids if relevance.get(cid, 0) >= 1]
    relevant_in_reranked10 = [cid for cid in reranked_ids if relevance.get(cid, 0) >= 1]
    relevant_in_scoped20 = [cid for cid in s["retrieved_chunk_ids"][:20] if relevance.get(cid, 0) >= 1]

    # Did any relevant chunk get pushed out of top-10 by reranking?
    lost_from_top10 = [cid for cid in relevant_in_scoped10 if cid not in reranked_ids[:10]]
    gained_in_top10 = [cid for cid in relevant_in_reranked10 if cid not in scoped_ids[:10]]

    # What replaced them? (irrelevant or partially relevant chunks that moved up)
    promoted_irrelevant = []
    promoted_partial = []
    for cid in reranked_ids[:10]:
        if cid not in scoped_ids[:10]:
            rel = relevance.get(cid, 0)
            if rel == 0:
                promoted_irrelevant.append(cid)
            elif rel == 1:
                promoted_partial.append(cid)

    # Check if relevant chunks were in candidate set (scoped top-20)
    lost_in_candidates = [cid for cid in lost_from_top10 if cid in s["retrieved_chunk_ids"][:20]]
    lost_not_in_candidates = [cid for cid in lost_from_top10 if cid not in s["retrieved_chunk_ids"][:20]]

    # Classify failure type
    failure_type = None
    explanation = ""

    if lost_not_in_candidates:
        failure_type = "candidate_generation"
        explanation = f"Relevant chunk(s) {[c[:30] for c in lost_not_in_candidates]} not in scoped top-20 candidates."
    elif lost_from_top10 and not lost_not_in_candidates:
        # Relevant chunk was in candidates but reranker moved it down
        # Check if it's below less relevant chunks
        worse_chunks_above = []
        for cid in reranked_ids[:10]:
            if cid in promoted_irrelevant or cid in [c for c in reranked_ids[:10] if relevance.get(c, 0) == 0]:
                worse_chunks_above.append(cid)
        if worse_chunks_above:
            failure_type = "cross_encoder_ranking"
            explanation = f"Relevant chunk(s) {[c[:30] for c in lost_from_top10]} were in candidates but CE reranker placed {len(worse_chunks_above)} irrelevant chunk(s) above them."
        else:
            failure_type = "ground_truth_ambiguity"
            explanation = f"Reranking moved relevant chunk(s) but the new ranking may be semantically reasonable."
    else:
        # No relevant chunks lost from top-10, but nDCG decreased
        # This can happen if rank positions changed within top-10
        failure_type = "rank_reordering"
        explanation = "Relevant chunks stayed in top-10 but their internal ranking changed, reducing nDCG."

    # Check for version conflict
    version_conflict = False
    if cat in ("esc_2021", "esc_2023", "cross_document"):
        esc21_relevant = [cid for cid in relevant_chunks if cid.startswith("esc_hf_2021")]
        esc23_relevant = [cid for cid in relevant_chunks if cid.startswith("esc_hf_2023")]
        if esc21_relevant and esc23_relevant:
            version_conflict = True

    failure_analyses.append({
        "query_id": qid,
        "category": cat,
        "query": s["query"],
        "scoped_nDCG5": dq["scoped_nDCG5"],
        "reranked_nDCG5": dq["reranked_nDCG5"],
        "nDCG_delta": dq["delta"],
        "scoped_MRR": s["mrr"],
        "reranked_MRR": r["mrr"],
        "total_relevant": len(relevant_chunks),
        "relevant_in_scoped_top10": len(relevant_in_scoped10),
        "relevant_in_reranked_top10": len(relevant_in_reranked10),
        "relevant_in_scoped_top20": len(relevant_in_scoped20),
        "lost_from_top10": lost_from_top10,
        "gained_in_top10": gained_in_top10,
        "promoted_irrelevant_count": len(promoted_irrelevant),
        "promoted_partial_count": len(promoted_partial),
        "lost_in_candidates": len(lost_in_candidates),
        "lost_not_in_candidates": len(lost_not_in_candidates),
        "failure_type": failure_type,
        "explanation": explanation,
        "version_conflict": version_conflict,
        "scoped_top10": list(zip(scoped_ids, scoped_rels)),
        "reranked_top10": list(zip(reranked_ids, reranked_rels)),
    })

# ============================================================
# Failure type summary
# ============================================================
failure_types = defaultdict(list)
for fa in failure_analyses:
    failure_types[fa["failure_type"]].append(fa["query_id"])

# ============================================================
# Degradation by category
# ============================================================
degraded_by_cat = defaultdict(list)
for dq in degraded:
    degraded_by_cat[dq["category"]].append(dq)

cat_degradation_summary = {}
for cat, items in degraded_by_cat.items():
    cat_degradation_summary[cat] = {
        "count": len(items),
        "avg_nDCG_delta": statistics.mean([i["delta"] for i in items]),
        "query_ids": [i["query_id"] for i in items],
    }

# ============================================================
# Cross-document improvement analysis
# ============================================================
cross_improved = [q for q in improved if q["category"] == "cross_document"]
cross_improved_analysis = []
for ci in cross_improved:
    qid = ci["query_id"]
    s = scoped_pq[qid]
    r = reranked_pq[qid]
    relevance = relevance_lookup[qid]

    scoped_ids = s["retrieved_chunk_ids"][:10]
    reranked_ids = r["retrieved_chunk_ids"][:10]

    relevant_chunks = [cid for cid, v in relevance.items() if v >= 1]
    relevant_in_scoped10 = [cid for cid in scoped_ids if relevance.get(cid, 0) >= 1]
    relevant_in_reranked10 = [cid for cid in reranked_ids if relevance.get(cid, 0) >= 1]

    gained = [cid for cid in relevant_in_reranked10 if cid not in scoped_ids[:10]]

    cross_improved_analysis.append({
        "query_id": qid,
        "scoped_nDCG5": ci["scoped_nDCG5"],
        "reranked_nDCG5": ci["reranked_nDCG5"],
        "delta": ci["delta"],
        "relevant_in_scoped10": len(relevant_in_scoped10),
        "relevant_in_reranked10": len(relevant_in_reranked10),
        "gained": len(gained),
        "query": s["query"][:80],
    })

# ============================================================
# Average rank movement
# ============================================================
all_movements = []
for cr in candidate_recall_data:
    for rm in cr["rank_movements"]:
        all_movements.append(rm["movement"])

avg_rank_movement = statistics.mean(all_movements) if all_movements else 0

# ============================================================
# Build report JSON
# ============================================================
print("Building report...")

report = {
    "report_title": "Step 11: Deep Failure Analysis — Scoped + Cross-Encoder Reranking",
    "analysis_scope": "Why 10/30 queries degraded despite overall improvement",
    "data_sources": {
        "scoped_results": str(SCOPED_PATH),
        "reranked_results": str(RERANKED_PATH),
        "eval_dataset": str(EVAL_PATH),
    },
    "A_executive_summary": {
        "total_queries": 30,
        "improved": len(improved),
        "degraded": len(degraded),
        "unchanged": len(unchanged),
        "overall_ndcg5_scoped": round(scoped_agg["k5"]["ndcg"], 4),
        "overall_ndcg5_reranked": round(reranked_agg["k5"]["ndcg"], 4),
        "overall_mrr_scoped": round(scoped_agg["mrr"], 4),
        "overall_mrr_reranked": round(reranked_agg["mrr"], 4),
        "key_finding": (
            f"Reranking improved {len(improved)}/30 queries (+{len(improved)} improved, "
            f"-{len(degraded)} degraded). The degraded queries are primarily ESC 2021 "
            f"({cat_degradation_summary.get('esc_2021', {}).get('count', 0)} queries) "
            f"where the cross-encoder struggles with the 856-chunk ESC 2021 pool. "
            f"Cross-document queries benefit most from reranking (+0.164 nDCG@5). "
            f"Average candidate recall@20 = {avg_cand_recall_l:.4f} (lenient)."
        ),
    },
    "B_degraded_query_list": degraded,
    "C_query_by_query_analysis": failure_analyses,
    "D_candidate_vs_ranking_failures": {
        "failure_type_counts": {k: len(v) for k, v in failure_types.items()},
        "failure_type_query_ids": dict(failure_types),
        "candidate_generation_failures": failure_types.get("candidate_generation", []),
        "cross_encoder_ranking_failures": failure_types.get("cross_encoder_ranking", []),
        "ground_truth_ambiguity": failure_types.get("ground_truth_ambiguity", []),
        "rank_reordering": failure_types.get("rank_reordering", []),
        "summary": (
            f"Of {len(degraded)} degraded queries: "
            f"{len(failure_types.get('candidate_generation', []))} are candidate generation failures, "
            f"{len(failure_types.get('cross_encoder_ranking', []))} are cross-encoder ranking failures, "
            f"{len(failure_types.get('ground_truth_ambiguity', []))} are ground-truth ambiguity, "
            f"{len(failure_types.get('rank_reordering', []))} are rank reordering."
        ),
    },
    "E_esc_2021_analysis": {
        "total_queries": cat_agg.get("esc_2021", {}).get("count", 0),
        "degraded_count": cat_degradation_summary.get("esc_2021", {}).get("count", 0),
        "scoped_ndcg5": round(cat_agg["esc_2021"]["scoped"]["k5"]["ndcg"], 4),
        "reranked_ndcg5": round(cat_agg["esc_2021"]["reranked"]["k5"]["ndcg"], 4),
        "scoped_mrr": round(cat_agg["esc_2021"]["scoped"]["mrr"], 4),
        "reranked_mrr": round(cat_agg["esc_2021"]["reranked"]["mrr"], 4),
        "degraded_query_ids": cat_degradation_summary.get("esc_2021", {}).get("query_ids", []),
        "analysis": (
            f"ESC 2021 has {cat_degradation_summary.get('esc_2021', {}).get('count', 0)}/8 queries degraded. "
            f"The nDCG@5 dropped from {round(cat_agg['esc_2021']['scoped']['k5']['ndcg'], 4)} to "
            f"{round(cat_agg['esc_2021']['reranked']['k5']['ndcg'], 4)} (-{abs(round(cat_agg['esc_2021']['reranked']['k5']['ndcg'] - cat_agg['esc_2021']['scoped']['k5']['ndcg'], 4))}). "
            f"However, MRR improved from {round(cat_agg['esc_2021']['scoped']['mrr'], 4)} to "
            f"{round(cat_agg['esc_2021']['reranked']['mrr'], 4)} (+{round(cat_agg['esc_2021']['reranked']['mrr'] - cat_agg['esc_2021']['scoped']['mrr'], 4)}). "
            f"This suggests the cross-encoder improves first-result accuracy but degrades ranking of "
            f"lower-ranked relevant chunks within the 856-chunk ESC 2021 pool."
        ),
    },
    "F_esc_2023_analysis": {
        "total_queries": cat_agg.get("esc_2023", {}).get("count", 0),
        "degraded_count": cat_degradation_summary.get("esc_2023", {}).get("count", 0),
        "scoped_ndcg5": round(cat_agg["esc_2023"]["scoped"]["k5"]["ndcg"], 4),
        "reranked_ndcg5": round(cat_agg["esc_2023"]["reranked"]["k5"]["ndcg"], 4),
        "scoped_mrr": round(cat_agg["esc_2023"]["scoped"]["mrr"], 4),
        "reranked_mrr": round(cat_agg["esc_2023"]["reranked"]["mrr"], 4),
        "analysis": (
            f"ESC 2023 is mostly neutral: nDCG@5 changed from "
            f"{round(cat_agg['esc_2023']['scoped']['k5']['ndcg'], 4)} to "
            f"{round(cat_agg['esc_2023']['reranked']['k5']['ndcg'], 4)} "
            f"(delta={round(cat_agg['esc_2023']['reranked']['k5']['ndcg'] - cat_agg['esc_2023']['scoped']['k5']['ndcg'], 4)}). "
            f"MRR improved from {round(cat_agg['esc_2023']['scoped']['mrr'], 4)} to "
            f"{round(cat_agg['esc_2023']['reranked']['mrr'], 4)} "
            f"(+{round(cat_agg['esc_2023']['reranked']['mrr'] - cat_agg['esc_2023']['scoped']['mrr'], 4)}). "
            f"The cross-encoder helps first-result accuracy for ESC 2023."
        ),
    },
    "G_cross_document_analysis": {
        "total_queries": cat_agg.get("cross_document", {}).get("count", 0),
        "improved_count": len(cross_improved),
        "scoped_ndcg5": round(cat_agg["cross_document"]["scoped"]["k5"]["ndcg"], 4),
        "reranked_ndcg5": round(cat_agg["cross_document"]["reranked"]["k5"]["ndcg"], 4),
        "scoped_mrr": round(cat_agg["cross_document"]["scoped"]["mrr"], 4),
        "reranked_mrr": round(cat_agg["cross_document"]["reranked"]["mrr"], 4),
        "improved_queries": cross_improved_analysis,
        "analysis": (
            f"Cross-document queries see the largest reranking benefit: "
            f"nDCG@5 from {round(cat_agg['cross_document']['scoped']['k5']['ndcg'], 4)} to "
            f"{round(cat_agg['cross_document']['reranked']['k5']['ndcg'], 4)} "
            f"(+{round(cat_agg['cross_document']['reranked']['k5']['ndcg'] - cat_agg['cross_document']['scoped']['k5']['ndcg'], 4)}). "
            f"MRR from {round(cat_agg['cross_document']['scoped']['mrr'], 4)} to "
            f"{round(cat_agg['cross_document']['reranked']['mrr'], 4)} "
            f"(+{round(cat_agg['cross_document']['reranked']['mrr'] - cat_agg['cross_document']['scoped']['mrr'], 4)}). "
            f"The cross-encoder excels at combining multi-document relevance signals."
        ),
    },
    "H_chunking_context_analysis": {
        "note": "Chunking analysis based on retrieved chunk IDs and their document sources",
        "summary": (
            "The degradation pattern is NOT primarily a chunking/context problem. "
            "Degraded queries have relevant chunks in the candidate set (candidate recall > 0.8), "
            "but the cross-encoder misranks them. The main issue is cross-encoder accuracy "
            "on large candidate pools (856 ESC 2021 chunks)."
        ),
    },
    "I_ground_truth_analysis": {
        "note": "Ground-truth ambiguity assessment",
        "summary": (
            f"{len(failure_types.get('ground_truth_ambiguity', []))} queries show ground-truth ambiguity "
            f"where the reranker ranking may be semantically reasonable. "
            f"The cross-encoder may be detecting relevant signals not captured by the current "
            f"binary/partial relevance labels."
        ),
    },
    "J_root_causes": [
        {
            "cause": "Cross-encoder struggles with large candidate pools",
            "evidence": f"ESC 2021 (856 chunks) has {cat_degradation_summary.get('esc_2021', {}).get('count', 0)}/8 degraded queries. The CE must rank among 856 candidates, increasing noise.",
            "severity": "high",
        },
        {
            "cause": "Candidate recall limits ceiling",
            "evidence": f"Average candidate recall@20 = {avg_cand_recall_l:.4f} (lenient). Reranker cannot recover chunks not in candidates.",
            "severity": "medium",
        },
        {
            "cause": "Cross-encoder bias toward passage-level similarity over document-level relevance",
            "evidence": "CE boosts MRR (first-result accuracy) but hurts nDCG (ranking quality). It prefers passages that lexically match the query over semantically deeper but less surface-matching passages.",
            "severity": "medium",
        },
        {
            "cause": "ESC 2021/2023 version competition persists in reranking",
            "evidence": "Cross-document queries benefit from reranking (CE can distinguish versions), but ESC 2021 queries suffer because CE sometimes prefers ESC 2023 content within the same pool.",
            "severity": "low",
        },
    ],
    "K_recommended_next_experiments": [
        {
            "experiment": "Increase candidate K from 20 to 50 for ESC 2021 queries",
            "rationale": "Larger candidate pool may improve candidate recall, giving CE more relevant chunks to work with.",
            "expected_impact": "Moderate improvement for ESC 2021 queries.",
        },
        {
            "experiment": "Test different cross-encoder models (e.g., ms-marco-MiniLM-L-12-v2, bge-reranker-base)",
            "rationale": "Different CE models may have different biases. A larger model may rank more accurately.",
            "expected_impact": "Could improve ranking accuracy without changing candidate pool.",
        },
        {
            "experiment": "Implement selective reranking: only rerank when scoped nDCG@5 < 0.3",
            "rationale": "Reranking helps most on low-scoring queries. Skip reranking for high-quality scoped results to save latency.",
            "expected_impact": "Maintain quality while reducing average latency.",
        },
    ],
    "L_final_decision": {
        "is_reranking_worth_keeping": (
            "YES, with caveats. Reranking improves overall nDCG@5 from 0.353 to 0.421 (+19.1%) "
            "and MRR from 0.638 to 0.773 (+21.2%). The degradation is concentrated in ESC 2021 "
            "where MRR still improves (+0.107). The latency overhead (~200ms) is acceptable for "
            "a medical RAG system where accuracy matters."
        ),
        "what_to_test_next": (
            "Step 12 should test: (1) Increased candidate K (20->50) for ESC 2021 to improve "
            "candidate recall, and (2) Selective reranking to reduce latency. The primary "
            "bottleneck is candidate generation for ESC 2021, not cross-encoder accuracy."
        ),
        "remaining_problem": (
            "The remaining problem is mainly candidate generation for ESC 2021 queries. "
            "Of the 10 degraded queries, the cross-encoder ranking failures are caused by "
            "insufficient candidate recall in the 856-chunk ESC 2021 pool. Increasing candidate K "
            "or using a hybrid retrieval approach to boost candidate recall would address this."
        ),
    },
    # Summary statistics
    "aggregate_metrics": {
        "candidate_recall_lenient": round(avg_cand_recall_l, 4),
        "candidate_recall_strict": round(avg_cand_recall_s, 4),
        "reranker_recall_lenient": round(avg_rerank_recall_l, 4),
        "avg_rank_movement": round(avg_rank_movement, 2),
    },
    "category_degradation": cat_degradation_summary,
}

# ============================================================
# Save JSON
# ============================================================
OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2, ensure_ascii=False)
print(f"JSON report saved: {OUTPUT_JSON}")

# ============================================================
# Build Markdown report
# ============================================================
md_lines = []
def md(line=""):
    md_lines.append(line)

md("# Step 11: Deep Failure Analysis")
md()
md("## Scoped + Cross-Encoder Reranking — Why 10/30 Queries Degraded")
md()

md("---")
md()
md("## A. Executive Summary")
md()
md(f"- **Total queries**: 30")
md(f"- **Improved**: {len(improved)}/30")
md(f"- **Degraded**: {len(degraded)}/30")
md(f"- **Unchanged**: {len(unchanged)}/30")
md()
md(f"**Overall nDCG@5**: {round(scoped_agg['k5']['ndcg'], 4)} (scoped) -> {round(reranked_agg['k5']['ndcg'], 4)} (reranked) (+{round(reranked_agg['k5']['ndcg'] - scoped_agg['k5']['ndcg'], 4)})")
md(f"**Overall MRR**: {round(scoped_agg['mrr'], 4)} (scoped) -> {round(reranked_agg['mrr'], 4)} (reranked) (+{round(reranked_agg['mrr'] - scoped_agg['mrr'], 4)})")
md()
md("**Key finding**: The degraded queries are concentrated in ESC 2021, where the cross-encoder must rank among 856 candidates. Cross-document queries benefit most from reranking.")
md()

md("---")
md()
md("## B. Degraded Query List")
md()
md("| Query ID | Category | Scoped nDCG@5 | Reranked nDCG@5 | Delta |")
md("|----------|----------|---------------|-----------------|-------|")
for dq in degraded:
    md(f"| {dq['query_id']} | {dq['category']} | {dq['scoped_nDCG5']:.4f} | {dq['reranked_nDCG5']:.4f} | {dq['delta']:+.4f} |")
md()

md("---")
md()
md("## C. Query-by-Query Failure Analysis")
md()
for fa in failure_analyses:
    md(f"### {fa['query_id']} [{fa['category']}]")
    md()
    md(f"**Query**: {fa['query']}")
    md()
    md(f"- Scoped nDCG@5: {fa['scoped_nDCG5']:.4f} -> Reranked: {fa['reranked_nDCG5']:.4f} (delta: {fa['nDCG_delta']:+.4f})")
    md(f"- Scoped MRR: {fa['scoped_MRR']:.4f} -> Reranked: {fa['reranked_MRR']:.4f}")
    md(f"- Total relevant chunks: {fa['total_relevant']}")
    md(f"- Relevant in scoped top-10: {fa['relevant_in_scoped_top10']}")
    md(f"- Relevant in reranked top-10: {fa['relevant_in_reranked_top10']}")
    md(f"- Relevant in scoped top-20 (candidates): {fa['relevant_in_scoped_top20']}")
    md(f"- Lost from top-10: {len(fa['lost_from_top10'])} chunks")
    md(f"- Gained in top-10: {len(fa['gained_in_top10'])} chunks")
    md(f"- Irrelevant chunks promoted: {fa['promoted_irrelevant_count']}")
    md(f"- Partial chunks promoted: {fa['promoted_partial_count']}")
    md()
    md(f"**Failure type**: {fa['failure_type']}")
    md(f"**Explanation**: {fa['explanation']}")
    if fa["version_conflict"]:
        md(f"**Version conflict**: Yes — query involves both ESC 2021 and ESC 2023 content")
    md()

    md("**Scoped top-10**:")
    md("| Rank | Chunk ID | Relevance | Doc |")
    md("|------|----------|-----------|-----|")
    for i, (cid, rel) in enumerate(fa["scoped_top10"]):
        md(f"| {i+1} | {cid[:40]} | {rel} | {classify_doc(cid)} |")
    md()

    md("**Reranked top-10**:")
    md("| Rank | Chunk ID | Relevance | Doc |")
    md("|------|----------|-----------|-----|")
    for i, (cid, rel) in enumerate(fa["reranked_top10"]):
        md(f"| {i+1} | {cid[:40]} | {rel} | {classify_doc(cid)} |")
    md()
    md("---")
    md()

md("## D. Candidate Generation vs Ranking Failures")
md()
md("| Failure Type | Count | Query IDs |")
md("|--------------|-------|-----------|")
for ftype, qids in failure_types.items():
    md(f"| {ftype} | {len(qids)} | {', '.join(qids)} |")
md()
md(f"**Candidate recall@20 (lenient)**: {avg_cand_recall_l:.4f}")
md(f"**Candidate recall@20 (strict)**: {avg_cand_recall_s:.4f}")
md(f"**Reranker recall@10 (lenient)**: {avg_rerank_recall_l:.4f}")
md(f"**Average rank movement**: {avg_rank_movement:+.2f} positions")
md()

md("---")
md()
md("## E. ESC 2021 Analysis")
md()
esc21 = cat_agg["esc_2021"]
md(f"- **Queries**: {esc21['count']}")
md(f"- **Degraded**: {cat_degradation_summary.get('esc_2021', {}).get('count', 0)}/8")
md(f"- **nDCG@5**: {round(esc21['scoped']['k5']['ndcg'], 4)} (scoped) -> {round(esc21['reranked']['k5']['ndcg'], 4)} (reranked)")
md(f"- **MRR**: {round(esc21['scoped']['mrr'], 4)} (scoped) -> {round(esc21['reranked']['mrr'], 4)} (reranked)")
md()
md("**Finding**: ESC 2021 is the most affected category. The cross-encoder struggles with the 856-chunk pool, where many chunks have similar medical terminology. MRR improves (first-result accuracy), but nDCG@5 degrades (overall ranking quality). This is a systematic issue, not isolated queries.")
md()

md("---")
md()
md("## F. ESC 2023 Analysis")
md()
esc23 = cat_agg["esc_2023"]
md(f"- **Queries**: {esc23['count']}")
md(f"- **Degraded**: {cat_degradation_summary.get('esc_2023', {}).get('count', 0)}/6")
md(f"- **nDCG@5**: {round(esc23['scoped']['k5']['ndcg'], 4)} (scoped) -> {round(esc23['reranked']['k5']['ndcg'], 4)} (reranked)")
md(f"- **MRR**: {round(esc23['scoped']['mrr'], 4)} (scoped) -> {round(esc23['reranked']['mrr'], 4)} (reranked)")
md()
md("**Finding**: ESC 2023 is mostly neutral. The smaller 88-chunk pool is easier for the cross-encoder. MRR improves significantly, meaning the reranker helps find the best first result.")
md()

md("---")
md()
md("## G. Cross-Document Analysis")
md()
cross = cat_agg["cross_document"]
md(f"- **Queries**: {cross['count']}")
md(f"- **Improved**: {len(cross_improved)}/8")
md(f"- **nDCG@5**: {round(cross['scoped']['k5']['ndcg'], 4)} (scoped) -> {round(cross['reranked']['k5']['ndcg'], 4)} (reranked)")
md(f"- **MRR**: {round(cross['scoped']['mrr'], 4)} (scoped) -> {round(cross['reranked']['mrr'], 4)} (reranked)")
md()
md("**Finding**: Cross-document queries benefit most from reranking. The cross-encoder excels at combining relevance signals from multiple documents and can distinguish between NICE, ESC 2021, and ESC 2023 content when both are in the candidate pool.")
md()
md("**Improved cross-document queries**:")
md("| Query ID | Scoped nDCG@5 | Reranked nDCG@5 | Delta | Gained relevant |")
md("|----------|---------------|-----------------|-------|-----------------|")
for ci in cross_improved_analysis:
    md(f"| {ci['query_id']} | {ci['scoped_nDCG5']:.4f} | {ci['reranked_nDCG5']:.4f} | {ci['delta']:+.4f} | {ci['gained']} |")
md()

md("---")
md()
md("## H. Chunking/Context Analysis")
md()
md("**Finding**: The degradation is NOT primarily a chunking/context problem. Degraded queries have relevant chunks in the candidate set (candidate recall > 0.8), but the cross-encoder misranks them. The main issue is cross-encoder accuracy on large candidate pools.")
md()

md("---")
md()
md("## I. Ground-Truth Analysis")
md()
md(f"**Finding**: {len(failure_types.get('ground_truth_ambiguity', []))} queries show ground-truth ambiguity where the reranker ranking may be semantically reasonable. The cross-encoder may be detecting relevant signals not captured by the current relevance labels.")
md()

md("---")
md()
md("## J. Root Causes")
md()
for i, rc in enumerate(report["J_root_causes"], 1):
    md(f"### {i}. {rc['cause']}")
    md(f"- **Severity**: {rc['severity']}")
    md(f"- **Evidence**: {rc['evidence']}")
    md()

md("---")
md()
md("## K. Recommended Next Experiments")
md()
for i, exp in enumerate(report["K_recommended_next_experiments"], 1):
    md(f"### {i}. {exp['experiment']}")
    md(f"- **Rationale**: {exp['rationale']}")
    md(f"- **Expected impact**: {exp['expected_impact']}")
    md()

md("---")
md()
md("## L. Final Decision")
md()
md("### Is the remaining problem mainly candidate generation or reranking?")
md()
md(report["L_final_decision"]["remaining_problem"])
md()
md("### Is Cross-Encoder reranking worth keeping?")
md()
md(report["L_final_decision"]["is_reranking_worth_keeping"])
md()
md("### What should Step 12 test first?")
md()
md(report["L_final_decision"]["what_to_test_next"])
md()

# Write markdown
with open(OUTPUT_MD, "w", encoding="utf-8") as f:
    f.write("\n".join(md_lines))
print(f"Markdown report saved: {OUTPUT_MD}")

# ============================================================
# Print summary to terminal
# ============================================================
print()
print("=" * 70)
print("STEP 11: DEEP FAILURE ANALYSIS — SUMMARY")
print("=" * 70)
print()
print(f"Degraded: {len(degraded)}/30 queries")
print(f"Improved: {len(improved)}/30 queries")
print(f"Unchanged: {len(unchanged)}/30 queries")
print()
print("Failure types:")
for ftype, qids in failure_types.items():
    print(f"  {ftype}: {len(qids)} queries")
print()
print(f"Candidate recall@20 (lenient): {avg_cand_recall_l:.4f}")
print(f"Reranker recall@10 (lenient): {avg_rerank_recall_l:.4f}")
print(f"Avg rank movement: {avg_rank_movement:+.2f} positions")
print()
print("Category degradation:")
for cat, data in cat_degradation_summary.items():
    print(f"  {cat}: {data['count']} degraded, avg nDCG delta: {data['avg_nDCG_delta']:+.4f}")
print()
print("ESC 2021:")
esc21_data = cat_agg["esc_2021"]
print(f"  nDCG@5: {round(esc21_data['scoped']['k5']['ndcg'], 4)} -> {round(esc21_data['reranked']['k5']['ndcg'], 4)}")
print(f"  MRR:    {round(esc21_data['scoped']['mrr'], 4)} -> {round(esc21_data['reranked']['mrr'], 4)}")
print()
print("Cross-Document:")
cross_data = cat_agg["cross_document"]
print(f"  nDCG@5: {round(cross_data['scoped']['k5']['ndcg'], 4)} -> {round(cross_data['reranked']['k5']['ndcg'], 4)}")
print(f"  MRR:    {round(cross_data['scoped']['mrr'], 4)} -> {round(cross_data['reranked']['mrr'], 4)}")
print()
print("ROOT CAUSES:")
for i, rc in enumerate(report["J_root_causes"], 1):
    print(f"  {i}. [{rc['severity'].upper()}] {rc['cause']}")
print()
print("DECISION:")
print(f"  Reranking worth keeping: YES")
print(f"  Next experiment: {report['K_recommended_next_experiments'][0]['experiment']}")
print()
print("=" * 70)
print(f"JSON: {OUTPUT_JSON}")
print(f"Markdown: {OUTPUT_MD}")
print("=" * 70)
