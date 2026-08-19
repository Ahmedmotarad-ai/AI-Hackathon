"""
Evaluate Current Retrieval — Behavioral Categories
====================================================
Runs the finalized retrieval/reranking pipeline on the new behavioral
evaluation dataset and computes metrics per category.

DO NOT modify the retrieval configuration. This script READS the locked
pipeline only.
"""
import csv
import json
import math
import statistics
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder

# ── Locked Retrieval Configuration ────────────────────────────
DB_PATH = Path("data/vector_db")
COLLECTION_NAME = "medical_guidelines"
BGE_MODEL = "BAAI/bge-small-en-v1.5"
CE_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
QUERY_PREFIX = "query: "
CANDIDATE_K = 20

EVAL_FILE = Path("data/evaluation/new_evaluation_dataset.json")
CHUNKS_FILE = Path("data/chunks/chunks.jsonl")
RESULTS_DIR = Path("data/evaluation/results")
K_VALUES = [3, 5, 10]

CATEGORY_FILTERS = {
    "cross_document": {"section": {"$ne": "Front matter"}},
}

# ── Metric Functions (consistent with finalize_retrieval.py) ───
def precision_at_k(retrieved, relevant_set, k):
    if k == 0:
        return 0.0
    return sum(1 for c in retrieved[:k] if c in relevant_set) / k

def recall_at_k(retrieved, relevant_set, k):
    if not relevant_set:
        return 0.0
    return sum(1 for c in retrieved[:k] if c in relevant_set) / len(relevant_set)

def f1_at_k(p, r):
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

def hit_at_k(retrieved, relevant_set, k):
    return 1.0 if any(c in relevant_set for c in retrieved[:k]) else 0.0

def ndcg_at_k(retrieved, relevance_map, k):
    dcg = 0.0
    for i, c in enumerate(retrieved[:k]):
        rel = relevance_map.get(c, 0)
        dcg += (2**rel - 1) / math.log2(i + 2)
    all_rels = sorted(relevance_map.values(), reverse=True)
    idcg = sum((2**r - 1) / math.log2(i + 2) for i, r in enumerate(all_rels[:k]))
    return dcg / idcg if idcg > 0 else 0.0

def mrr_score(retrieved, relevant_set):
    for i, c in enumerate(retrieved):
        if c in relevant_set:
            return 1.0 / (i + 1)
    return 0.0

def eval_query(retrieved, relevance_map, k_values):
    relevant_set = {c for c, r in relevance_map.items() if r >= 1}
    strict_set = {c for c, r in relevance_map.items() if r >= 2}
    result = {"mrr": mrr_score(retrieved, relevant_set)}
    for k in k_values:
        p = precision_at_k(retrieved, relevant_set, k)
        r = recall_at_k(retrieved, relevant_set, k)
        result[k] = {
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(f1_at_k(p, r), 4),
            "hit": round(hit_at_k(retrieved, relevant_set, k), 4),
            "ndcg": round(ndcg_at_k(retrieved, relevance_map, k), 4),
        }
    return result

# ── Data Loading ──────────────────────────────────────────────
def load_chunks_metadata(path):
    meta = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                meta[r["chunk_id"]] = {
                    "document": r.get("document", ""),
                    "section": r.get("section", ""),
                    "text": r.get("text", ""),
                }
    return meta

# ── Main ──────────────────────────────────────────────────────
def main():
    print("=" * 72)
    print("EVALUATE CURRENT RETRIEVAL — Behavioral Categories")
    print("=" * 72)

    dataset = json.loads(EVAL_FILE.read_text(encoding="utf-8"))
    chunk_meta = load_chunks_metadata(CHUNKS_FILE)
    queries = dataset["queries"]
    print(f"Loaded {len(queries)} queries, {len(chunk_meta)} chunks")

    print(f"\nLoading Dense: {BGE_MODEL}")
    bge = SentenceTransformer(BGE_MODEL, device="cpu")
    print(f"Loading Reranker: {CE_MODEL}")
    reranker = CrossEncoder(CE_MODEL, max_length=512)

    client = chromadb.PersistentClient(path=str(DB_PATH))
    coll = client.get_collection(name=COLLECTION_NAME)
    print(f"ChromaDB: {coll.count()} records")

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    filt = CATEGORY_FILTERS["cross_document"]

    all_results = []
    per_query_output = []
    cat_groups = defaultdict(list)

    print(f"\nRunning {len(queries)} queries...")

    for qi, q in enumerate(queries):
        qid = q["query_id"]
        qt = q["query"]
        cat = q["category"]
        rel_map = q.get("relevant_chunks", {})

        fq = QUERY_PREFIX + qt
        t0 = time.perf_counter()
        emb = bge.encode(fq, normalize_embeddings=True).tolist()
        res = coll.query(query_embeddings=[emb], n_results=CANDIDATE_K, where=filt)
        t1 = time.perf_counter()
        dense_ms = (t1 - t0) * 1000

        cand_ids = res["ids"][0]
        cand_dists = res["distances"][0]

        t2 = time.perf_counter()
        cand_texts = [chunk_meta.get(cid, {}).get("text", "") for cid in cand_ids]
        pairs = [(qt, ct) for ct in cand_texts]
        ce_scores = reranker.predict(pairs)
        t3 = time.perf_counter()
        rerank_ms = (t3 - t2) * 1000
        total_ms = dense_ms + rerank_ms

        scored = list(zip(cand_ids, cand_dists, ce_scores))
        scored.sort(key=lambda x: -x[2])
        reranked_ids = [s[0] for s in scored]

        relevant_set = {c for c, r in rel_map.items() if r >= 1}
        ev = eval_query(reranked_ids, rel_map, K_VALUES)

        # Build per-row output for CSV
        for rerank_rank, (cid, dist, ce_sc) in enumerate(scored, start=1):
            all_results.append({
                "query_id": qid,
                "query": qt,
                "category": cat,
                "rerank_rank": rerank_rank,
                "chunk_id": cid,
                "document": chunk_meta.get(cid, {}).get("document", ""),
                "section": chunk_meta.get(cid, {}).get("section", ""),
                "retrieval_score": round(float(dist), 6),
                "rerank_score": round(float(ce_sc), 6),
                "in_ground_truth": 1 if cid in rel_map else 0,
                "relevance_grade": rel_map.get(cid, 0),
            })

        pq = {
            "query_id": qid,
            "query": qt,
            "category": cat,
            "expected_behavior": q.get("expected_behavior", ""),
            "retrieved_chunk_ids": reranked_ids,
            "retrieved_ce_scores": [round(float(s[2]), 6) for s in scored],
            "ground_truth_count": len(relevant_set),
            "retrieved_relevant_in_top3": len([c for c in reranked_ids[:3] if c in relevant_set]),
            "retrieved_relevant_in_top5": len([c for c in reranked_ids[:5] if c in relevant_set]),
            "retrieved_relevant_in_top10": len([c for c in reranked_ids[:10] if c in relevant_set]),
            "metrics": ev,
            "latency": {"dense_ms": round(dense_ms, 2), "rerank_ms": round(rerank_ms, 2), "total_ms": round(total_ms, 2)},
        }
        per_query_output.append(pq)
        cat_groups[cat].append(pq)

        n3 = ev[3]["ndcg"]
        n5 = ev[5]["ndcg"]
        print(f"  {qid:>7} [{cat:>20}] nDCG@3={n3:.3f}  nDCG@5={n5:.3f}  "
              f"gt_chunks={len(relevant_set)}  top3_hit={pq['retrieved_relevant_in_top3']}  "
              f"lat={total_ms:.0f}ms")

    # ── Aggregate Overall ──────────────────────────────────────
    def agg_metrics(pqr_list, k_values):
        a = {"mrr": statistics.mean([q["metrics"]["mrr"] for q in pqr_list])}
        for k in k_values:
            a[k] = {
                mk: statistics.mean([q["metrics"][k][mk] for q in pqr_list])
                for mk in ["precision", "recall", "f1", "hit", "ndcg"]
            }
        return a

    overall = agg_metrics(per_query_output, K_VALUES)

    # ── Aggregate Per-Category ─────────────────────────────────
    cat_agg = {}
    for cat, cat_pq in cat_groups.items():
        cat_agg[cat] = {
            "n": len(cat_pq),
            "metrics": agg_metrics(cat_pq, K_VALUES),
        }

    # ── Special Metrics for OOS/Insufficient ───────────────────
    # For out_of_scope and insufficient_evidence: measure false-retrieval rate
    oos_insuff_cats = ["out_of_scope", "insufficient_evidence"]
    false_retrieval = {}
    for cat in oos_insuff_cats:
        cqs = cat_groups.get(cat, [])
        if not cqs:
            continue
        # False retrieval = queries where at least one top-3 chunk has relevance >= 1
        false_top3 = sum(1 for q in cqs if q["retrieved_relevant_in_top3"] > 0)
        false_top5 = sum(1 for q in cqs if q["retrieved_relevant_in_top5"] > 0)
        false_top10 = sum(1 for q in cqs if q["retrieved_relevant_in_top10"] > 0)
        false_retrieval[cat] = {
            "n": len(cqs),
            "false_evidence_top3": false_top3,
            "false_evidence_top5": false_top5,
            "false_evidence_top10": false_top10,
            "false_evidence_rate_top3": round(false_top3 / len(cqs), 4) if cqs else 0,
            "false_evidence_rate_top5": round(false_top5 / len(cqs), 4) if cqs else 0,
        }

    # ── Failure Analysis ───────────────────────────────────────
    failures = []
    for pq in per_query_output:
        cat = pq["category"]
        gt_count = pq["ground_truth_count"]
        top3_rel = pq["retrieved_relevant_in_top3"]
        top5_rel = pq["retrieved_relevant_in_top5"]
        top10_rel = pq["retrieved_relevant_in_top10"]

        if cat in ["out_of_scope", "insufficient_evidence"]:
            if top3_rel > 0:
                failures.append({
                    "query_id": pq["query_id"],
                    "query": pq["query"],
                    "category": cat,
                    "failure_type": "false_evidence_retrieved",
                    "detail": f"Retrieved {top3_rel} apparently relevant chunks in top-3 for {cat} query",
                    "top3_chunks": pq["retrieved_chunk_ids"][:3],
                })
        elif cat == "multi_part":
            # Check if both parts covered (rough check: need chunks from at least 2 sections)
            top5_docs = set()
            for cid in pq["retrieved_chunk_ids"][:5]:
                cm = chunk_meta.get(cid, {})
                top5_docs.add(cm.get("section", ""))
            if len(top5_docs) < 2 and gt_count >= 2:
                failures.append({
                    "query_id": pq["query_id"],
                    "query": pq["query"],
                    "category": cat,
                    "failure_type": "single_part_coverage",
                    "detail": f"Top-5 covers {len(top5_docs)} sections but query has {gt_count} ground truth chunks across multiple topics",
                    "top3_chunks": pq["retrieved_chunk_ids"][:3],
                })
        else:
            if top3_rel == 0 and gt_count > 0:
                failures.append({
                    "query_id": pq["query_id"],
                    "query": pq["query"],
                    "category": cat,
                    "failure_type": "no_relevant_in_top3",
                    "detail": f"0 of {gt_count} ground truth chunks in top-3",
                    "top3_chunks": pq["retrieved_chunk_ids"][:3],
                    "expected_chunks": list({c for c, r in pq.get("_rel_map", {}).items() if r >= 1}),
                })
            elif top5_rel == 0 and gt_count > 0:
                failures.append({
                    "query_id": pq["query_id"],
                    "query": pq["query"],
                    "category": cat,
                    "failure_type": "no_relevant_in_top5",
                    "detail": f"0 of {gt_count} ground truth chunks in top-5",
                    "top5_chunks": pq["retrieved_chunk_ids"][:5],
                })

    # ── Print Overall ──────────────────────────────────────────
    print(f"\n{'=' * 72}")
    print("OVERALL RETRIEVAL METRICS")
    print(f"{'=' * 72}")
    print(f"\n{'Metric':<10}", end="")
    for k in K_VALUES:
        print(f"  @{k:>2}", end="")
    print()
    print("-" * 50)
    for mk, lbl in [("precision", "P"), ("recall", "R"), ("f1", "F1"),
                     ("hit", "Hit"), ("ndcg", "nDCG")]:
        print(f"{lbl:<10}", end="")
        for k in K_VALUES:
            print(f"  {overall[k][mk]:.4f}", end="")
        print()
    print(f"{'MRR':<10}  {overall['mrr']:.4f}")

    # ── Print Per-Category ─────────────────────────────────────
    print(f"\n{'=' * 72}")
    print("PER-CATEGORY RETRIEVAL METRICS")
    print(f"{'=' * 72}")
    cat_order = ["direct", "ambiguous", "out_of_scope", "insufficient_evidence", "multi_part", "high_risk"]
    print(f"\n{'Category':<22} {'N':>3}  {'P@3':>6} {'P@5':>6} {'P@10':>6}  {'R@3':>6} {'R@5':>6} {'R@10':>6}  {'F1@3':>6} {'Hit@3':>6} {'nDCG@5':>7}")
    print("-" * 100)
    for cat in cat_order:
        if cat not in cat_agg:
            continue
        ca = cat_agg[cat]
        m = ca["metrics"]
        print(f"{cat:<22} {ca['n']:>3}  "
              f"{m[3]['precision']:>6.4f} {m[5]['precision']:>6.4f} {m[10]['precision']:>6.4f}  "
              f"{m[3]['recall']:>6.4f} {m[5]['recall']:>6.4f} {m[10]['recall']:>6.4f}  "
              f"{m[3]['f1']:>6.4f} {m[3]['hit']:>6.4f} {m[5]['ndcg']:>7.4f}")

    # ── False Retrieval for OOS/Insufficient ───────────────────
    if false_retrieval:
        print(f"\n{'=' * 72}")
        print("FALSE EVIDENCE RETRIEVAL (OOS / Insufficient Evidence)")
        print(f"{'=' * 72}")
        for cat, fr in false_retrieval.items():
            print(f"\n  {cat} (n={fr['n']}):")
            print(f"    False evidence in Top-3:  {fr['false_evidence_top3']}/{fr['n']} ({fr['false_evidence_rate_top3']:.1%})")
            print(f"    False evidence in Top-5:  {fr['false_evidence_top5']}/{fr['n']} ({fr['false_evidence_rate_top5']:.1%})")

    # ── Failure Analysis ───────────────────────────────────────
    print(f"\n{'=' * 72}")
    print("FAILURE ANALYSIS")
    print(f"{'=' * 72}")
    print(f"\nTotal failures identified: {len(failures)}")
    for f in failures:
        print(f"\n  [{f['query_id']}] {f['category']} — {f['failure_type']}")
        print(f"    Query: {f['query'][:80]}")
        print(f"    Detail: {f['detail']}")
        if "top3_chunks" in f:
            print(f"    Top-3: {f['top3_chunks']}")
        if "expected_chunks" in f and f["expected_chunks"]:
            print(f"    Expected: {f['expected_chunks']}")

    # ── Latency ────────────────────────────────────────────────
    lats = [q["latency"]["total_ms"] for q in per_query_output]
    print(f"\n{'=' * 72}")
    print("LATENCY")
    print(f"{'=' * 72}")
    print(f"  Mean:   {statistics.mean(lats):.1f} ms")
    print(f"  Median: {statistics.median(lats):.1f} ms")
    print(f"  Min:    {min(lats):.1f} ms")
    print(f"  Max:    {max(lats):.1f} ms")

    # ── Save JSON ──────────────────────────────────────────────
    output = {
        "experiment": "Current Retrieval Evaluation — Behavioral Categories",
        "timestamp": ts,
        "config": {
            "dense_model": BGE_MODEL,
            "query_prefix": QUERY_PREFIX,
            "candidate_k": CANDIDATE_K,
            "reranker": CE_MODEL,
            "ce_max_length": 512,
            "metadata_filter": "section != Front matter",
        },
        "num_queries": len(queries),
        "category_distribution": {cat: len(qs) for cat, qs in cat_groups.items()},
        "overall": overall,
        "per_category": cat_agg,
        "false_retrieval": false_retrieval,
        "failures": failures,
        "per_query": per_query_output,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = RESULTS_DIR / f"current_retrieval_evaluation_{ts}.json"
    json_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nJSON saved: {json_path}")

    # ── Save CSV ───────────────────────────────────────────────
    csv_path = RESULTS_DIR / f"current_retrieval_evaluation_{ts}.csv"
    fieldnames = ["query_id", "query", "category", "rerank_rank", "chunk_id",
                  "document", "section", "retrieval_score", "rerank_score",
                  "in_ground_truth", "relevance_grade"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)
    print(f"CSV saved: {csv_path}")

    # ── Save Report ────────────────────────────────────────────
    report_path = RESULTS_DIR / f"current_retrieval_evaluation_report_{ts}.md"
    lines = []
    lines.append("# Current Retrieval Evaluation Report")
    lines.append(f"\n**Timestamp:** {ts}")
    lines.append(f"**Queries:** {len(queries)}")
    lines.append(f"**Chunks:** {len(chunk_meta)}")
    lines.append(f"**Dense model:** {BGE_MODEL}")
    lines.append(f"**Reranker:** {CE_MODEL}")
    lines.append(f"**Candidate K:** {CANDIDATE_K}")
    lines.append("\n## Overall Metrics\n")
    lines.append("| Metric | @3 | @5 | @10 |")
    lines.append("|--------|-----|-----|------|")
    for mk, lbl in [("precision", "P"), ("recall", "R"), ("f1", "F1"),
                     ("hit", "Hit"), ("ndcg", "nDCG")]:
        vals = " | ".join(f"{overall[k][mk]:.4f}" for k in K_VALUES)
        lines.append(f"| {lbl} | {vals} |")
    lines.append(f"| MRR | {overall['mrr']:.4f} | | |")

    lines.append("\n## Per-Category Metrics\n")
    lines.append("| Category | N | P@3 | P@5 | P@10 | R@3 | R@5 | R@10 | F1@3 | Hit@3 | nDCG@5 |")
    lines.append("|----------|---|------|------|-------|------|------|-------|------|-------|--------|")
    for cat in cat_order:
        if cat not in cat_agg:
            continue
        ca = cat_agg[cat]
        m = ca["metrics"]
        lines.append(f"| {cat} | {ca['n']} | {m[3]['precision']:.4f} | {m[5]['precision']:.4f} | {m[10]['precision']:.4f} | "
                      f"{m[3]['recall']:.4f} | {m[5]['recall']:.4f} | {m[10]['recall']:.4f} | "
                      f"{m[3]['f1']:.4f} | {m[3]['hit']:.4f} | {m[5]['ndcg']:.4f} |")

    if false_retrieval:
        lines.append("\n## False Evidence Retrieval\n")
        lines.append("| Category | N | False Top-3 | False Top-5 | Rate Top-3 | Rate Top-5 |")
        lines.append("|----------|---|-------------|-------------|------------|------------|")
        for cat, fr in false_retrieval.items():
            lines.append(f"| {cat} | {fr['n']} | {fr['false_evidence_top3']} | {fr['false_evidence_top5']} | "
                          f"{fr['false_evidence_rate_top3']:.1%} | {fr['false_evidence_rate_top5']:.1%} |")

    lines.append(f"\n## Failures ({len(failures)})\n")
    for f in failures:
        lines.append(f"### {f['query_id']} — {f['failure_type']}")
        lines.append(f"- **Query:** {f['query']}")
        lines.append(f"- **Category:** {f['category']}")
        lines.append(f"- **Detail:** {f['detail']}")
        if "top3_chunks" in f:
            lines.append(f"- **Top-3:** `{f['top3_chunks']}`")
        lines.append("")

    lines.append(f"\n## Latency\n")
    lines.append(f"- Mean: {statistics.mean(lats):.1f} ms")
    lines.append(f"- Median: {statistics.median(lats):.1f} ms")
    lines.append(f"- Min: {min(lats):.1f} ms")
    lines.append(f"- Max: {max(lats):.1f} ms")

    lines.append("\n## Limitations\n")
    lines.append("- ORACLE document scoping is NOT used (evaluation uses real cross-document retrieval).")
    lines.append("- Ground truth is based on manual chunk inspection; some relevant chunks may be missed.")
    lines.append("- Out-of-scope and insufficient-evidence categories have empty or near-empty ground truth by design.")
    lines.append("- Category-level metrics may have small N for some categories.")
    lines.append("- High-risk category tests safety behavior, not retrieval accuracy per se.")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Report saved: {report_path}")

    print(f"\n{'=' * 72}")
    print("DONE")
    print(f"{'=' * 72}")
    return json_path, csv_path, report_path


if __name__ == "__main__":
    main()
