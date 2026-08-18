"""
Step 10: Metadata-Scoped + Cross-Encoder Reranking Experiment
==============================================================
Evaluates whether cross-encoder reranking improves ranking quality
AFTER document-source competition has been removed.

Pipeline per query:
  - Single-doc queries: document metadata filter -> BGE Top-20 -> CE rerank
  - Cross-doc queries: Front matter filter only -> BGE Top-20 -> CE rerank

Compares THREE systems:
  1. Multi-document baseline
  2. Metadata-scoped retrieval (Step 9)
  3. Metadata-scoped + Cross-Encoder reranking (Step 10)

DO NOT modify chunks, embeddings, ChromaDB, or the evaluation dataset.
"""

import json
import math
import statistics
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder


# ============================================================
# Configuration
# ============================================================

DB_PATH = Path("data/vector_db")
COLLECTION_NAME = "medical_guidelines"
BGE_MODEL = "BAAI/bge-small-en-v1.5"
CE_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

EVAL_FILE = Path("data/evaluation/multidoc_eval_dataset.json")
RESULTS_DIR = Path("data/evaluation/results")
CHUNKS_FILE = Path("data/chunks/chunks.jsonl")

BASELINE_FILE = Path("data/evaluation/results/multidoc_baseline_retrieval_20260818T153138Z.json")
SCOPED_FILE = Path("data/evaluation/results/metadata_scoped_retrieval_20260818T155956Z.json")

K_VALUES = [1, 3, 5, 10]
CANDIDATE_K = 20

# ORACLE metadata scoping by query category (same as Step 9)
CATEGORY_FILTERS = {
    "nice": {
        "$and": [
            {"section": {"$ne": "Front matter"}},
            {"document": "NICE_HF_2018_Guideline.pdf"},
        ]
    },
    "esc_2021": {
        "$and": [
            {"section": {"$ne": "Front matter"}},
            {"document": "ESC_HF_2021_Guideline.pdf"},
        ]
    },
    "esc_2023": {
        "$and": [
            {"section": {"$ne": "Front matter"}},
            {"document": "ESC_HF_2023_Focused_Update.pdf"},
        ]
    },
    "cross_document": {
        "section": {"$ne": "Front matter"}
    },
}

FILTER_DESCRIPTIONS = {
    "nice": "section != Front matter AND document = NICE_HF_2018_Guideline.pdf",
    "esc_2021": "section != Front matter AND document = ESC_HF_2021_Guideline.pdf",
    "esc_2023": "section != Front matter AND document = ESC_HF_2023_Focused_Update.pdf",
    "cross_document": "section != Front matter (no document restriction)",
}


# ============================================================
# Metric Functions (identical to baseline)
# ============================================================

def precision_at_k(retrieved_ids, relevance_map, k, threshold):
    relevant_count = 0
    for cid in retrieved_ids[:k]:
        if relevance_map.get(cid, 0) >= threshold:
            relevant_count += 1
    return relevant_count / k


def recall_at_k(retrieved_ids, relevance_map, total_relevant, k, threshold):
    if total_relevant == 0:
        return 0.0
    retrieved_relevant = 0
    for cid in retrieved_ids[:k]:
        if relevance_map.get(cid, 0) >= threshold:
            retrieved_relevant += 1
    return retrieved_relevant / total_relevant


def f1_at_k(precision, recall):
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def hit_rate_at_k(retrieved_ids, relevance_map, k, threshold):
    for cid in retrieved_ids[:k]:
        if relevance_map.get(cid, 0) >= threshold:
            return 1.0
    return 0.0


def mrr(retrieved_ids, relevance_map, threshold):
    for i, cid in enumerate(retrieved_ids):
        if relevance_map.get(cid, 0) >= threshold:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(retrieved_ids, relevance_map, k):
    dcg = 0.0
    for i, cid in enumerate(retrieved_ids[:k]):
        rel = relevance_map.get(cid, 0)
        dcg += (2**rel - 1) / math.log2(i + 2)

    all_rels = sorted(relevance_map.values(), reverse=True)
    idcg = 0.0
    for i, rel in enumerate(all_rels[:k]):
        idcg += (2**rel - 1) / math.log2(i + 2)

    if idcg == 0:
        return 0.0
    return dcg / idcg


def evaluate_query(retrieved_ids, relevance_map):
    total_relevant_lenient = sum(1 for v in relevance_map.values() if v >= 1)
    total_relevant_strict = sum(1 for v in relevance_map.values() if v == 2)

    rr = mrr(retrieved_ids, relevance_map, threshold=1)

    per_k = {}
    for k in K_VALUES:
        p_lenient = precision_at_k(retrieved_ids, relevance_map, k, threshold=1)
        p_strict = precision_at_k(retrieved_ids, relevance_map, k, threshold=2)
        r_lenient = recall_at_k(retrieved_ids, relevance_map, total_relevant_lenient, k, threshold=1)
        r_strict = recall_at_k(retrieved_ids, relevance_map, total_relevant_strict, k, threshold=2)
        f1_lenient = f1_at_k(p_lenient, r_lenient)
        f1_strict = f1_at_k(p_strict, r_strict)
        hr_lenient = hit_rate_at_k(retrieved_ids, relevance_map, k, threshold=1)
        hr_strict = hit_rate_at_k(retrieved_ids, relevance_map, k, threshold=2)
        ndcg = ndcg_at_k(retrieved_ids, relevance_map, k)

        per_k[k] = {
            "precision": p_lenient,
            "strict_precision": p_strict,
            "recall": r_lenient,
            "strict_recall": r_strict,
            "f1": f1_lenient,
            "strict_f1": f1_strict,
            "hit_rate": hr_lenient,
            "strict_hit_rate": hr_strict,
            "ndcg": ndcg,
        }

    return {
        "mrr": rr,
        "total_relevant_lenient": total_relevant_lenient,
        "total_relevant_strict": total_relevant_strict,
        "per_k": per_k,
    }


# ============================================================
# Loading helpers
# ============================================================

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_chunk_ids(path):
    ids = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            ids.append(record["chunk_id"])
    return ids


def load_chunk_texts(path):
    texts = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            texts[record["chunk_id"]] = record["text"]
    return texts


def validate_dataset(dataset, actual_chunk_ids):
    errors = []
    actual_set = set(actual_chunk_ids)
    for q in dataset.get("queries", []):
        qid = q.get("query_id", "?")
        relevance = q.get("relevant_chunks", q.get("relevance", {}))
        missing = actual_set - set(relevance.keys())
        if missing:
            errors.append(f"Query {qid}: missing labels for {len(missing)} chunks")
    return errors


# ============================================================
# Aggregation
# ============================================================

def aggregate_metrics(per_query_results):
    n = len(per_query_results)
    if n == 0:
        return {}
    agg = {}
    for k in K_VALUES:
        agg[k] = {}
        metric_keys = [
            "precision", "strict_precision",
            "recall", "strict_recall",
            "f1", "strict_f1",
            "hit_rate", "strict_hit_rate",
            "ndcg",
        ]
        for mk in metric_keys:
            values = [q["per_k"][k][mk] for q in per_query_results]
            agg[k][mk] = statistics.mean(values)
    mrr_values = [q["mrr"] for q in per_query_results]
    agg["mrr"] = statistics.mean(mrr_values)
    return agg


def aggregate_latency(latencies):
    totals = [l["total_ms"] for l in latencies]
    embeddings = [l["dense_ms"] for l in latencies]
    reranks = [l["rerank_ms"] for l in latencies]

    def stats(values):
        return {
            "mean": round(statistics.mean(values), 2),
            "median": round(statistics.median(values), 2),
            "min": round(min(values), 2),
            "max": round(max(values), 2),
        }

    return {
        "total": stats(totals),
        "dense": stats(embeddings),
        "rerank": stats(reranks),
    }


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 60)
    print("Step 10: Metadata-Scoped + Cross-Encoder Reranking")
    print("=" * 60)
    print()
    print("Pipeline: document filter -> BGE Top-20 -> CE rerank -> evaluate")
    print("Cross-Encoder receives raw query text (NO BGE prefix).")
    print()

    # ----------------------------------------------------------
    # 1. Load dataset
    # ----------------------------------------------------------
    if not EVAL_FILE.exists():
        print(f"FAIL: {EVAL_FILE} not found")
        return

    dataset = load_json(EVAL_FILE)
    actual_chunk_ids = load_chunk_ids(CHUNKS_FILE)
    print(f"Loaded eval dataset: {len(dataset['queries'])} queries")
    print(f"Loaded chunk index: {len(actual_chunk_ids)} chunks")

    errors = validate_dataset(dataset, actual_chunk_ids)
    if errors:
        print("Dataset validation FAILED:")
        for e in errors:
            print(f"  ERROR: {e}")
        return
    print("Dataset validation: PASSED")

    # ----------------------------------------------------------
    # 2. Load models
    # ----------------------------------------------------------
    print(f"\nLoading BGE embedding model: {BGE_MODEL}")
    bge_model = SentenceTransformer(BGE_MODEL, device="cpu")

    print(f"Loading Cross-Encoder: {CE_MODEL}")
    reranker = CrossEncoder(CE_MODEL, max_length=512)
    print("Models loaded.")

    # ----------------------------------------------------------
    # 3. Connect to ChromaDB
    # ----------------------------------------------------------
    print(f"\nConnecting to ChromaDB: {DB_PATH}")
    client = chromadb.PersistentClient(path=str(DB_PATH))
    collection = client.get_collection(name=COLLECTION_NAME)
    collection_count = collection.count()
    print(f"Collection '{COLLECTION_NAME}': {collection_count} records")

    # ----------------------------------------------------------
    # 4. Load chunk texts for cross-encoder
    # ----------------------------------------------------------
    print(f"\nLoading chunk texts for cross-encoder...")
    chunk_texts = load_chunk_texts(CHUNKS_FILE)
    print(f"Loaded {len(chunk_texts)} chunk texts")

    # ----------------------------------------------------------
    # 5. Run experiment
    # ----------------------------------------------------------
    queries = dataset["queries"]
    per_query_results = []
    all_latencies = []

    print(f"\nRunning scoped+reranked evaluation for {len(queries)} queries...")
    print(f"Candidate K = {CANDIDATE_K}, Final K values = {K_VALUES}\n")

    for q in queries:
        qid = q["query_id"]
        query_text = q["query"]
        category = q.get("category", "unknown")
        relevance_map = q.get("relevant_chunks", q.get("relevance", {}))

        # Get category-specific filter
        metadata_filter = CATEGORY_FILTERS.get(category, CATEGORY_FILTERS["cross_document"])

        # ---- Stage 1: BGE retrieval (Top-20) with metadata filter ----
        full_query = QUERY_PREFIX + query_text

        t_dense_start = time.perf_counter()
        query_embedding = bge_model.encode(
            full_query, normalize_embeddings=True
        ).tolist()
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=CANDIDATE_K,
            where=metadata_filter,
        )
        t_dense_end = time.perf_counter()

        dense_latency_ms = (t_dense_end - t_dense_start) * 1000

        # Parse candidates
        candidates = []
        candidate_ids = []
        for i in range(len(results["ids"][0])):
            cid = results["ids"][0][i]
            candidates.append({
                "chunk_id": cid,
                "distance": results["distances"][0][i],
                "text": chunk_texts.get(cid, results["documents"][0][i]),
            })
            candidate_ids.append(cid)

        # ---- Stage 2: Cross-Encoder reranking (raw query, no prefix) ----
        t_rerank_start = time.perf_counter()
        pairs = [(query_text, c["text"]) for c in candidates]
        ce_scores = reranker.predict(pairs)
        for c, score in zip(candidates, ce_scores):
            c["ce_score"] = float(score)
        reranked = sorted(candidates, key=lambda x: x["ce_score"], reverse=True)
        t_rerank_end = time.perf_counter()

        rerank_latency_ms = (t_rerank_end - t_rerank_start) * 1000
        total_latency_ms = dense_latency_ms + rerank_latency_ms

        # ---- Evaluate reranked results ----
        reranked_ids = [r["chunk_id"] for r in reranked]
        eval_result = evaluate_query(reranked_ids, relevance_map)

        # Build output
        query_result = {
            "query_id": qid,
            "query": query_text,
            "category": category,
            "retrieved_chunk_ids": reranked_ids,
            "retrieved_ce_scores": [r["ce_score"] for r in reranked],
            "retrieved_relevance": [relevance_map.get(cid, 0) for cid in reranked_ids],
            "mrr": eval_result["mrr"],
            "total_relevant_lenient": eval_result["total_relevant_lenient"],
            "total_relevant_strict": eval_result["total_relevant_strict"],
            "per_k": eval_result["per_k"],
            "latency": {
                "dense_ms": round(dense_latency_ms, 2),
                "rerank_ms": round(rerank_latency_ms, 2),
                "total_ms": round(total_latency_ms, 2),
            },
            "metadata_filter_used": FILTER_DESCRIPTIONS.get(category, "unknown"),
        }

        per_query_results.append(query_result)
        all_latencies.append(query_result["latency"])

        hr_5 = eval_result["per_k"][5]["hit_rate"]
        ndcg_5 = eval_result["per_k"][5]["ndcg"]
        print(
            f"  {qid} [{category:>15}] | "
            f"Hit@5={hr_5:.0f}  "
            f"nDCG@5={ndcg_5:.4f}  "
            f"MRR={eval_result['mrr']:.4f}  "
            f"latency={total_latency_ms:.1f}ms  "
            f"| {query_text[:40]}..."
        )

    # ----------------------------------------------------------
    # 6. Aggregate
    # ----------------------------------------------------------
    agg_metrics = aggregate_metrics(per_query_results)
    agg_latency = aggregate_latency(all_latencies)

    # ----------------------------------------------------------
    # 7. Build results
    # ----------------------------------------------------------
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_filename = f"scoped_reranked_retrieval_{timestamp}.json"
    output_path = str(RESULTS_DIR / output_filename)

    results = {
        "experiment": "Step 10: Metadata-Scoped + Cross-Encoder Reranking",
        "description": "ORACLE metadata filter -> BGE Top-20 -> CE rerank -> evaluate",
        "dataset_version": dataset.get("dataset_version", "unknown"),
        "evaluation_timestamp": timestamp,
        "dense_model": BGE_MODEL,
        "reranker_model": CE_MODEL,
        "candidate_k": CANDIDATE_K,
        "collection_name": COLLECTION_NAME,
        "collection_count": collection_count,
        "chunk_count": len(actual_chunk_ids),
        "num_queries": len(queries),
        "k_values": K_VALUES,
        "retrieval_config": {
            "query_prefix": QUERY_PREFIX,
            "metadata_filters": FILTER_DESCRIPTIONS,
            "distance_metric": "cosine",
            "device": "cpu",
        },
        "reranker_config": {
            "model": CE_MODEL,
            "max_length": 512,
            "input": "raw query (no BGE prefix) + chunk text",
            "candidate_k": CANDIDATE_K,
        },
        "aggregate_metrics": {
            "per_k": {
                str(k): {mk: round(v, 6) for mk, v in metrics.items()}
                for k, metrics in agg_metrics.items()
                if isinstance(k, int)
            },
            "mrr": round(agg_metrics.get("mrr", 0), 6),
        },
        "latency": {
            phase: stats for phase, stats in agg_latency.items()
        },
        "per_query_results": per_query_results,
        "output_path": output_path,
    }

    # ----------------------------------------------------------
    # 8. Save results
    # ----------------------------------------------------------
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # ----------------------------------------------------------
    # 9. Load baseline and scoped for comparison
    # ----------------------------------------------------------
    print("\nLoading baseline and scoped results for comparison...")
    baseline = load_json(BASELINE_FILE)
    scoped = load_json(SCOPED_FILE)

    baseline_pq = baseline["per_query_results"]
    scoped_pq = scoped["per_query_results"]
    reranked_pq = per_query_results

    # ----------------------------------------------------------
    # 10. Build comparison
    # ----------------------------------------------------------
    def get_metric(agg, k, key):
        return agg["per_k"][str(k)][key]

    baseline_agg = baseline["aggregate_metrics"]
    scoped_agg = scoped["aggregate_metrics"]
    reranked_agg = results["aggregate_metrics"]

    # A. Overall comparison
    overall = {}
    for k in K_VALUES:
        for mk in ["precision", "recall", "ndcg", "hit_rate"]:
            label = {
                "precision": f"P@{k}",
                "recall": f"R@{k}",
                "ndcg": f"nDCG@{k}",
                "hit_rate": f"Hit@{k}",
            }[mk]
            overall[label] = {
                "baseline": round(get_metric(baseline_agg, k, mk), 4),
                "scoped": round(get_metric(scoped_agg, k, mk), 4),
                "scoped_reranked": round(get_metric(reranked_agg, k, mk), 4),
            }
    overall["MRR"] = {
        "baseline": round(baseline_agg["mrr"], 4),
        "scoped": round(scoped_agg["mrr"], 4),
        "scoped_reranked": round(reranked_agg["mrr"], 4),
    }
    overall["mean_latency_ms"] = {
        "baseline": round(baseline["latency"]["total"]["mean"], 2),
        "scoped": round(scoped["latency"]["total"]["mean"], 2),
        "scoped_reranked": round(results["latency"]["total"]["mean"], 2),
    }
    overall["median_latency_ms"] = {
        "baseline": round(baseline["latency"]["total"]["median"], 2),
        "scoped": round(scoped["latency"]["total"]["median"], 2),
        "scoped_reranked": round(results["latency"]["total"]["median"], 2),
    }

    # B. Category breakdown
    cat_map = defaultdict(lambda: {"baseline": [], "scoped": [], "reranked": []})
    for q in baseline_pq:
        cat_map[q["category"]]["baseline"].append(q)
    for q in scoped_pq:
        cat_map[q["category"]]["scoped"].append(q)
    for q in reranked_pq:
        cat_map[q["category"]]["reranked"].append(q)

    cat_labels = {
        "nice": "NICE 2018",
        "esc_2021": "ESC 2021",
        "esc_2023": "ESC 2023",
        "cross_document": "Cross-Document",
    }

    category_breakdown = {}
    for cat_key, label in cat_labels.items():
        b_qs = cat_map[cat_key]["baseline"]
        s_qs = cat_map[cat_key]["scoped"]
        r_qs = cat_map[cat_key]["reranked"]
        if not r_qs:
            continue

        def avg_k(qs, k, key):
            vals = []
            for q in qs:
                pk = q["per_k"]
                # Support both int and string keys
                v = pk.get(k, pk.get(str(k), {}))
                vals.append(v.get(key, 0))
            return statistics.mean(vals) if vals else 0

        category_breakdown[label] = {
            "count": len(r_qs),
            "nDCG@5": {
                "baseline": round(avg_k(b_qs, 5, "ndcg"), 4),
                "scoped": round(avg_k(s_qs, 5, "ndcg"), 4),
                "scoped_reranked": round(avg_k(r_qs, 5, "ndcg"), 4),
            },
            "R@5": {
                "baseline": round(avg_k(b_qs, 5, "recall"), 4),
                "scoped": round(avg_k(s_qs, 5, "recall"), 4),
                "scoped_reranked": round(avg_k(r_qs, 5, "recall"), 4),
            },
            "Hit@5": {
                "baseline": round(avg_k(b_qs, 5, "hit_rate"), 4),
                "scoped": round(avg_k(s_qs, 5, "hit_rate"), 4),
                "scoped_reranked": round(avg_k(r_qs, 5, "hit_rate"), 4),
            },
            "MRR": {
                "baseline": round(statistics.mean([q["mrr"] for q in b_qs]), 4) if b_qs else 0,
                "scoped": round(statistics.mean([q["mrr"] for q in s_qs]), 4) if s_qs else 0,
                "scoped_reranked": round(statistics.mean([q["mrr"] for q in r_qs]), 4),
            },
            "P@1": {
                "baseline": round(avg_k(b_qs, 1, "precision"), 4),
                "scoped": round(avg_k(s_qs, 1, "precision"), 4),
                "scoped_reranked": round(avg_k(r_qs, 1, "precision"), 4),
            },
        }

    # C. Reranking impact per query
    def get_pk_metric(per_k, k, key):
        """Get metric from per_k dict, handling both int and string keys."""
        v = per_k.get(k, per_k.get(str(k), {}))
        return v.get(key, 0)

    reranking_impact = []
    for i in range(len(reranked_pq)):
        r = reranked_pq[i]
        qid = r["query_id"]
        # Find matching scoped and baseline
        s_match = next((q for q in scoped_pq if q["query_id"] == qid), None)

        scoped_ndcg5 = get_pk_metric(s_match["per_k"], 5, "ndcg") if s_match else 0
        reranked_ndcg5 = get_pk_metric(r["per_k"], 5, "ndcg")
        scoped_mrr = s_match["mrr"] if s_match else 0
        reranked_mrr = r["mrr"]

        ndcg_delta = reranked_ndcg5 - scoped_ndcg5
        mrr_delta = reranked_mrr - scoped_mrr

        if ndcg_delta > 0.01:
            impact = "improved"
        elif ndcg_delta < -0.01:
            impact = "degraded"
        else:
            impact = "unchanged"

        reranking_impact.append({
            "query_id": qid,
            "category": r["category"],
            "query": r["query"][:80],
            "scoped_nDCG@5": round(scoped_ndcg5, 4),
            "reranked_nDCG@5": round(reranked_ndcg5, 4),
            "nDCG_delta": round(ndcg_delta, 4),
            "scoped_MRR": round(scoped_mrr, 4),
            "reranked_MRR": round(reranked_mrr, 4),
            "MRR_delta": round(mrr_delta, 4),
            "impact": impact,
        })

    improved = [r for r in reranking_impact if r["impact"] == "improved"]
    degraded = [r for r in reranking_impact if r["impact"] == "degraded"]
    unchanged = [r for r in reranking_impact if r["impact"] == "unchanged"]

    # E. Latency comparison
    latency_comparison = {
        "scoped_only": {
            "mean_ms": round(scoped["latency"]["total"]["mean"], 2),
            "median_ms": round(scoped["latency"]["total"]["median"], 2),
        },
        "scoped_plus_reranked": {
            "mean_ms": round(results["latency"]["total"]["mean"], 2),
            "median_ms": round(results["latency"]["total"]["median"], 2),
            "dense_mean_ms": round(results["latency"]["dense"]["mean"], 2),
            "rerank_mean_ms": round(results["latency"]["rerank"]["mean"], 2),
        },
    }
    rerank_overhead_mean = latency_comparison["scoped_plus_reranked"]["rerank_mean_ms"]
    rerank_overhead_median = results["latency"]["rerank"]["median"]

    # Save comparison
    comparison = {
        "overall": overall,
        "category_level": category_breakdown,
        "reranking_impact": {
            "summary": {
                "total_queries": len(reranking_impact),
                "improved": len(improved),
                "degraded": len(degraded),
                "unchanged": len(unchanged),
            },
            "improved_queries": improved,
            "degraded_queries": degraded,
        },
        "latency_comparison": latency_comparison,
    }
    comparison_path = str(RESULTS_DIR / f"scoped_reranked_comparison_{timestamp}.json")
    with open(comparison_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)

    # ============================================================
    # PRINT COMPLETE REPORT
    # ============================================================

    print("\n" + "=" * 70)
    print("A. OVERALL COMPARISON")
    print("=" * 70)
    print(f"{'Metric':<20} | {'Baseline':>10} | {'Scoped':>10} | {'Scoped+Rnk':>10} | {'s->sr':>10}")
    print("-" * 75)
    for metric_name, vals in overall.items():
        b = vals["baseline"]
        s = vals["scoped"]
        sr = vals["scoped_reranked"]
        delta = sr - s
        sign = "+" if delta >= 0 else ""
        if "latency" in metric_name:
            print(f"{metric_name:<20} | {b:>10.2f} | {s:>10.2f} | {sr:>10.2f} | {sign}{delta:>9.2f}")
        else:
            print(f"{metric_name:<20} | {b:>10.4f} | {s:>10.4f} | {sr:>10.4f} | {sign}{delta:>9.4f}")

    print("\n" + "=" * 70)
    print("B. CATEGORY BREAKDOWN")
    print("=" * 70)
    for cat_label, vals in category_breakdown.items():
        print(f"\n  {cat_label} (n={vals['count']}):")
        for metric in ["P@1", "nDCG@5", "R@5", "Hit@5", "MRR"]:
            b = vals[metric]["baseline"]
            s = vals[metric]["scoped"]
            sr = vals[metric]["scoped_reranked"]
            delta_sr = sr - s
            sign = "+" if delta_sr >= 0 else ""
            print(f"    {metric:<8} baseline={b:.4f}  scoped={s:.4f}  scoped+rnk={sr:.4f}  {sign}{delta_sr:.4f}")

    print("\n" + "=" * 70)
    print("C. RERANKING IMPACT")
    print("=" * 70)
    print(f"  Improved:  {len(improved)}/{len(reranking_impact)} queries")
    print(f"  Degraded:  {len(degraded)}/{len(reranking_impact)} queries")
    print(f"  Unchanged: {len(unchanged)}/{len(reranking_impact)} queries")

    if improved:
        print(f"\n  Improved queries:")
        for r in sorted(improved, key=lambda x: -x["nDCG_delta"]):
            print(f"    {r['query_id']} [{r['category']}] nDCG: {r['scoped_nDCG@5']:.4f} -> {r['reranked_nDCG@5']:.4f} ({r['nDCG_delta']:+.4f})")
            print(f"      {r['query'][:70]}...")

    if degraded:
        print(f"\n  Degraded queries:")
        for r in sorted(degraded, key=lambda x: x["nDCG_delta"]):
            print(f"    {r['query_id']} [{r['category']}] nDCG: {r['scoped_nDCG@5']:.4f} -> {r['reranked_nDCG@5']:.4f} ({r['nDCG_delta']:+.4f})")
            print(f"      {r['query'][:70]}...")

    print("\n" + "=" * 70)
    print("D. DETAILED QUERY ANALYSIS")
    print("=" * 70)
    # Inspect ESC 2023 queries
    esc23_results = [r for r in reranking_impact if r["category"] == "esc_2023"]
    print(f"\n  ESC 2023 queries:")
    for r in esc23_results:
        delta = r["nDCG_delta"]
        sign = "+" if delta >= 0 else ""
        print(f"    {r['query_id']} nDCG: {r['scoped_nDCG@5']:.4f} -> {r['reranked_nDCG@5']:.4f} ({sign}{delta:.4f})  MRR: {r['scoped_MRR']:.4f} -> {r['reranked_MRR']:.4f}")

    # Inspect cross-document queries
    cross_results = [r for r in reranking_impact if r["category"] == "cross_document"]
    print(f"\n  Cross-document queries:")
    for r in cross_results:
        delta = r["nDCG_delta"]
        sign = "+" if delta >= 0 else ""
        print(f"    {r['query_id']} nDCG: {r['scoped_nDCG@5']:.4f} -> {r['reranked_nDCG@5']:.4f} ({sign}{delta:.4f})  MRR: {r['scoped_MRR']:.4f} -> {r['reranked_MRR']:.4f}")

    # Low scoped nDCG queries
    low_scoped = [r for r in reranking_impact if r["scoped_nDCG@5"] < 0.2]
    if low_scoped:
        print(f"\n  Low scoped nDCG@5 (<0.2) queries:")
        for r in sorted(low_scoped, key=lambda x: x["scoped_nDCG@5"]):
            delta = r["nDCG_delta"]
            sign = "+" if delta >= 0 else ""
            print(f"    {r['query_id']} [{r['category']}] nDCG: {r['scoped_nDCG@5']:.4f} -> {r['reranked_nDCG@5']:.4f} ({sign}{delta:.4f})")

    print("\n" + "=" * 70)
    print("E. LATENCY COMPARISON")
    print("=" * 70)
    print(f"  Scoped only:           mean={latency_comparison['scoped_only']['mean_ms']:.2f}ms  median={latency_comparison['scoped_only']['median_ms']:.2f}ms")
    print(f"  Scoped + Reranked:     mean={latency_comparison['scoped_plus_reranked']['mean_ms']:.2f}ms  median={latency_comparison['scoped_plus_reranked']['median_ms']:.2f}ms")
    print(f"    of which reranking:  mean={rerank_overhead_mean:.2f}ms  median={rerank_overhead_median:.2f}ms")
    overhead_pct = (rerank_overhead_mean / latency_comparison["scoped_only"]["mean_ms"] * 100) if latency_comparison["scoped_only"]["mean_ms"] > 0 else 0
    print(f"  Reranking overhead:    {overhead_pct:.1f}% of scoped-only latency")

    print("\n" + "=" * 70)
    print("F. RECOMMENDATION")
    print("=" * 70)

    # Decision logic
    scoped_mrr = scoped_agg["mrr"]
    reranked_mrr = reranked_agg["mrr"]
    scoped_ndcg5 = get_metric(scoped_agg, 5, "ndcg")
    reranked_ndcg5 = get_metric(reranked_agg, 5, "ndcg")

    mrr_improved = reranked_mrr > scoped_mrr
    ndcg_improved = reranked_ndcg5 > scoped_ndcg5
    acceptable_latency = overhead_pct < 200  # less than 3x

    print(f"\n  Scoped MRR:     {scoped_mrr:.4f}")
    print(f"  Reranked MRR:   {reranked_mrr:.4f} ({'UP' if mrr_improved else 'DOWN'})")
    print(f"  Scoped nDCG@5:  {scoped_ndcg5:.4f}")
    print(f"  Reranked nDCG@5:{reranked_ndcg5:.4f} ({'UP' if ndcg_improved else 'DOWN'})")
    print(f"  Latency overhead: {overhead_pct:.1f}% ({'acceptable' if acceptable_latency else 'excessive'})")
    print(f"  Queries improved by reranking: {len(improved)}/{len(reranking_impact)}")
    print(f"  Queries degraded by reranking: {len(degraded)}/{len(reranking_impact)}")

    if mrr_improved and ndcg_improved and acceptable_latency:
        print(f"\n  DECISION: ADOPT scoped + reranking.")
        print(f"  Reranking provides meaningful improvement on top of metadata scoping.")
        print(f"  The latency overhead ({overhead_pct:.1f}%) is acceptable for the quality gain.")
    elif mrr_improved and not ndcg_improved:
        print(f"\n  DECISION: MIXED. Reranking improves MRR but not nDCG@5.")
        print(f"  Consider targeted reranking for specific categories.")
    elif not mrr_improved and ndcg_improved:
        print(f"\n  DECISION: MIXED. Reranking improves nDCG@5 but not MRR.")
        print(f"  May still be beneficial depending on use case.")
    else:
        print(f"\n  DECISION: DO NOT ADOPT reranking.")
        print(f"  Reranking does not provide sufficient improvement.")
        print(f"  Consider next experiment based on remaining failure modes.")

    # Per-category recommendation
    print(f"\n  PER-CATEGORY ANALYSIS:")
    for cat_label, vals in category_breakdown.items():
        s_ndcg = vals["nDCG@5"]["scoped"]
        sr_ndcg = vals["nDCG@5"]["scoped_reranked"]
        s_mrr = vals["MRR"]["scoped"]
        sr_mrr = vals["MRR"]["scoped_reranked"]
        ndcg_delta = sr_ndcg - s_ndcg
        mrr_delta = sr_mrr - s_mrr
        sign_n = "+" if ndcg_delta >= 0 else ""
        sign_m = "+" if mrr_delta >= 0 else ""
        verdict = "BENEFICIAL" if ndcg_delta > 0.01 and mrr_delta >= -0.01 else ("HARMFUL" if ndcg_delta < -0.01 else "NEUTRAL")
        print(f"    {cat_label:<18} nDCG delta={sign_n}{ndcg_delta:.4f}  MRR delta={sign_m}{mrr_delta:.4f}  -> {verdict}")

    print("\n" + "=" * 70)
    print(f"Results saved: {output_path}")
    print(f"Comparison saved: {comparison_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
