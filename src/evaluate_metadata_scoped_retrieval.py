"""
Step 9: Metadata-Scoped Retrieval Experiment
=============================================
Controlled experiment: restrict ChromaDB search to the expected document
based on query category (ORACLE scoping).

This isolates the effect of document-source metadata filtering on retrieval quality.

DO NOT modify chunks, embeddings, ChromaDB data, or the evaluation dataset.
"""

import json
import math
import statistics
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


# ============================================================
# Configuration (identical to baseline except metadata filter)
# ============================================================

DB_PATH = Path("data/vector_db")
COLLECTION_NAME = "medical_guidelines"
MODEL_NAME = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

EVAL_FILE = Path("data/evaluation/multidoc_eval_dataset.json")
RESULTS_DIR = Path("data/evaluation/results")
CHUNKS_FILE = Path("data/chunks/chunks.jsonl")
BASELINE_FILE = Path("data/evaluation/results/multidoc_baseline_retrieval_20260818T153138Z.json")

K_VALUES = [1, 3, 5, 10]

# ORACLE metadata scoping by query category
# This is an experiment: we use the known category to pick the document filter.
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
    "nice": 'section != Front matter AND document = NICE_HF_2018_Guideline.pdf',
    "esc_2021": 'section != Front matter AND document = ESC_HF_2021_Guideline.pdf',
    "esc_2023": 'section != Front matter AND document = ESC_HF_2023_Focused_Update.pdf',
    "cross_document": 'section != Front matter (no document restriction)',
}


# ============================================================
# Metric Calculators (identical to baseline)
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


def evaluate_query(query_data, retrieved, relevance_map):
    retrieved_ids = [r["chunk_id"] for r in retrieved]

    total_relevant_lenient = sum(1 for v in relevance_map.values() if v >= 1)
    total_relevant_strict = sum(1 for v in relevance_map.values() if v == 2)

    rr = mrr(retrieved_ids, relevance_map, threshold=1)

    results_per_k = {}
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

        results_per_k[k] = {
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
        "per_k": results_per_k,
    }


# ============================================================
# Dataset Loading & Validation
# ============================================================

def load_eval_dataset(path):
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


def validate_dataset(dataset, actual_chunk_ids):
    errors = []
    actual_set = set(actual_chunk_ids)

    queries = dataset.get("queries", [])
    if len(queries) == 0:
        errors.append("No queries found in dataset")
        return errors

    for q in queries:
        qid = q.get("query_id", "UNKNOWN")
        relevance = q.get("relevant_chunks", q.get("relevance", {}))

        if not relevance:
            errors.append(f"Query {qid}: no relevance labels")
            continue

        missing = actual_set - set(relevance.keys())
        if missing:
            errors.append(f"Query {qid}: missing labels for {len(missing)} chunks")

    return errors


# ============================================================
# Retrieval (with category-specific filter)
# ============================================================

def retrieve(query_text, model, collection, max_k, metadata_filter):
    full_query = QUERY_PREFIX + query_text

    t0 = time.perf_counter()
    query_embedding = model.encode(
        full_query, normalize_embeddings=True
    ).tolist()
    t1 = time.perf_counter()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=max_k,
        where=metadata_filter,
    )
    t2 = time.perf_counter()

    retrieved = []
    for i in range(len(results["ids"][0])):
        retrieved.append({
            "chunk_id": results["ids"][0][i],
            "distance": results["distances"][0][i],
            "rank": i + 1,
        })

    return retrieved, {
        "embedding_ms": (t1 - t0) * 1000,
        "search_ms": (t2 - t1) * 1000,
        "total_ms": (t2 - t0) * 1000,
    }


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
    embeddings = [l["embedding_ms"] for l in latencies]
    searches = [l["search_ms"] for l in latencies]

    def stats(values):
        return {
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "min": min(values),
            "max": max(values),
        }

    return {
        "total": stats(totals),
        "embedding": stats(embeddings),
        "search": stats(searches),
    }


# ============================================================
# Document classification
# ============================================================

def classify_chunk_doc(cid):
    if cid.startswith("nice_hf_2018"):
        return "NICE"
    elif cid.startswith("esc_hf_2021"):
        return "ESC 2021"
    elif cid.startswith("esc_hf_2023"):
        return "ESC 2023"
    return "unknown"


# ============================================================
# Source Confusion Analysis
# ============================================================

def analyze_source_confusion(per_query_results, eval_lookup):
    """For each query, count wrong-document results in top-1 and top-5."""
    results = []
    for q in per_query_results:
        qid = q["query_id"]
        cat = q["category"]
        top1_doc = classify_chunk_doc(q["retrieved_chunk_ids"][0])
        top5_docs = [classify_chunk_doc(cid) for cid in q["retrieved_chunk_ids"][:5]]

        # Expected docs = docs with any relevant chunk (score >= 1)
        exp_docs = set()
        rel_map = eval_lookup.get(qid, {})
        for cid, score in rel_map.items():
            if score >= 1:
                exp_docs.add(classify_chunk_doc(cid))

        wrong_top1 = 1 if top1_doc not in exp_docs else 0
        wrong_top5 = sum(1 for d in top5_docs if d not in exp_docs)

        results.append({
            "query_id": qid,
            "category": cat,
            "expected_documents": sorted(exp_docs),
            "top1_document": top1_doc,
            "top5_documents": top5_docs,
            "wrong_doc_top1": wrong_top1,
            "wrong_doc_top5": wrong_top5,
        })

    return results


# ============================================================
# Comparison against baseline
# ============================================================

def build_comparison(scoped_results, baseline_results, scoped_confusion, baseline_confusion):
    scoped_agg = scoped_results["aggregate_metrics"]
    baseline_agg = baseline_results["aggregate_metrics"]

    def get_metric(agg, k, key):
        return agg["per_k"][str(k)][key]

    overall_comparison = {}
    for k in K_VALUES:
        overall_comparison[f"P@{k}"] = {
            "baseline": round(get_metric(baseline_agg, k, "precision"), 4),
            "scoped": round(get_metric(scoped_agg, k, "precision"), 4),
        }
        overall_comparison[f"R@{k}"] = {
            "baseline": round(get_metric(baseline_agg, k, "recall"), 4),
            "scoped": round(get_metric(scoped_agg, k, "recall"), 4),
        }
        overall_comparison[f"nDCG@{k}"] = {
            "baseline": round(get_metric(baseline_agg, k, "ndcg"), 4),
            "scoped": round(get_metric(scoped_agg, k, "ndcg"), 4),
        }
        overall_comparison[f"Hit@{k}"] = {
            "baseline": round(get_metric(baseline_agg, k, "hit_rate"), 4),
            "scoped": round(get_metric(scoped_agg, k, "hit_rate"), 4),
        }

    overall_comparison["MRR"] = {
        "baseline": round(baseline_agg["mrr"], 4),
        "scoped": round(scoped_agg["mrr"], 4),
    }
    overall_comparison["mean_latency_ms"] = {
        "baseline": round(baseline_results["latency"]["total"]["mean"], 2),
        "scoped": round(scoped_results["latency"]["total"]["mean"], 2),
    }
    overall_comparison["median_latency_ms"] = {
        "baseline": round(baseline_results["latency"]["total"]["median"], 2),
        "scoped": round(scoped_results["latency"]["total"]["median"], 2),
    }

    return overall_comparison


def build_category_comparison(scoped_per_query, baseline_per_query):
    cat_map = defaultdict(lambda: {"scoped": [], "baseline": []})
    for q in scoped_per_query:
        cat_map[q["category"]]["scoped"].append(q)
    for q in baseline_per_query:
        cat_map[q["category"]]["baseline"].append(q)

    cat_labels = {
        "nice": "NICE 2018",
        "esc_2021": "ESC 2021",
        "esc_2023": "ESC 2023",
        "cross_document": "Cross-Document",
    }

    result = {}
    for cat_key, label in cat_labels.items():
        scoped_qs = cat_map[cat_key]["scoped"]
        baseline_qs = cat_map[cat_key]["baseline"]
        if not scoped_qs:
            continue

    def avg5(qs, key):
        return statistics.mean([q["per_k"]["5"][key] for q in qs])

    def avg_k(qs, k, key):
        return statistics.mean([q["per_k"][str(k)][key] for q in qs])

        result[label] = {
            "count": len(scoped_qs),
            "P@5": {
                "baseline": round(avg5(baseline_qs, "precision"), 4),
                "scoped": round(avg5(scoped_qs, "precision"), 4),
            },
            "R@5": {
                "baseline": round(avg5(baseline_qs, "recall"), 4),
                "scoped": round(avg5(scoped_qs, "recall"), 4),
            },
            "nDCG@5": {
                "baseline": round(avg5(baseline_qs, "ndcg"), 4),
                "scoped": round(avg5(scoped_qs, "ndcg"), 4),
            },
            "Hit@5": {
                "baseline": round(avg5(baseline_qs, "hit_rate"), 4),
                "scoped": round(avg5(scoped_qs, "hit_rate"), 4),
            },
            "MRR": {
                "baseline": round(statistics.mean([q["mrr"] for q in baseline_qs]), 4),
                "scoped": round(statistics.mean([q["mrr"] for q in scoped_qs]), 4),
            },
            "P@1": {
                "baseline": round(avg_k(baseline_qs, 1, "precision"), 4),
                "scoped": round(avg_k(scoped_qs, 1, "precision"), 4),
            },
        }

    return result


def build_confusion_comparison(scoped_confusion, baseline_confusion):
    def summarize(confusion):
        total_q = len(confusion)
        wrong_top1 = sum(c["wrong_doc_top1"] for c in confusion)
        wrong_top5_total = sum(c["wrong_doc_top5"] for c in confusion)
        wrong_top5_slots = total_q * 5
        return {
            "total_queries": total_q,
            "wrong_doc_top1_count": wrong_top1,
            "wrong_doc_top1_rate": round(wrong_top1 / total_q, 4) if total_q else 0,
            "wrong_doc_top5_total_slots": wrong_top5_slots,
            "wrong_doc_top5_count": wrong_top5_total,
            "wrong_doc_top5_rate": round(wrong_top5_total / wrong_top5_slots, 4) if wrong_top5_slots else 0,
        }

    # Split by single-doc vs cross-doc
    scoped_single = [c for c in scoped_confusion if len(c["expected_documents"]) == 1]
    scoped_cross = [c for c in scoped_confusion if len(c["expected_documents"]) > 1]
    baseline_single = [c for c in baseline_confusion if len(c["expected_documents"]) == 1]
    baseline_cross = [c for c in baseline_confusion if len(c["expected_documents"]) > 1]

    return {
        "all": {
            "baseline": summarize(baseline_confusion),
            "scoped": summarize(scoped_confusion),
        },
        "single_document_queries": {
            "baseline": summarize(baseline_single),
            "scoped": summarize(scoped_single),
        },
        "cross_document_queries": {
            "baseline": summarize(baseline_cross),
            "scoped": summarize(scoped_cross),
        },
    }


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 60)
    print("METADATA-SCOPED RETRIEVAL EXPERIMENT (Step 9)")
    print("=" * 60)
    print()
    print("ORACLE scoping: query category determines document filter.")
    print("This is NOT a production routing solution.")
    print()

    # 1. Load dataset
    if not EVAL_FILE.exists():
        print(f"FAIL: Evaluation file not found: {EVAL_FILE}")
        return

    dataset = load_eval_dataset(EVAL_FILE)
    actual_chunk_ids = load_chunk_ids(CHUNKS_FILE)
    print(f"Loaded evaluation dataset: {len(dataset['queries'])} queries")
    print(f"Loaded chunk index: {len(actual_chunk_ids)} chunks")

    errors = validate_dataset(dataset, actual_chunk_ids)
    if errors:
        print("\nDataset validation FAILED:")
        for e in errors:
            print(f"  ERROR: {e}")
        return
    print("Dataset validation: PASSED")

    # 2. Load model and ChromaDB
    print(f"\nLoading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME, device="cpu")

    print(f"Connecting to ChromaDB: {DB_PATH}")
    client = chromadb.PersistentClient(path=str(DB_PATH))
    collection = client.get_collection(name=COLLECTION_NAME)
    collection_count = collection.count()
    print(f"Collection '{COLLECTION_NAME}': {collection_count} records")

    # 3. Validate metadata
    print("\n-- Metadata Validation --")
    doc_counter = {}
    results_all = collection.get(include=["metadatas"], limit=collection_count)
    for meta in results_all["metadatas"]:
        doc = meta.get("document", "MISSING")
        doc_counter[doc] = doc_counter.get(doc, 0) + 1

    print(f"  Total records: {collection_count}")
    for doc, cnt in sorted(doc_counter.items()):
        print(f"  {doc}: {cnt}")

    # Validate expected docs exist
    expected_doc_names = [
        "NICE_HF_2018_Guideline.pdf",
        "ESC_HF_2021_Guideline.pdf",
        "ESC_HF_2023_Focused_Update.pdf",
    ]
    for ed in expected_doc_names:
        if ed not in doc_counter:
            print(f"  FAIL: Expected document '{ed}' not found in ChromaDB metadata")
            return

    print("\n-- Filters for each category --")
    for cat, filt_desc in FILTER_DESCRIPTIONS.items():
        print(f"  {cat}: {filt_desc}")

    # Validate filter chunk counts
    print("\n-- Filtered chunk counts --")
    for cat, filt in CATEGORY_FILTERS.items():
        res = collection.get(where=filt, include=["metadatas"])
        print(f"  {cat}: {len(res['ids'])} chunks")

    # 4. Build eval lookup
    eval_lookup = {}
    for q in dataset["queries"]:
        eval_lookup[q["query_id"]] = q.get("relevant_chunks", q.get("relevance", {}))

    # 5. Run evaluation
    queries = dataset["queries"]
    per_query_results = []
    all_latencies = []
    max_k = max(K_VALUES)

    print(f"\nRunning metadata-scoped evaluation for {len(queries)} queries (K={K_VALUES})...\n")

    for q in queries:
        qid = q["query_id"]
        query_text = q["query"]
        category = q.get("category", "unknown")
        relevance_map = q.get("relevant_chunks", q.get("relevance", {}))

        # Get category-specific filter
        metadata_filter = CATEGORY_FILTERS.get(category, CATEGORY_FILTERS["cross_document"])

        # Retrieve
        retrieved, latency = retrieve(query_text, model, collection, max_k, metadata_filter)

        # Evaluate
        eval_result = evaluate_query(query_text, retrieved, relevance_map)

        # Build output
        retrieved_ids = [r["chunk_id"] for r in retrieved]
        retrieved_distances = [r["distance"] for r in retrieved]
        retrieved_relevance = [relevance_map.get(cid, 0) for cid in retrieved_ids]

        query_result = {
            "query_id": qid,
            "query": query_text,
            "category": category,
            "difficulty": q.get("difficulty", "unknown"),
            "retrieved_chunk_ids": retrieved_ids,
            "retrieved_distances": retrieved_distances,
            "retrieved_relevance": retrieved_relevance,
            "mrr": eval_result["mrr"],
            "total_relevant_lenient": eval_result["total_relevant_lenient"],
            "total_relevant_strict": eval_result["total_relevant_strict"],
            "per_k": eval_result["per_k"],
            "latency": latency,
            "metadata_filter_used": FILTER_DESCRIPTIONS.get(category, "unknown"),
        }

        per_query_results.append(query_result)
        all_latencies.append(latency)

        hr_5 = eval_result["per_k"].get(5, {}).get("hit_rate", 0)
        ndcg_5 = eval_result["per_k"].get(5, {}).get("ndcg", 0)
        print(
            f"  {qid} [{category:>15}] | "
            f"Hit@5={hr_5:.0f}  "
            f"nDCG@5={ndcg_5:.4f}  "
            f"MRR={eval_result['mrr']:.4f}  "
            f"latency={latency['total_ms']:.1f}ms  "
            f"| {query_text[:40]}..."
        )

    # 6. Aggregate
    agg_metrics = aggregate_metrics(per_query_results)
    agg_latency = aggregate_latency(all_latencies)

    # 7. Build results
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_filename = f"metadata_scoped_retrieval_{timestamp}.json"
    output_path = str(RESULTS_DIR / output_filename)

    results = {
        "experiment": "metadata_scoped_retrieval",
        "description": "ORACLE metadata-scoped retrieval: query category determines document filter",
        "dataset_version": dataset.get("dataset_version", "unknown"),
        "evaluation_timestamp": timestamp,
        "embedding_model": MODEL_NAME,
        "collection_name": COLLECTION_NAME,
        "collection_count": collection_count,
        "chunk_count": len(actual_chunk_ids),
        "num_queries": len(queries),
        "k_values": K_VALUES,
        "retrieval_config": {
            "query_prefix": QUERY_PREFIX,
            "metadata_filters": {k: v for k, v in FILTER_DESCRIPTIONS.items()},
            "distance_metric": "cosine",
            "device": "cpu",
        },
        "metric_definitions": {
            "precision": "lenient: relevance >= 1",
            "strict_precision": "strict: relevance == 2",
            "recall": "lenient: relevance >= 1",
            "strict_recall": "strict: relevance == 2",
            "f1": "harmonic mean of precision and recall",
            "hit_rate": "lenient: any relevance >= 1 in top-K",
            "strict_hit_rate": "strict: any relevance == 2 in top-K",
            "mrr": "1/rank of first relevant (relevance >= 1)",
            "ndcg": "graded relevance (0/1/2), DCG = sum((2^rel-1)/log2(rank+1))",
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
            phase: {stat: round(val, 2) for stat, val in stats.items()}
            for phase, stats in agg_latency.items()
        },
        "per_query_results": per_query_results,
        "output_path": output_path,
    }

    # 8. Save scoped results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # 9. Load baseline for comparison
    print("\nLoading baseline results for comparison...")
    with open(BASELINE_FILE, "r", encoding="utf-8") as f:
        baseline = json.load(f)

    baseline_per_query = baseline["per_query_results"]

    # 10. Source confusion analysis
    scoped_confusion = analyze_source_confusion(per_query_results, eval_lookup)
    baseline_confusion = analyze_source_confusion(baseline_per_query, eval_lookup)

    # 11. Build comparison
    overall_comp = build_comparison(results, baseline, scoped_confusion, baseline_confusion)
    cat_comp = build_category_comparison(per_query_results, baseline_per_query)
    confusion_comp = build_confusion_comparison(scoped_confusion, baseline_confusion)

    # Save comparison
    comparison = {
        "overall": overall_comp,
        "category_level": cat_comp,
        "source_confusion": confusion_comp,
        "per_query_confusion_scoped": scoped_confusion,
    }
    comparison_path = str(RESULTS_DIR / f"metadata_scoped_comparison_{timestamp}.json")
    with open(comparison_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)

    # ============================================================
    # PRINT COMPARISON SUMMARY
    # ============================================================

    print("\n" + "=" * 60)
    print("A. OVERALL COMPARISON")
    print("=" * 60)
    print(f"{'Metric':<20} | {'Baseline':>10} | {'Scoped':>10} | {'Delta':>10} | {'% Change':>10}")
    print("-" * 70)
    for metric_name, vals in overall_comp.items():
        b = vals["baseline"]
        s = vals["scoped"]
        delta = s - b
        pct = (delta / b * 100) if b != 0 else float("inf")
        sign = "+" if delta >= 0 else ""
        print(f"{metric_name:<20} | {b:>10.4f} | {s:>10.4f} | {sign}{delta:>9.4f} | {sign}{pct:>9.1f}%")

    print("\n" + "=" * 60)
    print("B. CATEGORY COMPARISON (P@5, R@5, nDCG@5, Hit@5, MRR)")
    print("=" * 60)
    for cat_label, vals in cat_comp.items():
        print(f"\n  {cat_label} (n={vals['count']}):")
        for metric in ["P@1", "P@5", "R@5", "nDCG@5", "Hit@5", "MRR"]:
            b = vals[metric]["baseline"]
            s = vals[metric]["scoped"]
            delta = s - b
            sign = "+" if delta >= 0 else ""
            print(f"    {metric:<8} baseline={b:.4f}  scoped={s:.4f}  {sign}{delta:.4f}")

    print("\n" + "=" * 60)
    print("C. SOURCE CONFUSION COMPARISON")
    print("=" * 60)
    for scope_label in ["all", "single_document_queries", "cross_document_queries"]:
        vals = confusion_comp[scope_label]
        b = vals["baseline"]
        s = vals["scoped"]
        print(f"\n  {scope_label}:")
        print(f"    Top-1 wrong doc:  baseline={b['wrong_doc_top1_rate']:.4f} ({b['wrong_doc_top1_count']}/{b['total_queries']})"
              f"  scoped={s['wrong_doc_top1_rate']:.4f} ({s['wrong_doc_top1_count']}/{s['total_queries']})")
        print(f"    Top-5 wrong slots: baseline={b['wrong_doc_top5_rate']:.4f} ({b['wrong_doc_top5_count']}/{b['wrong_doc_top5_total_slots']})"
              f"  scoped={s['wrong_doc_top5_rate']:.4f} ({s['wrong_doc_top5_count']}/{s['wrong_doc_top5_total_slots']})")

    print("\n" + "=" * 60)
    print("D. LATENCY COMPARISON")
    print("=" * 60)
    b_lat = baseline["latency"]["total"]
    s_lat = results["latency"]["total"]
    print(f"  Baseline: mean={b_lat['mean']:.2f}ms  median={b_lat['median']:.2f}ms")
    print(f"  Scoped:   mean={s_lat['mean']:.2f}ms  median={s_lat['median']:.2f}ms")
    print(f"  Delta:    mean={s_lat['mean']-b_lat['mean']:+.2f}ms  median={s_lat['median']-b_lat['median']:+.2f}ms")

    print("\n" + "=" * 60)
    print("E. HYPOTHESIS EVALUATION")
    print("=" * 60)

    # Evaluate hypothesis
    nice_b = cat_comp.get("NICE 2018", {}).get("MRR", {}).get("baseline", 0)
    nice_s = cat_comp.get("NICE 2018", {}).get("MRR", {}).get("scoped", 0)
    esc21_b = cat_comp.get("ESC 2021", {}).get("MRR", {}).get("baseline", 0)
    esc21_s = cat_comp.get("ESC 2021", {}).get("MRR", {}).get("scoped", 0)
    esc23_b = cat_comp.get("ESC 2023", {}).get("MRR", {}).get("baseline", 0)
    esc23_s = cat_comp.get("ESC 2023", {}).get("MRR", {}).get("scoped", 0)
    cross_b = cat_comp.get("Cross-Document", {}).get("MRR", {}).get("baseline", 0)
    cross_s = cat_comp.get("Cross-Document", {}).get("MRR", {}).get("scoped", 0)

    single_b = confusion_comp["single_document_queries"]["baseline"]["wrong_doc_top1_rate"]
    single_s = confusion_comp["single_document_queries"]["scoped"]["wrong_doc_top1_rate"]

    nice_improved = nice_s > nice_b
    esc23_improved = esc23_s > esc23_b
    cross_similar = abs(cross_s - cross_b) < 0.1

    hypothesis_confirmed = nice_improved and single_s < single_b

    print(f"  NICE MRR:     {nice_b:.4f} -> {nice_s:.4f} ({'IMPROVED' if nice_improved else 'NO CHANGE'})")
    print(f"  ESC 2021 MRR: {esc21_b:.4f} -> {esc21_s:.4f}")
    print(f"  ESC 2023 MRR: {esc23_b:.4f} -> {esc23_s:.4f} ({'IMPROVED' if esc23_improved else 'NO CHANGE'})")
    print(f"  Cross-doc MRR: {cross_b:.4f} -> {cross_s:.4f} ({'SIMILAR' if cross_similar else 'CHANGED'})")
    print(f"  Single-doc wrong top-1: {single_b:.4f} -> {single_s:.4f}")
    print()

    if hypothesis_confirmed:
        print("  HYPOTHESIS: CONFIRMED")
        print("  Corpus dilution / source competition is the dominant failure mode.")
        print("  Metadata filtering eliminates wrong-document contamination for single-doc queries.")
        if cross_similar:
            print("  Cross-document queries remain similar (expected: no filter applied).")
    else:
        print("  HYPOTHESIS: NOT CONFIRMED")
        print("  Metadata filtering did not produce the expected improvement.")

    print("\n" + "=" * 60)
    print("F. RECOMMENDATION FOR NEXT EXPERIMENT")
    print("=" * 60)
    if hypothesis_confirmed:
        print("  The experiment confirmed that corpus dilution is the primary issue.")
        print("  Recommended next step: Implement document-source routing.")
        print("  Options:")
        print("    1. Metadata-based routing (keyword/rule-based query classification)")
        print("    2. LLM-based query routing (classify query -> document)")
        print("    3. Cross-encoder reranking (re-score to boost correct-document chunks)")
        print("  The metadata-scoped results show the ceiling achievable with perfect routing.")
    else:
        print("  Results are mixed. Consider cross-encoder reranking as the next experiment,")
        print("  which can re-score results without needing explicit routing.")

    print("\n" + "=" * 60)
    print(f"Results saved to: {output_path}")
    print(f"Comparison saved to: {comparison_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
