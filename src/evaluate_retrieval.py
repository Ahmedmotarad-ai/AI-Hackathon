"""
Retrieval Baseline Evaluation Engine

Evaluates the current retrieval pipeline against a validated ground-truth
dataset. Measures Precision@K, Recall@K, F1@K, Hit Rate@K, MRR, nDCG@K,
and retrieval latency across multiple K values.

Retrieval configuration (DO NOT CHANGE during baseline evaluation):
  - Embedding model: BAAI/bge-small-en-v1.5 (384-dim, normalized)
  - Query prefix: "Represent this sentence for searching relevant passages: "
  - Vector DB: ChromaDB persistent, cosine distance, HNSW
  - Collection: medical_guidelines
  - Metadata filter: section != "Front matter"
"""

import json
import math
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


# ============================================================
# Configuration (mirrors existing retrieval pipeline exactly)
# ============================================================

DB_PATH = Path("data/vector_db")
COLLECTION_NAME = "medical_guidelines"
MODEL_NAME = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
METADATA_FILTER = {"section": {"$ne": "Front matter"}}

EVAL_FILE = Path("data/evaluation/multidoc_eval_dataset.json")
RESULTS_DIR = Path("data/evaluation/results")
CHUNKS_FILE = Path("data/chunks/chunks.jsonl")

K_VALUES = [1, 3, 5, 10]


# ============================================================
# Metric Definitions
# ============================================================
#
# The ground-truth dataset uses graded relevance:
#   0 = Not Relevant
#   1 = Partially Relevant
#   2 = Relevant
#
# Binary metrics (Precision, Recall, F1, Hit Rate) use two thresholds:
#   - Lenient: relevance >= 1  (Partially Relevant or Relevant)
#   - Strict:  relevance == 2  (Relevant only)
#
# Ranking metrics use graded relevance directly:
#   - nDCG uses 0/1/2 as gain values
#   - MRR uses relevance >= 1 as the threshold for "relevant"
#
# Formulas:
#   Precision@K  = (count of relevant in top-K) / K
#   Recall@K     = (count of relevant in top-K) / (total relevant in corpus)
#   F1@K         = 2 * Precision * Recall / (Precision + Recall)
#   Hit Rate@K   = 1 if any relevant chunk in top-K, else 0
#   MRR          = 1 / rank_of_first_relevant (or 0 if none)
#   nDCG@K       = DCG@K / IDCG@K
#     DCG@K      = sum((2^rel_i - 1) / log2(i + 1)) for i in 1..K
#     IDCG@K     = DCG of ideal ranking for that query
# ============================================================


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

    for q in queries:
        qid = q.get("query_id", "?")
        relevance = q.get("relevant_chunks", q.get("relevance", {}))

        # Check all chunks labeled
        missing = actual_set - set(relevance.keys())
        if missing:
            errors.append(f"Query {qid}: missing labels for {missing}")

        # Check no unknown IDs
        unknown = set(relevance.keys()) - actual_set
        if unknown:
            errors.append(f"Query {qid}: unknown chunk IDs {unknown}")

        # Check all labels valid
        for cid, label in relevance.items():
            if label not in {0, 1, 2}:
                errors.append(
                    f"Query {qid}, chunk {cid}: invalid label {label}"
                )

    return errors


# ============================================================
# Retrieval
# ============================================================

def retrieve(query_text, model, collection, max_k=10):
    """
    Execute retrieval matching the existing pipeline exactly.

    Returns:
        list of dicts with chunk_id, distance, rank
        timing dict with embedding_ms and search_ms
    """
    full_query = QUERY_PREFIX + query_text

    # Embed query
    t0 = time.perf_counter()
    query_embedding = model.encode(
        full_query, normalize_embeddings=True
    ).tolist()
    t1 = time.perf_counter()

    # ChromaDB search
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=max_k,
        where=METADATA_FILTER,
    )
    t2 = time.perf_counter()

    # Parse results
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
# Metric Calculators
# ============================================================

def precision_at_k(retrieved_ids, relevance_map, k, threshold):
    """Precision@K: fraction of top-K results that are relevant."""
    relevant_count = 0
    for cid in retrieved_ids[:k]:
        if relevance_map.get(cid, 0) >= threshold:
            relevant_count += 1
    return relevant_count / k


def recall_at_k(retrieved_ids, relevance_map, total_relevant, k, threshold):
    """Recall@K: fraction of all relevant chunks found in top-K."""
    if total_relevant == 0:
        return 0.0
    retrieved_relevant = 0
    for cid in retrieved_ids[:k]:
        if relevance_map.get(cid, 0) >= threshold:
            retrieved_relevant += 1
    return retrieved_relevant / total_relevant


def f1_at_k(precision, recall):
    """F1@K: harmonic mean of precision and recall."""
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def hit_rate_at_k(retrieved_ids, relevance_map, k, threshold):
    """Hit Rate@K: 1 if any relevant chunk in top-K, else 0."""
    for cid in retrieved_ids[:k]:
        if relevance_map.get(cid, 0) >= threshold:
            return 1.0
    return 0.0


def mrr(retrieved_ids, relevance_map, threshold):
    """Mean Reciprocal Rank: 1/rank of first relevant result."""
    for i, cid in enumerate(retrieved_ids):
        if relevance_map.get(cid, 0) >= threshold:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(retrieved_ids, relevance_map, k):
    """
    nDCG@K using graded relevance (0/1/2).

    DCG@K = sum((2^rel - 1) / log2(rank + 1)) for ranks 1..K
    IDCG@K = DCG of ideal ranking for that query
    """
    # DCG of retrieved results
    dcg = 0.0
    for i, cid in enumerate(retrieved_ids[:k]):
        rel = relevance_map.get(cid, 0)
        dcg += (2**rel - 1) / math.log2(i + 2)  # rank = i+1, so log2(rank+1)

    # IDCG: ideal ranking sorted by relevance descending
    all_rels = sorted(relevance_map.values(), reverse=True)
    idcg = 0.0
    for i, rel in enumerate(all_rels[:k]):
        idcg += (2**rel - 1) / math.log2(i + 2)

    if idcg == 0:
        return 0.0
    return dcg / idcg


# ============================================================
# Per-Query Evaluation
# ============================================================

def evaluate_query(query_data, retrieved, relevance_map):
    """Compute all metrics for a single query at all K values."""
    retrieved_ids = [r["chunk_id"] for r in retrieved]

    # Count total relevant at each threshold
    total_relevant_lenient = sum(
        1 for v in relevance_map.values() if v >= 1
    )
    total_relevant_strict = sum(
        1 for v in relevance_map.values() if v == 2
    )

    # MRR (uses lenient threshold, not K-dependent)
    rr = mrr(retrieved_ids, relevance_map, threshold=1)

    results_per_k = {}
    for k in K_VALUES:
        p_lenient = precision_at_k(
            retrieved_ids, relevance_map, k, threshold=1
        )
        p_strict = precision_at_k(
            retrieved_ids, relevance_map, k, threshold=2
        )
        r_lenient = recall_at_k(
            retrieved_ids, relevance_map, total_relevant_lenient, k, threshold=1
        )
        r_strict = recall_at_k(
            retrieved_ids, relevance_map, total_relevant_strict, k, threshold=2
        )
        f1_lenient = f1_at_k(p_lenient, r_lenient)
        f1_strict = f1_at_k(p_strict, r_strict)
        hr_lenient = hit_rate_at_k(
            retrieved_ids, relevance_map, k, threshold=1
        )
        hr_strict = hit_rate_at_k(
            retrieved_ids, relevance_map, k, threshold=2
        )
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
# Aggregation
# ============================================================

def aggregate_metrics(per_query_results):
    """Macro-average metrics across all queries."""
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
            values = [
                q["per_k"][k][mk] for q in per_query_results
            ]
            agg[k][mk] = statistics.mean(values)

    # MRR
    mrr_values = [q["mrr"] for q in per_query_results]
    agg["mrr"] = statistics.mean(mrr_values)

    return agg


def aggregate_latency(latencies):
    """Compute latency statistics."""
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
# Results Persistence
# ============================================================

def save_results(results, output_path):
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


# ============================================================
# Terminal Summary
# ============================================================

def print_summary(results):
    agg = results["aggregate_metrics"]
    agg_per_k = agg["per_k"]
    lat = results["latency"]

    print()
    print("=" * 60)
    print("RETRIEVAL BASELINE EVALUATION")
    print("=" * 60)
    print(f"\nQueries evaluated: {results['num_queries']}")
    print(f"Chunks in corpus: {results['chunk_count']}")
    print(f"Embedding model: {results['embedding_model']}")
    print(f"Collection: {results['collection_name']}")
    print(f"Distance metric: cosine")
    print(f"Metadata filter: section != Front matter")
    print(f"Query prefix: \"{QUERY_PREFIX}\"")

    for k in K_VALUES:
        m = agg_per_k[str(k)]
        print(f"\n{'=' * 60}")
        print(f"K = {k}")
        print(f"{'=' * 60}")
        print(f"  Precision@{k}:           {m['precision']:.4f}")
        print(f"  Strict Precision@{k}:    {m['strict_precision']:.4f}")
        print(f"  Recall@{k}:              {m['recall']:.4f}")
        print(f"  Strict Recall@{k}:       {m['strict_recall']:.4f}")
        print(f"  F1@{k}:                  {m['f1']:.4f}")
        print(f"  Strict F1@{k}:           {m['strict_f1']:.4f}")
        print(f"  Hit Rate@{k}:            {m['hit_rate']:.4f}")
        print(f"  Strict Hit Rate@{k}:     {m['strict_hit_rate']:.4f}")
        print(f"  nDCG@{k}:                {m['ndcg']:.4f}")

    print(f"\n{'=' * 60}")
    print(f"RANKING METRICS")
    print(f"{'=' * 60}")
    print(f"  MRR (relevance >= 1):    {agg['mrr']:.4f}")

    print(f"\n{'=' * 60}")
    print(f"LATENCY (ms)")
    print(f"{'=' * 60}")
    print(f"  Embedding:  mean={lat['embedding']['mean']:.2f}  "
          f"median={lat['embedding']['median']:.2f}  "
          f"min={lat['embedding']['min']:.2f}  "
          f"max={lat['embedding']['max']:.2f}")
    print(f"  Search:     mean={lat['search']['mean']:.2f}  "
          f"median={lat['search']['median']:.2f}  "
          f"min={lat['search']['min']:.2f}  "
          f"max={lat['search']['max']:.2f}")
    print(f"  Total:      mean={lat['total']['mean']:.2f}  "
          f"median={lat['total']['median']:.2f}  "
          f"min={lat['total']['min']:.2f}  "
          f"max={lat['total']['max']:.2f}")

    print(f"\n{'=' * 60}")
    print(f"RESULTS SAVED")
    print(f"{'=' * 60}")
    print(f"  {results['output_path']}")


# ============================================================
# Main
# ============================================================

def main():
    # ----------------------------------------------------------
    # 1. Load and validate dataset
    # ----------------------------------------------------------
    print("=" * 60)
    print("Retrieval Baseline Evaluation")
    print("=" * 60)

    if not EVAL_FILE.exists():
        print(f"\nFAIL: Evaluation file not found: {EVAL_FILE}")
        return

    dataset = load_eval_dataset(EVAL_FILE)
    actual_chunk_ids = load_chunk_ids(CHUNKS_FILE)
    print(f"\nLoaded evaluation dataset: {len(dataset['queries'])} queries")
    print(f"Loaded chunk index: {len(actual_chunk_ids)} chunks")

    errors = validate_dataset(dataset, actual_chunk_ids)
    if errors:
        print("\nDataset validation FAILED:")
        for e in errors:
            print(f"  ERROR: {e}")
        return
    print("Dataset validation: PASSED")

    # ----------------------------------------------------------
    # 2. Load retrieval components
    # ----------------------------------------------------------
    print(f"\nLoading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME, device="cpu")

    print(f"Connecting to ChromaDB: {DB_PATH}")
    client = chromadb.PersistentClient(path=str(DB_PATH))

    try:
        collection = client.get_collection(name=COLLECTION_NAME)
    except Exception as e:
        print(f"\nFAIL: Could not get collection '{COLLECTION_NAME}': {e}")
        return

    collection_count = collection.count()
    print(f"Collection '{COLLECTION_NAME}': {collection_count} records")

    if collection_count == 0:
        print("\nFAIL: Collection is empty")
        return

    # ----------------------------------------------------------
    # 3. Run evaluation
    # ----------------------------------------------------------
    queries = dataset["queries"]
    per_query_results = []
    all_latencies = []
    max_k = max(K_VALUES)

    print(f"\nRunning evaluation for {len(queries)} queries "
          f"(K={K_VALUES})...\n")

    for q in queries:
        qid = q["query_id"]
        query_text = q["query"]
        relevance_map = q.get("relevant_chunks", q.get("relevance", {}))

        # Retrieve top-max_k results
        retrieved, latency = retrieve(query_text, model, collection, max_k)

        # Evaluate
        eval_result = evaluate_query(query_text, retrieved, relevance_map)

        # Build per-query output
        retrieved_ids = [r["chunk_id"] for r in retrieved]
        retrieved_distances = [r["distance"] for r in retrieved]
        retrieved_relevance = [
            relevance_map.get(cid, 0) for cid in retrieved_ids
        ]

        query_result = {
            "query_id": qid,
            "query": query_text,
            "category": q.get("category", "unknown"),
            "difficulty": q.get("difficulty", "unknown"),
            "retrieved_chunk_ids": retrieved_ids,
            "retrieved_distances": retrieved_distances,
            "retrieved_relevance": retrieved_relevance,
            "mrr": eval_result["mrr"],
            "total_relevant_lenient": eval_result["total_relevant_lenient"],
            "total_relevant_strict": eval_result["total_relevant_strict"],
            "per_k": eval_result["per_k"],
            "latency": latency,
        }

        per_query_results.append(query_result)
        all_latencies.append(latency)

        # Print progress
        hr_5 = eval_result["per_k"].get(5, {}).get("hit_rate", 0)
        ndcg_5 = eval_result["per_k"].get(5, {}).get("ndcg", 0)
        print(
            f"  {qid} | "
            f"Hit@5={hr_5:.0f}  "
            f"nDCG@5={ndcg_5:.4f}  "
            f"MRR={eval_result['mrr']:.4f}  "
            f"latency={latency['total_ms']:.1f}ms  "
            f"| {query_text[:50]}..."
        )

    # ----------------------------------------------------------
    # 4. Aggregate
    # ----------------------------------------------------------
    agg_metrics = aggregate_metrics(per_query_results)
    agg_latency = aggregate_latency(all_latencies)

    # ----------------------------------------------------------
    # 5. Build results structure
    # ----------------------------------------------------------
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_filename = f"multidoc_baseline_retrieval_{timestamp}.json"
    output_path = str(RESULTS_DIR / output_filename)

    results = {
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
            "metadata_filter": METADATA_FILTER,
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

    # ----------------------------------------------------------
    # 6. Save and print
    # ----------------------------------------------------------
    save_results(results, output_path)
    print_summary(results)


if __name__ == "__main__":
    main()
