"""
Step 4: Cross-Encoder Reranking Experiment

Dense Top-20 → Cross-Encoder Reranking → Top-K evaluation

Compares against baseline_retrieval_*.json to determine whether
cross-encoder reranking improves recall@5 and nDCG@5 without
degrading P@1 and MRR.
"""

import json
import math
import statistics
import time
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
METADATA_FILTER = {"section": {"$ne": "Front matter"}}

EVAL_FILE = Path("data/evaluation/eval_dataset.json")
RESULTS_DIR = Path("data/evaluation/results")
CHUNKS_FILE = Path("data/chunks/chunks.jsonl")

K_VALUES = [1, 3, 5, 10]
CANDIDATE_K = 20


# ============================================================
# Metric Functions (identical to evaluate_retrieval.py)
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
        r_lenient = recall_at_k(
            retrieved_ids, relevance_map, total_relevant_lenient, k, threshold=1
        )
        r_strict = recall_at_k(
            retrieved_ids, relevance_map, total_relevant_strict, k, threshold=2
        )
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


def load_chunk_texts(path):
    """Load chunk_id -> text mapping for cross-encoder."""
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
        relevance = q.get("relevance", {})
        missing = actual_set - set(relevance.keys())
        if missing:
            errors.append(f"Query {qid}: missing labels for {missing}")
        unknown = set(relevance.keys()) - actual_set
        if unknown:
            errors.append(f"Query {qid}: unknown chunk IDs {unknown}")
        for cid, label in relevance.items():
            if label not in {0, 1, 2}:
                errors.append(f"Query {qid}, chunk {cid}: invalid label {label}")
    return errors


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 60)
    print("Step 4: Cross-Encoder Reranking Experiment")
    print("=" * 60)

    # ----------------------------------------------------------
    # 1. Load dataset
    # ----------------------------------------------------------
    if not EVAL_FILE.exists():
        print(f"FAIL: {EVAL_FILE} not found")
        return

    dataset = load_eval_dataset(EVAL_FILE)
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
    try:
        collection = client.get_collection(name=COLLECTION_NAME)
    except Exception as e:
        print(f"FAIL: Could not get collection: {e}")
        return
    collection_count = collection.count()
    print(f"Collection '{COLLECTION_NAME}': {collection_count} records")
    if collection_count == 0:
        print("FAIL: Collection is empty")
        return

    # ----------------------------------------------------------
    # 4. Run experiment
    # ----------------------------------------------------------
    queries = dataset["queries"]
    per_query_results = []
    all_dense_latencies = []
    all_rerank_latencies = []
    all_total_latencies = []

    print(f"\nRunning reranked evaluation for {len(queries)} queries...")
    print(f"Candidate K = {CANDIDATE_K}, Final K values = {K_VALUES}\n")

    for q in queries:
        qid = q["query_id"]
        query_text = q["query"]
        relevance_map = q["relevance"]

        # ---- Stage 1: Dense retrieval (Top-20) ----
        full_query = QUERY_PREFIX + query_text

        t_dense_start = time.perf_counter()
        query_embedding = bge_model.encode(
            full_query, normalize_embeddings=True
        ).tolist()
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=CANDIDATE_K,
            where=METADATA_FILTER,
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
                "text": results["documents"][0][i],
                "section": results["metadatas"][0][i]["section"],
                "page": results["metadatas"][0][i]["page"],
            })
            candidate_ids.append(cid)

        # ---- Stage 2: Cross-Encoder reranking ----
        t_rerank_start = time.perf_counter()
        pairs = [(query_text, c["text"]) for c in candidates]
        ce_scores = reranker.predict(pairs)
        for c, score in zip(candidates, ce_scores):
            c["ce_score"] = float(score)
        reranked = sorted(candidates, key=lambda x: x["ce_score"], reverse=True)
        t_rerank_end = time.perf_counter()

        rerank_latency_ms = (t_rerank_end - t_rerank_start) * 1000
        total_latency_ms = dense_latency_ms + rerank_latency_ms

        # ---- Build candidate recall@20 ----
        candidate_relevance = [
            relevance_map.get(cid, 0) for cid in candidate_ids
        ]
        candidate_relevant_lenient = sum(1 for r in candidate_relevance if r >= 1)
        candidate_relevant_strict = sum(1 for r in candidate_relevance if r == 2)
        total_relevant_lenient = sum(1 for v in relevance_map.values() if v >= 1)
        total_relevant_strict = sum(1 for v in relevance_map.values() if v == 2)

        cand_recall_lenient = (
            candidate_relevant_lenient / total_relevant_lenient
            if total_relevant_lenient > 0 else 0.0
        )
        cand_recall_strict = (
            candidate_relevant_strict / total_relevant_strict
            if total_relevant_strict > 0 else 0.0
        )

        # ---- Evaluate reranked results at all K ----
        reranked_ids = [r["chunk_id"] for r in reranked]
        eval_result = evaluate_query(reranked_ids, relevance_map)

        # Per-query output
        query_result = {
            "query_id": qid,
            "query": query_text,
            "category": q.get("category", "unknown"),
            "difficulty": q.get("difficulty", "unknown"),
            # Dense results (Top-20)
            "dense_chunk_ids": candidate_ids,
            "dense_distances": [c["distance"] for c in candidates],
            # Reranked results
            "reranked_chunk_ids": reranked_ids,
            "reranked_ce_scores": [r["ce_score"] for r in reranked],
            "reranked_distances": [r["distance"] for r in reranked],
            "reranked_relevance": [
                relevance_map.get(cid, 0) for cid in reranked_ids
            ],
            # Metrics
            "mrr": eval_result["mrr"],
            "total_relevant_lenient": eval_result["total_relevant_lenient"],
            "total_relevant_strict": eval_result["total_relevant_strict"],
            "per_k": eval_result["per_k"],
            # Candidate recall
            "candidate_recall": {
                "lenient": round(cand_recall_lenient, 6),
                "strict": round(cand_recall_strict, 6),
                "relevant_in_candidates": candidate_relevant_lenient,
                "relevant_in_candidates_strict": candidate_relevant_strict,
                "total_relevant_lenient": total_relevant_lenient,
                "total_relevant_strict": total_relevant_strict,
            },
            # Latency
            "latency": {
                "dense_ms": round(dense_latency_ms, 2),
                "rerank_ms": round(rerank_latency_ms, 2),
                "total_ms": round(total_latency_ms, 2),
            },
        }

        per_query_results.append(query_result)
        all_dense_latencies.append(dense_latency_ms)
        all_rerank_latencies.append(rerank_latency_ms)
        all_total_latencies.append(total_latency_ms)

        # Progress
        r5 = eval_result["per_k"][5]["recall"]
        ndcg5 = eval_result["per_k"][5]["ndcg"]
        cand_r = cand_recall_lenient
        print(
            f"  {qid} | "
            f"R@5={r5:.4f}  nDCG@5={ndcg5:.4f}  "
            f"CandR@20={cand_r:.4f}  "
            f"MRR={eval_result['mrr']:.4f}  "
            f"latency={total_latency_ms:.1f}ms  "
            f"| {query_text[:45]}..."
        )

    # ----------------------------------------------------------
    # 5. Aggregate metrics
    # ----------------------------------------------------------
    def stats(values):
        return {
            "mean": round(statistics.mean(values), 2),
            "median": round(statistics.median(values), 2),
            "min": round(min(values), 2),
            "max": round(max(values), 2),
        }

    agg_per_k = {}
    for k in K_VALUES:
        metric_keys = [
            "precision", "strict_precision",
            "recall", "strict_recall",
            "f1", "strict_f1",
            "hit_rate", "strict_hit_rate",
            "ndcg",
        ]
        agg_per_k[str(k)] = {}
        for mk in metric_keys:
            values = [q["per_k"][k][mk] for q in per_query_results]
            agg_per_k[str(k)][mk] = round(statistics.mean(values), 6)

    mrr_values = [q["mrr"] for q in per_query_results]
    agg_mrr = round(statistics.mean(mrr_values), 6)

    # Candidate recall aggregate
    cand_r_lenient = [q["candidate_recall"]["lenient"] for q in per_query_results]
    cand_r_strict = [q["candidate_recall"]["strict"] for q in per_query_results]

    # ----------------------------------------------------------
    # 6. Load baseline for comparison
    # ----------------------------------------------------------
    baseline_files = sorted(RESULTS_DIR.glob("baseline_retrieval_*.json"))
    if not baseline_files:
        print("\nWARNING: No baseline file found for comparison")
        baseline = None
    else:
        baseline_path = baseline_files[-1]
        print(f"\nLoading baseline: {baseline_path.name}")
        with open(baseline_path, "r", encoding="utf-8") as f:
            baseline = json.load(f)

    # ----------------------------------------------------------
    # 7. Build results
    # ----------------------------------------------------------
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_filename = f"reranked_retrieval_{timestamp}.json"
    output_path = str(RESULTS_DIR / output_filename)

    results = {
        "experiment": "Step 4: Cross-Encoder Reranking",
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
            "metadata_filter": METADATA_FILTER,
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
            "per_k": agg_per_k,
            "mrr": agg_mrr,
        },
        "candidate_recall": {
            "lenient": {
                "mean": round(statistics.mean(cand_r_lenient), 6),
                "median": round(statistics.median(cand_r_lenient), 6),
            },
            "strict": {
                "mean": round(statistics.mean(cand_r_strict), 6),
                "median": round(statistics.median(cand_r_strict), 6),
            },
        },
        "latency": {
            "dense": stats(all_dense_latencies),
            "rerank": stats(all_rerank_latencies),
            "total": stats(all_total_latencies),
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
    print(f"\nResults saved: {output_path}")

    # ----------------------------------------------------------
    # 9. Print comparison table
    # ----------------------------------------------------------
    print()
    print("=" * 72)
    print("BASELINE vs RERANKED COMPARISON")
    print("=" * 72)

    if baseline:
        bpk = baseline["aggregate_metrics"]["per_k"]
        rpk = agg_per_k

        header = f"{'Metric':<28} {'Baseline':>10} {'Reranked':>10} {'Delta':>10}"
        print(f"\n{header}")
        print("-" * 72)

        rows = []
        for k in K_VALUES:
            sk = str(k)
            for mk, label in [
                ("precision", f"Precision@{k}"),
                ("strict_precision", f"Strict Precision@{k}"),
                ("recall", f"Recall@{k}"),
                ("strict_recall", f"Strict Recall@{k}"),
                ("f1", f"F1@{k}"),
                ("ndcg", f"nDCG@{k}"),
                ("hit_rate", f"Hit Rate@{k}"),
            ]:
                bval = bpk[sk][mk]
                rval = rpk[sk][mk]
                delta = rval - bval
                sign = "+" if delta >= 0 else ""
                print(f"  {label:<26} {bval:>10.4f} {rval:>10.4f} {sign}{delta:>9.4f}")
                rows.append((label, bval, rval, delta))

        # MRR
        b_mrr = baseline["aggregate_metrics"]["mrr"]
        r_mrr = agg_mrr
        d_mrr = r_mrr - b_mrr
        sign = "+" if d_mrr >= 0 else ""
        print(f"  {'MRR':<26} {b_mrr:>10.4f} {r_mrr:>10.4f} {sign}{d_mrr:>9.4f}")

        # Latency
        b_lat = baseline["latency"]["total"]["mean"]
        r_lat = results["latency"]["total"]["mean"]
        d_lat = r_lat - b_lat
        sign = "+" if d_lat >= 0 else ""
        print(f"  {'Mean Latency (ms)':<26} {b_lat:>10.2f} {r_lat:>10.2f} {sign}{d_lat:>9.2f}")

        b_lat_m = baseline["latency"]["total"]["median"]
        r_lat_m = results["latency"]["total"]["median"]
        d_lat_m = r_lat_m - b_lat_m
        sign = "+" if d_lat_m >= 0 else ""
        print(f"  {'Median Latency (ms)':<26} {b_lat_m:>10.2f} {r_lat_m:>10.2f} {sign}{d_lat_m:>9.2f}")

    # Candidate recall
    print(f"\n{'=' * 72}")
    print("CANDIDATE RECALL (how much relevant info reached the reranker)")
    print("=" * 72)
    cr = results["candidate_recall"]
    print(f"  Candidate Recall@20 (lenient): mean={cr['lenient']['mean']:.4f}  "
          f"median={cr['lenient']['median']:.4f}")
    print(f"  Candidate Recall@20 (strict):  mean={cr['strict']['mean']:.4f}  "
          f"median={cr['strict']['median']:.4f}")
    print(f"  Final Recall@5 (lenient):      {agg_per_k['5']['recall']:.4f}")
    print(f"  Final Recall@10 (lenient):     {agg_per_k['10']['recall']:.4f}")

    # Per-query Recall@5 comparison
    if baseline:
        print(f"\n{'=' * 72}")
        print("PER-QUERY RECALL@5 AND nDCG@5 COMPARISON")
        print("=" * 72)
        bqr = baseline["per_query_results"]
        header = f"  {'Query':<6} {'Cat':<15} {'B_R@5':>7} {'R_R@5':>7} {'dR@5':>7} {'B_n5':>7} {'R_n5':>7} {'dn5':>7}"
        print(header)
        print("  " + "-" * 68)

        improved = 0
        unchanged = 0
        degraded = 0

        for bq, rq in zip(bqr, per_query_results):
            b_r5 = bq["per_k"]["5"]["recall"] if "5" in bq["per_k"] else bq["per_k"][5]["recall"]
            r_r5 = rq["per_k"][5]["recall"]
            d_r5 = r_r5 - b_r5

            b_n5 = bq["per_k"]["5"]["ndcg"] if "5" in bq["per_k"] else bq["per_k"][5]["ndcg"]
            r_n5 = rq["per_k"][5]["ndcg"]
            d_n5 = r_n5 - b_n5

            sign_r = "+" if d_r5 >= 0 else ""
            sign_n = "+" if d_n5 >= 0 else ""

            print(
                f"  {rq['query_id']:<6} {rq['category']:<15} "
                f"{b_r5:>7.4f} {r_r5:>7.4f} {sign_r}{d_r5:>6.4f} "
                f"{b_n5:>7.4f} {r_n5:>7.4f} {sign_n}{d_n5:>6.4f}"
            )

            if d_r5 > 0.001:
                improved += 1
            elif d_r5 < -0.001:
                degraded += 1
            else:
                unchanged += 1

        print(f"\n  Improved: {improved}  |  Unchanged: {unchanged}  |  Degraded: {degraded}")

    # ----------------------------------------------------------
    # 10. q005 analysis
    # ----------------------------------------------------------
    print(f"\n{'=' * 72}")
    print("q005 ANALYSIS: Calcium channel blockers query")
    print("=" * 72)

    q005 = [r for r in per_query_results if r["query_id"] == "q005"][0]
    q005_baseline = None
    if baseline:
        q005_baseline = [r for r in baseline["per_query_results"] if r["query_id"] == "q005"][0]

    if q005_baseline:
        print(f"\nDense ranking (baseline Top-10):")
        for i, cid in enumerate(q005_baseline["retrieved_chunk_ids"][:10]):
            rel = q005_baseline["retrieved_relevance"][i]
            marker = " <<<relevant>>>" if rel >= 1 else ""
            print(f"  #{i+1} {cid} dist={q005_baseline['retrieved_distances'][i]:.4f} rel={rel}{marker}")

    print(f"\nReranked ranking (Top-10):")
    for i, cid in enumerate(q005["reranked_chunk_ids"][:10]):
        rel = q005["reranked_relevance"][i]
        ce = q005["reranked_ce_scores"][i]
        marker = " <<<relevant>>>" if rel >= 1 else ""
        print(f"  #{i+1} {cid} ce={ce:.4f} rel={rel}{marker}")

    # Check chunk_0020 position
    reranked_pos = None
    for i, cid in enumerate(q005["reranked_chunk_ids"]):
        if cid == "nice_hf_2018_chunk_0020":
            reranked_pos = i + 1
            break

    if q005_baseline:
        dense_pos = None
        for i, cid in enumerate(q005_baseline["retrieved_chunk_ids"]):
            if cid == "nice_hf_2018_chunk_0020":
                dense_pos = i + 1
                break
        print(f"\nchunk_0020 dense position: #{dense_pos}")
    print(f"chunk_0020 reranked position: #{reranked_pos}")
    if reranked_pos and reranked_pos == 1:
        print("RESULT: Reranker PROMOTED chunk_0020 to #1 - q005 FIXED")
    elif reranked_pos and reranked_pos <= 3:
        print(f"RESULT: chunk_0020 moved to #{reranked_pos} but not #1")
    else:
        print(f"RESULT: chunk_0020 at #{reranked_pos}")

    # ----------------------------------------------------------
    # 11. Low-recall query analysis
    # ----------------------------------------------------------
    low_recall_ids = ["q003", "q006", "q007", "q011", "q015", "q019"]
    print(f"\n{'=' * 72}")
    print("LOW-RECALL QUERY ANALYSIS")
    print("=" * 72)

    for lqid in low_recall_ids:
        rq = [r for r in per_query_results if r["query_id"] == lqid][0]
        bq = None
        if baseline:
            bq = [r for r in baseline["per_query_results"] if r["query_id"] == lqid][0]

        print(f"\n--- {lqid}: {rq['query'][:60]}... ---")

        if bq:
            b_top5 = bq["retrieved_chunk_ids"][:5]
            b_r5 = bq["per_k"]["5"]["recall"] if "5" in bq["per_k"] else bq["per_k"][5]["recall"]
            b_p5 = bq["per_k"]["5"]["precision"] if "5" in bq["per_k"] else bq["per_k"][5]["precision"]
            b_n5 = bq["per_k"]["5"]["ndcg"] if "5" in bq["per_k"] else bq["per_k"][5]["ndcg"]
        else:
            b_top5 = []
            b_r5 = b_p5 = b_n5 = 0

        r_top5 = rq["reranked_chunk_ids"][:5]
        r_r5 = rq["per_k"][5]["recall"]
        r_p5 = rq["per_k"][5]["precision"]
        r_n5 = rq["per_k"][5]["ndcg"]

        # Identify relevant chunks in each
        rel_map = {}
        for q in dataset["queries"]:
            if q["query_id"] == lqid:
                rel_map = q["relevance"]
                break
        relevant_chunks = {cid for cid, r in rel_map.items() if r >= 1}

        b_relevant_in_top5 = [cid for cid in b_top5 if cid in relevant_chunks]
        r_relevant_in_top5 = [cid for cid in r_top5 if cid in relevant_chunks]

        if bq:
            d_r5 = r_r5 - b_r5
            d_n5 = r_n5 - b_n5
            sign_r = "+" if d_r5 >= 0 else ""
            sign_n = "+" if d_n5 >= 0 else ""
            print(f"  Dense  Top-5: R@5={b_r5:.4f} P@5={b_p5:.4f} nDCG@5={b_n5:.4f}")
            print(f"  Rerank Top-5: R@5={r_r5:.4f} P@5={r_p5:.4f} nDCG@5={r_n5:.4f}")
            print(f"  Delta:        R@5={sign_r}{d_r5:.4f}  nDCG@5={sign_n}{d_n5:.4f}")
        else:
            print(f"  Rerank Top-5: R@5={r_r5:.4f} P@5={r_p5:.4f} nDCG@5={r_n5:.4f}")

        print(f"  Relevant in Dense  Top-5:  {b_relevant_in_top5}")
        print(f"  Relevant in Rerank Top-5:  {r_relevant_in_top5}")

        # Candidate recall for this query
        cr = rq["candidate_recall"]
        print(f"  Candidate Recall@20: {cr['lenient']:.4f} "
              f"({cr['relevant_in_candidates']}/{cr['total_relevant_lenient']})")

        # What relevant chunks are still missing from top-5
        still_missed = relevant_chunks - set(r_top5)
        # Check if they were in top-20
        in_top20 = [cid for cid in still_missed if cid in rq["reranked_chunk_ids"][:20]]
        not_in_top20 = [cid for cid in still_missed if cid not in rq["reranked_chunk_ids"][:20]]
        print(f"  Still missed from Top-5: {len(still_missed)} chunks")
        if in_top20:
            print(f"    Case A (in Top-20 but ranked low by CE): {in_top20}")
        if not_in_top20:
            print(f"    Case B (not in Top-20 at all): {not_in_top20}")

    # ----------------------------------------------------------
    # 12. Failure decomposition
    # ----------------------------------------------------------
    print(f"\n{'=' * 72}")
    print("FAILURE DECOMPOSITION")
    print("=" * 72)

    total_missed_baseline = 0
    total_missed_reranked = 0
    case_a_count = 0
    case_b_count = 0
    for rq in per_query_results:
        rel_map = {}
        for q in dataset["queries"]:
            if q["query_id"] == rq["query_id"]:
                rel_map = q["relevance"]
                break
        relevant_chunks = {cid for cid, r in rel_map.items() if r >= 1}
        if not relevant_chunks:
            continue

        # Reranked top-5 missed
        r_top5 = set(rq["reranked_chunk_ids"][:5])
        missed = relevant_chunks - r_top5
        total_missed_reranked += len(missed)

        for cid in missed:
            if cid in rq["reranked_chunk_ids"][:CANDIDATE_K]:
                case_a_count += 1
            else:
                case_b_count += 1

        # Also count from baseline if available
        if baseline:
            bq = [b for b in baseline["per_query_results"] if b["query_id"] == rq["query_id"]]
            if bq:
                b_top5 = set(bq[0]["retrieved_chunk_ids"][:5])
                total_missed_baseline += len(relevant_chunks - b_top5)

    print(f"\nTotal relevant chunks missed from Top-5:")
    if baseline:
        print(f"  Baseline:  {total_missed_baseline}")
    print(f"  Reranked:  {total_missed_reranked}")
    print(f"\nOf reranked misses:")
    print(f"  Case A (in Top-20, ranked low by CE): {case_a_count}")
    print(f"  Case B (not in Top-20 at all):        {case_b_count}")

    # ----------------------------------------------------------
    # 13. Regression check
    # ----------------------------------------------------------
    print(f"\n{'=' * 72}")
    print("REGRESSION ANALYSIS")
    print("=" * 72)

    if baseline:
        b_mrr = baseline["aggregate_metrics"]["mrr"]
        b_p1 = baseline["aggregate_metrics"]["per_k"]["1"]["precision"]
        r_p1 = agg_per_k["1"]["precision"]

        d_mrr = agg_mrr - b_mrr
        d_p1 = r_p1 - b_p1

        print(f"\n  MRR:         baseline={b_mrr:.4f}  reranked={agg_mrr:.4f}  "
              f"Delta={'+' if d_mrr >= 0 else ''}{d_mrr:.4f}")

        print(f"  Precision@1: baseline={b_p1:.4f}  reranked={r_p1:.4f}  "
              f"Delta={'+' if d_p1 >= 0 else ''}{d_p1:.4f}")
        # Check per-query regressions in MRR and P@1
        mrr_regressions = 0
        p1_regressions = 0
        for bq, rq in zip(baseline["per_query_results"], per_query_results):
            if rq["mrr"] < bq["mrr"] - 0.001:
                mrr_regressions += 1
                print(f"  MRR regression: {rq['query_id']} {bq['mrr']:.4f} -> {rq['mrr']:.4f}")
            if rq["per_k"][1]["precision"] < (bq["per_k"]["1"]["precision"] if "1" in bq["per_k"] else bq["per_k"][1]["precision"]) - 0.001:
                p1_regressions += 1
                print(f"  P@1 regression: {rq['query_id']}")

        print(f"\n  MRR regressions:   {mrr_regressions}/20")
        print(f"  P@1 regressions:   {p1_regressions}/20")

    # ----------------------------------------------------------
    # 14. Latency breakdown
    # ----------------------------------------------------------
    print(f"\n{'=' * 72}")
    print("LATENCY ANALYSIS")
    print("=" * 72)
    lat = results["latency"]
    print(f"\n  Dense retrieval:    mean={lat['dense']['mean']:.2f}ms  "
          f"median={lat['dense']['median']:.2f}ms")
    print(f"  CE reranking:       mean={lat['rerank']['mean']:.2f}ms  "
          f"median={lat['rerank']['median']:.2f}ms")
    print(f"  Total:              mean={lat['total']['mean']:.2f}ms  "
          f"median={lat['total']['median']:.2f}ms")
    if baseline:
        b_median = baseline["latency"]["total"]["median"]
        print(f"\n  Baseline median latency:  {b_median:.2f}ms")
        print(f"  Reranked median latency:  {lat['total']['median']:.2f}ms")
        print(f"  Overhead:                 +{lat['total']['median'] - b_median:.2f}ms")


if __name__ == "__main__":
    main()
