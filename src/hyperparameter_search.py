"""
Hyperparameter Search: Retrieval Pipeline
==========================================
Systematic grid search over tuneable parameters of the RAG retrieval pipeline.

Search space:
  1. candidate_k: [20, 50, 75]          -- how many BGE candidates to pass to CE
  2. query_prefix: ["", "Represent...", "query: "]
  3. ce_max_length: [128, 512]           -- CE reranker token window

Primary metric: nDCG@5 (graded, lenient)
Secondary: MRR, P@1, latency

All experiments use the 30-query multi-document eval dataset.
"""
import json, math, statistics, time, itertools
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder


# ============================================================
# Constants
# ============================================================

DB_PATH = Path("data/vector_db")
COLLECTION_NAME = "medical_guidelines"
BGE_MODEL = "BAAI/bge-small-en-v1.5"
CE_MODEL_BASE = "cross-encoder/ms-marco-MiniLM-L-6-v2"

EVAL_FILE = Path("data/evaluation/multidoc_eval_dataset.json")
RESULTS_DIR = Path("data/evaluation/results")
CHUNKS_FILE = Path("data/chunks/chunks.jsonl")
K_VALUES = [1, 3, 5, 10]

CATEGORY_FILTERS = {
    "nice":           {"$and": [{"section": {"$ne": "Front matter"}}, {"document": "NICE_HF_2018_Guideline.pdf"}]},
    "esc_2021":       {"$and": [{"section": {"$ne": "Front matter"}}, {"document": "ESC_HF_2021_Guideline.pdf"}]},
    "esc_2023":       {"$and": [{"section": {"$ne": "Front matter"}}, {"document": "ESC_HF_2023_Focused_Update.pdf"}]},
    "cross_document": {"section": {"$ne": "Front matter"}},
}

# ============================================================
# Search Space
# ============================================================

SEARCH_SPACE = {
    "candidate_k":    [20, 50, 75],
    "query_prefix":   ["", "Represent this sentence for searching relevant passages: ", "query: "],
    "ce_max_length":  [128, 512],
}
# Total configs: 3 x 3 x 2 = 18

PRIMARY_METRIC = "nDCG@5"

# ============================================================
# Metric Functions (from existing framework)
# ============================================================

def precision_at_k(ids, rels, k, th):
    return sum(1 for c in ids[:k] if rels.get(c, 0) >= th) / k

def recall_at_k(ids, rels, total, k, th):
    if total == 0: return 0.0
    return sum(1 for c in ids[:k] if rels.get(c, 0) >= th) / total

def hit_at_k(ids, rels, k, th):
    return 1.0 if any(rels.get(c, 0) >= th for c in ids[:k]) else 0.0

def mrr(ids, rels, th):
    for i, c in enumerate(ids):
        if rels.get(c, 0) >= th: return 1.0 / (i + 1)
    return 0.0

def ndcg_at_k(ids, rels, k):
    dcg = sum((2**rels.get(c, 0) - 1) / math.log2(i + 2) for i, c in enumerate(ids[:k]))
    all_r = sorted(rels.values(), reverse=True)
    idcg = sum((2**r - 1) / math.log2(i + 2) for i, r in enumerate(all_r[:k]))
    return dcg / idcg if idcg > 0 else 0.0

def evaluate(ids, rels):
    nrel = sum(1 for v in rels.values() if v >= 1)
    pk = {}
    for k in K_VALUES:
        pk[k] = {
            "precision": precision_at_k(ids, rels, k, 1),
            "recall":    recall_at_k(ids, rels, nrel, k, 1),
            "hit_rate":  hit_at_k(ids, rels, k, 1),
            "ndcg":      ndcg_at_k(ids, rels, k),
        }
    return {"mrr": mrr(ids, rels, 1), "total_relevant_lenient": nrel, "per_k": pk}

def agg_metrics(pqr):
    if not pqr: return {}
    a = {}
    for k in K_VALUES:
        a[k] = {mk: statistics.mean([q["per_k"][k][mk] for q in pqr]) for mk in ["precision","recall","hit_rate","ndcg"]}
    a["mrr"] = statistics.mean([q["mrr"] for q in pqr])
    return a

def agg_lat(lats):
    t = [l.get("total_ms", l.get("dense_ms", 0)) for l in lats]
    if not t: return {"mean": 0, "median": 0, "min": 0, "max": 0}
    return {"mean": round(statistics.mean(t), 2), "median": round(statistics.median(t), 2),
            "min": round(min(t), 2), "max": round(max(t), 2)}

# ============================================================
# Data loading
# ============================================================

def load_json(p):
    with open(p, "r", encoding="utf-8") as f: return json.load(f)

def load_chunks(p):
    t = {}
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                t[r["chunk_id"]] = r["text"]
    return t

# ============================================================
# Single configuration evaluation
# ============================================================

def run_config(queries, chunk_texts, bge, collection, config):
    """
    Run all 30 queries through one retrieval config.
    Returns: per_query_results, latencies
    """
    candidate_k  = config["candidate_k"]
    query_prefix = config["query_prefix"]
    ce_max_len   = config["ce_max_length"]

    ce = CrossEncoder(CE_MODEL_BASE, max_length=ce_max_len)

    per_query = []
    latencies = []

    for q in queries:
        qid = q["query_id"]
        qt  = q["query"]
        cat = q.get("category", "unknown")
        rels = q.get("relevant_chunks", q.get("relevance", {}))
        filt = CATEGORY_FILTERS.get(cat, CATEGORY_FILTERS["cross_document"])

        # Stage 1: Dense retrieval
        fq = query_prefix + qt
        t0 = time.perf_counter()
        emb = bge.encode(fq, normalize_embeddings=True).tolist()
        res = collection.query(query_embeddings=[emb], n_results=candidate_k, where=filt)
        t1 = time.perf_counter()
        dense_ms = (t1 - t0) * 1000

        sc_ids = res["ids"][0]

        # Stage 2: CE reranking
        cands = [{"chunk_id": c, "text": chunk_texts.get(c, "")} for c in sc_ids]
        t2 = time.perf_counter()
        pairs = [(qt, c["text"]) for c in cands]
        ce_scores = ce.predict(pairs)
        for c, s in zip(cands, ce_scores): c["ce_score"] = float(s)
        reranked = sorted(cands, key=lambda x: x["ce_score"], reverse=True)
        t3 = time.perf_counter()
        rerank_ms = (t3 - t2) * 1000

        rr_ids = [r["chunk_id"] for r in reranked]
        total_ms = dense_ms + rerank_ms

        ev = evaluate(rr_ids, rels)
        pq = {
            "query_id": qid, "query": qt, "category": cat,
            "retrieved_chunk_ids": rr_ids,
            "retrieved_relevance": [rels.get(c, 0) for c in rr_ids],
            "mrr": ev["mrr"],
            "total_relevant_lenient": ev["total_relevant_lenient"],
            "per_k": ev["per_k"],
            "latency": {"dense_ms": round(dense_ms, 2), "rerank_ms": round(rerank_ms, 2), "total_ms": round(total_ms, 2)},
        }
        per_query.append(pq)
        latencies.append(pq["latency"])

    agg = agg_metrics(per_query)
    lat = agg_lat(latencies)
    return per_query, agg, lat

# ============================================================
# Grid search
# ============================================================

def grid_search(queries, chunk_texts, bge, collection):
    keys = list(SEARCH_SPACE.keys())
    combos = list(itertools.product(*[SEARCH_SPACE[k] for k in keys]))
    n_configs = len(combos)
    n_queries = len(queries)

    print(f"\nSearch space: {len(keys)} parameters, {n_configs} configurations")
    print(f"Queries per config: {n_queries}")
    print(f"Total evaluations: {n_configs * n_queries}\n")

    all_results = []

    for idx, values in enumerate(combos):
        config = dict(zip(keys, values))
        label  = " | ".join(f"{k}={v}" for k, v in config.items())

        print(f"[{idx+1}/{n_configs}] {label}  ", end="", flush=True)
        t_start = time.perf_counter()

        per_query, agg, lat = run_config(queries, chunk_texts, bge, collection, config)
        total_time = time.perf_counter() - t_start

        nDCG5 = agg.get(5, {}).get("ndcg", 0)
        mrr_v = agg.get("mrr", 0)
        P1    = agg.get(1, {}).get("precision", 0)

        print(f"nDCG@5={nDCG5:.4f}  MRR={mrr_v:.4f}  P@1={P1:.4f}  time={total_time:.1f}s")

        all_results.append({
            "config": config,
            "aggregate": agg,
            "latency": lat,
            "per_query": per_query,
            "wall_time_s": round(total_time, 2),
        })

    return all_results

# ============================================================
# Analysis & Reporting
# ============================================================

def analyze(all_results):
    # Rank by primary metric
    ranked = sorted(all_results, key=lambda r: r["aggregate"].get(5, {}).get("ndcg", 0), reverse=True)

    print("\n" + "=" * 80)
    print("HYPERPARAMETER SEARCH RESULTS")
    print("=" * 80)

    # Summary table
    print(f"\n{'Rank':<5} {'candidate_k':<12} {'query_prefix':<15} {'ce_max_len':<12} {'nDCG@5':>8} {'MRR':>8} {'P@1':>8} {'Latency':>10}")
    print("-" * 95)
    for i, r in enumerate(ranked):
        c = r["config"]
        a = r["aggregate"]
        nDCG5 = a.get(5, {}).get("ndcg", 0)
        mrr_v = a.get("mrr", 0)
        P1    = a.get(1, {}).get("precision", 0)
        lat   = r["latency"]["mean"]
        prefix_short = c["query_prefix"][:12] if c["query_prefix"] else "(empty)"
        print(f"{i+1:<5} {c['candidate_k']:<12} {prefix_short:<15} {c['ce_max_length']:<12} {nDCG5:>8.4f} {mrr_v:>8.4f} {P1:>8.4f} {lat:>8.1f}ms")

    # Best config
    best = ranked[0]
    worst = ranked[-1]
    print(f"\n{'='*80}")
    print("BEST CONFIGURATION")
    print(f"{'='*80}")
    print(f"  candidate_k      = {best['config']['candidate_k']}")
    print(f"  query_prefix     = '{best['config']['query_prefix']}'")
    print(f"  ce_max_length    = {best['config']['ce_max_length']}")
    print(f"  nDCG@5           = {best['aggregate'].get(5,{}).get('ndcg',0):.4f}")
    print(f"  MRR              = {best['aggregate'].get('mrr',0):.4f}")
    print(f"  P@1              = {best['aggregate'].get(1,{}).get('precision',0):.4f}")
    print(f"  Latency (mean)   = {best['latency']['mean']:.1f}ms")

    print(f"\n{'='*80}")
    print("WORST CONFIGURATION")
    print(f"{'='*80}")
    print(f"  candidate_k      = {worst['config']['candidate_k']}")
    print(f"  query_prefix     = '{worst['config']['query_prefix']}'")
    print(f"  ce_max_length    = {worst['config']['ce_max_length']}")
    print(f"  nDCG@5           = {worst['aggregate'].get(5,{}).get('ndcg',0):.4f}")

    # Improvement over worst
    delta = best["aggregate"].get(5,{}).get("ndcg",0) - worst["aggregate"].get(5,{}).get("ndcg",0)
    print(f"\n  Best - Worst nDCG@5 = {delta:+.4f}")

    # Per-parameter analysis
    print(f"\n{'='*80}")
    print("PER-PARAMETER ANALYSIS (nDCG@5)")
    print(f"{'='*80}")
    for param in ["candidate_k", "query_prefix", "ce_max_length"]:
        print(f"\n  {param}:")
        param_vals = SEARCH_SPACE[param]
        for val in param_vals:
            matching = [r for r in ranked if r["config"][param] == val]
            if matching:
                scores = [r["aggregate"].get(5,{}).get("ndcg",0) for r in matching]
                lats   = [r["latency"]["mean"] for r in matching]
                print(f"    {str(val):<20}  nDCG@5 mean={statistics.mean(scores):.4f}  latency mean={statistics.mean(lats):.1f}ms")

    # Category breakdown of best config
    best_pq = best["per_query"]
    cats = defaultdict(list)
    for q in best_pq:
        cats[q["category"]].append(q)
    print(f"\n{'='*80}")
    print("BEST CONFIG - PER-CATEGORY BREAKDOWN")
    print(f"{'='*80}")
    cat_labels = {"nice": "NICE 2018", "esc_2021": "ESC 2021", "esc_2023": "ESC 2023", "cross_document": "Cross-Document"}
    for cat_key, label in cat_labels.items():
        qs = cats.get(cat_key, [])
        if not qs: continue
        a = agg_metrics(qs)
        print(f"  {label:<18} (n={len(qs):>2})  nDCG@5={a.get(5,{}).get('ndcg',0):.4f}  MRR={a.get('mrr',0):.4f}")

    return ranked

# ============================================================
# Main
# ============================================================

def main():
    print("=" * 80)
    print("Hyperparameter Search: Retrieval Pipeline")
    print("=" * 80)

    dataset = load_json(EVAL_FILE)
    chunk_texts = load_chunks(CHUNKS_FILE)
    queries = dataset["queries"]
    print(f"Loaded {len(queries)} queries, {len(chunk_texts)} chunks")

    # Load models
    print(f"\nLoading BGE: {BGE_MODEL}")
    bge = SentenceTransformer(BGE_MODEL, device="cpu")
    print(f"Loading CE:  {CE_MODEL_BASE}")
    print("  (CE models loaded per-config with different max_length)")

    # ChromaDB
    client = chromadb.PersistentClient(path=str(DB_PATH))
    coll = client.get_collection(name=COLLECTION_NAME)
    print(f"ChromaDB: {coll.count()} records")

    # Run grid search
    t0 = time.perf_counter()
    all_results = grid_search(queries, chunk_texts, bge, coll)
    total_wall = time.perf_counter() - t0

    # Analyze
    ranked = analyze(all_results)

    # Save
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = {
        "experiment": "Hyperparameter Search: Retrieval Pipeline",
        "description": "Grid search over candidate_k, query_prefix, ce_max_length",
        "search_space": SEARCH_SPACE,
        "primary_metric": PRIMARY_METRIC,
        "evaluation_timestamp": ts,
        "total_configs": len(all_results),
        "total_queries": len(queries),
        "total_evaluations": len(all_results) * len(queries),
        "wall_time_s": round(total_wall, 2),
        "ranked_results": [
            {
                "rank": i+1,
                "config": r["config"],
                "nDCG@5": round(r["aggregate"].get(5,{}).get("ndcg",0), 4),
                "MRR": round(r["aggregate"].get("mrr",0), 4),
                "P@1": round(r["aggregate"].get(1,{}).get("precision",0), 4),
                "latency_mean_ms": r["latency"]["mean"],
                "wall_time_s": r["wall_time_s"],
            }
            for i, r in enumerate(ranked)
        ],
        "best_config": ranked[0]["config"],
        "best_metrics": {
            "nDCG@5": round(ranked[0]["aggregate"].get(5,{}).get("ndcg",0), 4),
            "MRR": round(ranked[0]["aggregate"].get("mrr",0), 4),
            "P@1": round(ranked[0]["aggregate"].get(1,{}).get("precision",0), 4),
        },
        "best_latency": ranked[0]["latency"],
        "full_results": [
            {"config": r["config"], "aggregate": r["aggregate"], "latency": r["latency"],
             "wall_time_s": r["wall_time_s"], "per_query_ids": [q["query_id"] for q in r["per_query"]]}
            for r in ranked
        ],
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    op = RESULTS_DIR / f"hyperparameter_search_{ts}.json"
    with open(op, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*80}")
    print(f"Total search time: {total_wall:.1f}s ({total_wall/60:.1f} min)")
    print(f"Results saved: {op}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
