"""
Final Reranked Retrieval — Validated Configuration
===================================================
Produces the final retrieval output for downstream LLM generation.

Configuration (validated by Steps 7–15 + Hyperparameter Search):
  Dense:      BAAI/bge-small-en-v1.5, query_prefix="query: ", normalized
  Candidates: Top-20 from ChromaDB (ORACLE doc-scoping, section != Front matter)
  Reranker:   cross-encoder/ms-marco-MiniLM-L-6-v2, max_length=512
"""
import csv, json, math, statistics, time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder

# ── Configuration ────────────────────────────────────────────────
DB_PATH = Path("data/vector_db")
COLLECTION_NAME = "medical_guidelines"
BGE_MODEL = "BAAI/bge-small-en-v1.5"
CE_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
QUERY_PREFIX = "query: "
CANDIDATE_K = 20

EVAL_FILE = Path("data/evaluation/multidoc_eval_dataset.json")
CHUNKS_FILE = Path("data/chunks/chunks.jsonl")
RESULTS_DIR = Path("data/evaluation/results")
K_VALUES = [1, 3, 5, 10]

CATEGORY_FILTERS = {
    "nice":           {"$and": [{"section": {"$ne": "Front matter"}}, {"document": "NICE_HF_2018_Guideline.pdf"}]},
    "esc_2021":       {"$and": [{"section": {"$ne": "Front matter"}}, {"document": "ESC_HF_2021_Guideline.pdf"}]},
    "esc_2023":       {"$and": [{"section": {"$ne": "Front matter"}}, {"document": "ESC_HF_2023_Focused_Update.pdf"}]},
    "cross_document": {"section": {"$ne": "Front matter"}},
}

# ── Metric Functions ─────────────────────────────────────────────
def precision_at_k(ids, rels, k, th):
    return sum(1 for c in ids[:k] if rels.get(c, 0) >= th) / k

def recall_at_k(ids, rels, total, k, th):
    return sum(1 for c in ids[:k] if rels.get(c, 0) >= th) / total if total > 0 else 0.0

def f1_at_k(p, r):
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

def hit_at_k(ids, rels, k, th):
    return 1.0 if any(rels.get(c, 0) >= th for c in ids[:k]) else 0.0

def mrr_score(ids, rels, th):
    for i, c in enumerate(ids):
        if rels.get(c, 0) >= th:
            return 1.0 / (i + 1)
    return 0.0

def ndcg_at_k(ids, rels, k):
    dcg = sum((2 ** rels.get(c, 0) - 1) / math.log2(i + 2) for i, c in enumerate(ids[:k]))
    all_r = sorted(rels.values(), reverse=True)
    idcg = sum((2 ** r - 1) / math.log2(i + 2) for i, r in enumerate(all_r[:k]))
    return dcg / idcg if idcg > 0 else 0.0

def evaluate_full(ids, rels):
    nrel = sum(1 for v in rels.values() if v >= 1)
    nstr = sum(1 for v in rels.values() if v == 2)
    pk = {}
    for k in K_VALUES:
        p_l = precision_at_k(ids, rels, k, 1)
        p_s = precision_at_k(ids, rels, k, 2)
        r_l = recall_at_k(ids, rels, nrel, k, 1)
        r_s = recall_at_k(ids, rels, nstr, k, 2)
        pk[k] = {
            "precision": p_l, "strict_precision": p_s,
            "recall": r_l, "strict_recall": r_s,
            "f1": f1_at_k(p_l, r_l), "strict_f1": f1_at_k(p_s, r_s),
            "hit_rate": hit_at_k(ids, rels, k, 1),
            "strict_hit_rate": hit_at_k(ids, rels, k, 2),
            "ndcg": ndcg_at_k(ids, rels, k),
        }
    return {"mrr": mrr_score(ids, rels, 1), "total_relevant_lenient": nrel, "total_relevant_strict": nstr, "per_k": pk}

def agg_metrics(pqr):
    if not pqr:
        return {}
    a = {}
    for k in K_VALUES:
        a[k] = {
            mk: statistics.mean([q["per_k"][k][mk] for q in pqr])
            for mk in ["precision", "strict_precision", "recall", "strict_recall",
                        "f1", "strict_f1", "hit_rate", "strict_hit_rate", "ndcg"]
        }
    a["mrr"] = statistics.mean([q["mrr"] for q in pqr])
    return a

def agg_lat(lats):
    def stats(vals):
        if not vals:
            return {"mean": 0, "median": 0, "min": 0, "max": 0}
        return {"mean": round(statistics.mean(vals), 2), "median": round(statistics.median(vals), 2),
                "min": round(min(vals), 2), "max": round(max(vals), 2)}
    return {k: stats([l[k] for l in lats]) for k in lats[0]}

# ── Data Loading ─────────────────────────────────────────────────
def load_json(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def load_chunks_metadata(p):
    meta = {}
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                meta[r["chunk_id"]] = {
                    "document": r.get("document", ""),
                    "section": r.get("section", ""),
                    "text": r.get("text", ""),
                }
    return meta

# ── Main ─────────────────────────────────────────────────────────
def main():
    print("=" * 80)
    print("FINAL RERANKED RETRIEVAL — Validated Configuration")
    print("=" * 80)

    # Load data
    dataset = load_json(EVAL_FILE)
    chunk_meta = load_chunks_metadata(CHUNKS_FILE)
    queries = dataset["queries"]
    print(f"Loaded {len(queries)} queries, {len(chunk_meta)} chunks")

    # Load models
    print(f"\nLoading Dense: {BGE_MODEL}")
    bge = SentenceTransformer(BGE_MODEL, device="cpu")
    print(f"Loading Reranker: {CE_MODEL}")
    reranker = CrossEncoder(CE_MODEL, max_length=512)

    # ChromaDB
    client = chromadb.PersistentClient(path=str(DB_PATH))
    coll = client.get_collection(name=COLLECTION_NAME)
    print(f"ChromaDB: {coll.count()} records")

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    all_rows = []
    per_query_results = []
    all_latencies = []
    cat_groups = defaultdict(list)

    print(f"\nRunning {len(queries)} queries...")

    for qi, q in enumerate(queries):
        qid = q["query_id"]
        qt = q["query"]
        cat = q.get("category", "unknown")
        rels = q.get("relevant_chunks", {})
        filt = CATEGORY_FILTERS.get(cat, CATEGORY_FILTERS["cross_document"])

        # Dense retrieval
        fq = QUERY_PREFIX + qt
        t0 = time.perf_counter()
        emb = bge.encode(fq, normalize_embeddings=True).tolist()
        res = coll.query(query_embeddings=[emb], n_results=CANDIDATE_K, where=filt)
        t1 = time.perf_counter()
        dense_ms = (t1 - t0) * 1000

        cand_ids = res["ids"][0]
        cand_dists = res["distances"][0]

        # CE reranking
        t2 = time.perf_counter()
        cand_texts = [chunk_meta.get(cid, {}).get("text", "") for cid in cand_ids]
        pairs = [(qt, ct) for ct in cand_texts]
        ce_scores = reranker.predict(pairs)
        t3 = time.perf_counter()
        rerank_ms = (t3 - t2) * 1000
        total_ms = dense_ms + rerank_ms

        # Sort by CE score (descending)
        scored = list(zip(cand_ids, cand_dists, ce_scores))
        scored.sort(key=lambda x: -x[2])

        # Evaluate
        reranked_ids = [s[0] for s in scored]
        ev = evaluate_full(reranked_ids, rels)

        # Build per-row output
        query_rows = []
        for rerank_rank, (cid, dist, ce_sc) in enumerate(scored, start=1):
            candidate_rank = cand_ids.index(cid) + 1
            cm = chunk_meta.get(cid, {})
            query_rows.append({
                "query_id": qid,
                "query": qt,
                "category": cat,
                "candidate_rank": candidate_rank,
                "rerank_rank": rerank_rank,
                "chunk_id": cid,
                "document": cm.get("document", ""),
                "section": cm.get("section", ""),
                "retrieval_score": round(float(dist), 6),
                "rerank_score": round(float(ce_sc), 6),
                "relevance_grade": rels.get(cid, 0),
            })
        all_rows.extend(query_rows)

        # Per-query result
        pq = {
            "query_id": qid,
            "query": qt,
            "category": cat,
            "retrieved_chunk_ids": reranked_ids,
            "retrieved_ce_scores": [round(float(s[2]), 6) for s in scored],
            "retrieved_relevance": [rels.get(cid, 0) for cid in reranked_ids],
            "mrr": ev["mrr"],
            "total_relevant_lenient": ev["total_relevant_lenient"],
            "total_relevant_strict": ev["total_relevant_strict"],
            "per_k": ev["per_k"],
            "latency": {"dense_ms": round(dense_ms, 2), "rerank_ms": round(rerank_ms, 2), "total_ms": round(total_ms, 2)},
        }
        per_query_results.append(pq)
        cat_groups[cat].append(pq)
        all_latencies.append(pq["latency"])

        # Top-K shorthand
        top1 = reranked_ids[0] if len(reranked_ids) > 0 else None
        top1_rel = rels.get(top1, 0) if top1 else 0
        top5_rels = [rels.get(c, 0) for c in reranked_ids[:5]]
        n5 = ev["per_k"][5]["ndcg"]

        print(f"  {qid:>7} [{cat:>15}] nDCG@5={n5:.4f}  MRR={ev['mrr']:.4f}  "
              f"top1_rel={top1_rel}  latency={total_ms:.0f}ms")

    # ── Aggregate ────────────────────────────────────────────────
    overall = agg_metrics(per_query_results)
    lat_stats = agg_lat(all_latencies)

    cat_labels = {
        "nice": "NICE 2018", "esc_2021": "ESC 2021",
        "esc_2023": "ESC 2023", "cross_document": "Cross-Doc"
    }
    cat_agg = {ck: agg_metrics(cq) for ck, cq in cat_groups.items()}

    # ── Print Overall ────────────────────────────────────────────
    print(f"\n{'=' * 80}")
    print("OVERALL METRICS")
    print(f"{'=' * 80}")
    print(f"\n{'Metric':<12}", end="")
    for k in K_VALUES:
        print(f"  @{k:>2}", end="")
    print()
    print("-" * 60)

    for mk, lbl in [("precision", "P"), ("recall", "R"), ("f1", "F1"),
                     ("hit_rate", "Hit"), ("ndcg", "nDCG")]:
        print(f"{lbl:<12}", end="")
        for k in K_VALUES:
            print(f"  {overall[k][mk]:.4f}", end="")
        print()

    print(f"{'MRR':<12}  {overall['mrr']:.4f}")
    print(f"{'Latency ms':<12}  {lat_stats['total_ms']['mean']:.1f}")

    # ── Print Per-Category ───────────────────────────────────────
    print(f"\n{'=' * 80}")
    print("PER-CATEGORY METRICS")
    print(f"{'=' * 80}")
    print(f"\n{'Category':<18} {'n':>3} {'P@1':>7} {'P@5':>7} {'nDCG@5':>8} {'MRR':>7} {'Lat ms':>8}")
    print("-" * 65)
    for ck, label in cat_labels.items():
        cq = cat_groups.get(ck, [])
        if not cq:
            continue
        ca = cat_agg.get(ck, {})
        print(f"{label:<18} {len(cq):>3} "
              f"{ca.get(5, {}).get('precision', 0):>7.4f} "
              f"{ca.get(5, {}).get('precision', 0):>7.4f} "
              f"{ca.get(5, {}).get('ndcg', 0):>8.4f} "
              f"{ca.get('mrr', 0):>7.4f} "
              f"{statistics.mean([q['latency']['total_ms'] for q in cq]):>8.1f}")

    # ── Save JSON ────────────────────────────────────────────────
    output = {
        "experiment": "Final Reranked Retrieval — Validated Configuration",
        "timestamp": ts,
        "config": {
            "dense_model": BGE_MODEL,
            "query_prefix": QUERY_PREFIX,
            "candidate_k": CANDIDATE_K,
            "reranker": CE_MODEL,
            "ce_max_length": 512,
            "metadata_filter": "ORACLE doc-scoping + section != Front matter",
        },
        "num_queries": len(queries),
        "aggregate": overall,
        "latency": lat_stats,
        "per_category": {
            cat_labels.get(ck, ck): {
                "n": len(cat_groups.get(ck, [])),
                "metrics": cat_agg.get(ck, {}),
            }
            for ck in cat_labels
        },
        "per_query": per_query_results,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = RESULTS_DIR / f"final_reranked_retrieval_{ts}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nJSON saved: {json_path}")

    # ── Save CSV ─────────────────────────────────────────────────
    csv_path = RESULTS_DIR / f"final_reranked_retrieval_{ts}.csv"
    fieldnames = ["query_id", "query", "category", "candidate_rank", "rerank_rank",
                  "chunk_id", "document", "section", "retrieval_score", "rerank_score",
                  "relevance_grade"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"CSV saved: {csv_path}")

    # ── Failure Check ────────────────────────────────────────────
    print(f"\n{'=' * 80}")
    print("FINAL FAILURE CHECK")
    print(f"{'=' * 80}")

    # Queries with nDCG@5 < 0.20
    low_n5 = [pq for pq in per_query_results if pq["per_k"][5]["ndcg"] < 0.20]
    print(f"\nQueries with nDCG@5 < 0.20: {len(low_n5)}/{len(per_query_results)}")
    for pq in low_n5:
        print(f"  {pq['query_id']} [{pq['category']}] nDCG@5={pq['per_k'][5]['ndcg']:.4f}")

    # Queries with 0 relevant in top-20
    no_rel = [pq for pq in per_query_results if not any(r >= 1 for r in pq["retrieved_relevance"])]
    print(f"\nQueries with 0 relevant in top-20: {len(no_rel)}/{len(per_query_results)}")
    for pq in no_rel:
        print(f"  {pq['query_id']} [{pq['category']}] nRel={pq['total_relevant_lenient']}")

    # Queries where CE ranks relevant chunks poorly
    ce_poor = []
    for pq in per_query_results:
        rels_in_cands = [(i, cid, pq["retrieved_relevance"][i])
                         for i, cid in enumerate(pq["retrieved_chunk_ids"])
                         if pq["retrieved_relevance"][i] >= 1]
        if rels_in_cands:
            worst_rank = max(r[0] for r in rels_in_cands)
            if worst_rank >= 5:
                ce_poor.append((pq, rels_in_cands))
    print(f"\nQueries with relevant chunks ranked >= 5: {len(ce_poor)}/{len(per_query_results)}")
    for pq, rels in ce_poor:
        rel_positions = [f"rank {r+1}(rel={rel})" for r, _, rel in rels if r >= 5]
        print(f"  {pq['query_id']} [{pq['category']}] pushed down: {', '.join(rel_positions)}")

    # Bait chunk detection: chunks that appear in top-3 with rel=0 across ≥3 queries
    bait_counts = defaultdict(int)
    for pq in per_query_results:
        for cid in pq["retrieved_chunk_ids"][:3]:
            if pq["retrieved_relevance"][pq["retrieved_chunk_ids"].index(cid)] == 0:
                bait_counts[cid] += 1
    top_bait = [(cid, cnt) for cid, cnt in bait_counts.items() if cnt >= 3]
    top_bait.sort(key=lambda x: -x[1])
    print(f"\nBait chunks (rel=0 in top-3 across >=3 queries): {len(top_bait)}")
    for cid, cnt in top_bait[:10]:
        cm = chunk_meta.get(cid, {})
        print(f"  {cid} ({cnt}x) [{cm.get('section', '?')}]")

    # ── Comparison with HP Search Baseline ───────────────────────
    hp_path = RESULTS_DIR / "hyperparameter_search_20260819T101302Z.json"
    if hp_path.exists():
        hp = load_json(hp_path)
        hp_best = hp.get("best_config", {})
        hp_agg = hp.get("aggregate", {})

        print(f"\n{'=' * 80}")
        print("COMPARISON vs HYPERSEARCH BEST")
        print(f"{'=' * 80}")
        print(f"\nHP best config: candidate_k={hp_best.get('candidate_k')}, "
              f"prefix={hp_best.get('query_prefix')}, ce_max_length={hp_best.get('ce_max_length')}")
        print(f"\n{'Metric':<12} {'HP Best':>10} {'Final':>10} {'Delta':>10}")
        print("-" * 50)
        for k in K_VALUES:
            for mk in ["precision", "recall", "ndcg"]:
                lbl = {"precision": f"P@{k}", "recall": f"R@{k}", "ndcg": f"nDCG@{k}"}[mk]
                hp_v = hp_agg.get(k, {}).get(mk, 0)
                fi_v = overall.get(k, {}).get(mk, 0)
                print(f"{lbl:<12} {hp_v:>10.4f} {fi_v:>10.4f} {fi_v - hp_v:>+10.4f}")
        hp_mrr = hp_agg.get("mrr", 0)
        fi_mrr = overall.get("mrr", 0)
        print(f"{'MRR':<12} {hp_mrr:>10.4f} {fi_mrr:>10.4f} {fi_mrr - hp_mrr:>+10.4f}")

        hp_lat = hp.get("latency", {}).get("mean_total_ms", 0)
        fi_lat = lat_stats["total_ms"]["mean"]
        print(f"{'Latency ms':<12} {hp_lat:>10.1f} {fi_lat:>10.1f} {fi_lat - hp_lat:>+10.1f}")

    print(f"\n{'=' * 80}")
    print("DONE")
    print(f"{'=' * 80}")

    return json_path, csv_path, overall, lat_stats, per_query_results


if __name__ == "__main__":
    main()
