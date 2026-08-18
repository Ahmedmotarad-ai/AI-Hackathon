"""
Step 12: Candidate Pool Size Experiment
Tests whether increasing candidate K from 20 to 50 improves retrieval quality.
"""
import json, math, statistics, time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder

DB_PATH = Path("data/vector_db")
COLLECTION_NAME = "medical_guidelines"
BGE_MODEL = "BAAI/bge-small-en-v1.5"
CE_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
EVAL_FILE = Path("data/evaluation/multidoc_eval_dataset.json")
RESULTS_DIR = Path("data/evaluation/results")
CHUNKS_FILE = Path("data/chunks/chunks.jsonl")
K_VALUES = [1, 3, 5, 10]
CANDIDATE_K_VALUES = [20, 50]

CATEGORY_FILTERS = {
    "nice": {"$and": [{"section": {"$ne": "Front matter"}}, {"document": "NICE_HF_2018_Guideline.pdf"}]},
    "esc_2021": {"$and": [{"section": {"$ne": "Front matter"}}, {"document": "ESC_HF_2021_Guideline.pdf"}]},
    "esc_2023": {"$and": [{"section": {"$ne": "Front matter"}}, {"document": "ESC_HF_2023_Focused_Update.pdf"}]},
    "cross_document": {"section": {"$ne": "Front matter"}},
}


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
            "recall": recall_at_k(ids, rels, nrel, k, 1),
            "hit_rate": hit_at_k(ids, rels, k, 1),
            "ndcg": ndcg_at_k(ids, rels, k),
        }
    return {"mrr": mrr(ids, rels, 1), "total_relevant_lenient": nrel, "per_k": pk}

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

def agg_metrics(pqr):
    if not pqr: return {}
    a = {}
    for k in K_VALUES:
        a[k] = {mk: statistics.mean([q["per_k"][k][mk] for q in pqr]) for mk in ["precision", "recall", "hit_rate", "ndcg"]}
    a["mrr"] = statistics.mean([q["mrr"] for q in pqr])
    return a

def agg_lat(lats):
    t = [l.get("total_ms", l.get("dense_ms", 0)) for l in lats]
    return {"total": {"mean": round(statistics.mean(t), 2), "median": round(statistics.median(t), 2), "min": round(min(t), 2), "max": round(max(t), 2)}}


def main():
    print("=" * 70)
    print("Step 12: Candidate Pool Size Experiment (K=20 vs K=50)")
    print("=" * 70)

    dataset = load_json(EVAL_FILE)
    chunk_texts = load_chunks(CHUNKS_FILE)
    print(f"Loaded {len(dataset['queries'])} queries, {len(chunk_texts)} chunks")

    print(f"Loading BGE: {BGE_MODEL}")
    bge = SentenceTransformer(BGE_MODEL, device="cpu")
    print(f"Loading CE: {CE_MODEL}")
    ce = CrossEncoder(CE_MODEL, max_length=512)

    client = chromadb.PersistentClient(path=str(DB_PATH))
    coll = client.get_collection(name=COLLECTION_NAME)
    print(f"ChromaDB: {coll.count()} records")

    queries = dataset["queries"]
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    results = {}
    for ck in CANDIDATE_K_VALUES:
        results[f"scoped_k{ck}"] = {"pq": [], "lat": []}
        results[f"scoped_reranked_k{ck}"] = {"pq": [], "lat": []}

    print(f"\nRunning {len(queries)} queries x 2 candidate K values...\n")

    for q in queries:
        qid = q["query_id"]
        qt = q["query"]
        cat = q.get("category", "unknown")
        rels = q.get("relevant_chunks", q.get("relevance", {}))
        filt = CATEGORY_FILTERS.get(cat, CATEGORY_FILTERS["cross_document"])

        for ck in CANDIDATE_K_VALUES:
            fq = QUERY_PREFIX + qt
            t0 = time.perf_counter()
            emb = bge.encode(fq, normalize_embeddings=True).tolist()
            res = coll.query(query_embeddings=[emb], n_results=ck, where=filt)
            t1 = time.perf_counter()
            dense_ms = (t1 - t0) * 1000

            sc_ids = res["ids"][0]
            ev = evaluate(sc_ids, rels)
            sr = {"query_id": qid, "query": qt, "category": cat, "retrieved_chunk_ids": sc_ids,
                  "retrieved_relevance": [rels.get(c, 0) for c in sc_ids], "mrr": ev["mrr"],
                  "total_relevant_lenient": ev["total_relevant_lenient"], "per_k": ev["per_k"],
                  "latency": {"dense_ms": round(dense_ms, 2)}, "candidate_k": ck}
            results[f"scoped_k{ck}"]["pq"].append(sr)
            results[f"scoped_k{ck}"]["lat"].append(sr["latency"])

            cands = [{"chunk_id": c, "text": chunk_texts.get(c, "")} for c in sc_ids]
            t2 = time.perf_counter()
            pairs = [(qt, c["text"]) for c in cands]
            sc = ce.predict(pairs)
            for c, s in zip(cands, sc): c["ce_score"] = float(s)
            reranked = sorted(cands, key=lambda x: x["ce_score"], reverse=True)
            t3 = time.perf_counter()
            rr_ids = [r["chunk_id"] for r in reranked]
            rerank_ms = (t3 - t2) * 1000
            total_ms = dense_ms + rerank_ms

            rev = evaluate(rr_ids, rels)
            rr = {"query_id": qid, "query": qt, "category": cat, "retrieved_chunk_ids": rr_ids,
                  "retrieved_relevance": [rels.get(c, 0) for c in rr_ids], "mrr": rev["mrr"],
                  "total_relevant_lenient": rev["total_relevant_lenient"], "per_k": rev["per_k"],
                  "latency": {"dense_ms": round(dense_ms, 2), "rerank_ms": round(rerank_ms, 2), "total_ms": round(total_ms, 2)},
                  "candidate_k": ck}
            results[f"scoped_reranked_k{ck}"]["pq"].append(rr)
            results[f"scoped_reranked_k{ck}"]["lat"].append(rr["latency"])

        hr20 = results["scoped_k20"]["pq"][-1]["per_k"][5]["hit_rate"]
        nd20 = results["scoped_k20"]["pq"][-1]["per_k"][5]["ndcg"]
        hr50 = results["scoped_k50"]["pq"][-1]["per_k"][5]["hit_rate"]
        nd50 = results["scoped_k50"]["pq"][-1]["per_k"][5]["ndcg"]
        print(f"  {qid} [{cat:>15}] | K20: nDCG={nd20:.4f} Hit={hr20:.0f}  K50: nDCG={nd50:.4f} Hit={hr50:.0f}")

    # Aggregate
    aggs = {}
    for key, data in results.items():
        aggs[key] = {"aggregate": agg_metrics(data["pq"]), "latency": agg_lat(data["lat"])}

    # Build overall comparison
    def gm(a, k, mk):
        return a.get(k, a.get(str(k), {})).get(mk, 0)

    overall = {}
    conds = ["scoped_k20", "scoped_k50", "scoped_reranked_k20", "scoped_reranked_k50"]
    for k in K_VALUES:
        for mk in ["precision", "recall", "hit_rate", "ndcg"]:
            lbl = {"precision": f"P@{k}", "recall": f"R@{k}", "hit_rate": f"Hit@{k}", "ndcg": f"nDCG@{k}"}[mk]
            overall[lbl] = {c: round(gm(aggs[c]["aggregate"], k, mk), 4) for c in conds}
    overall["MRR"] = {c: round(aggs[c]["aggregate"].get("mrr", 0), 4) for c in conds}
    overall["mean_latency_ms"] = {c: round(aggs[c]["latency"]["total"]["mean"], 2) for c in conds}

    # Category breakdown
    cat_labels = {"nice": "NICE 2018", "esc_2021": "ESC 2021", "esc_2023": "ESC 2023", "cross_document": "Cross-Document"}
    cg = defaultdict(lambda: defaultdict(list))
    for ck in conds:
        for q in results[ck]["pq"]:
            cg[q["category"]][ck].append(q)

    cat_bd = {}
    for ck_key, label in cat_labels.items():
        cat_bd[label] = {"count": 0}
        for ck in conds:
            qs = cg[ck_key][ck]
            if not qs: continue
            if ck == "scoped_k20": cat_bd[label]["count"] = len(qs)
            a = agg_metrics(qs)
            cat_bd[label][f"{ck}_nDCG5"] = round(gm(a, 5, "ndcg"), 4)
            cat_bd[label][f"{ck}_MRR"] = round(a.get("mrr", 0), 4)

    # Candidate recall
    cr = []
    for qi, q in enumerate(queries):
        rels = q.get("relevant_chunks", q.get("relevance", {}))
        nrel = sum(1 for v in rels.values() if v >= 1)
        s20 = results["scoped_k20"]["pq"][qi]["retrieved_chunk_ids"][:20]
        s50top20 = results["scoped_k50"]["pq"][qi]["retrieved_chunk_ids"][:20]
        s50full = results["scoped_k50"]["pq"][qi]["retrieved_chunk_ids"][:50]
        cr.append({
            "query_id": q["query_id"], "category": q["category"], "total_relevant": nrel,
            "k20_recall": round(sum(1 for c in s20 if rels.get(c, 0) >= 1) / nrel if nrel else 0, 4),
            "k50_recall_top20": round(sum(1 for c in s50top20 if rels.get(c, 0) >= 1) / nrel if nrel else 0, 4),
            "k50_recall_top50": round(sum(1 for c in s50full if rels.get(c, 0) >= 1) / nrel if nrel else 0, 4),
        })
    avg_k20r = statistics.mean([c["k20_recall"] for c in cr])
    avg_k50r20 = statistics.mean([c["k50_recall_top20"] for c in cr])
    avg_k50r50 = statistics.mean([c["k50_recall_top50"] for c in cr])

    # Per-query improvement
    k50_imp = []
    for qi, q in enumerate(queries):
        s20n = results["scoped_k20"]["pq"][qi]["per_k"][5]["ndcg"]
        s50n = results["scoped_k50"]["pq"][qi]["per_k"][5]["ndcg"]
        r20n = results["scoped_reranked_k20"]["pq"][qi]["per_k"][5]["ndcg"]
        r50n = results["scoped_reranked_k50"]["pq"][qi]["per_k"][5]["ndcg"]
        k50_imp.append({
            "query_id": q["query_id"], "category": q["category"],
            "scoped_k20": round(s20n, 4), "scoped_k50": round(s50n, 4), "scoped_delta": round(s50n - s20n, 4),
            "reranked_k20": round(r20n, 4), "reranked_k50": round(r50n, 4), "reranked_delta": round(r50n - r20n, 4),
        })

    # Save
    output = {
        "experiment": "Step 12: Candidate Pool Size Experiment",
        "description": "Testing candidate K=20 vs K=50 for scoped and scoped+reranked pipelines",
        "evaluation_timestamp": ts,
        "conditions": {"scoped_k20": "BGE Top-20 + metadata filter", "scoped_k50": "BGE Top-50 + metadata filter",
                       "scoped_reranked_k20": "BGE Top-20 -> CE rerank", "scoped_reranked_k50": "BGE Top-50 -> CE rerank"},
        "overall_comparison": overall,
        "category_breakdown": cat_bd,
        "candidate_recall": {"avg_k20_recall": round(avg_k20r, 4), "avg_k50_recall_top20": round(avg_k50r20, 4),
                             "avg_k50_recall_top50": round(avg_k50r50, 4), "per_query": cr},
        "k50_improvement_per_query": k50_imp,
        "latency": {c: aggs[c]["latency"]["total"] for c in conds},
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    op = RESULTS_DIR / f"step12_candidate_pool_size_{ts}.json"
    with open(op, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    # Print
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"\n{'Metric':<15} | {'Scoped K20':>10} | {'Scoped K50':>10} | {'S+R K20':>10} | {'S+R K50':>10}")
    print("-" * 65)
    for lbl in ["P@1", "P@5", "R@5", "nDCG@5", "Hit@5", "MRR", "mean_latency_ms"]:
        v = overall[lbl]
        print(f"{lbl:<15} | {v['scoped_k20']:>10.4f} | {v['scoped_k50']:>10.4f} | {v['scoped_reranked_k20']:>10.4f} | {v['scoped_reranked_k50']:>10.4f}")

    print(f"\n--- Category nDCG@5 ---")
    for lbl, v in cat_bd.items():
        cnt = v.get("count", 0)
        s20 = v.get("scoped_k20_nDCG5", 0)
        s50 = v.get("scoped_k50_nDCG5", 0)
        r20 = v.get("scoped_reranked_k20_nDCG5", 0)
        r50 = v.get("scoped_reranked_k50_nDCG5", 0)
        print(f"  {lbl:<18} (n={cnt})  Scoped: {s20:.4f}->{s50:.4f} ({s50-s20:+.4f})  Reranked: {r20:.4f}->{r50:.4f} ({r50-r20:+.4f})")

    print(f"\n--- Candidate Recall ---")
    print(f"  K=20 avg recall@20: {avg_k20r:.4f}")
    print(f"  K=50 avg recall@20: {avg_k50r20:.4f} (delta: {avg_k50r20-avg_k20r:+.4f})")
    print(f"  K=50 avg recall@50: {avg_k50r50:.4f} (delta: {avg_k50r50-avg_k20r:+.4f})")

    print(f"\n--- Latency ---")
    for c in conds:
        lt = aggs[c]["latency"]["total"]
        print(f"  {c:<25} mean={lt['mean']:.2f}ms  median={lt['median']:.2f}ms")

    # Decision
    sn20 = overall["nDCG@5"]["scoped_k20"]
    sn50 = overall["nDCG@5"]["scoped_k50"]
    rn20 = overall["nDCG@5"]["scoped_reranked_k20"]
    rn50 = overall["nDCG@5"]["scoped_reranked_k50"]
    sm20 = overall["MRR"]["scoped_k20"]
    sm50 = overall["MRR"]["scoped_k50"]
    rm20 = overall["MRR"]["scoped_reranked_k20"]
    rm50 = overall["MRR"]["scoped_reranked_k50"]

    print(f"\n--- Decision ---")
    print(f"  Scoped K50 vs K20:   nDCG {sn20:.4f}->{sn50:.4f} ({sn50-sn20:+.4f})  MRR {sm20:.4f}->{sm50:.4f} ({sm50-sm20:+.4f})")
    print(f"  Reranked K50 vs K20: nDCG {rn20:.4f}->{rn50:.4f} ({rn50-rn20:+.4f})  MRR {rm20:.4f}->{rm50:.4f} ({rm50-rm20:+.4f})")

    if sn50 > sn20 and rn50 >= rn20:
        print("\n  DECISION: K=50 is BETTER. Adopt candidate K=50.")
    elif sn50 > sn20 and rn50 < rn20:
        print("\n  DECISION: Mixed. K=50 helps scoped but hurts reranked.")
    elif sn50 <= sn20 and rn50 > rn20:
        print("\n  DECISION: Mixed. K=50 helps reranked but not scoped.")
    else:
        print("\n  DECISION: K=20 is sufficient. Keep candidate K=20.")

    print(f"\n  Results saved: {op}")
    print("=" * 70)


if __name__ == "__main__":
    main()
