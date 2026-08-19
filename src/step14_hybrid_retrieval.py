"""
Step 14: Hybrid Dense + Sparse Retrieval Experiment
====================================================
Tests whether combining BGE dense retrieval with BM25 sparse retrieval
improves candidate recall and final retrieval quality, especially for ESC 2021.

Pipeline:
  Query -> BGE Dense (top-K_pool) + BM25 Sparse (top-K_pool)
       -> RRF Fusion -> Top-20 Candidates
       -> CE Reranking -> Final Results

This is an isolated experiment. It does NOT modify existing pipelines.
"""
import json, math, statistics, time, re, sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder


# ============================================================
# Configuration
# ============================================================

DB_PATH = Path("data/vector_db")
COLLECTION_NAME = "medical_guidelines"
BGE_MODEL = "BAAI/bge-small-en-v1.5"
CE_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
QUERY_PREFIX = "query: "

EVAL_FILE = Path("data/evaluation/multidoc_eval_dataset.json")
RESULTS_DIR = Path("data/evaluation/results")
CHUNKS_FILE = Path("data/chunks/chunks.jsonl")

# Dense-only baseline results (latest run, apples-to-apples)
DENSE_BASELINE_FILE = None  # Will run inline

K_VALUES = [1, 3, 5, 10]
CANDIDATE_K = 20       # Final candidates after fusion
POOL_K = 40            # How many each retriever returns before fusion

RRF_K = 60             # RRF constant

CATEGORY_FILTERS = {
    "nice":           {"$and": [{"section": {"$ne": "Front matter"}}, {"document": "NICE_HF_2018_Guideline.pdf"}]},
    "esc_2021":       {"$and": [{"section": {"$ne": "Front matter"}}, {"document": "ESC_HF_2021_Guideline.pdf"}]},
    "esc_2023":       {"$and": [{"section": {"$ne": "Front matter"}}, {"document": "ESC_HF_2023_Focused_Update.pdf"}]},
    "cross_document": {"section": {"$ne": "Front matter"}},
}


# ============================================================
# Metric Functions (identical to existing framework)
# ============================================================

def precision_at_k(ids, rels, k, th):
    return sum(1 for c in ids[:k] if rels.get(c, 0) >= th) / k

def recall_at_k(ids, rels, total, k, th):
    if total == 0: return 0.0
    return sum(1 for c in ids[:k] if rels.get(c, 0) >= th) / total

def f1_at_k(p, r):
    return 2*p*r/(p+r) if (p+r) > 0 else 0.0

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
    return {"mrr": mrr(ids, rels, 1), "total_relevant_lenient": nrel, "total_relevant_strict": nstr, "per_k": pk}

def candidate_recall(cand_ids, rels):
    nrel = sum(1 for v in rels.values() if v >= 1)
    if nrel == 0: return 0.0
    return sum(1 for c in cand_ids if rels.get(c, 0) >= 1) / nrel


# ============================================================
# Data Loading
# ============================================================

def load_json(p):
    with open(p, "r", encoding="utf-8") as f: return json.load(f)

def load_chunks(p):
    texts = {}
    ids = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                texts[r["chunk_id"]] = r["text"]
                ids.append(r["chunk_id"])
    return texts, ids

def load_chunks_full(p):
    """Load chunk_id, text, document, section."""
    chunks = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                chunks.append({
                    "chunk_id": r["chunk_id"],
                    "text": r["text"],
                    "document": r.get("document", ""),
                    "section": r.get("section", ""),
                })
    return chunks


# ============================================================
# BM25 Index
# ============================================================

def tokenize(text):
    """Simple whitespace + lowercase tokenization for BM25."""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s\-/]', ' ', text)
    return text.split()

def build_bm25_index(chunks, metadata_filter=None):
    """
    Build BM25 index over chunks, optionally filtered by metadata.
    Returns: (bm25, chunk_ids_in_index, chunk_texts_in_index)
    """
    filtered = chunks
    if metadata_filter:
        # Apply document filter if present
        doc_filter = None
        if "$and" in metadata_filter:
            for cond in metadata_filter["$and"]:
                if "document" in cond:
                    doc_filter = cond["document"]
        elif "document" in metadata_filter:
            doc_filter = metadata_filter["document"]
        
        if doc_filter:
            filtered = [c for c in filtered if c["document"] == doc_filter]
        
        # Always filter front matter
        filtered = [c for c in filtered if c["section"] != "Front matter"]
    
    corpus_texts = [tokenize(c["text"]) for c in filtered]
    chunk_ids = [c["chunk_id"] for c in filtered]
    chunk_texts = {c["chunk_id"]: c["text"] for c in filtered}
    
    bm25 = BM25Okapi(corpus_texts)
    return bm25, chunk_ids, chunk_texts

def bm25_retrieve(bm25, chunk_ids, query_text, k):
    """Retrieve top-k chunks using BM25."""
    tokens = tokenize(query_text)
    scores = bm25.get_scores(tokens)
    top_indices = scores.argsort()[::-1][:k]
    return [(chunk_ids[i], float(scores[i])) for i in top_indices]


# ============================================================
# Reciprocal Rank Fusion
# ============================================================

def rrf_fusion(dense_results, sparse_results, k=RRF_K, top_n=CANDIDATE_K):
    """
    Fuse dense and sparse results using Reciprocal Rank Fusion.
    
    Args:
        dense_results: list of (chunk_id, rank) tuples (rank 1-based)
        sparse_results: list of (chunk_id, rank) tuples (rank 1-based)
        k: RRF constant (default 60)
        top_n: number of results to return
    
    Returns:
        list of (chunk_id, rrf_score) tuples, sorted descending
    """
    scores = defaultdict(float)
    
    for rank, (cid, _) in enumerate(dense_results, start=1):
        scores[cid] += 1.0 / (k + rank)
    
    for rank, (cid, _) in enumerate(sparse_results, start=1):
        scores[cid] += 1.0 / (k + rank)
    
    ranked = sorted(scores.items(), key=lambda x: -x[1])
    return ranked[:top_n]


# ============================================================
# Aggregation
# ============================================================

def agg_metrics(pqr):
    if not pqr: return {}
    a = {}
    for k in K_VALUES:
        a[k] = {mk: statistics.mean([q["per_k"][k][mk] for q in pqr]) for mk in
                 ["precision","strict_precision","recall","strict_recall","f1","strict_f1",
                  "hit_rate","strict_hit_rate","ndcg"]}
    a["mrr"] = statistics.mean([q["mrr"] for q in pqr])
    return a

def agg_lat(lats):
    def stats(vals):
        if not vals: return {"mean":0,"median":0,"min":0,"max":0}
        return {"mean":round(statistics.mean(vals),2),"median":round(statistics.median(vals),2),
                "min":round(min(vals),2),"max":round(max(vals),2)}
    return {k: stats([l[k] for l in lats]) for k in lats[0]}


# ============================================================
# Main Experiment
# ============================================================

def main():
    print("=" * 80)
    print("Step 14: Hybrid Dense + Sparse Retrieval Experiment")
    print("=" * 80)
    
    # 1. Load data
    dataset = load_json(EVAL_FILE)
    chunk_texts, chunk_id_list = load_chunks(CHUNKS_FILE)
    chunks_full = load_chunks_full(CHUNKS_FILE)
    queries = dataset["queries"]
    print(f"Loaded {len(queries)} queries, {len(chunk_texts)} chunks")
    
    # 2. Load models
    print(f"\nLoading BGE: {BGE_MODEL}")
    bge = SentenceTransformer(BGE_MODEL, device="cpu")
    print(f"Loading CE:  {CE_MODEL}")
    reranker = CrossEncoder(CE_MODEL, max_length=512)
    
    # 3. ChromaDB
    client = chromadb.PersistentClient(path=str(DB_PATH))
    coll = client.get_collection(name=COLLECTION_NAME)
    print(f"ChromaDB: {coll.count()} records")
    
    # 4. Build BM25 indices (one per document scope + cross-doc)
    print("\nBuilding BM25 indices...")
    bm25_indices = {}
    
    # Cross-document BM25 (all docs, no front matter)
    bm25_xdoc, ids_xdoc, texts_xdoc = build_bm25_index(chunks_full)
    bm25_indices["cross_document"] = (bm25_xdoc, ids_xdoc, texts_xdoc)
    print(f"  cross_document: {len(ids_xdoc)} chunks")
    
    # Per-document BM25
    for doc_name in ["NICE_HF_2018_Guideline.pdf", "ESC_HF_2021_Guideline.pdf", "ESC_HF_2023_Focused_Update.pdf"]:
        short = doc_name.split("_")[0].lower()
        if "NICE" in doc_name: short = "nice"
        elif "2021" in doc_name: short = "esc_2021"
        elif "2023" in doc_name: short = "esc_2023"
        
        filtered = [c for c in chunks_full if c["document"] == doc_name and c["section"] != "Front matter"]
        corpus_t = [tokenize(c["text"]) for c in filtered]
        cids = [c["chunk_id"] for c in filtered]
        bm25_doc = BM25Okapi(corpus_t)
        bm25_indices[short] = (bm25_doc, cids, {c["chunk_id"]: c["text"] for c in filtered})
        print(f"  {short}: {len(cids)} chunks")
    
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    
    # Storage
    dense_pq = []    # Dense-only per-query
    hybrid_pq = []   # Hybrid per-query
    dense_lat = []
    hybrid_lat = []
    candidate_recall_data = []
    failure_analysis = []
    
    print(f"\nRunning {len(queries)} queries...")
    print(f"  Dense pool: top-{POOL_K} per retriever, fusion to top-{CANDIDATE_K}, CE rerank\n")
    
    for qi, q in enumerate(queries):
        qid = q["query_id"]
        qt = q["query"]
        cat = q.get("category", "unknown")
        rels = q.get("relevant_chunks", {})
        filt = CATEGORY_FILTERS.get(cat, CATEGORY_FILTERS["cross_document"])
        
        # ---- Dense Retrieval ----
        fq = QUERY_PREFIX + qt
        t0 = time.perf_counter()
        emb = bge.encode(fq, normalize_embeddings=True).tolist()
        dense_res = coll.query(query_embeddings=[emb], n_results=POOL_K, where=filt)
        t1 = time.perf_counter()
        dense_ms = (t1 - t0) * 1000
        
        dense_ids = dense_res["ids"][0]
        dense_dists = dense_res["distances"][0]
        dense_cands = [(cid, dist) for cid, dist in zip(dense_ids, dense_dists)]
        
        # Dense-only CE rerank
        t2 = time.perf_counter()
        dense_texts = [{"chunk_id": c, "text": chunk_texts.get(c, "")} for c in dense_ids[:CANDIDATE_K]]
        pairs_d = [(qt, ct["text"]) for ct in dense_texts]
        ce_scores_d = reranker.predict(pairs_d)
        for ct, s in zip(dense_texts, ce_scores_d): ct["ce_score"] = float(s)
        dense_reranked = sorted(dense_texts, key=lambda x: -x["ce_score"])
        t3 = time.perf_counter()
        dense_rerank_ms = (t3 - t2) * 1000
        dense_total_ms = dense_ms + dense_rerank_ms
        
        dr_ids = [r["chunk_id"] for r in dense_reranked]
        dr_ev = evaluate_full(dr_ids, rels)
        dense_cr = candidate_recall(dense_ids[:CANDIDATE_K], rels)
        
        dense_pq.append({
            "query_id": qid, "query": qt, "category": cat,
            "retrieved_chunk_ids": dr_ids,
            "retrieved_ce_scores": [r["ce_score"] for r in dense_reranked],
            "retrieved_relevance": [rels.get(c, 0) for c in dr_ids],
            "mrr": dr_ev["mrr"], "total_relevant_lenient": dr_ev["total_relevant_lenient"],
            "total_relevant_strict": dr_ev["total_relevant_strict"], "per_k": dr_ev["per_k"],
            "candidate_recall": round(dense_cr, 4),
            "latency": {"dense_ms": round(dense_ms, 2), "rerank_ms": round(dense_rerank_ms, 2), "total_ms": round(dense_total_ms, 2)},
        })
        dense_lat.append(dense_pq[-1]["latency"])
        
        # ---- BM25 Retrieval ----
        bm25_for_cat = bm25_indices.get(cat, bm25_indices["cross_document"])
        bm25_idx, bm25_ids, bm25_texts_dict = bm25_for_cat
        
        t4 = time.perf_counter()
        sparse_cands = bm25_retrieve(bm25_idx, bm25_ids, qt, POOL_K)
        t5 = time.perf_counter()
        sparse_ms = (t5 - t4) * 1000
        
        sparse_ids = [cid for cid, _ in sparse_cands]
        
        # ---- RRF Fusion ----
        t6 = time.perf_counter()
        dense_ranks = [(cid, i+1) for i, cid in enumerate(dense_ids[:POOL_K])]
        sparse_ranks = [(cid, i+1) for i, cid in enumerate(sparse_ids[:POOL_K])]
        fused = rrf_fusion(dense_ranks, sparse_ranks, k=RRF_K, top_n=CANDIDATE_K)
        t7 = time.perf_counter()
        fusion_ms = (t7 - t6) * 1000
        
        fused_ids = [cid for cid, _ in fused]
        hybrid_cr = candidate_recall(fused_ids, rels)
        
        # ---- Hybrid CE Rerank ----
        t8 = time.perf_counter()
        hybrid_texts = [{"chunk_id": c, "text": chunk_texts.get(c, "")} for c in fused_ids]
        pairs_h = [(qt, ct["text"]) for ct in hybrid_texts]
        ce_scores_h = reranker.predict(pairs_h)
        for ct, s in zip(hybrid_texts, ce_scores_h): ct["ce_score"] = float(s)
        hybrid_reranked = sorted(hybrid_texts, key=lambda x: -x["ce_score"])
        t9 = time.perf_counter()
        hybrid_rerank_ms = (t9 - t8) * 1000
        hybrid_total_ms = dense_ms + sparse_ms + fusion_ms + hybrid_rerank_ms
        
        hr_ids = [r["chunk_id"] for r in hybrid_reranked]
        hr_ev = evaluate_full(hr_ids, rels)
        
        hybrid_pq.append({
            "query_id": qid, "query": qt, "category": cat,
            "retrieved_chunk_ids": hr_ids,
            "retrieved_ce_scores": [r["ce_score"] for r in hybrid_reranked],
            "retrieved_relevance": [rels.get(c, 0) for c in hr_ids],
            "mrr": hr_ev["mrr"], "total_relevant_lenient": hr_ev["total_relevant_lenient"],
            "total_relevant_strict": hr_ev["total_relevant_strict"], "per_k": hr_ev["per_k"],
            "candidate_recall": round(hybrid_cr, 4),
            "dense_top5": dense_ids[:5],
            "sparse_top5": sparse_ids[:5],
            "fused_top5": fused_ids[:5],
            "latency": {"dense_ms": round(dense_ms, 2), "sparse_ms": round(sparse_ms, 2),
                        "fusion_ms": round(fusion_ms, 2), "rerank_ms": round(hybrid_rerank_ms, 2),
                        "total_ms": round(hybrid_total_ms, 2)},
        })
        hybrid_lat.append(hybrid_pq[-1]["latency"])
        
        # Candidate recall comparison
        candidate_recall_data.append({
            "query_id": qid, "category": cat,
            "n_relevant": sum(1 for v in rels.values() if v >= 1),
            "dense_candidate_recall": round(dense_cr, 4),
            "hybrid_candidate_recall": round(hybrid_cr, 4),
            "delta": round(hybrid_cr - dense_cr, 4),
            "dense_top5": dense_ids[:5],
            "hybrid_top5": fused_ids[:5],
        })
        
        # Failure query analysis
        d_n5 = dr_ev["per_k"][5]["ndcg"]
        h_n5 = hr_ev["per_k"][5]["ndcg"]
        improved = h_n5 - d_n5
        
        print(f"  {qid} [{cat:>15}] Dense: nDCG@5={d_n5:.4f} CR={dense_cr:.2f}  Hybrid: nDCG@5={h_n5:.4f} CR={hybrid_cr:.2f}  delta={improved:+.4f}")
    
    # ---- Aggregate ----
    d_agg = agg_metrics(dense_pq)
    h_agg = agg_metrics(hybrid_pq)
    d_lat = agg_lat(dense_lat)
    h_lat = agg_lat(hybrid_lat)
    
    # ---- Overall comparison ----
    print("\n" + "=" * 80)
    print("OVERALL RESULTS")
    print("=" * 80)
    
    def gm(a, k, mk):
        return a.get(k, {}).get(mk, 0)
    
    print(f"\n{'Metric':<20} {'Dense+CE':>12} {'Hybrid+CE':>12} {'Delta':>12}")
    print("-" * 60)
    for k in K_VALUES:
        for mk in ["precision", "recall", "ndcg"]:
            lbl = {"precision": f"P@{k}", "recall": f"R@{k}", "ndcg": f"nDCG@{k}"}[mk]
            dv = gm(d_agg, k, mk)
            hv = gm(h_agg, k, mk)
            print(f"{lbl:<20} {dv:>12.4f} {hv:>12.4f} {hv-dv:>+12.4f}")
    print(f"{'MRR':<20} {d_agg.get('mrr',0):>12.4f} {h_agg.get('mrr',0):>12.4f} {h_agg.get('mrr',0)-d_agg.get('mrr',0):>+12.4f}")
    d_total_lat = d_lat.get('total_ms', d_lat.get('total', {})).get('mean', 0)
    h_total_lat = h_lat.get('total_ms', h_lat.get('total', {})).get('mean', 0)
    print(f"{'mean_latency_ms':<20} {d_total_lat:>12.1f} {h_total_lat:>12.1f} {h_total_lat-d_total_lat:>+12.1f}")
    
    # Candidate recall
    avg_d_cr = statistics.mean([c["dense_candidate_recall"] for c in candidate_recall_data])
    avg_h_cr = statistics.mean([c["hybrid_candidate_recall"] for c in candidate_recall_data])
    print(f"\n{'Avg Candidate Recall':<30} {avg_d_cr:>12.4f} {avg_h_cr:>12.4f} {avg_h_cr-avg_d_cr:>+12.4f}")
    
    # ---- Per-category ----
    print(f"\n{'='*80}")
    print("PER-CATEGORY nDCG@5")
    print(f"{'='*80}")
    
    cat_labels = {"nice": "NICE 2018", "esc_2021": "ESC 2021", "esc_2023": "ESC 2023", "cross_document": "Cross-Doc"}
    d_cg = defaultdict(list)
    h_cg = defaultdict(list)
    cr_cg = defaultdict(list)
    for pq_d, pq_h, cr in zip(dense_pq, hybrid_pq, candidate_recall_data):
        d_cg[pq_d["category"]].append(pq_d)
        h_cg[pq_h["category"]].append(pq_h)
        cr_cg[cr["category"]].append(cr)
    
    print(f"\n{'Category':<18} {'n':>3} {'Dense_nDCG5':>13} {'Hybrid_nDCG5':>14} {'Delta':>8} {'Dense_CR':>10} {'Hybrid_CR':>11} {'CR_Delta':>10}")
    print("-" * 95)
    for ck, label in cat_labels.items():
        dq = d_cg.get(ck, [])
        hq = h_cg.get(ck, [])
        cq = cr_cg.get(ck, [])
        if not dq: continue
        da = agg_metrics(dq)
        ha = agg_metrics(hq)
        d_n5 = da.get(5, {}).get("ndcg", 0)
        h_n5 = ha.get(5, {}).get("ndcg", 0)
        d_cr = statistics.mean([c["dense_candidate_recall"] for c in cq]) if cq else 0
        h_cr = statistics.mean([c["hybrid_candidate_recall"] for c in cq]) if cq else 0
        print(f"{label:<18} {len(dq):>3} {d_n5:>13.4f} {h_n5:>14.4f} {h_n5-d_n5:>+8.4f} {d_cr:>10.4f} {h_cr:>11.4f} {h_cr-d_cr:>+10.4f}")
    
    # ---- ESC 2021 Focused ----
    print(f"\n{'='*80}")
    print("ESC 2021 FOCUSED ANALYSIS")
    print(f"{'='*80}")
    
    esc_d = [q for q in dense_pq if q["category"] == "esc_2021"]
    esc_h = [q for q in hybrid_pq if q["category"] == "esc_2021"]
    esc_cr = [c for c in candidate_recall_data if c["category"] == "esc_2021"]
    
    esc_da = agg_metrics(esc_d) if esc_d else {}
    esc_ha = agg_metrics(esc_h) if esc_h else {}
    
    print(f"\n{'Metric':<20} {'Dense':>12} {'Hybrid':>12} {'Delta':>12}")
    print("-" * 60)
    for k in K_VALUES:
        for mk in ["precision", "recall", "ndcg"]:
            lbl = {"precision": f"P@{k}", "recall": f"R@{k}", "ndcg": f"nDCG@{k}"}[mk]
            dv = gm(esc_da, k, mk)
            hv = gm(esc_ha, k, mk)
            print(f"{lbl:<20} {dv:>12.4f} {hv:>12.4f} {hv-dv:>+12.4f}")
    print(f"{'MRR':<20} {esc_da.get('mrr',0):>12.4f} {esc_ha.get('mrr',0):>12.4f} {esc_ha.get('mrr',0)-esc_da.get('mrr',0):>+12.4f}")
    
    if esc_cr:
        esc_d_cr = statistics.mean([c["dense_candidate_recall"] for c in esc_cr])
        esc_h_cr = statistics.mean([c["hybrid_candidate_recall"] for c in esc_cr])
        print(f"{'Avg Candidate Recall':<20} {esc_d_cr:>12.4f} {esc_h_cr:>12.4f} {esc_h_cr-esc_d_cr:>+12.4f}")
    
    print(f"\nESC 2021 per-query:")
    print(f"{'Query':<10} {'nRel':>5} {'Dense_nDCG5':>13} {'Hybrid_nDCG5':>14} {'Dense_CR':>10} {'Hybrid_CR':>11} {'Fixed?':>8}")
    print("-" * 75)
    for dq, hq, cr in zip(esc_d, esc_h, esc_cr):
        d_n5 = dq["per_k"][5]["ndcg"]
        h_n5 = hq["per_k"][5]["ndcg"]
        fixed = "YES" if h_n5 > d_n5 + 0.01 else ("MAYBE" if h_n5 > d_n5 - 0.01 else "NO")
        print(f"{dq['query_id']:<10} {cr['n_relevant']:>5} {d_n5:>13.4f} {h_n5:>14.4f} {cr['dense_candidate_recall']:>10.4f} {cr['hybrid_candidate_recall']:>11.4f} {fixed:>8}")
    
    # ---- Failure query analysis ----
    print(f"\n{'='*80}")
    print("TARGET FAILURE QUERY ANALYSIS")
    print(f"{'='*80}")
    
    failure_ids = ["mdq009", "mdq010", "mdq012", "mdq014", "mdq016", "mdq024"]
    for fq_id in failure_ids:
        dq = next((q for q in dense_pq if q["query_id"] == fq_id), None)
        hq = next((q for q in hybrid_pq if q["query_id"] == fq_id), None)
        cr = next((c for c in candidate_recall_data if c["query_id"] == fq_id), None)
        if not dq or not hq: continue
        
        print(f"\n--- {fq_id}: {dq['query'][:80]}... ---")
        print(f"  Relevant chunks: {cr['n_relevant']}")
        print(f"  Dense candidate recall:  {cr['dense_candidate_recall']:.4f}")
        print(f"  Hybrid candidate recall: {cr['hybrid_candidate_recall']:.4f}  (delta: {cr['delta']:+.4f})")
        print(f"  Dense  nDCG@5={dq['per_k'][5]['ndcg']:.4f}  MRR={dq['mrr']:.4f}")
        print(f"  Hybrid nDCG@5={hq['per_k'][5]['ndcg']:.4f}  MRR={hq['mrr']:.4f}")
        
        # Show dense vs hybrid top candidates before reranking
        print(f"  Dense  top-5 candidates: {hq['dense_top5'][:5]}")
        print(f"  Hybrid top-5 candidates: {hq['fused_top5'][:5]}")
        print(f"  Hybrid final top-5 (post-CE): {hq['retrieved_chunk_ids'][:5]}")
        print(f"  Hybrid final relevance:       {hq['retrieved_relevance'][:5]}")
        
        # Check which relevant chunks were recovered
        rel_ids = [cid for cid, v in dq.get("retrieved_relevance_map", {}).items() if v >= 1] if "retrieved_relevance_map" in dq else []
    
    # ---- Latency ----
    print(f"\n{'='*80}")
    print("LATENCY COMPARISON")
    print(f"{'='*80}")
    print(f"\n{'Component':<20} {'Dense+CE':>12} {'Hybrid+CE':>12}")
    print("-" * 48)
    for comp_name, dv_val, hv_val in [
        ("dense_ms", d_lat.get("dense_ms",{}).get("mean",0), h_lat.get("dense_ms",{}).get("mean",0)),
        ("sparse_ms", 0, h_lat.get("sparse_ms",{}).get("mean",0)),
        ("fusion_ms", 0, h_lat.get("fusion_ms",{}).get("mean",0)),
        ("rerank_ms", d_lat.get("rerank_ms",{}).get("mean",0), h_lat.get("rerank_ms",{}).get("mean",0)),
        ("total_ms", d_lat.get("total_ms",{}).get("mean",0), h_lat.get("total_ms",{}).get("mean",0)),
    ]:
        print(f"{comp_name:<20} {dv_val:>12.1f} {hv_val:>12.1f}")
    
    # ---- What BM25 recovered ----
    print(f"\n{'='*80}")
    print("WHAT BM25 RECOVERED (queries where hybrid CR > dense CR)")
    print(f"{'='*80}")
    
    improved_cr = [c for c in candidate_recall_data if c["delta"] > 0.05]
    improved_cr.sort(key=lambda x: -x["delta"])
    
    for c in improved_cr:
        print(f"\n  {c['query_id']} [{c['category']}] nRel={c['n_relevant']}")
        print(f"    Dense CR:  {c['dense_candidate_recall']:.4f}")
        print(f"    Hybrid CR: {c['hybrid_candidate_recall']:.4f}  (+{c['delta']:.4f})")
        print(f"    Dense  top-5: {c['dense_top5'][:5]}")
        print(f"    Hybrid top-5: {c['hybrid_top5'][:5]}")
    
    if not improved_cr:
        print("  No queries showed significant candidate recall improvement.")
    
    # ---- What Hybrid failed to fix ----
    print(f"\n{'='*80}")
    print("WHAT HYBRID FAILED TO FIX")
    print(f"{'='*80}")
    
    degraded = [(dq, hq) for dq, hq in zip(dense_pq, hybrid_pq) if hq["per_k"][5]["ndcg"] < dq["per_k"][5]["ndcg"] - 0.01]
    for dq, hq in degraded:
        print(f"\n  {hq['query_id']} [{hq['category']}]")
        print(f"    Dense  nDCG@5: {dq['per_k'][5]['ndcg']:.4f}")
        print(f"    Hybrid nDCG@5: {hq['per_k'][5]['ndcg']:.4f}")
        print(f"    Hybrid top-5: {hq['retrieved_chunk_ids'][:5]}")
        print(f"    Hybrid relevance: {hq['retrieved_relevance'][:5]}")
    
    if not degraded:
        print("  No queries degraded with hybrid retrieval.")
    
    # ---- Save results ----
    output = {
        "experiment": "Step 14: Hybrid Dense + Sparse Retrieval",
        "description": "BGE dense + BM25 sparse + RRF fusion + CE reranking",
        "evaluation_timestamp": ts,
        "config": {
            "dense_model": BGE_MODEL, "query_prefix": QUERY_PREFIX,
            "ce_model": CE_MODEL, "ce_max_length": 512,
            "candidate_k": CANDIDATE_K, "pool_k": POOL_K,
            "rrf_k": RRF_K, "bm25": "rank_bm25.BM25Okapi",
        },
        "num_queries": len(queries),
        "aggregate_dense": d_agg,
        "aggregate_hybrid": h_agg,
        "latency_dense": d_lat,
        "latency_hybrid": h_lat,
        "candidate_recall": {
            "dense_avg": round(avg_d_cr, 4),
            "hybrid_avg": round(avg_h_cr, 4),
            "delta": round(avg_h_cr - avg_d_cr, 4),
            "per_query": candidate_recall_data,
        },
        "per_category": {
            cat_labels.get(ck, ck): {
                "dense_nDCG5": round(gm(agg_metrics(d_cg.get(ck, [])), 5, "ndcg"), 4) if d_cg.get(ck) else 0,
                "hybrid_nDCG5": round(gm(agg_metrics(h_cg.get(ck, [])), 5, "ndcg"), 4) if h_cg.get(ck) else 0,
                "dense_CR": round(statistics.mean([c["dense_candidate_recall"] for c in cr_cg.get(ck, [])]), 4) if cr_cg.get(ck) else 0,
                "hybrid_CR": round(statistics.mean([c["hybrid_candidate_recall"] for c in cr_cg.get(ck, [])]), 4) if cr_cg.get(ck) else 0,
            }
            for ck in cat_labels
        },
        "per_query_dense": dense_pq,
        "per_query_hybrid": hybrid_pq,
    }
    
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    op = RESULTS_DIR / f"step14_hybrid_retrieval_{ts}.json"
    with open(op, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*80}")
    print(f"Results saved: {op}")
    print(f"{'='*80}")
    
    # ---- Decision ----
    overall_d_n5 = gm(d_agg, 5, "ndcg")
    overall_h_n5 = gm(h_agg, 5, "ndcg")
    overall_d_mrr = d_agg.get("mrr", 0)
    overall_h_mrr = h_agg.get("mrr", 0)
    
    print(f"\n{'='*80}")
    print("DECISION")
    print(f"{'='*80}")
    print(f"\n  Overall nDCG@5:  Dense={overall_d_n5:.4f}  Hybrid={overall_h_n5:.4f}  delta={overall_h_n5-overall_d_n5:+.4f}")
    print(f"  Overall MRR:     Dense={overall_d_mrr:.4f}  Hybrid={overall_h_mrr:.4f}  delta={overall_h_mrr-overall_d_mrr:+.4f}")
    d_total = d_lat.get('total_ms', d_lat.get('total', {})).get('mean', 0)
    h_total = h_lat.get('total_ms', h_lat.get('total', {})).get('mean', 0)
    print(f"  Avg Latency:     Dense={d_total:.1f}ms  Hybrid={h_total:.1f}ms")
    print(f"  Candidate Recall: Dense={avg_d_cr:.4f}  Hybrid={avg_h_cr:.4f}  delta={avg_h_cr-avg_d_cr:+.4f}")
    
    if esc_cr:
        print(f"\n  ESC 2021 nDCG@5: Dense={gm(esc_da,5,'ndcg'):.4f}  Hybrid={gm(esc_ha,5,'ndcg'):.4f}")
        print(f"  ESC 2021 CR:     Dense={esc_d_cr:.4f}  Hybrid={esc_h_cr:.4f}")
    
    improvement = overall_h_n5 - overall_d_n5
    if improvement > 0.03:
        print(f"\n  DECISION: STRONG IMPROVEMENT. Hybrid significantly improves retrieval.")
    elif improvement > 0.01:
        print(f"\n  DECISION: MARGINAL IMPROVEMENT. Hybrid helps some queries.")
    elif improvement > -0.01:
        print(f"\n  DECISION: NO MEANINGFUL IMPROVEMENT. Keep Dense + CE.")
    else:
        print(f"\n  DECISION: HYBRID HURTS. Dense + CE is better.")
    
    print(f"\n{'='*80}")


if __name__ == "__main__":
    main()
