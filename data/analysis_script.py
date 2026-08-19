import json

BASE = r'C:\Users\ASUS\Desktop\ahmed\Ai-hackathon\embeddings'

# Load eval dataset
with open(f'{BASE}\\data\\evaluation\\multidoc_eval_dataset.json', 'r') as f:
    data = json.load(f)

queries = data['queries']
print(f"Total queries: {len(queries)}")
print()
print("="*120)
print("QUERY SUMMARY")
print("="*120)
for q in queries:
    rel_strict = sum(1 for v in q['relevant_chunks'].values() if v == 2)
    rel_lenient = sum(1 for v in q['relevant_chunks'].values() if v >= 1)
    total = len(q['relevant_chunks'])
    print(f"{q['query_id']} | {q['category']:15s} | strict={rel_strict:3d} lenient={rel_lenient:3d} total_chunks={total} | {q['query'][:90]}")

# Load all result files
result_files = {
    'baseline': f'{BASE}\\data\\evaluation\\results\\multidoc_baseline_retrieval_20260818T153138Z.json',
    'scoped': f'{BASE}\\data\\evaluation\\results\\metadata_scoped_retrieval_20260818T155956Z.json',
    'scoped_reranked': f'{BASE}\\data\\evaluation\\results\\scoped_reranked_retrieval_20260818T163657Z.json',
}

results = {}
for name, path in result_files.items():
    with open(path, 'r') as f:
        results[name] = json.load(f)

print()
print("="*120)
print("PER-QUERY METRICS COMPARISON (nDCG@5, strict)")
print("="*120)
print(f"{'Query':8s} {'Cat':15s} {'Baseline':>10s} {'Scoped':>10s} {'Reranked':>10s} {'Delta':>10s}")
print("-"*70)

baseline_pq = {r['query_id']: r for r in results['baseline']['per_query_results']}
scoped_pq = {r['query_id']: r for r in results['scoped']['per_query_results']}
reranked_pq = {r['query_id']: r for r in results['scoped_reranked']['per_query_results']}

for q in queries:
    qid = q['query_id']
    cat = q['category']
    b = baseline_pq[qid]['per_k']['5']['ndcg']
    s = scoped_pq[qid]['per_k']['5']['ndcg']
    r = reranked_pq[qid]['per_k']['5']['ndcg']
    delta = r - b
    marker = " **" if delta < -0.05 else (" ++" if delta > 0.05 else "")
    print(f"{qid:8s} {cat:15s} {b:10.4f} {s:10.4f} {r:10.4f} {delta:+10.4f}{marker}")

# Print overall metrics
print()
print("="*120)
print("AGGREGATE METRICS")
print("="*120)
for name, label in [('baseline','Baseline'), ('scoped','Metadata Scoped'), ('scoped_reranked','Scoped + Reranked')]:
    agg = results[name]['aggregate_metrics']
    print(f"\n{label}:")
    for k in ['1','3','5','10']:
        m = agg['per_k'][k]
        print(f"  @{k}: nDCG={m['ndcg']:.4f}  precision={m['precision']:.4f}  recall={m['recall']:.4f}  hit_rate={m['hit_rate']:.4f}")

# Print per-category
print()
print("="*120)
print("PER-CATEGORY METRICS")
print("="*120)
for name, label in [('baseline','Baseline'), ('scoped','Metadata Scoped'), ('scoped_reranked','Scoped + Reranked')]:
    print(f"\n--- {label} ---")
    for cat in ['nice', 'esc_2021', 'esc_2023', 'cross_document']:
        cat_pqs = [pq for pq in results[name]['per_query_results'] if pq['category'] == cat]
        if not cat_pqs:
            continue
        ndcg5_vals = [pq['per_k']['5']['ndcg'] for pq in cat_pqs]
        avg_ndcg = sum(ndcg5_vals) / len(ndcg5_vals)
        print(f"  {cat:15s}: nDCG@5={avg_ndcg:.4f}  (n={len(cat_pqs)})")

# Now print the relevant chunks for each query (just the IDs that are relevant)
print()
print("="*120)
print("RELEVANT CHUNK IDS PER QUERY (strict=2 only)")
print("="*120)
for q in queries:
    strict_chunks = [cid for cid, rel in q['relevant_chunks'].items() if rel == 2]
    lenient_chunks = [cid for cid, rel in q['relevant_chunks'].items() if rel == 1]
    print(f"\n{q['query_id']} ({q['category']}): {q['query'][:80]}")
    print(f"  STRICT relevant (rel=2): {strict_chunks}")
    print(f"  LENIENT relevant (rel=1): {lenient_chunks}")

# Print retrieved chunks per query for each pipeline
print()
print("="*120)
print("RETRIEVED CHUNK IDS PER QUERY")
print("="*120)
for q in queries:
    qid = q['query_id']
    print(f"\n{qid} ({q['category']}): {q['query'][:80]}")
    
    b = baseline_pq[qid]
    s = scoped_pq[qid]
    r = reranked_pq[qid]
    
    print(f"  BASELINE:  {b['retrieved_chunk_ids']}")
    print(f"  SCOPED:    {s['retrieved_chunk_ids']}")
    print(f"  RERANKED:  {r['retrieved_chunk_ids']}")
    
    # Show overlap and differences
    b_set = set(b['retrieved_chunk_ids'][:5])
    r_set = set(r['retrieved_chunk_ids'][:5])
    print(f"  Top-5 overlap baseline vs reranked: {len(b_set & r_set)} chunks in common")
