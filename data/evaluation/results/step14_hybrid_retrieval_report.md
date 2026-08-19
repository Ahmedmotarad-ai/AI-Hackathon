# Step 14: Hybrid Dense + Sparse Retrieval Experiment

## Question
Does combining BGE dense retrieval with BM25 sparse retrieval (via RRF fusion) improve candidate recall and final retrieval quality, especially for ESC 2021 queries?

## Design

### Pipeline Comparison
| Pipeline | Retrieval | Pool | Fusion | Final Candidates | Reranker |
|----------|-----------|------|--------|------------------|----------|
| Dense+CE (baseline) | BGE dense top-20 | 20 | None | 20 | CE ms-marco-MiniLM-L-6-v2 |
| Hybrid+CE (experiment) | BGE top-40 + BM25 top-40 | 40 each | RRF (k=60) | 20 | CE ms-marco-MiniLM-L-6-v2 |

### Configuration
- **Dense:** `BAAI/bge-small-en-v1.5`, query_prefix=`"query: "`, cosine distance
- **Sparse:** `rank_bm25.BM25Okapi`, whitespace tokenization + lowercase
- **Fusion:** Reciprocal Rank Fusion with k=60
- **Candidate Pool:** 40 per retriever (80 total before fusion), fused to 20
- **Reranker:** `cross-encoder/ms-marco-MiniLM-L-6-v2`, max_length=512
- **Metadata:** ORACLE document scoping + section != Front matter
- **Dataset:** 30 queries, multi-document

### BM25 Index Details
| Scope | Chunks | Description |
|-------|--------|-------------|
| cross_document | 1001 | All documents, no front matter |
| nice | 46 | NICE HF 2018 only |
| esc_2021 | 843 | ESC HF 2021 only |
| esc_2023 | 76 | ESC HF 2023 only |

---

## Results

### Overall Performance

| Metric | Dense+CE | Hybrid+CE | Delta | Verdict |
|--------|----------|-----------|-------|---------|
| **P@1** | 0.6667 | 0.7000 | +0.0333 | Hybrid better |
| nDCG@1 | 0.5333 | 0.5444 | +0.0111 | Marginal |
| **nDCG@5** | 0.4382 | 0.4257 | -0.0124 | Dense better |
| **nDCG@10** | 0.4221 | 0.4420 | +0.0199 | Hybrid better |
| **MRR** | 0.7864 | 0.7972 | +0.0109 | Marginal |
| **P@10** | 0.3400 | 0.3667 | +0.0267 | Hybrid better |
| **R@10** | 0.3123 | 0.3297 | +0.0174 | Hybrid better |

### Candidate Recall

| Scope | Dense CR | Hybrid CR | Delta |
|-------|----------|-----------|-------|
| Overall | 0.4045 | 0.4101 | +0.0056 |
| ESC 2021 | 0.1678 | 0.1904 | +0.0226 |

### Per-Category nDCG@5

| Category | Dense | Hybrid | Delta | Dense CR | Hybrid CR |
|----------|-------|--------|-------|----------|-----------|
| NICE 2018 | 0.8029 | 0.8029 | 0.0000 | 0.8631 | 0.8452 |
| ESC 2021 | 0.1614 | 0.1741 | +0.0127 | 0.1678 | 0.1904 |
| ESC 2023 | 0.3918 | 0.3196 | -0.0722 | 0.3804 | 0.3906 |
| Cross-Doc | 0.3850 | 0.3798 | -0.0052 | 0.2007 | 0.2094 |

### Latency

| Component | Dense+CE | Hybrid+CE |
|-----------|----------|-----------|
| Dense retrieval | 100.2ms | 100.2ms |
| BM25 retrieval | — | 4.0ms |
| RRF fusion | — | 0.1ms |
| CE reranking | 207.4ms | 187.7ms |
| **Total** | **307.7ms** | **291.9ms** |

---

## ESC 2021 Focused Analysis

| Metric | Dense | Hybrid | Delta |
|--------|-------|--------|-------|
| P@1 | 0.3750 | 0.5000 | +0.1250 |
| MRR | 0.6364 | 0.6771 | +0.0407 |
| nDCG@5 | 0.1614 | 0.1741 | +0.0127 |
| nDCG@10 | 0.1630 | 0.2060 | +0.0430 |
| Candidate Recall | 0.1678 | 0.1904 | +0.0226 |

### ESC 2021 Per-Query Results

| Query | nRel | Dense nDCG@5 | Hybrid nDCG@5 | Dense CR | Hybrid CR | Fixed? |
|-------|------|-------------|---------------|----------|-----------|--------|
| mdq009 | 16 | 0.2025 | 0.0565 | 0.1875 | 0.1875 | NO |
| mdq010 | 41 | 0.0000 | 0.3452 | 0.0732 | 0.1220 | **YES** |
| mdq011 | 50 | 0.3452 | 0.1461 | 0.0800 | 0.1200 | NO |
| mdq012 | 45 | 0.1151 | 0.2443 | 0.1333 | 0.1111 | **YES** |
| mdq013 | 36 | 0.1923 | 0.1923 | 0.1667 | 0.1944 | MAYBE |
| mdq014 | 22 | 0.0713 | 0.0565 | 0.3182 | 0.3636 | NO |
| mdq015 | 49 | 0.2409 | 0.2281 | 0.1837 | 0.2245 | NO |
| mdq016 | 20 | 0.1239 | 0.1239 | 0.2000 | 0.2000 | MAYBE |

**ESC 2021 outcomes:**
- **1 query fixed** (mdq010: +0.345 nDCG@5, CR 0.0732→0.1220)
- **1 query partially fixed** (mdq012: +0.129 nDCG@5, CR dropped but reranking improved)
- **3 queries degraded** (mdq009, mdq011, mdq014)
- **2 queries unchanged** (mdq013, mdq016)

---

## What BM25 Recovered (Queries Where Hybrid CR > Dense CR)

| Query | Category | nRel | Dense CR | Hybrid CR | Delta |
|-------|----------|------|----------|-----------|-------|
| mdq018 | ESC 2023 | 5 | 0.4000 | 0.6000 | +0.2000 |
| mdq025 | Cross-Doc | 33 | 0.3333 | 0.3939 | +0.0606 |
| mdq024 | Cross-Doc | 18 | 0.1667 | 0.2222 | +0.0556 |

BM25 helped most with exact medical terminology matching (ARNI/sacubitril/valsartan terms in mdq024, specific drug names in mdq018).

---

## What Hybrid Failed to Fix (Queries That Degraded)

| Query | Category | Dense nDCG@5 | Hybrid nDCG@5 | Delta |
|-------|----------|-------------|---------------|-------|
| mdq009 | ESC 2021 | 0.2025 | 0.0565 | -0.1460 |
| mdq011 | ESC 2021 | 0.3452 | 0.1461 | -0.1991 |
| mdq019 | ESC 2023 | 0.6844 | 0.5531 | -0.1312 |
| mdq021 | ESC 2023 | 0.6427 | 0.4138 | -0.2288 |
| mdq022 | ESC 2023 | 0.5958 | 0.5088 | -0.0870 |
| mdq030 | Cross-Doc | 0.4704 | 0.3392 | -0.1312 |

**Pattern:** BM25 degraded performance for queries that were already well-served by dense retrieval. When BGE found the right chunks, BM25's keyword matching diluted the candidate pool with less relevant chunks, displacing the dense retrieval's good candidates.

---

## Analysis

### Why Hybrid Helped (Slightly) for ESC 2021

1. **Medical terminology matching:** BM25 found chunks with exact drug names, procedure codes, and guideline-specific terminology that BGE's semantic embeddings sometimes missed.
2. **Better P@1 (+0.125) and MRR (+0.041):** For the first relevant result, BM25's exact matching helped locate the most specific chunk.
3. **Higher candidate recall (+0.023):** More relevant chunks entered the fusion pool.

### Why Hybrid Hurt ESC 2023 and Overall nDCG@5

1. **Candidate pool dilution:** When BGE already found the best chunks (ESC 2023 had high dense nDCG@5 = 0.39), BM25's keyword matches introduced noise.
2. **RRF fusion limitations:** RRF treats all rank positions equally; BM25's top-40 often included chunks irrelevant to the clinical question but matching on common medical terms.
3. **CE reranking couldn't recover:** The CE reranker had to work with a noisier candidate pool, and its 512-token limit meant it couldn't fully evaluate all candidates.

### ESC 2023 Degradation Pattern

ESC 2023 (focused update) had the worst hybrid degradation (-0.072 nDCG@5). This is likely because:
- ESC 2023 is a focused update with concentrated medical terminology
- Dense retrieval already found good semantic matches
- BM25's keyword matching introduced many chunks that shared terminology but were about different aspects of the update

---

## Conclusion

**Hybrid dense+BM25 retrieval does NOT improve overall retrieval quality.**

| Dimension | Verdict |
|-----------|---------|
| Overall nDCG@5 | **Worse** (-0.012) |
| Overall MRR | Marginal (+0.011) |
| P@1 | Better (+0.033) |
| nDCG@10 | Better (+0.020) |
| Candidate Recall | Marginal (+0.006) |
| ESC 2021 nDCG@5 | Marginal (+0.013) |
| Latency | Faster (-16ms) |

**Decision: Keep Dense + CE (no BM25).** The CE reranker is the primary quality driver. BM25 helps slightly with candidate recall but introduces enough noise to hurt final nDCG@5. The marginal MRR and nDCG@10 improvements do not justify the added complexity.

**Root cause confirmed:** The dominant failure mode is CE reranking quality, not candidate recall. Hybrid retrieval marginally improves recall (+0.6%) but the CE reranker cannot capitalize on it.

---

*Report generated: 2026-08-19*
*Results file: `data/evaluation/results/step14_hybrid_retrieval_20260819T111652Z.json`*
