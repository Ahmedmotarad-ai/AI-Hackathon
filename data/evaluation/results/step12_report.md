# Step 12: Candidate Pool Size Experiment

## K=20 vs K=50 for Scoped and Scoped+Reranked Pipelines

---

## A. Executive Summary

| Metric | Scoped K20 | Scoped K50 | Delta | S+R K20 | S+R K50 | Delta |
|--------|-----------|-----------|-------|---------|---------|-------|
| P@1 | 0.5000 | 0.5000 | +0.0000 | 0.6667 | 0.7000 | +0.0333 |
| nDCG@5 | 0.3534 | 0.3534 | +0.0000 | 0.4207 | 0.4282 | +0.0075 |
| MRR | 0.6381 | 0.6395 | +0.0014 | 0.7733 | 0.8017 | +0.0284 |
| Hit@5 | 0.8667 | 0.8667 | +0.0000 | 0.9333 | 0.9333 | +0.0000 |
| Latency (mean) | 54.92ms | 41.17ms | -13.75ms | 254.68ms | 449.76ms | +195.08ms |

**Key finding**: K=50 has zero impact on scoped-only retrieval (identical results) but provides a modest improvement when combined with cross-encoder reranking (+0.0075 nDCG@5, +0.0284 MRR). The cross-encoder benefits from the larger candidate pool.

---

## B. Candidate Recall

| Metric | K=20 | K=50 | Delta |
|--------|------|------|-------|
| Avg recall@20 | 0.4037 | 0.4037 | +0.0000 |
| Avg recall@50 | — | 0.6261 | +0.2224 |

The K=50 pool contains 55% more relevant chunks than K=20 (recall 0.40 → 0.63). However, the BGE top-20 is unchanged — the additional relevant chunks are at ranks 21-50, which only the cross-encoder can leverage.

---

## C. Category Breakdown (nDCG@5)

| Category | Scoped K20 | Scoped K50 | Delta | S+R K20 | S+R K50 | Delta |
|----------|-----------|-----------|-------|---------|---------|-------|
| NICE 2018 | 0.6915 | 0.6915 | +0.0000 | 0.8029 | 0.8029 | +0.0000 |
| ESC 2021 | 0.1524 | 0.1524 | +0.0000 | 0.1324 | 0.1955 | **+0.0631** |
| ESC 2023 | 0.3478 | 0.3478 | +0.0000 | 0.3440 | 0.3196 | -0.0244 |
| Cross-Doc | 0.2203 | 0.2203 | +0.0000 | 0.3844 | 0.3676 | -0.0168 |

**ESC 2021** benefits most from K=50 (+0.0631 nDCG) — this confirms Step 11's hypothesis that the 856-chunk ESC 2021 pool needed more candidates.

**ESC 2023 and Cross-Doc** slightly degrade with K=50, likely because the CE reranker introduces noise from the 30 additional lower-ranked candidates.

---

## D. Latency Analysis

| Condition | Mean | Median | Overhead vs K20 |
|-----------|------|--------|-----------------|
| Scoped K20 | 54.92ms | 38.47ms | — |
| Scoped K50 | 41.17ms | 39.09ms | -25% (faster!) |
| S+R K20 | 254.68ms | 218.53ms | — |
| S+R K50 | 449.76ms | 456.46ms | **+77%** |

The scoped-only K=50 is actually faster (ChromaDB optimization). But the reranked K=50 adds 195ms mean latency (+77%) because the CE must score 50 pairs instead of 20.

---

## E. Decision

**K=50 helps reranked but not scoped.**

The cross-encoder benefits from the larger candidate pool (+0.0075 nDCG, +0.0284 MRR), particularly for ESC 2021 queries (+0.0631 nDCG). However, the 77% latency increase (254ms → 450ms) is significant.

**Recommendation**: Keep K=20 as default. Use K=50 only for ESC 2021 queries if document routing is available. Alternatively, test K=30 as a compromise.

---

## F. Files

- Results: `data/evaluation/results/step12_candidate_pool_size_20260818T172258Z.json`
- Script: `src/step12_candidate_pool_size.py`
