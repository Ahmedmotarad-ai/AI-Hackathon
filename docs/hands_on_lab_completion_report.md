# Hands-on Lab Completion Report

This report maps the team's "Hands-on lab" checklist to the evidence already
present in this repository. No new experiments were run to produce this
report — every number below is pulled directly from existing files in
`data/evaluation/` and `src/`. Two checklist items are **not** covered by
existing artifacts; they are flagged explicitly rather than filled in with
invented numbers.

Dataset: 3 NICE/ESC heart-failure guideline PDFs, 1,001 chunks total
(`data/evaluation/results/chunk_quality_audit.json`).
Embedding model: `BAAI/bge-small-en-v1.5`, ChromaDB vector store.

---

## Step 4 — Mini evaluation set

**Checklist**: Create 15–20 questions (direct, paraphrased, abbreviation,
threshold, out-of-scope) with labeled expected evidence.

**Status: Partially done**

- `data/evaluation/eval_dataset.json` — 20 labeled questions across 3
  guideline documents (NICE 2018), each with per-chunk graded relevance
  (0 = not relevant, 1 = partially relevant, 2 = directly relevant) instead
  of a single expected-chunk ID. Categories used: `diagnosis`, `treatment`,
  `medication`, `comorbidity`, `management`, `advanced_care`, `lifestyle`;
  difficulty: `easy` / `medium`.
- `data/evaluation/multidoc_eval_dataset.json` — 30 additional cross-document
  questions spanning NICE 2018, ESC 2021, and ESC 2023 (`nice`: 8,
  `esc_2021`: 8, `esc_2023`: 6, `cross_document`: 8), same graded-relevance
  labeling scheme.

**Gap**: none of the 50 questions are explicitly tagged as *paraphrased*,
*abbreviation*, or *out-of-scope* the way the checklist names them — the
existing taxonomy is by clinical category and difficulty instead. If the
team needs that specific taxonomy for the submission, a small labeled subset
(e.g. 4–5 questions) covering those types would need to be added; this
report does not add them so as not to introduce untested data.

---

## Step 3 — Compare retrieval strategies

**Checklist**: Understand and test semantic, keyword, hybrid, and reranked
retrieval.

**Status: Done for semantic + reranked; keyword/hybrid not present**

- Semantic (dense) retrieval: `src/evaluate_retrieval.py`,
  results in `data/evaluation/results/baseline_retrieval_20260818T101625Z.json`.
- Cross-encoder reranking on top of semantic retrieval:
  `src/evaluate_reranked_retrieval.py`, `src/rerank_experiment.py`,
  4 result files (`reranked_retrieval_*.json`), plus a scoped variant
  (`src/evaluate_scoped_reranked_retrieval.py`,
  `scoped_reranked_*.json`).
- Metadata-scoped retrieval (filtering by document/guideline before
  ranking): `src/evaluate_metadata_scoped_retrieval.py`.

**Gap**: no pure keyword (BM25/lexical) baseline and no explicit hybrid
(keyword + semantic fusion) run exists in the results folder — only
semantic and semantic+reranking are compared.

---

## Hands-on lab checklist

### ✅ Run Top-5 search on 15–20 questions
`baseline_retrieval_20260818T101625Z.json` runs all 20 `eval_dataset.json`
questions through the retriever with `k_values: [1, 3, 5, 10]`, so Top-5
results exist for every question.

### ❌ Compare at least two chunk settings
**Not done.** `src/chunker.py` uses a single fixed configuration (max 450
tokens/chunk, ~677-token target derived from a BGE tokenizer analysis,
hard ceiling `MAX_TOKENS`). There is no second chunk-size/overlap
configuration anywhere in `src/` or `data/` to compare against. This is a
genuine gap — closing it means re-chunking the corpus with a second setting
(e.g. smaller chunks or added overlap) and re-running retrieval, which
wasn't done here since it would mean generating new data rather than
reporting on what already exists.

### ✅ Display retrieved chunks with metadata and scores
Every `per_query_results` entry in the result JSONs includes
`retrieved_chunk_ids`, `retrieved_distances` (similarity/rerank scores),
and `retrieved_relevance` (the graded label) side by side, e.g. for `q001`:

| Rank | Chunk ID | Distance | Relevance |
|---|---|---|---|
| 1 | nice_hf_2018_chunk_0042 | 0.2521 | 2 |
| 2 | nice_hf_2018_chunk_0014 | 0.2672 | 2 |
| 3 | nice_hf_2018_chunk_0027 | 0.2754 | 1 |

`data/evaluation/results/step11_failure_analysis.md` additionally renders
these as readable ranked tables per query for failure-analysis review.

### ✅ Try Top-3 vs Top-5 vs Top-10
All evaluation runs compute metrics at `k = 1, 3, 5, 10` (see
`aggregate_metrics.per_k` in every results file). A related but distinct
experiment, `src/step12_candidate_pool_size.py`
(`step12_candidate_pool_size_20260818T172258Z.json`,
`data/evaluation/results/step12_report.md`), separately compares candidate
pool sizes K=20 vs K=50 *before* reranking.

### ✅ Label each retrieved chunk as relevant or not
Both eval datasets label every candidate chunk per question on a 0/1/2
graded scale (`relevance_scale` field), not just binary — this is a
superset of the checklist's relevant/not-relevant requirement.

### ✅ Optionally test hybrid search or reranking (bonus)
Cross-encoder reranking was tested extensively (see Step 3 above) and shown
to be stable: on the 30-question multi-document set, reranking improved
17/30 queries, degraded 10/30, left 3/30 unchanged
(`data/evaluation/results/step11_failure_analysis.md`), with overall
nDCG@5 0.3534 → 0.4207 and MRR 0.6381 → 0.7733.

### ✅ Calculate Precision@3 and Precision@5
From `baseline_retrieval_20260818T101625Z.json` (`aggregate_metrics.per_k`,
lenient = relevance ≥ 1, strict = relevance == 2):

| Metric | Precision@3 | Precision@5 |
|---|---|---|
| Baseline (semantic, lenient) | 0.6167 | 0.5000 |
| Baseline (semantic, strict) | 0.4500 | 0.3200 |
| Reranked (lenient) | 0.6167 | 0.4800 |
| Reranked (strict) | 0.5167 | 0.3300 |

### ✅ Document which setup performs best and why
- `data/evaluation/results/step11_failure_analysis.md` — deep dive into
  why cross-encoder reranking degrades 10/30 queries (concentrated in
  ESC 2021, the largest and most repetitive document at 856 chunks).
- `data/evaluation/results/step12_report.md` — K=20 vs K=50 candidate
  pool decision: **recommendation is to keep K=20 as default**; K=50 helps
  the reranked pipeline modestly (+0.0075 nDCG@5, +0.0284 MRR) but adds 77%
  latency (254ms → 450ms mean) and gives zero benefit to scoped-only
  retrieval.

---

## Summary

| Checklist item | Status |
|---|---|
| Top-5 search on 15–20 questions | ✅ Done |
| Compare ≥2 chunk settings | ❌ Not done — single fixed chunk config only |
| Display chunks with metadata/scores | ✅ Done |
| Top-3 vs Top-5 vs Top-10 | ✅ Done |
| Label chunks relevant/not | ✅ Done (graded 0/1/2, superset of binary) |
| Hybrid or reranking (bonus) | ✅ Reranking done; no lexical/hybrid baseline |
| Precision@3 / Precision@5 | ✅ Done |
| Document best setup + why | ✅ Done |

**Stop-point check**: per the lab instructions ("do not finalize generation
prompts until retrieval can surface the correct evidence consistently") —
retrieval is not yet fully stable: Step 11 shows reranking degrades 10 of 30
multi-document queries, concentrated in the ESC 2021 document. The team's
own recommendation (Step 12) is K=20 as default, with reranking used
cautiously rather than as a blanket improvement. This suggests generation
prompt work can proceed for NICE/ESC-2023-scoped queries, but ESC 2021
retrieval quality should be revisited first if it's in scope for the final
deliverable.
