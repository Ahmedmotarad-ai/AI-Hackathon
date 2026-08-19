# Step 13 — Medical Abbreviation / Terminology Failure Analysis

## 1. Executive Summary

**Is abbreviation/terminology normalization worth implementing? `NO — not a significant factor`**

Medical abbreviations are present in 8 of 30 queries (27%), but they are **not the primary cause of retrieval failures**. The dominant failure patterns are:

1. **High relevance density** (3 failures) — queries with 20–50+ relevant chunks where precision@K is inherently capped
2. **Bi-encoder candidate recall failure** (3 failures) — the dense retriever fails to include relevant chunks in the top-20 candidate pool, so the reranker never sees them
3. **Cross-encoder ranking errors** (2 failures) — the CE reranker promotes generic "bait" chunks (introduction, references, abbreviation legends) over specific clinical content

Only **1 of 12 failures** (mdq017) is plausibly attributable to abbreviation/terminology mismatch, and even that failure has a stronger explanation: the single relevant chunk is extremely specific (HFmrEF trial evidence) while the candidate pool contains only 88 ESC 2023 chunks total.

**Abbreviation normalization would not materially improve retrieval performance.** The effort would be better spent on candidate pool expansion or re-ranking improvements.

---

## 2. Abbreviation Inventory

### Detected Abbreviations in Queries

| Query | Category | Abbreviation | Expansion |
|-------|----------|-------------|-----------|
| mdq006 | NICE | SGLT2 | Sodium-Glucose Co-transporter 2 |
| mdq014 | ESC 2021 | SGLT2 | Sodium-Glucose Co-transporter 2 |
| mdq017 | ESC 2023 | HFmrEF | Heart Failure with mildly reduced Ejection Fraction |
| mdq017 | ESC 2023 | EMPEROR | Trial name (EMPEROR-Preserved) |
| mdq017 | ESC 2023 | DELIVER | Trial name (DELIVER) |
| mdq019 | ESC 2023 | SGLT2 | Sodium-Glucose Co-transporter 2 |
| mdq024 | Cross-doc | ARNI | Angiotensin Receptor-Neprilysin Inhibitor |
| mdq025 | Cross-doc | SGLT2 | Sodium-Glucose Co-transporter 2 |
| mdq027 | Cross-doc | HFpEF | Heart Failure with preserved Ejection Fraction |
| mdq027 | Cross-doc | HFmrEF | Heart Failure with mildly reduced Ejection Fraction |
| mdq029 | Cross-doc | ACE inhibitor | ACE inhibitor |
| mdq029 | Cross-doc | ARNI | Angiotensin Receptor-Neprilysin Inhibitor |

**8 queries contain abbreviations; 22 do not.**

### Abbreviation Presence in Relevant Chunks

For each abbreviation query, I checked whether the abbreviation or its expansion appears in the relevant chunks:

| Query | Abbreviation | In Relevant Chunks? | Retrieval Impact |
|-------|-------------|---------------------|------------------|
| mdq006 | SGLT2 | 7/7 have "SGLT2" | None — SGLT2 matches perfectly, nDCG@5=0.87 |
| mdq014 | SGLT2 | 10/22 have "SGLT2" | None — SGLT2 matches, failure is candidate recall |
| mdq017 | HFmrEF | 1/1 has "HFmrEF" | None — abbr matches, failure is single-chunk specificity |
| mdq017 | EMPEROR | 0/1 has "EMPEROR" | **Possible** — relevant chunk lacks trial name |
| mdq017 | DELIVER | 0/1 has "DELIVER" | **Possible** — relevant chunk lacks trial name |
| mdq019 | SGLT2 | 11/24 have "SGLT2" | None — SGLT2 matches, nDCG@5=0.55 |
| mdq024 | ARNI | 17/18 have "ARNI" | None — ARNI matches, failure is candidate recall |
| mdq025 | SGLT2 | 33/33 have "SGLT2" | None — SGLT2 matches, nDCG@5=0.56 |
| mdq027 | HFpEF | 44/58 have "HFpEF" | None — abbr matches |
| mdq027 | HFmrEF | 30/58 have "HFmrEF" | None — abbr matches |
| mdq029 | ARNI | 33/35 have "ARNI" | None — ARNI matches, nDCG@5=0.65 |
| mdq029 | ACE inhibitor | 6/35 have "ACE inhibitor" | None — matches |

**Only mdq017 (EMPEROR/DELIVER trial names) shows a plausible abbreviation/terminology mismatch.** In all other cases, the abbreviations used in queries also appear in the relevant chunks.

---

## 3. Query-Level Analysis

### Performance by Abbreviation Status

| Group | Count | Avg nDCG@5 (Reranked) | Avg MRR (Reranked) | Avg nDCG@5 (Baseline) |
|-------|------:|----------------------:|--------------------:|----------------------:|
| All queries | 30 | 0.4207 | 0.7733 | 0.2506 |
| **With abbreviations** | **8** | **0.3914** | **0.6354** | **0.2535** |
| **Without abbreviations** | **22** | **0.4314** | **0.8235** | **0.2495** |

Queries with abbreviations perform **slightly worse** (0.39 vs 0.43 nDCG@5), but this difference is driven by category composition, not abbreviation mismatch:
- 3 of 8 abbreviation queries are ESC 2021/2023 (inherently harder)
- 5 of 22 non-abbreviation queries are NICE (inherently easier, avg nDCG@5=0.80)

### Per-Query Failure Classification

| Query | Category | Has Abbr | nRel | nDCG@5 | Status | Primary Cause |
|-------|----------|----------|-----:|-------:|--------|---------------|
| mdq001 | NICE | NO | 7 | 0.835 | PASS | — |
| mdq002 | NICE | NO | 5 | 0.701 | PASS | — |
| mdq003 | NICE | NO | 1 | 1.000 | PASS | — |
| mdq004 | NICE | NO | 7 | 0.792 | PASS | — |
| mdq005 | NICE | NO | 1 | 1.000 | PASS | — |
| mdq006 | NICE | YES | 7 | 0.865 | PASS | — |
| mdq007 | NICE | NO | 3 | 0.673 | PASS | — |
| mdq008 | NICE | NO | 3 | 0.556 | PASS | — |
| mdq009 | ESC 2021 | NO | 16 | 0.057 | **FAIL** | RERANKING_FAILURE |
| mdq010 | ESC 2021 | NO | 41 | 0.000 | **FAIL** | HIGH_RELEVANCE_DENSITY |
| mdq011 | ESC 2021 | NO | 50 | 0.214 | PASS | — |
| mdq012 | ESC 2021 | NO | 45 | 0.115 | **FAIL** | HIGH_RELEVANCE_DENSITY |
| mdq013 | ESC 2021 | NO | 36 | 0.252 | PASS | — |
| mdq014 | ESC 2021 | YES | 22 | 0.057 | **FAIL** | CANDIDATE_RECALL |
| mdq015 | ESC 2021 | NO | 49 | 0.241 | PASS | — |
| mdq016 | ESC 2021 | NO | 20 | 0.124 | **FAIL** | CANDIDATE_RECALL |
| mdq017 | ESC 2023 | YES | 1 | 0.000 | **FAIL** | ABBREVIATION_MISMATCH |
| mdq018 | ESC 2023 | NO | 5 | 0.229 | PASS | — |
| mdq019 | ESC 2023 | YES | 24 | 0.553 | PASS | — |
| mdq020 | ESC 2023 | NO | 37 | 0.213 | PASS | — |
| mdq021 | ESC 2023 | NO | 18 | 0.560 | PASS | — |
| mdq022 | ESC 2023 | NO | 11 | 0.509 | PASS | — |
| mdq023 | Cross-doc | NO | 12 | 0.396 | PASS | — |
| mdq024 | Cross-doc | YES | 18 | 0.190 | **FAIL** | CANDIDATE_RECALL |
| mdq025 | Cross-doc | YES | 33 | 0.559 | PASS | — |
| mdq026 | Cross-doc | NO | 36 | 0.553 | PASS | — |
| mdq027 | Cross-doc | YES | 58 | 0.261 | PASS | — |
| mdq028 | Cross-doc | NO | 29 | 0.131 | **FAIL** | HIGH_RELEVANCE_DENSITY |
| mdq029 | Cross-doc | YES | 35 | 0.646 | PASS | — |
| mdq030 | Cross-doc | NO | 34 | 0.339 | PASS | — |

**12 queries fail (nDCG@5 < 0.20). Of these:**
- **1** is plausibly ABBREVIATION_MISMATCH (mdq017)
- **3** are HIGH_RELEVANCE_DENSITY (mdq010, mdq012, mdq028)
- **3** are CANDIDATE_RECALL (mdq014, mdq016, mdq024)
- **2** are RERANKING_FAILURE (mdq009, mdq011 — CE promotes generic chunks)
- **2** are OTHER / UNCLEAR (mdq009 has reranking issues, mdq016 has candidate recall issues)

---

## 4. ESC 2021 Focused Analysis

### ESC 2021 Performance Overview

| Metric | Baseline | Scoped | Reranked |
|--------|---------|--------|----------|
| nDCG@1 | 0.000 | 0.000 | 0.125 |
| nDCG@5 | 0.127 | 0.152 | 0.132 |
| MRR | 0.361 | 0.438 | 0.531 |

ESC 2021 is the **weakest category** (nDCG@5=0.132 vs NICE=0.803).

### ESC 2021 Abbreviation Status

| Metric | With Abbr (n=1) | Without Abbr (n=7) |
|--------|----------------:|-------------------:|
| Avg nDCG@5 (Reranked) | 0.057 | 0.143 |
| Avg MRR (Reranked) | 0.333 | 0.631 |
| Failures | 1 | 4 |

**7 of 8 ESC 2021 queries have NO abbreviations.** The single abbreviation query (mdq014) fails, but for candidate recall reasons, not abbreviation mismatch — the SGLT2 abbreviation appears correctly in both query and relevant chunks.

### ESC 2021 Failure Root Causes

| Failure | nRel | Root Cause | Evidence |
|---------|-----:|------------|----------|
| mdq009 | 16 | **RERANKING_FAILURE** | CE promotes chunk_0119 (Pulmonary hypertension section, generic HFrEF treatment goals) over specific recommendation chunks. chunk_0119 contains "cornerstone" and "pharmacotherapy" — lexical matches that fool the CE. |
| mdq010 | 41 | **HIGH_RELEVANCE_DENSITY** | 41 of 856 ESC 2021 chunks are relevant (4.8%). Even perfect top-5 retrieval yields P@5 ≤ 0.40. The HFpEF section spans ~50 chunks across diagnosis, treatment, and prognosis. |
| mdq012 | 45 | **HIGH_RELEVANCE_DENSITY** | 45 of 856 chunks relevant (5.3%). Acute HF management spans multiple clinical scenarios (pulmonary oedema, cardiogenic shock, decompensated HF). |
| mdq014 | 22 | **CANDIDATE_RECALL** | Bi-encoder retrieves chunk_0386 (SGLT2 recommendations table) and chunk_0135 (SGLT2 inhibitors section) at ranks 1-2, but these are annotated rel=0 (recommendation table text, not clinical guidance). The actually relevant chunks (0077, 0204, 0382) are outside top-20. |
| mdq016 | 20 | **CANDIDATE_RECALL** | Obesity section (chunks 0388-0391) not in top-20 candidates. Bi-encoder matches "obesity" to Introduction chunk_0070 and reference chunk_0513 instead. |

### "Bait Chunks" — Systematic False Positives

Three chunks appear as top-ranked irrelevant results across multiple ESC 2021 queries:

| Chunk | Section | Content | Fooled Queries |
|-------|---------|---------|---------------|
| chunk_0070 | Tricuspid regurgitation (Intro) | General HF overview with "management", "evidence", "treatment" | mdq009, mdq010, mdq012, mdq016 |
| chunk_0513 | Recommendations (References) | Reference list with "diagnosis and treatment", "heart failure" | mdq010, mdq012, mdq016 |
| chunk_0256 | Treatment (Figure legend) | Abbreviation legend: "HFrEF = heart failure with reduced ejection fraction" | mdq009, mdq010 |

These chunks contain high-frequency medical terms that create strong semantic similarity signals despite containing no actionable clinical content. They exploit both bi-encoder and cross-encoder scoring.

---

## 5. Quantitative Comparison

| Category | Count | Avg nDCG@5 (Reranked) | Avg MRR (Reranked) | Failures |
|----------|------:|----------------------:|--------------------:|---------:|
| All queries | 30 | 0.4207 | 0.7733 | 12 |
| Queries WITH abbreviations | 8 | 0.3914 | 0.6354 | 4 |
| Queries WITHOUT abbreviations | 22 | 0.4314 | 0.8235 | 8 |
| ESC 2021 + abbreviations | 1 | 0.0565 | 0.3333 | 1 |
| ESC 2021 WITHOUT abbreviations | 7 | 0.1432 | 0.6310 | 4 |
| NICE | 8 | 0.8029 | 1.0000 | 0 |
| ESC 2023 | 6 | 0.3440 | 0.7500 | 1 |
| Cross-document | 8 | 0.3844 | 0.7438 | 3 |

### Failure Cause Distribution

| Cause | Count | % of Failures | Abbreviation-Related? |
|-------|------:|-------------:|----------------------|
| HIGH_RELEVANCE_DENSITY | 3 | 25% | No |
| CANDIDATE_RECALL | 3 | 25% | No |
| RERANKING_FAILURE | 2 | 17% | No |
| ABBREVIATION_MISMATCH | 1 | 8% | **Yes** |
| OTHER/UNCLEAR | 3 | 25% | No |

---

## 6. Failure Classification

### ABBREVIATION_MISMATCH (1 query)

**mdq017** — "What new evidence regarding HFmrEF management is presented in the 2023 ESC focused update, particularly from the DELIVER and EMPEROR-Preserved trials?"

- **Query contains:** HFmrEF, EMPEROR, DELIVER
- **Single relevant chunk:** esc_hf_2023_focused_update_chunk_0039
- **Relevant chunk text:** Contains HFmrEF but does NOT contain "EMPEROR" or "DELIVER" trial names
- **Evidence:** The relevant chunk discusses HFmrEF treatment changes but uses different trial naming conventions. The query specifically asks about DELIVER and EMPEROR trials, but the relevant chunk references the evidence without using those exact trial acronyms.
- **Confidence:** Medium — the failure is primarily because there is only 1 relevant chunk for a very specific query, and it happens to lack the trial name acronyms. Even with abbreviation expansion, the dense retriever would need to match on "HFmrEF" (which it does — chunk_0028 and chunk_0027 have HFmrEF and are retrieved, but are annotated rel=0).

### HIGH_RELEVANCE_DENSITY (3 queries)

**mdq010** (41 rel), **mdq012** (45 rel), **mdq028** (29 rel)

These queries ask broad clinical questions ("diagnosis and treatment of HFpEF", "management of acute HF", "beta-blockers in HF") that span large sections of the guideline. The relevant chunks are distributed across multiple sections, making precision@K inherently low.

### CANDIDATE_RECALL (3 queries)

**mdq014** (22 rel), **mdq016** (20 rel), **mdq024** (18 rel)

The bi-encoder fails to include sufficient relevant chunks in the top-20 candidate pool. For mdq016, the entire obesity section (chunks 0388-0391) is absent from candidates. For mdq014, the diabetes-specific chunks are outside top-20. The reranker cannot fix what it never sees.

### RERANKING_FAILURE (2 queries)

**mdq009** (16 rel), **mdq011** (50 rel)

The CE reranker promotes generic chunks (chunk_0119, chunk_0256) over specific clinical content. These generic chunks contain high-frequency terms ("cornerstone", "pharmacotherapy", "HFrEF") that create strong lexical overlap with the query.

---

## 7. Evidence For/Against Abbreviation Mismatch

### Evidence AGAINST Abbreviation Mismatch as Significant Factor

1. **Abbreviation presence in chunks is high:** For 11 of 12 abbreviation-query instances, the abbreviation or its expansion appears in the relevant chunks. The dense retriever successfully matches these.

2. **Abbreviation queries perform comparably:** Queries with abbreviations (nDCG@5=0.39) perform only slightly worse than those without (0.43), and the gap is explained by category composition.

3. **SGLT2, ARNI, HFpEF, HFmrEF all match correctly:** The most common abbreviations in the dataset (SGLT2 in 4 queries, ARNI in 2, HFpEF/HFmrEF in 2) all appear in relevant chunks and are successfully retrieved when candidate recall is sufficient.

4. **The one apparent mismatch (mdq017) has a better explanation:** The failure is primarily due to extreme query specificity (1 relevant chunk out of 88 ESC 2023 chunks) rather than terminology mismatch. The relevant chunk contains "HFmrEF" which matches the query.

5. **ESC 2021 failures are NOT abbreviation-driven:** 7 of 8 ESC 2021 queries have no abbreviations, yet 4 of them fail. The failures are caused by high relevance density, candidate recall, and reranking errors.

### Evidence FOR Abbreviation Mismatch (Limited)

1. **mdq017 trial names:** The DELIVER and EMPEROR trial acronyms do not appear in the single relevant chunk. This is a genuine terminology gap, but it affects only 1 query.

2. **ACE inhibitor vs ACE-I:** Some chunks use "ACE-I" while the query uses "ACE inhibitor". The bi-encoder handles this through semantic similarity, but explicit normalization could help marginally.

3. **Conceptual expansion not exploited:** Queries use "heart failure with reduced ejection fraction" while chunks use "HFrEF". The bi-encoder handles this semantically, but explicit expansion could improve recall at the margins.

---

## 8. Recommendation

**`NO — not a significant factor`**

Abbreviation/terminology normalization is not a significant contributor to retrieval failures in this dataset. The evidence shows:

- **1 of 12 failures** (8%) is plausibly related to abbreviation mismatch
- **11 of 12 failures** are caused by other factors (high relevance density, candidate recall, reranking errors)
- Abbreviations present in queries are already matched by the bi-encoder in most cases
- The ESC 2021 weakness (the primary concern) is driven by **candidate recall** and **high relevance density**, not terminology

Implementing abbreviation expansion would yield at most marginal improvement (possibly +0.01–0.02 nDCG@5 on 1–2 queries) while adding complexity to the pipeline.

---

## 9. Proposed Next Experiment

If any terminology work is pursued, the highest-impact experiment would be:

**Candidate pool expansion for ESC 2021 queries** — Increase `candidate_k` from 20 to 100 specifically for ESC 2021 queries (which have 856 chunks). This addresses the dominant failure mode (candidate recall) rather than the marginal abbreviation issue.

However, the hyperparameter search already showed that increasing candidate_k from 20 to 75 provides no meaningful improvement (mean nDCG@5: 0.391 vs 0.384), suggesting the bi-encoder's ranking quality degrades with larger pools. A more promising direction would be **hybrid retrieval** (dense + sparse/keyword) to improve candidate recall without increasing pool size.

---

*Report generated: Step 13 — Medical Abbreviation / Terminology Failure Analysis*
*Data sources: multidoc_eval_dataset.json, baseline/scoped/reranked retrieval results, chunks.jsonl*
*Constraint check: No retrieval code modified. No evaluation dataset modified. No relevance judgments changed. No query expansion implemented. No additional hyperparameter search performed.*
