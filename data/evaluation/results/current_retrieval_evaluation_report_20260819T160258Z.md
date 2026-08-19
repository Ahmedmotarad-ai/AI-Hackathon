# Current Retrieval Evaluation Report

**Timestamp:** 20260819T160258Z
**Queries:** 30
**Chunks:** 1001
**Dense model:** BAAI/bge-small-en-v1.5
**Reranker:** cross-encoder/ms-marco-MiniLM-L-6-v2
**Candidate K:** 20

## Overall Metrics

| Metric | @3 | @5 | @10 |
|--------|-----|-----|------|
| P | 0.1444 | 0.1333 | 0.0967 |
| R | 0.1500 | 0.2389 | 0.3333 |
| F1 | 0.1433 | 0.1641 | 0.1462 |
| Hit | 0.4000 | 0.5000 | 0.5667 |
| nDCG | 0.1603 | 0.2013 | 0.2312 |
| MRR | 0.3225 | | |

## Per-Category Metrics

| Category | N | P@3 | P@5 | P@10 | R@3 | R@5 | R@10 | F1@3 | Hit@3 | nDCG@5 |
|----------|---|------|------|-------|------|------|-------|------|-------|--------|
| direct | 6 | 0.2222 | 0.2000 | 0.1833 | 0.1667 | 0.2500 | 0.4722 | 0.1878 | 0.6667 | 0.2613 |
| ambiguous | 5 | 0.2000 | 0.1600 | 0.0800 | 0.1833 | 0.2500 | 0.2500 | 0.1905 | 0.6000 | 0.1835 |
| out_of_scope | 5 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| insufficient_evidence | 5 | 0.1333 | 0.1200 | 0.0600 | 0.2000 | 0.4000 | 0.4000 | 0.1600 | 0.2000 | 0.2861 |
| multi_part | 4 | 0.0833 | 0.2000 | 0.1250 | 0.0625 | 0.2708 | 0.3542 | 0.0714 | 0.2500 | 0.1329 |
| high_risk | 5 | 0.2000 | 0.1200 | 0.1200 | 0.2667 | 0.2667 | 0.5000 | 0.2267 | 0.6000 | 0.3185 |

## False Evidence Retrieval

| Category | N | False Top-3 | False Top-5 | Rate Top-3 | Rate Top-5 |
|----------|---|-------------|-------------|------------|------------|
| out_of_scope | 5 | 0 | 0 | 0.0% | 0.0% |
| insufficient_evidence | 5 | 1 | 2 | 20.0% | 40.0% |

## Failures (8)

### eval005 — no_relevant_in_top3
- **Query:** What is the recommended approach for rate control in heart failure patients with atrial fibrillation?
- **Category:** direct
- **Detail:** 0 of 4 ground truth chunks in top-3
- **Top-3:** `['esc_hf_2021_chunk_0059', 'esc_hf_2021_chunk_0698', 'esc_hf_2021_chunk_0695']`

### eval006 — no_relevant_in_top3
- **Query:** How is obesity managed in heart failure patients according to the ESC guidelines?
- **Category:** direct
- **Detail:** 0 of 3 ground truth chunks in top-3
- **Top-3:** `['esc_hf_2021_chunk_0070', 'esc_hf_2021_chunk_0610', 'esc_hf_2021_chunk_0745']`

### eval009 — no_relevant_in_top3
- **Query:** Is this patient a candidate for device therapy?
- **Category:** ambiguous
- **Detail:** 0 of 2 ground truth chunks in top-3
- **Top-3:** `['esc_hf_2021_chunk_0492', 'esc_hf_2021_chunk_0491', 'esc_hf_2021_chunk_0253']`

### eval010 — no_relevant_in_top3
- **Query:** What are the sleep problems in heart failure?
- **Category:** ambiguous
- **Detail:** 0 of 2 ground truth chunks in top-3
- **Top-3:** `['esc_hf_2021_chunk_0786', 'esc_hf_2021_chunk_0062', 'nice_hf_2018_chunk_0054']`

### eval018 — false_evidence_retrieved
- **Query:** What are the long-term outcomes of heart transplantation in patients over 70 years old?
- **Category:** insufficient_evidence
- **Detail:** Retrieved 2 apparently relevant chunks in top-3 for insufficient_evidence query
- **Top-3:** `['esc_hf_2021_chunk_0251', 'esc_hf_2021_chunk_0254', 'esc_hf_2021_chunk_0240']`

### eval023 — single_part_coverage
- **Query:** How should beta-blockers be initiated and what are the monitoring requirements?
- **Category:** multi_part
- **Detail:** Top-5 covers 1 sections but query has 4 ground truth chunks across multiple topics
- **Top-3:** `['esc_hf_2021_chunk_0126', 'nice_hf_2018_chunk_0053', 'esc_hf_2021_chunk_0347']`

### eval026 — no_relevant_in_top3
- **Query:** What is the starting dose of carvedilol for a patient with HFrEF?
- **Category:** high_risk
- **Detail:** 0 of 2 ground truth chunks in top-3
- **Top-3:** `['esc_hf_2021_chunk_0352', 'esc_hf_2021_chunk_0502', 'esc_hf_2021_chunk_0120']`

### eval030 — no_relevant_in_top3
- **Query:** How do I manage acute pulmonary oedema in the emergency department?
- **Category:** high_risk
- **Detail:** 0 of 2 ground truth chunks in top-3
- **Top-3:** `['esc_hf_2021_chunk_0276', 'esc_hf_2021_chunk_0289', 'esc_hf_2021_chunk_0020']`


## Latency

- Mean: 241.7 ms
- Median: 214.5 ms
- Min: 199.7 ms
- Max: 949.5 ms

## Limitations

- ORACLE document scoping is NOT used (evaluation uses real cross-document retrieval).
- Ground truth is based on manual chunk inspection; some relevant chunks may be missed.
- Out-of-scope and insufficient-evidence categories have empty or near-empty ground truth by design.
- Category-level metrics may have small N for some categories.
- High-risk category tests safety behavior, not retrieval accuracy per se.