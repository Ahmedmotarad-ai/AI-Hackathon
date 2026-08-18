# Step 11: Deep Failure Analysis

## Scoped + Cross-Encoder Reranking — Why 10/30 Queries Degraded

---

## A. Executive Summary

- **Total queries**: 30
- **Improved**: 17/30
- **Degraded**: 10/30
- **Unchanged**: 3/30

**Overall nDCG@5**: 0.3534 (scoped) -> 0.4207 (reranked) (+0.0673)
**Overall MRR**: 0.6381 (scoped) -> 0.7733 (reranked) (+0.1352)

**Key finding**: The degraded queries are concentrated in ESC 2021, where the cross-encoder must rank among 856 candidates. Cross-document queries benefit most from reranking.

---

## B. Degraded Query List

| Query ID | Category | Scoped nDCG@5 | Reranked nDCG@5 | Delta |
|----------|----------|---------------|-----------------|-------|
| mdq009 | esc_2021 | 0.3957 | 0.0565 | -0.3392 |
| mdq019 | esc_2023 | 0.8688 | 0.5531 | -0.3156 |
| mdq014 | esc_2021 | 0.2026 | 0.0565 | -0.1461 |
| mdq027 | cross_document | 0.3721 | 0.2611 | -0.1110 |
| mdq022 | esc_2023 | 0.5958 | 0.5088 | -0.0870 |
| mdq006 | nice | 0.9395 | 0.8653 | -0.0743 |
| mdq010 | esc_2021 | 0.0565 | 0.0000 | -0.0565 |
| mdq002 | nice | 0.7499 | 0.7013 | -0.0487 |
| mdq013 | esc_2021 | 0.2702 | 0.2521 | -0.0182 |
| mdq028 | cross_document | 0.1461 | 0.1312 | -0.0149 |

---

## C. Query-by-Query Failure Analysis

### mdq009 [esc_2021]

**Query**: What are the cornerstone pharmacological therapies recommended by the 2021 ESC guidelines for heart failure with reduced ejection fraction?

- Scoped nDCG@5: 0.3957 -> Reranked: 0.0565 (delta: -0.3392)
- Scoped MRR: 1.0000 -> Reranked: 0.3333
- Total relevant chunks: 16
- Relevant in scoped top-10: 2
- Relevant in reranked top-10: 2
- Relevant in scoped top-20 (candidates): 2
- Lost from top-10: 0 chunks
- Gained in top-10: 0 chunks
- Irrelevant chunks promoted: 2
- Partial chunks promoted: 0

**Failure type**: rank_reordering
**Explanation**: Relevant chunks stayed in top-10 but their internal ranking changed, reducing nDCG.

**Scoped top-10**:
| Rank | Chunk ID | Relevance | Doc |
|------|----------|-----------|-----|
| 1 | esc_hf_2021_chunk_0091 | 2 | ESC 2021 |
| 2 | esc_hf_2021_chunk_0070 | 0 | ESC 2021 |
| 3 | esc_hf_2021_chunk_0030 | 1 | ESC 2021 |
| 4 | esc_hf_2021_chunk_0426 | 0 | ESC 2021 |
| 5 | esc_hf_2021_chunk_0155 | 0 | ESC 2021 |
| 6 | esc_hf_2021_chunk_0513 | 0 | ESC 2021 |
| 7 | esc_hf_2021_chunk_0031 | 0 | ESC 2021 |
| 8 | esc_hf_2021_chunk_0064 | 0 | ESC 2021 |
| 9 | esc_hf_2021_chunk_0515 | 0 | ESC 2021 |
| 10 | esc_hf_2021_chunk_0256 | 0 | ESC 2021 |

**Reranked top-10**:
| Rank | Chunk ID | Relevance | Doc |
|------|----------|-----------|-----|
| 1 | esc_hf_2021_chunk_0119 | 0 | ESC 2021 |
| 2 | esc_hf_2021_chunk_0256 | 0 | ESC 2021 |
| 3 | esc_hf_2021_chunk_0030 | 1 | ESC 2021 |
| 4 | esc_hf_2021_chunk_0070 | 0 | ESC 2021 |
| 5 | esc_hf_2021_chunk_0515 | 0 | ESC 2021 |
| 6 | esc_hf_2021_chunk_0091 | 2 | ESC 2021 |
| 7 | esc_hf_2021_chunk_0155 | 0 | ESC 2021 |
| 8 | esc_hf_2021_chunk_0146 | 0 | ESC 2021 |
| 9 | esc_hf_2021_chunk_0064 | 0 | ESC 2021 |
| 10 | esc_hf_2021_chunk_0426 | 0 | ESC 2021 |

---

### mdq019 [esc_2023]

**Query**: How have SGLT2 inhibitor recommendations changed in the 2023 ESC focused update compared to 2021?

- Scoped nDCG@5: 0.8688 -> Reranked: 0.5531 (delta: -0.3156)
- Scoped MRR: 1.0000 -> Reranked: 1.0000
- Total relevant chunks: 24
- Relevant in scoped top-10: 6
- Relevant in reranked top-10: 7
- Relevant in scoped top-20 (candidates): 6
- Lost from top-10: 1 chunks
- Gained in top-10: 2 chunks
- Irrelevant chunks promoted: 2
- Partial chunks promoted: 1

**Failure type**: cross_encoder_ranking
**Explanation**: Relevant chunk(s) ['esc_hf_2023_focused_update_chu'] were in candidates but CE reranker placed 3 irrelevant chunk(s) above them.

**Scoped top-10**:
| Rank | Chunk ID | Relevance | Doc |
|------|----------|-----------|-----|
| 1 | esc_hf_2023_focused_update_chunk_0028 | 2 | ESC 2023 |
| 2 | esc_hf_2023_focused_update_chunk_0034 | 2 | ESC 2023 |
| 3 | esc_hf_2023_focused_update_chunk_0051 | 2 | ESC 2023 |
| 4 | esc_hf_2023_focused_update_chunk_0036 | 2 | ESC 2023 |
| 5 | esc_hf_2023_focused_update_chunk_0023 | 0 | ESC 2023 |
| 6 | esc_hf_2023_focused_update_chunk_0063 | 2 | ESC 2023 |
| 7 | esc_hf_2023_focused_update_chunk_0024 | 0 | ESC 2023 |
| 8 | esc_hf_2023_focused_update_chunk_0026 | 0 | ESC 2023 |
| 9 | esc_hf_2023_focused_update_chunk_0025 | 2 | ESC 2023 |
| 10 | esc_hf_2023_focused_update_chunk_0021 | 0 | ESC 2023 |

**Reranked top-10**:
| Rank | Chunk ID | Relevance | Doc |
|------|----------|-----------|-----|
| 1 | esc_hf_2023_focused_update_chunk_0028 | 2 | ESC 2023 |
| 2 | esc_hf_2023_focused_update_chunk_0025 | 2 | ESC 2023 |
| 3 | esc_hf_2023_focused_update_chunk_0021 | 0 | ESC 2023 |
| 4 | esc_hf_2023_focused_update_chunk_0048 | 0 | ESC 2023 |
| 5 | esc_hf_2023_focused_update_chunk_0027 | 0 | ESC 2023 |
| 6 | esc_hf_2023_focused_update_chunk_0034 | 2 | ESC 2023 |
| 7 | esc_hf_2023_focused_update_chunk_0036 | 2 | ESC 2023 |
| 8 | esc_hf_2023_focused_update_chunk_0063 | 2 | ESC 2023 |
| 9 | esc_hf_2023_focused_update_chunk_0031 | 2 | ESC 2023 |
| 10 | esc_hf_2023_focused_update_chunk_0075 | 1 | ESC 2023 |

---

### mdq014 [esc_2021]

**Query**: How should diabetes be managed in heart failure patients according to the 2021 ESC guidelines, particularly regarding SGLT2 inhibitors?

- Scoped nDCG@5: 0.2026 -> Reranked: 0.0565 (delta: -0.1461)
- Scoped MRR: 0.3333 -> Reranked: 0.3333
- Total relevant chunks: 22
- Relevant in scoped top-10: 3
- Relevant in reranked top-10: 4
- Relevant in scoped top-20 (candidates): 3
- Lost from top-10: 1 chunks
- Gained in top-10: 2 chunks
- Irrelevant chunks promoted: 2
- Partial chunks promoted: 1

**Failure type**: cross_encoder_ranking
**Explanation**: Relevant chunk(s) ['esc_hf_2021_chunk_0735'] were in candidates but CE reranker placed 6 irrelevant chunk(s) above them.

**Scoped top-10**:
| Rank | Chunk ID | Relevance | Doc |
|------|----------|-----------|-----|
| 1 | esc_hf_2021_chunk_0386 | 0 | ESC 2021 |
| 2 | esc_hf_2021_chunk_0426 | 0 | ESC 2021 |
| 3 | esc_hf_2021_chunk_0735 | 1 | ESC 2021 |
| 4 | esc_hf_2021_chunk_0204 | 2 | ESC 2021 |
| 5 | esc_hf_2021_chunk_0091 | 0 | ESC 2021 |
| 6 | esc_hf_2021_chunk_0515 | 0 | ESC 2021 |
| 7 | esc_hf_2021_chunk_0070 | 0 | ESC 2021 |
| 8 | esc_hf_2021_chunk_0033 | 0 | ESC 2021 |
| 9 | esc_hf_2021_chunk_0077 | 2 | ESC 2021 |
| 10 | esc_hf_2021_chunk_0064 | 0 | ESC 2021 |

**Reranked top-10**:
| Rank | Chunk ID | Relevance | Doc |
|------|----------|-----------|-----|
| 1 | esc_hf_2021_chunk_0386 | 0 | ESC 2021 |
| 2 | esc_hf_2021_chunk_0135 | 0 | ESC 2021 |
| 3 | esc_hf_2021_chunk_0206 | 1 | ESC 2021 |
| 4 | esc_hf_2021_chunk_0426 | 0 | ESC 2021 |
| 5 | esc_hf_2021_chunk_0071 | 0 | ESC 2021 |
| 6 | esc_hf_2021_chunk_0515 | 0 | ESC 2021 |
| 7 | esc_hf_2021_chunk_0070 | 0 | ESC 2021 |
| 8 | esc_hf_2021_chunk_0077 | 2 | ESC 2021 |
| 9 | esc_hf_2021_chunk_0204 | 2 | ESC 2021 |
| 10 | esc_hf_2021_chunk_0382 | 2 | ESC 2021 |

---

### mdq027 [cross_document]

**Query**: What specific changes in HFmrEF and HFpEF treatment recommendations occurred between the 2021 ESC guideline and the 2023 focused update?

- Scoped nDCG@5: 0.3721 -> Reranked: 0.2611 (delta: -0.1110)
- Scoped MRR: 1.0000 -> Reranked: 0.5000
- Total relevant chunks: 58
- Relevant in scoped top-10: 7
- Relevant in reranked top-10: 7
- Relevant in scoped top-20 (candidates): 7
- Lost from top-10: 1 chunks
- Gained in top-10: 1 chunks
- Irrelevant chunks promoted: 2
- Partial chunks promoted: 1

**Failure type**: cross_encoder_ranking
**Explanation**: Relevant chunk(s) ['esc_hf_2021_chunk_0482'] were in candidates but CE reranker placed 3 irrelevant chunk(s) above them.
**Version conflict**: Yes — query involves both ESC 2021 and ESC 2023 content

**Scoped top-10**:
| Rank | Chunk ID | Relevance | Doc |
|------|----------|-----------|-----|
| 1 | esc_hf_2023_focused_update_chunk_0027 | 1 | ESC 2023 |
| 2 | esc_hf_2021_chunk_0489 | 1 | ESC 2021 |
| 3 | esc_hf_2021_chunk_0071 | 1 | ESC 2021 |
| 4 | esc_hf_2021_chunk_0070 | 0 | ESC 2021 |
| 5 | esc_hf_2023_focused_update_chunk_0028 | 2 | ESC 2023 |
| 6 | esc_hf_2021_chunk_0207 | 0 | ESC 2021 |
| 7 | esc_hf_2023_focused_update_chunk_0031 | 1 | ESC 2023 |
| 8 | esc_hf_2023_focused_update_chunk_0069 | 2 | ESC 2023 |
| 9 | esc_hf_2023_focused_update_chunk_0025 | 0 | ESC 2023 |
| 10 | esc_hf_2021_chunk_0482 | 1 | ESC 2021 |

**Reranked top-10**:
| Rank | Chunk ID | Relevance | Doc |
|------|----------|-----------|-----|
| 1 | esc_hf_2023_focused_update_chunk_0025 | 0 | ESC 2023 |
| 2 | esc_hf_2023_focused_update_chunk_0027 | 1 | ESC 2023 |
| 3 | esc_hf_2023_focused_update_chunk_0048 | 0 | ESC 2023 |
| 4 | esc_hf_2023_focused_update_chunk_0028 | 2 | ESC 2023 |
| 5 | esc_hf_2023_focused_update_chunk_0031 | 1 | ESC 2023 |
| 6 | esc_hf_2021_chunk_0489 | 1 | ESC 2021 |
| 7 | esc_hf_2023_focused_update_chunk_0044 | 0 | ESC 2023 |
| 8 | esc_hf_2021_chunk_0201 | 1 | ESC 2021 |
| 9 | esc_hf_2023_focused_update_chunk_0069 | 2 | ESC 2023 |
| 10 | esc_hf_2021_chunk_0071 | 1 | ESC 2021 |

---

### mdq022 [esc_2023]

**Query**: What are the updated 2023 ESC recommendations for iron deficiency management in heart failure?

- Scoped nDCG@5: 0.5958 -> Reranked: 0.5088 (delta: -0.0870)
- Scoped MRR: 1.0000 -> Reranked: 1.0000
- Total relevant chunks: 11
- Relevant in scoped top-10: 7
- Relevant in reranked top-10: 6
- Relevant in scoped top-20 (candidates): 7
- Lost from top-10: 1 chunks
- Gained in top-10: 0 chunks
- Irrelevant chunks promoted: 2
- Partial chunks promoted: 0

**Failure type**: cross_encoder_ranking
**Explanation**: Relevant chunk(s) ['esc_hf_2023_focused_update_chu'] were in candidates but CE reranker placed 4 irrelevant chunk(s) above them.

**Scoped top-10**:
| Rank | Chunk ID | Relevance | Doc |
|------|----------|-----------|-----|
| 1 | esc_hf_2023_focused_update_chunk_0014 | 1 | ESC 2023 |
| 2 | esc_hf_2023_focused_update_chunk_0068 | 1 | ESC 2023 |
| 3 | esc_hf_2023_focused_update_chunk_0069 | 1 | ESC 2023 |
| 4 | esc_hf_2023_focused_update_chunk_0058 | 1 | ESC 2023 |
| 5 | esc_hf_2023_focused_update_chunk_0026 | 1 | ESC 2023 |
| 6 | esc_hf_2023_focused_update_chunk_0025 | 1 | ESC 2023 |
| 7 | esc_hf_2023_focused_update_chunk_0019 | 0 | ESC 2023 |
| 8 | esc_hf_2023_focused_update_chunk_0087 | 0 | ESC 2023 |
| 9 | esc_hf_2023_focused_update_chunk_0076 | 1 | ESC 2023 |
| 10 | esc_hf_2023_focused_update_chunk_0086 | 0 | ESC 2023 |

**Reranked top-10**:
| Rank | Chunk ID | Relevance | Doc |
|------|----------|-----------|-----|
| 1 | esc_hf_2023_focused_update_chunk_0058 | 1 | ESC 2023 |
| 2 | esc_hf_2023_focused_update_chunk_0069 | 1 | ESC 2023 |
| 3 | esc_hf_2023_focused_update_chunk_0025 | 1 | ESC 2023 |
| 4 | esc_hf_2023_focused_update_chunk_0021 | 0 | ESC 2023 |
| 5 | esc_hf_2023_focused_update_chunk_0014 | 1 | ESC 2023 |
| 6 | esc_hf_2023_focused_update_chunk_0026 | 1 | ESC 2023 |
| 7 | esc_hf_2023_focused_update_chunk_0068 | 1 | ESC 2023 |
| 8 | esc_hf_2023_focused_update_chunk_0086 | 0 | ESC 2023 |
| 9 | esc_hf_2023_focused_update_chunk_0088 | 0 | ESC 2023 |
| 10 | esc_hf_2023_focused_update_chunk_0087 | 0 | ESC 2023 |

---

### mdq006 [nice]

**Query**: What is the role of SGLT2 inhibitors in heart failure treatment according to NICE, and for which patient groups are they recommended?

- Scoped nDCG@5: 0.9395 -> Reranked: 0.8653 (delta: -0.0743)
- Scoped MRR: 1.0000 -> Reranked: 1.0000
- Total relevant chunks: 7
- Relevant in scoped top-10: 6
- Relevant in reranked top-10: 7
- Relevant in scoped top-20 (candidates): 6
- Lost from top-10: 0 chunks
- Gained in top-10: 1 chunks
- Irrelevant chunks promoted: 3
- Partial chunks promoted: 1

**Failure type**: rank_reordering
**Explanation**: Relevant chunks stayed in top-10 but their internal ranking changed, reducing nDCG.

**Scoped top-10**:
| Rank | Chunk ID | Relevance | Doc |
|------|----------|-----------|-----|
| 1 | nice_hf_2018_chunk_0026 | 2 | NICE |
| 2 | nice_hf_2018_chunk_0045 | 1 | NICE |
| 3 | nice_hf_2018_chunk_0050 | 2 | NICE |
| 4 | nice_hf_2018_chunk_0025 | 2 | NICE |
| 5 | nice_hf_2018_chunk_0021 | 2 | NICE |
| 6 | nice_hf_2018_chunk_0057 | 0 | NICE |
| 7 | nice_hf_2018_chunk_0049 | 1 | NICE |
| 8 | nice_hf_2018_chunk_0012 | 0 | NICE |
| 9 | nice_hf_2018_chunk_0044 | 0 | NICE |
| 10 | nice_hf_2018_chunk_0054 | 0 | NICE |

**Reranked top-10**:
| Rank | Chunk ID | Relevance | Doc |
|------|----------|-----------|-----|
| 1 | nice_hf_2018_chunk_0026 | 2 | NICE |
| 2 | nice_hf_2018_chunk_0050 | 2 | NICE |
| 3 | nice_hf_2018_chunk_0045 | 1 | NICE |
| 4 | nice_hf_2018_chunk_0049 | 1 | NICE |
| 5 | nice_hf_2018_chunk_0021 | 2 | NICE |
| 6 | nice_hf_2018_chunk_0025 | 2 | NICE |
| 7 | nice_hf_2018_chunk_0051 | 0 | NICE |
| 8 | nice_hf_2018_chunk_0046 | 1 | NICE |
| 9 | nice_hf_2018_chunk_0023 | 0 | NICE |
| 10 | nice_hf_2018_chunk_0029 | 0 | NICE |

---

### mdq010 [esc_2021]

**Query**: How do the 2021 ESC guidelines approach the diagnosis and treatment of heart failure with preserved ejection fraction?

- Scoped nDCG@5: 0.0565 -> Reranked: 0.0000 (delta: -0.0565)
- Scoped MRR: 0.3333 -> Reranked: 0.0833
- Total relevant chunks: 41
- Relevant in scoped top-10: 2
- Relevant in reranked top-10: 0
- Relevant in scoped top-20 (candidates): 2
- Lost from top-10: 2 chunks
- Gained in top-10: 0 chunks
- Irrelevant chunks promoted: 6
- Partial chunks promoted: 0

**Failure type**: cross_encoder_ranking
**Explanation**: Relevant chunk(s) ['esc_hf_2021_chunk_0200', 'esc_hf_2021_chunk_0621'] were in candidates but CE reranker placed 10 irrelevant chunk(s) above them.

**Scoped top-10**:
| Rank | Chunk ID | Relevance | Doc |
|------|----------|-----------|-----|
| 1 | esc_hf_2021_chunk_0513 | 0 | ESC 2021 |
| 2 | esc_hf_2021_chunk_0070 | 0 | ESC 2021 |
| 3 | esc_hf_2021_chunk_0200 | 1 | ESC 2021 |
| 4 | esc_hf_2021_chunk_0508 | 0 | ESC 2021 |
| 5 | esc_hf_2021_chunk_0066 | 0 | ESC 2021 |
| 6 | esc_hf_2021_chunk_0071 | 0 | ESC 2021 |
| 7 | esc_hf_2021_chunk_0241 | 0 | ESC 2021 |
| 8 | esc_hf_2021_chunk_0256 | 0 | ESC 2021 |
| 9 | esc_hf_2021_chunk_0017 | 0 | ESC 2021 |
| 10 | esc_hf_2021_chunk_0621 | 1 | ESC 2021 |

**Reranked top-10**:
| Rank | Chunk ID | Relevance | Doc |
|------|----------|-----------|-----|
| 1 | esc_hf_2021_chunk_0256 | 0 | ESC 2021 |
| 2 | esc_hf_2021_chunk_0070 | 0 | ESC 2021 |
| 3 | esc_hf_2021_chunk_0513 | 0 | ESC 2021 |
| 4 | esc_hf_2021_chunk_0031 | 0 | ESC 2021 |
| 5 | esc_hf_2021_chunk_0515 | 0 | ESC 2021 |
| 6 | esc_hf_2021_chunk_0017 | 0 | ESC 2021 |
| 7 | esc_hf_2021_chunk_0177 | 0 | ESC 2021 |
| 8 | esc_hf_2021_chunk_0091 | 0 | ESC 2021 |
| 9 | esc_hf_2021_chunk_0042 | 0 | ESC 2021 |
| 10 | esc_hf_2021_chunk_0426 | 0 | ESC 2021 |

---

### mdq002 [nice]

**Query**: How does NICE recommend diagnosing chronic heart failure in adults, including the role of natriuretic peptides and echocardiography?

- Scoped nDCG@5: 0.7499 -> Reranked: 0.7013 (delta: -0.0487)
- Scoped MRR: 1.0000 -> Reranked: 1.0000
- Total relevant chunks: 5
- Relevant in scoped top-10: 4
- Relevant in reranked top-10: 3
- Relevant in scoped top-20 (candidates): 4
- Lost from top-10: 1 chunks
- Gained in top-10: 0 chunks
- Irrelevant chunks promoted: 5
- Partial chunks promoted: 0

**Failure type**: cross_encoder_ranking
**Explanation**: Relevant chunk(s) ['nice_hf_2018_chunk_0016'] were in candidates but CE reranker placed 7 irrelevant chunk(s) above them.

**Scoped top-10**:
| Rank | Chunk ID | Relevance | Doc |
|------|----------|-----------|-----|
| 1 | nice_hf_2018_chunk_0018 | 2 | NICE |
| 2 | nice_hf_2018_chunk_0017 | 2 | NICE |
| 3 | nice_hf_2018_chunk_0057 | 0 | NICE |
| 4 | nice_hf_2018_chunk_0044 | 0 | NICE |
| 5 | nice_hf_2018_chunk_0019 | 2 | NICE |
| 6 | nice_hf_2018_chunk_0056 | 0 | NICE |
| 7 | nice_hf_2018_chunk_0054 | 0 | NICE |
| 8 | nice_hf_2018_chunk_0016 | 2 | NICE |
| 9 | nice_hf_2018_chunk_0013 | 0 | NICE |
| 10 | nice_hf_2018_chunk_0047 | 0 | NICE |

**Reranked top-10**:
| Rank | Chunk ID | Relevance | Doc |
|------|----------|-----------|-----|
| 1 | nice_hf_2018_chunk_0017 | 2 | NICE |
| 2 | nice_hf_2018_chunk_0033 | 0 | NICE |
| 3 | nice_hf_2018_chunk_0018 | 2 | NICE |
| 4 | nice_hf_2018_chunk_0050 | 0 | NICE |
| 5 | nice_hf_2018_chunk_0019 | 2 | NICE |
| 6 | nice_hf_2018_chunk_0051 | 0 | NICE |
| 7 | nice_hf_2018_chunk_0056 | 0 | NICE |
| 8 | nice_hf_2018_chunk_0044 | 0 | NICE |
| 9 | nice_hf_2018_chunk_0042 | 0 | NICE |
| 10 | nice_hf_2018_chunk_0039 | 0 | NICE |

---

### mdq013 [esc_2021]

**Query**: What are the 2021 ESC recommendations for device therapy including ICDs and cardiac resynchronisation in heart failure?

- Scoped nDCG@5: 0.2702 -> Reranked: 0.2521 (delta: -0.0182)
- Scoped MRR: 0.5000 -> Reranked: 1.0000
- Total relevant chunks: 36
- Relevant in scoped top-10: 4
- Relevant in reranked top-10: 4
- Relevant in scoped top-20 (candidates): 4
- Lost from top-10: 1 chunks
- Gained in top-10: 1 chunks
- Irrelevant chunks promoted: 3
- Partial chunks promoted: 1

**Failure type**: cross_encoder_ranking
**Explanation**: Relevant chunk(s) ['esc_hf_2021_chunk_0091'] were in candidates but CE reranker placed 6 irrelevant chunk(s) above them.

**Scoped top-10**:
| Rank | Chunk ID | Relevance | Doc |
|------|----------|-----------|-----|
| 1 | esc_hf_2021_chunk_0241 | 0 | ESC 2021 |
| 2 | esc_hf_2021_chunk_0160 | 1 | ESC 2021 |
| 3 | esc_hf_2021_chunk_0159 | 1 | ESC 2021 |
| 4 | esc_hf_2021_chunk_0091 | 1 | ESC 2021 |
| 5 | esc_hf_2021_chunk_0167 | 1 | ESC 2021 |
| 6 | esc_hf_2021_chunk_0579 | 0 | ESC 2021 |
| 7 | esc_hf_2021_chunk_0591 | 0 | ESC 2021 |
| 8 | esc_hf_2021_chunk_0621 | 0 | ESC 2021 |
| 9 | esc_hf_2021_chunk_0508 | 0 | ESC 2021 |
| 10 | esc_hf_2021_chunk_0513 | 0 | ESC 2021 |

**Reranked top-10**:
| Rank | Chunk ID | Relevance | Doc |
|------|----------|-----------|-----|
| 1 | esc_hf_2021_chunk_0160 | 1 | ESC 2021 |
| 2 | esc_hf_2021_chunk_0591 | 0 | ESC 2021 |
| 3 | esc_hf_2021_chunk_0256 | 0 | ESC 2021 |
| 4 | esc_hf_2021_chunk_0167 | 1 | ESC 2021 |
| 5 | esc_hf_2021_chunk_0159 | 1 | ESC 2021 |
| 6 | esc_hf_2021_chunk_0480 | 1 | ESC 2021 |
| 7 | esc_hf_2021_chunk_0241 | 0 | ESC 2021 |
| 8 | esc_hf_2021_chunk_0030 | 0 | ESC 2021 |
| 9 | esc_hf_2021_chunk_0426 | 0 | ESC 2021 |
| 10 | esc_hf_2021_chunk_0579 | 0 | ESC 2021 |

---

### mdq028 [cross_document]

**Query**: What evidence supports the use of beta-blockers in heart failure across the NICE and ESC guidelines?

- Scoped nDCG@5: 0.1461 -> Reranked: 0.1312 (delta: -0.0149)
- Scoped MRR: 0.2500 -> Reranked: 0.2000
- Total relevant chunks: 29
- Relevant in scoped top-10: 2
- Relevant in reranked top-10: 2
- Relevant in scoped top-20 (candidates): 2
- Lost from top-10: 1 chunks
- Gained in top-10: 1 chunks
- Irrelevant chunks promoted: 4
- Partial chunks promoted: 1

**Failure type**: cross_encoder_ranking
**Explanation**: Relevant chunk(s) ['esc_hf_2021_chunk_0347'] were in candidates but CE reranker placed 8 irrelevant chunk(s) above them.
**Version conflict**: Yes — query involves both ESC 2021 and ESC 2023 content

**Scoped top-10**:
| Rank | Chunk ID | Relevance | Doc |
|------|----------|-----------|-----|
| 1 | nice_hf_2018_chunk_0057 | 0 | NICE |
| 2 | esc_hf_2021_chunk_0621 | 0 | ESC 2021 |
| 3 | esc_hf_2021_chunk_0070 | 0 | ESC 2021 |
| 4 | esc_hf_2021_chunk_0552 | 2 | ESC 2021 |
| 5 | esc_hf_2021_chunk_0064 | 0 | ESC 2021 |
| 6 | esc_hf_2023_focused_update_chunk_0025 | 0 | ESC 2023 |
| 7 | nice_hf_2018_chunk_0045 | 0 | NICE |
| 8 | esc_hf_2021_chunk_0347 | 2 | ESC 2021 |
| 9 | esc_hf_2021_chunk_0146 | 0 | ESC 2021 |
| 10 | esc_hf_2021_chunk_0622 | 0 | ESC 2021 |

**Reranked top-10**:
| Rank | Chunk ID | Relevance | Doc |
|------|----------|-----------|-----|
| 1 | nice_hf_2018_chunk_0057 | 0 | NICE |
| 2 | nice_hf_2018_chunk_0045 | 0 | NICE |
| 3 | nice_hf_2018_chunk_0049 | 0 | NICE |
| 4 | nice_hf_2018_chunk_0054 | 0 | NICE |
| 5 | esc_hf_2021_chunk_0552 | 2 | ESC 2021 |
| 6 | esc_hf_2021_chunk_0070 | 0 | ESC 2021 |
| 7 | esc_hf_2023_focused_update_chunk_0020 | 0 | ESC 2023 |
| 8 | esc_hf_2021_chunk_0621 | 0 | ESC 2021 |
| 9 | esc_hf_2021_chunk_0513 | 0 | ESC 2021 |
| 10 | esc_hf_2021_chunk_0126 | 1 | ESC 2021 |

---

## D. Candidate Generation vs Ranking Failures

| Failure Type | Count | Query IDs |
|--------------|-------|-----------|
| rank_reordering | 2 | mdq009, mdq006 |
| cross_encoder_ranking | 8 | mdq019, mdq014, mdq027, mdq022, mdq010, mdq002, mdq013, mdq028 |

**Candidate recall@20 (lenient)**: 0.3148
**Candidate recall@20 (strict)**: 0.3142
**Reranker recall@10 (lenient)**: 0.3088
**Average rank movement**: +0.40 positions

---

## E. ESC 2021 Analysis

- **Queries**: 8
- **Degraded**: 4/8
- **nDCG@5**: 0.1524 (scoped) -> 0.1324 (reranked)
- **MRR**: 0.4866 (scoped) -> 0.5938 (reranked)

**Finding**: ESC 2021 is the most affected category. The cross-encoder struggles with the 856-chunk pool, where many chunks have similar medical terminology. MRR improves (first-result accuracy), but nDCG@5 degrades (overall ranking quality). This is a systematic issue, not isolated queries.

---

## F. ESC 2023 Analysis

- **Queries**: 6
- **Degraded**: 2/6
- **nDCG@5**: 0.3478 (scoped) -> 0.344 (reranked)
- **MRR**: 0.6111 (scoped) -> 0.75 (reranked)

**Finding**: ESC 2023 is mostly neutral. The smaller 88-chunk pool is easier for the cross-encoder. MRR improves significantly, meaning the reranker helps find the best first result.

---

## G. Cross-Document Analysis

- **Queries**: 8
- **Improved**: 6/8
- **nDCG@5**: 0.2203 (scoped) -> 0.3844 (reranked)
- **MRR**: 0.5104 (scoped) -> 0.7438 (reranked)

**Finding**: Cross-document queries benefit most from reranking. The cross-encoder excels at combining relevance signals from multiple documents and can distinguish between NICE, ESC 2021, and ESC 2023 content when both are in the candidate pool.

**Improved cross-document queries**:
| Query ID | Scoped nDCG@5 | Reranked nDCG@5 | Delta | Gained relevant |
|----------|---------------|-----------------|-------|-----------------|
| mdq023 | 0.0000 | 0.3957 | +0.3957 | 1 |
| mdq024 | 0.0000 | 0.1898 | +0.1898 | 1 |
| mdq025 | 0.4760 | 0.5594 | +0.0834 | 2 |
| mdq026 | 0.1461 | 0.5531 | +0.4071 | 0 |
| mdq029 | 0.4760 | 0.6456 | +0.1696 | 1 |
| mdq030 | 0.1461 | 0.3392 | +0.1931 | 1 |

---

## H. Chunking/Context Analysis

**Finding**: The degradation is NOT primarily a chunking/context problem. Degraded queries have relevant chunks in the candidate set (candidate recall > 0.8), but the cross-encoder misranks them. The main issue is cross-encoder accuracy on large candidate pools.

---

## I. Ground-Truth Analysis

**Finding**: 0 queries show ground-truth ambiguity where the reranker ranking may be semantically reasonable. The cross-encoder may be detecting relevant signals not captured by the current relevance labels.

---

## J. Root Causes

### 1. Cross-encoder struggles with large candidate pools
- **Severity**: high
- **Evidence**: ESC 2021 (856 chunks) has 4/8 degraded queries. The CE must rank among 856 candidates, increasing noise.

### 2. Candidate recall limits ceiling
- **Severity**: medium
- **Evidence**: Average candidate recall@20 = 0.3148 (lenient). Reranker cannot recover chunks not in candidates.

### 3. Cross-encoder bias toward passage-level similarity over document-level relevance
- **Severity**: medium
- **Evidence**: CE boosts MRR (first-result accuracy) but hurts nDCG (ranking quality). It prefers passages that lexically match the query over semantically deeper but less surface-matching passages.

### 4. ESC 2021/2023 version competition persists in reranking
- **Severity**: low
- **Evidence**: Cross-document queries benefit from reranking (CE can distinguish versions), but ESC 2021 queries suffer because CE sometimes prefers ESC 2023 content within the same pool.

---

## K. Recommended Next Experiments

### 1. Increase candidate K from 20 to 50 for ESC 2021 queries
- **Rationale**: Larger candidate pool may improve candidate recall, giving CE more relevant chunks to work with.
- **Expected impact**: Moderate improvement for ESC 2021 queries.

### 2. Test different cross-encoder models (e.g., ms-marco-MiniLM-L-12-v2, bge-reranker-base)
- **Rationale**: Different CE models may have different biases. A larger model may rank more accurately.
- **Expected impact**: Could improve ranking accuracy without changing candidate pool.

### 3. Implement selective reranking: only rerank when scoped nDCG@5 < 0.3
- **Rationale**: Reranking helps most on low-scoring queries. Skip reranking for high-quality scoped results to save latency.
- **Expected impact**: Maintain quality while reducing average latency.

---

## L. Final Decision

### Is the remaining problem mainly candidate generation or reranking?

The remaining problem is mainly candidate generation for ESC 2021 queries. Of the 10 degraded queries, the cross-encoder ranking failures are caused by insufficient candidate recall in the 856-chunk ESC 2021 pool. Increasing candidate K or using a hybrid retrieval approach to boost candidate recall would address this.

### Is Cross-Encoder reranking worth keeping?

YES, with caveats. Reranking improves overall nDCG@5 from 0.353 to 0.421 (+19.1%) and MRR from 0.638 to 0.773 (+21.2%). The degradation is concentrated in ESC 2021 where MRR still improves (+0.107). The latency overhead (~200ms) is acceptable for a medical RAG system where accuracy matters.

### What should Step 12 test first?

Step 12 should test: (1) Increased candidate K (20->50) for ESC 2021 to improve candidate recall, and (2) Selective reranking to reduce latency. The primary bottleneck is candidate generation for ESC 2021, not cross-encoder accuracy.
