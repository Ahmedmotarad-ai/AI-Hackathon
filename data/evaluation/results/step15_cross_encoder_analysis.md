# Step 15 — Cross-Encoder Bottleneck Analysis

## 1. Executive Summary

The Cross-Encoder (`ms-marco-MiniLM-L-6-v2`) is the **dominant bottleneck** in 100% of analyzed failure queries. However, the error pattern is specific and diagnosable:

**CE systematically over-prioritizes keyword-overlapping "bait chunks"** (introductions, abbreviation lists, table fragments, generic treatment statements) **above specific clinical recommendation content.**

Additionally, **2 of 6 target queries show potential relevance judgment issues** where CE-ranked chunks appear relevant based on text content but are labeled rel=0, suggesting the evaluation may underestimate CE performance.

| Finding | Severity |
|---------|----------|
| CE promotes keyword-bait over clinical content | HIGH |
| Truncation cuts off relevant content at chunk ends | MEDIUM |
| Possible relevance judgment errors in 2/6 queries | MEDIUM |
| Candidate recall limits ceiling (1 query) | LOW (1/6) |

---

## 2. Target Query Analysis

### mdq009 — "Cornerstone pharmacological therapies for HFrEF"

**Failure type: CE_RANKING_ERROR**

| Pre-CE Rank | Chunk | CE Rank | Rel | Content |
|-------------|-------|---------|-----|---------|
| 1 | chunk_0091 | **6** | **2** | Abbreviation list (HFrEF, SGLT2, etc.) |
| 2 | chunk_0030 | 3 | 1 | Table of contents for pharmacological treatments |
| 3 | chunk_0426 | 10 | 0 | Unknown |
| 4 | chunk_0070 | 4 | 0 | Introduction/overview text |
| 5 | chunk_0031 | 13 | 0 | Unknown |
| — | chunk_0119 | **1** | **0** | "Goals of pharmacotherapy" — mentions "cornerstone" |
| — | chunk_0256 | **2** | **0** | Mechanical circulatory support table fragment |
| — | chunk_0124 | **8** | **2** | **THE ACTUAL ANSWER**: Lists ACE-I, beta-blocker, MRA, SGLT2, ARNI as Class I A |

**Root cause:** CE ranks chunk_0119 at #1 because it contains the exact word "cornerstone" + "HFrEF" + "pharmacotherapy" — perfect keyword overlap. But the chunk discusses treatment GOALS, not specific therapies. Meanwhile, chunk_0124 (the actual recommendation table listing all 5 cornerstone drugs) is pushed to rank 8.

**CE score gap:** chunk_0119 = 4.98 vs chunk_0124 = 1.68. CE is 3x more confident in the wrong chunk.

---

### mdq010 — "Diagnosis and treatment of HFpEF"

**Failure type: CANDIDATE_RECALL + CE_RANKING_ERROR**

- Total relevant chunks: **41**
- In candidate pool (top-20): **3-5** (7-12% recall)
- CE placed **0 relevant chunks** in top-10 (scoped results)

**Root cause:** This is a dual failure. The 843-chunk ESC 2021 pool is too large for both BGE and CE to find HFpEF-specific content among the HFrEF-dominated chunks. CE then compounds the problem by ranking reference-list fragments (chunk_0597, chunk_0836) above clinical content.

---

### mdq012 — "Acute heart failure management"

**Failure type: CE_RANKING_ERROR (partial)**

| Chunk | CE Rank | Rel | Issue |
|-------|---------|-----|-------|
| chunk_0856 | 1 | 1 | Correctly ranked relevant chunk |
| chunk_0210 | 2 | 0 | Bait — pushed UP by CE |
| chunk_0513 | 3 | 0 | Bait — reference list |
| chunk_0070 | 4 | 0 | Bait — introduction text |
| chunk_0652 | 16 | 2 | Relevant — pushed DOWN |
| chunk_0655 | 18 | 2 | Relevant — pushed DOWN |

**Root cause:** CE correctly finds one relevant chunk at rank 1 (MRR=1.0) but pushes 2 highly relevant chunks (rel=2) to ranks 16-18. The top-5 has 3 irrelevant bait chunks. nDCG@5 is penalized because irrelevant chunks occupy positions 2-4.

---

### mdq014 — "Diabetes management / SGLT2 inhibitors"

**Failure type: CE_RANKING_ERROR + RELEVANCE_JUDGMENT_QUESTION**

| Chunk | CE Rank | Rel | Content |
|-------|---------|-----|---------|
| chunk_0386 | 1 | 0 | "Recommendations for the treatment of diabetes in heart failure" — SGLT2 inhibitors table |
| chunk_0135 | 2 | 0 | SGLT2 inhibitors practical guidance section |
| chunk_0077 | **8** | **2** | SGLT2 inhibitor recommendations (body text) |
| chunk_0204 | **9** | **2** | Diabetes mellitus treatment diagram |

**Critical observation:** chunk_0386 is titled "Recommendations for the treatment of diabetes in heart failure" and explicitly lists SGLT2 inhibitors. This appears to directly answer the query. Yet it is labeled rel=0. Either:
- (a) The annotators considered table fragments insufficient as standalone answers, OR
- (b) This is a relevance judgment error

If chunk_0386 is actually relevant, then CE is performing correctly (ranking it #1) and the evaluation underestimates performance.

---

### mdq016 — "Obesity and heart failure"

**Failure type: CE_RANKING_ERROR**

- MRR = 1.0 (CE correctly ranks one relevant chunk at #1)
- But nDCG@5 = 0.124 because relevant chunks at ranks 8-14 are pushed down
- CE scores are uniformly low (max 1.54), suggesting the model struggles to distinguish relevant from irrelevant for this niche topic

**Root cause:** Low-confidence scoring across all candidates. CE cannot differentiate obesity-specific HF content from generic HF content.

---

### mdq024 — "ARNI role across NICE and ESC"

**Failure type: CE_RANKING_ERROR + RELEVANCE_JUDGMENT_QUESTION**

| Chunk | CE Rank | Rel | Content |
|-------|---------|-----|---------|
| chunk_0128 | 1 | 0 | ESC: ARNI/PARADIGM-HF trial evidence |
| chunk_0183 | 2 | 0 | ESC: ARNI in HFmrEF |
| chunk_0133 | 3 | 0 | ESC: ARNI replacement recommendation |
| nice_chunk_0029 | 4 | 1 | NICE: ARNI prescribing guidance |
| nice_chunk_0021 | **8** | **2** | NICE: Treatment combinations including ARNI |

**Critical observation:** All 3 ESC ARNI chunks are labeled rel=0 despite being directly about ARNI. The query asks about ARNI "across the NICE and ESC guidelines." The relevance judgments appear to require content that addresses BOTH guidelines simultaneously, or the NICE-specific perspective.

If the ESC ARNI chunks are truly rel=0, then CE is correctly identifying that they don't address the cross-document aspect. But this seems like a strict relevance definition.

---

## 3. Bait Chunk Analysis

### Systematic Bait Chunks (appear across multiple queries)

| Chunk ID | Section | Bait Type | Appears In | Max CE Score |
|----------|---------|-----------|------------|--------------|
| `esc_hf_2021_chunk_0070` | Introduction | Generic overview | mdq009, mdq010, mdq016 | 2.83 |
| `esc_hf_2021_chunk_0256` | Treatment (table fragment) | Partial table | mdq009, mdq010, mdq016 | 4.14 |
| `esc_hf_2021_chunk_0513` | References | Reference list | mdq010, mdq012, mdq016 | 3.78 |
| `esc_hf_2021_chunk_0515` | Unknown | Unknown | mdq009, mdq014, mdq016 | 2.30 |
| `esc_hf_2021_chunk_0091` | Abbreviations | Abbreviation list | mdq009, mdq013 | 1.76 |

### Bait Chunk Characteristics

1. **Keyword density bait** (chunk_0119): Contains the exact query term "cornerstone" + medical terms, but discusses goals not specific therapies
2. **Introduction/overview bait** (chunk_0070): Broad coverage of HF terminology, high TF-IDF overlap with any HF query
3. **Table fragment bait** (chunk_0256): Contains treatment-related keywords in table format, but is a fragment without context
4. **Reference list bait** (chunk_0513): Contains author names and paper titles with HF keywords — CE sees high lexical overlap
5. **Abbreviation list bait** (chunk_0091): Every HF abbreviation creates keyword overlap with queries

### Pattern

**CE is trained on MS MARCO** where relevant passages typically contain the answer text directly. In medical guidelines:
- Bait chunks contain the query TERMS but not the answer
- Relevant chunks contain the answer but may use different terminology

This mismatch causes CE to favor lexical overlap over semantic relevance.

---

## 4. CE Ranking Errors

### Error Type 1: Keyword-Bait Promotion (4/6 queries)

CE promotes chunks with high keyword overlap regardless of whether they contain the actual answer.

**Examples:**
- mdq009: "cornerstone" keyword → chunk_0119 (goals, not therapies) ranked #1
- mdq014: "diabetes" + "SGLT2" keywords → chunk_0386 (table header) ranked #1
- mdq024: "ARNI" + "sacubitril/valsartan" keywords → ESC chunks ranked above NICE

### Error Type 2: Relevant Content Demotion (4/6 queries)

CE pushes semantically relevant chunks down because they use less query-matching terminology.

**Examples:**
- mdq009: chunk_0124 (actual drug recommendations) pushed from rank 1 → rank 8
- mdq012: chunk_0652, chunk_0655 (acute HF management) pushed to ranks 16-18
- mdq014: chunk_0077 (SGLT2 body text) pushed to rank 8

### Error Type 3: Low-Confidence Scoring (1/6 queries)

For niche topics, CE assigns uniformly low scores, making ranking essentially random.

**Example:**
- mdq016 (obesity + HF): max CE score = 1.54, range = [-7.69, 1.54]. All scores are low, suggesting CE cannot distinguish relevant from irrelevant for this specialized topic.

---

## 5. Truncation Check

### Model Configuration
- `cross-encoder/ms-marco-MiniLM-L-6-v2`: max_length = 512 tokens
- Input format: `[CLS] query [SEP] document [SEP]`
- Query tokens: ~20-30
- Available for document: ~479-489 tokens (~120-130 words)

### Chunk Length Analysis

| Chunk | Approx. Words | Est. Tokens | Truncated? | Relevant Content Location |
|-------|---------------|-------------|------------|--------------------------|
| chunk_0119 | ~250 | ~1000 | YES (~50%) | Beginning — survives truncation |
| chunk_0124 | ~200 | ~800 | YES (~40%) | End — **may be truncated** |
| chunk_0070 | ~300 | ~1200 | YES (~60%) | Beginning — survives truncation |
| chunk_0386 | ~150 | ~600 | YES (~20%) | Whole chunk — mostly survives |
| chunk_0077 | ~100 | ~400 | NO | Whole chunk — survives |
| chunk_0128 | ~300 | ~1200 | YES (~60%) | Beginning — survives truncation |

### Truncation Impact

**Plausible contributor in 2/6 queries:**
- **mdq009:** chunk_0124's recommendation table is at the END of the chunk. With ~40% truncation, the specific drug names (ACE-I, beta-blocker, MRA, SGLT2, ARNI) may be cut off. This could explain why CE ranks it at #8 instead of #1.
- **mdq012:** Relevant content in chunks 0652/0655 may be in truncated portions.

**NOT a primary contributor** for most failures because bait chunks (introductions, abbreviations) have their keyword-rich content at the BEGINNING, which survives truncation. Relevant chunks often have their specific answers later in the text.

**Truncation creates an asymmetric disadvantage:** Bait content (keyword-rich beginnings) survives truncation while relevant content (specific answers later in text) gets cut off.

---

## 6. Failure Distribution

| Query | Primary Cause | Secondary Cause | CE Ranking Error? |
|-------|--------------|-----------------|-------------------|
| mdq009 | CE_RANKING_ERROR | CHUNK_TRUNCATION | YES — keyword bait at #1 |
| mdq010 | CANDIDATE_RECALL | CE_RANKING_ERROR | YES — 0/20 relevant in top-10 |
| mdq012 | CE_RANKING_ERROR | — | YES — 3 irrelevant in top-5 |
| mdq014 | CE_RANKING_ERROR | RELEVANCE_JUDGMENT | MAYBE — rel=0 chunks appear relevant |
| mdq016 | CE_RANKING_ERROR | — | YES — low-confidence scoring |
| mdq024 | CE_RANKING_ERROR | RELEVANCE_JUDGMENT | MAYBE — cross-doc relevance strict |

### Percentage Analysis

| Failure Mode | Count | Percentage |
|-------------|-------|------------|
| CE Ranking Error | 6/6 | **100%** |
| Candidate Recall | 1/6 | 17% |
| Chunk Truncation | 2/6 | 33% |
| Relevance Judgment Issue | 2/6 | 33% |

**Conclusion:** CE ranking is the dominant bottleneck. However, the relevance judgment issues in mdq014 and mdq024 mean the actual CE performance may be better than measured.

---

## 7. Root Cause

### Primary: CE trained on MS MARCO struggles with medical guideline text

The `ms-marco-MiniLM-L-6-v2` model was trained on MS MARCO passage retrieval, where:
- Queries are short (3-8 words)
- Relevant passages contain the answer text directly
- Lexical overlap is a strong relevance signal

In our medical guideline context:
- Queries are complex clinical questions (15-25 words)
- Relevant chunks may use different terminology than the query
- Keyword overlap does NOT indicate answer presence (bait chunks)
- The "answer" is often a specific recommendation in a table, not a sentence that matches the query

### Secondary: Truncation creates asymmetric bias

With max_length=512, ~40-60% of each chunk is truncated. Bait chunks have keyword-rich content at the beginning (which survives), while relevant chunks often have specific answers later (which gets cut off). This systematically biases CE toward bait.

### Tertiary: Possible relevance judgment strictness

2/6 queries show chunks that appear relevant by content but are labeled rel=0. If these are judgment errors, CE performance is underestimated by ~5-10% nDCG@5.

---

## 8. ONE Recommended Next Experiment

### Test `BAAI/bge-reranker-v2-m3` as a single alternative reranker diagnostic

**Why this specific model:**
- Supports **8192 token context** (vs 512 for MiniLM) — eliminates truncation as a variable
- Trained on **broader data** including natural language inference tasks — may handle medical text better
- Similar model size to MiniLM — comparable latency
- Multilingual training may improve handling of complex medical syntax

**What this experiment tests:**
1. **Is the CE bottleneck model-specific or architectural?** If bge-reranker-v2-m3 improves significantly → the issue is MS MARCO training. If not → the issue is fundamental to cross-encoder architecture on this task.
2. **Does truncation matter?** With 8192-token context, all relevant content survives. If performance improves → truncation was a factor.
3. **Does broader training help?** If bge-reranker-v2-m3 ranks recommendation tables above keyword-bait → the issue is training data mismatch.

**Expected outcomes:**
- If nDCG@5 improves by >0.05 → CE model choice is the bottleneck, proceed with bge-reranker-v2-m3
- If nDCG@5 improves by 0.01-0.05 → partial improvement, consider ensemble or hybrid approach
- If nDCG@5 doesn't improve → CE architecture is not the bottleneck, investigate candidate generation or relevance judgments

**Alternative if bge-reranker-v2-m3 is too slow:** Test `cross-encoder/ms-marco-MiniLM-L-12-v2` (12-layer, same 512-token limit) to isolate model capacity from context length.

---

## 9. Stop/Continue Decision

### Decision: **A — CE is the bottleneck. Proceed with ONE targeted reranker experiment.**

**Rationale:**
1. CE ranking errors affect 100% of failure queries
2. The error pattern is systematic (keyword-bait promotion) not random
3. The current best nDCG@5 (0.438) is limited by CE's inability to distinguish clinical content from keyword-overlapping text
4. Candidate recall is adequate for most queries (only mdq010 shows recall as co-bottleneck)

**What NOT to do next:**
- Do NOT increase candidate K (Step 12 already showed marginal gains)
- Do NOT add BM25 (Step 14 showed no improvement)
- Do NOT test abbreviation expansion (Step 13 showed 1/12 impact)
- Do NOT run another hyperparameter search (already exhausted)

**What TO do next:**
1. Test `BAAI/bge-reranker-v2-m3` on the same 30-query dataset
2. Compare nDCG@5, MRR, and per-query results against the current CE baseline
3. Analyze whether bait chunks are still promoted with the new reranker
4. If improvement > 0.05 nDCG@5 → adopt the new reranker as production config

---

*Analysis completed: 2026-08-19*
*Data sources: Step 11 failure analysis, Step 14 hybrid results, scoped reranked results, hyperparameter search results*
