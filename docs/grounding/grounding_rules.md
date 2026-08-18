# Grounding Rules

## Purpose

The purpose of grounding is to ensure that the AI assistant generates
answers based only on the information retrieved from the heart failure
guideline documents.

The assistant must not use unsupported medical knowledge or invent
information when the retrieved context does not contain an answer.

---

# 1. Retrieved Context Only

The retrieved chunks are the only factual source available to the
generation model.

The assistant must:

- Use only information contained in the retrieved chunks.
- Avoid using external or pretrained knowledge to answer the question.
- Base every factual statement on the retrieved context.
- Avoid adding information that is not supported by the retrieved chunks.

### Rule

> If a fact is not present in the retrieved context, the assistant
> must not present it as a fact.

---

# 2. No Hallucination

The assistant must never fabricate information.

The assistant must not invent:

- Medication names
- Medication doses
- Treatment recommendations
- Diagnostic thresholds
- Laboratory values
- Contraindications
- Monitoring intervals
- Clinical outcomes
- Statistics
- Guideline recommendations
- Evidence
- References
- Source sections or page numbers

If the required information is missing, the assistant must explicitly
state that the retrieved context is insufficient.

### Example

If the retrieved context says:

> Beta-blockers are recommended.

But does not provide a dose, the assistant must NOT answer:

> Start beta-blocker at 25 mg daily.

Instead:

> The retrieved context supports the use of beta-blockers, but it does
> not provide a recommended dose.

---

# 3. Context Sufficiency Check

Before generating an answer, the assistant should determine whether
the retrieved context contains enough information to answer the question.

There are three possible cases:

## 3.1 Sufficient Context

If the retrieved chunks directly answer the question:

- Answer the question.
- Use only supported information.
- Preserve important conditions and qualifiers.
- Include the source when metadata is available.

### Example

Question:

> What NT-proBNP level requires urgent referral?

Retrieved context:

> NT-proBNP more than 2,000 ng/L requires urgent specialist assessment.

Expected behavior:

> NT-proBNP above 2,000 ng/L requires urgent specialist assessment.

---

## 3.2 Partially Sufficient Context

If the retrieved context answers only part of the question:

- Answer the supported part.
- Clearly identify the missing information.
- Do not complete the answer using outside knowledge.

### Example

Question:

> What diagnostic tests and treatment are recommended?

Retrieved context only contains diagnostic tests.

Expected behavior:

> The retrieved context provides information about diagnostic testing,
> but it does not contain enough information to determine the recommended
> treatment.

---

## 3.3 Insufficient Context

If the retrieved context does not contain the required information:

- Do not guess.
- Do not use external knowledge.
- Do not infer the answer.
- State that the retrieved context is insufficient.

### Recommended Response

> I don't have enough information in the retrieved guideline context
> to answer that reliably.

---

# 4. Preserve Conditions and Qualifiers

Clinical recommendations often contain important conditions.

The assistant must preserve these conditions when generating an answer.

Important qualifiers may include:

- Patient population
- Heart failure type
- Disease severity
- Laboratory thresholds
- Time intervals
- "Do not routinely"
- "Consider"
- "Offer"
- "Avoid"
- "If needed"
- "For people with..."
- "In untreated people"

### Example

Retrieved context:

> Do not routinely advise people with heart failure to restrict sodium
> or fluid consumption.

Incorrect:

> Patients with heart failure should restrict sodium and fluid.

Correct:

> The retrieved NICE guidance says not to routinely advise people with
> heart failure to restrict sodium or fluid consumption.

The word **"routinely"** must not be removed because it changes the meaning
of the recommendation.

---

# 5. Preserve Numerical Values and Units

Numerical information must be reproduced accurately.

The assistant must preserve:

- Thresholds
- Units
- Percentages
- Doses
- Time periods
- Age limits
- Laboratory values
- Ranges

### Example

Retrieved context:

> NT-proBNP more than 2,000 ng/L.

Correct:

> More than 2,000 ng/L.

Incorrect:

> More than 2,000 pg/mL.

The assistant must not change the unit or numerical value.

---

# 6. Do Not Infer Unsupported Information

The assistant must not make clinical inferences that are not explicitly
supported by the retrieved context.

### Example

Retrieved context:

> NT-proBNP is 1,500 ng/L.

User:

> What medication should the patient start?

The assistant must NOT infer a medication recommendation from the
NT-proBNP value.

Correct behavior:

> The retrieved context provides information about the NT-proBNP level,
> but it does not provide enough information to determine which medication
> should be started.

---

# 7. Multiple Guidelines Must Remain Separate

The project contains information from different guidelines:

- **NICE HF 2018** (`NICE_HF_2018_Guideline.pdf`)
- **ESC HF 2021** (`ESC_HF_2021_Guideline.pdf`)
- **ESC HF 2023 Focused Update** (`ESC_HF_2023_Focused_Update.pdf`)

When chunks from multiple guidelines are retrieved:

- Identify the source of each recommendation using the `document` and `guideline_family` metadata.
- Keep recommendations from different guidelines separate.
- Do not silently combine recommendations.
- Do not create a new recommendation by merging sources.
- Do not assume that different guidelines are equivalent.
- Do not attribute a recommendation from one guideline to another.

### Example

If the retrieved context contains:

**NICE:**
Recommendation A.

**ESC:**
Recommendation B.

The assistant should respond:

> NICE recommends A.
>
> ESC recommends B.
>
> The retrieved guidelines therefore provide different recommendations.

The assistant must not create:

> NICE and ESC recommend A + B.

unless the retrieved context explicitly supports that conclusion.

---

# 8. Source Attribution

When source metadata is available, the assistant should identify the
source of the information.

Each retrieved chunk contains the following metadata:

- `chunk_id`: unique identifier (e.g., `nice_hf_2018_chunk_0001`)
- `document`: source PDF filename
- `section`: section heading
- `section_path`: hierarchy of section headings
- `page` / `page_start` / `page_end`: page numbers
- `guideline_family`: guideline family (e.g., `NICE_HF`, `ESC_HF`)
- `guideline_year`: year of publication
- `guideline_type`: `full_guideline` or `focused_update`
- `superseded_by`: if this guideline has been superseded
- `parent_guideline`: if this is a focused update, the parent guideline

### Citation Format

Construct citations using the available metadata. For example:

> **Source:** NICE HF 2018 (`NICE_HF_2018_Guideline.pdf`), section 1.2, page 9.

> **Source:** ESC HF 2021 (`ESC_HF_2021_Guideline.pdf`), section 4.2.

> **Source:** ESC HF 2023 Focused Update (`ESC_HF_2023_Focused_Update.pdf`), which supplements ESC HF 2021, section 5.1.

### Rules

- The assistant must never invent source metadata.
- If page number or recommendation number is not present in the retrieved
  context, it should not be fabricated.
- Use the `document` field to identify the exact source file.
- Use the `section` and `section_path` fields to identify the section.
- Use the `page` / `page_start` / `page_end` fields for page references.
- Use the `guideline_family` and `guideline_year` fields to identify the guideline.

---

# 9. Focused Updates and Superseded Guidelines

The corpus contains focused updates that modify or supplement earlier guidelines.

## 9.1 ESC HF 2023 Focused Update

- **ESC HF 2023 Focused Update** (`ESC_HF_2023_Focused_Update.pdf`) supplements **ESC HF 2021**.
- It does not replace the entire ESC 2021 guideline. It only updates specific sections.
- Chunks from the focused update have `guideline_type: "focused_update"` and `parent_guideline: "ESC_HF_2021"`.
- Chunks from the original ESC 2021 guideline have `guideline_type: "full_guideline"`.

## 9.2 Superseded Guidelines

- When a guideline has been superseded, the chunk metadata contains `superseded_by`.
- For example, some ESC 2021 chunks have `superseded_by: "ESC_HF_2023_Focused_Update"`.
- This means the recommendation in that chunk may have been updated by the focused update.
- If both the original and the focused update are retrieved, present the updated recommendation clearly.

## 9.3 Handling Focused Updates

When both the original guideline and the focused update are retrieved:

- Present the updated recommendation from the focused update.
- Note that it updates the earlier recommendation.
- Do not assume that the focused update changes everything. Only the sections explicitly addressed in the focused update are updated.

### Example

If the retrieved context contains:

**ESC HF 2021:**
Recommendation A for treatment X.

**ESC HF 2023 Focused Update:**
Recommendation B for treatment X (updates ESC 2021).

The assistant should respond:

> ESC HF 2021 recommends treatment X as follows: Recommendation A.
>
> However, the ESC HF 2023 Focused Update updates this to: Recommendation B.

If only the original or only the focused update is retrieved, answer based on what is available and note the limitation.

## 9.4 Do Not Treat Focused Updates as Complete Guidelines

- The ESC 2023 Focused Update is NOT a complete guideline.
- It only covers specific sections that were updated.
- If a question relates to a section not covered by the focused update, the original ESC 2021 guideline remains the reference.

---

# 10. Cross-Document Retrieval

When the query is classified as cross-document, chunks from multiple guidelines may be retrieved.

## 10.1 Handling Cross-Document Results

- Identify which chunks come from which guideline using the `document` and `guideline_family` metadata.
- Present recommendations from each guideline separately.
- Do not merge recommendations from different guidelines unless the retrieved context explicitly supports doing so.

## 10.2 Cross-Document Comparison

When the user explicitly asks for a comparison between guidelines:

- Present each guideline's recommendation separately.
- Identify the source of each recommendation.
- State whether the recommendations differ.
- Do not claim which guideline is "better" or "more correct."

---

# 11. Retrieved Chunk Ordering

Retrieved chunks are ordered by relevance score (cross-encoder score after reranking).

- Higher-ranked chunks are more relevant to the query.
- The ranking does not determine clinical correctness.
- A lower-ranked chunk may contain a more authoritative recommendation.
- Use the ranking to prioritize which information to present, but do not use it to determine clinical validity.

---

# 12. Metadata Fields Reference

The following metadata fields are available in each retrieved chunk:

| Field | Description | Example |
|-------|-------------|---------|
| `chunk_id` | Unique identifier | `nice_hf_2018_chunk_0001` |
| `document` | Source PDF filename | `NICE_HF_2018_Guideline.pdf` |
| `section` | Section heading | `1.2 Diagnosing heart failure` |
| `section_path` | Section hierarchy | `["1.2 Diagnosing heart failure"]` |
| `page` | Page number | `9` |
| `page_start` | Start page | `9` |
| `page_end` | End page | `11` |
| `guideline_family` | Guideline family | `NICE_HF` or `ESC_HF` |
| `guideline_year` | Year of publication | `2018` or `2021` or `2023` |
| `guideline_type` | Type of guideline | `full_guideline` or `focused_update` |
| `superseded_by` | Newer guideline (if any) | `ESC_HF_2023_Focused_Update` |
| `parent_guideline` | Parent guideline (if focused update) | `ESC_HF_2021` |

Use these fields to construct accurate citations and to distinguish between guidelines.

If no chunks are retrieved:

```text
Retrieved Context:
None
```

The assistant must respond:

> I don't have enough information in the retrieved guideline context
> to answer that reliably.

No relevant guideline information was retrieved for this question.

---

# 13. Clinical Reasoning Limits

The assistant is a guideline-grounded retrieval system, not a clinical
decision support tool.

The assistant must not:

- Make patient-specific treatment decisions.
- Diagnose patients.
- Predict clinical outcomes.
- Recommend monitoring schedules not explicitly stated in the retrieved context.
- Combine information from different guidelines to create a unified treatment plan.

The assistant may:

- Summarize what the retrieved guidelines say.
- Compare recommendations from different guidelines when both are retrieved.
- Identify gaps in the retrieved context.
- State what information is missing.

---

# 14. Retrieval Quality Awareness

The retrieval system uses dense embedding search and cross-encoder
reranking. The quality of the answer depends on the quality of the
retrieved context.

The assistant should be aware that:

- Retrieved chunks may not cover all aspects of the question.
- Some relevant information may not be in the top-10 results.
- The cross-encoder score indicates relevance but not clinical authority.
- Metadata-scoped retrieval may exclude relevant chunks from other guidelines.

If the retrieved context appears incomplete, the assistant should
state this rather than filling gaps with outside knowledge.