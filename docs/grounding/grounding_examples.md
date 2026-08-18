# Grounding Generation — Input → Expected Output Examples

This file contains examples for evaluating grounded generation.
The assistant must answer using only the retrieved context.

---

## Example 1 — Fully Supported Question

### Input

**User Question:**

What NT-proBNP level requires urgent specialist assessment?

**Retrieved Context:**

Chunk: `nice_hf_2018_chunk_0002`
Document: `NICE_HF_2018_Guideline.pdf`
Section: `1.2.3 NT-proBNP thresholds`
Page: 9
Guideline Family: `NICE_HF`
Guideline Year: `2018`
Guideline Type: `full_guideline`

People with suspected heart failure and an NT-proBNP level
more than 2,000 nanogram per litre should be referred urgently
for specialist assessment and transthoracic echocardiography
within 2 weeks.

### Expected Output

According to the retrieved NICE guidance, people with suspected
heart failure and an NT-proBNP level above 2,000 ng/L should be
referred urgently for specialist assessment and transthoracic
echocardiography within 2 weeks.

**Source:** NICE HF 2018 (`NICE_HF_2018_Guideline.pdf`), section 1.2.3, page 9.

### Grounding Behavior

- The answer uses only information present in the retrieved context.
- The threshold and timeframe are preserved.
- No additional medical information is added.

---

## Example 2 — Fully Supported Question with Multiple Thresholds

### Input

**User Question:**

What are the NT-proBNP thresholds for suspected heart failure?

**Retrieved Context:**

Source: NICE Heart Failure Guideline

- NT-proBNP more than 2,000 ng/L:
  urgent specialist assessment and transthoracic echocardiography
  within 2 weeks.

- NT-proBNP between 400 and 2,000 ng/L:
  specialist assessment and transthoracic echocardiography
  within 6 weeks.

- NT-proBNP less than 400 ng/L in an untreated person:
  makes a diagnosis of heart failure less likely.

### Expected Output

According to the retrieved NICE guidance:

- **More than 2,000 ng/L:** urgent specialist assessment and
  transthoracic echocardiography within 2 weeks.
- **400–2,000 ng/L:** specialist assessment and transthoracic
  echocardiography within 6 weeks.
- **Less than 400 ng/L:** in an untreated person, this makes a
  diagnosis of heart failure less likely.

**Source:** NICE Heart Failure Guideline.

### Grounding Behavior

- All three statements are directly supported by the retrieved context.
- The answer does not introduce additional thresholds or interpretations.

---

## Example 3 — Missing Information / No Hallucination

### Input

**User Question:**

What dose of beta-blocker should be prescribed?

**Retrieved Context:**

Source: NICE Heart Failure Guideline

Beta-blockers are included in the treatment combination for
people with heart failure with reduced ejection fraction.

### Expected Output

The retrieved context supports the use of a beta-blocker in the
treatment combination for people with heart failure with reduced
ejection fraction, but it does not provide a beta-blocker dose.

I don't have enough information in the retrieved guideline context
to provide a reliable dose.

### Grounding Behavior

- Do not invent a medication dose.
- Do not use external medical knowledge.
- Explicitly identify the missing information.

---

## Example 4 — Partial Context

### Input

**User Question:**

What tests are recommended when heart failure is suspected,
and what do the results mean?

**Retrieved Context:**

Source: NICE Heart Failure Guideline

Measure NT-proBNP in people with suspected heart failure.

If NT-proBNP is more than 2,000 ng/L, refer urgently for
specialist assessment and transthoracic echocardiography
within 2 weeks.

### Expected Output

The retrieved context supports the following:

- NT-proBNP should be measured in people with suspected heart failure.
- If NT-proBNP is above 2,000 ng/L, urgent specialist assessment
  and transthoracic echocardiography should take place within 2 weeks.

The retrieved context does not provide enough information to
summarize all recommended diagnostic tests or interpret other
possible results.

### Grounding Behavior

- Answer only the supported part.
- Clearly state what information is missing.
- Do not add diagnostic tests that are not present in the context.

---

## Example 5 — Unsupported Question

### Input

**User Question:**

What is the mortality rate for patients with heart failure?

**Retrieved Context:**

Source: NICE Heart Failure Guideline

Heart failure is a complex clinical syndrome caused by impaired
heart function.

No mortality rate is provided in the retrieved context.

### Expected Output

The retrieved context does not provide a mortality rate for
patients with heart failure.

I don't have enough information in the retrieved guideline context
to answer this reliably.

### Grounding Behavior

- Do not provide a mortality percentage from general knowledge.
- Do not estimate or infer a value.
- Explicitly state that the information is unavailable.

---

## Example 6 — Preserve Conditions in a Recommendation

### Input

**User Question:**

Which medicines should be offered?

**Retrieved Context:**

Chunk: `nice_hf_2018_chunk_0015`
Document: `NICE_HF_2018_Guideline.pdf`
Section: `1.4.1 Pharmacological treatment`
Page: 25
Guideline Family: `NICE_HF`
Guideline Year: `2018`
Guideline Type: `full_guideline`

Offer an angiotensin-converting enzyme (ACE) inhibitor,
a beta-blocker, a mineralocorticoid receptor antagonist (MRA)
and a sodium-glucose cotransporter-2 (SGLT2) inhibitor to people
with heart failure with reduced ejection fraction.

### Expected Output

For people with heart failure with reduced ejection fraction,
the retrieved NICE guidance recommends offering:

- an ACE inhibitor
- a beta-blocker
- an MRA
- an SGLT2 inhibitor

**Source:** NICE HF 2018 (`NICE_HF_2018_Guideline.pdf`), section 1.4.1, page 25.

### Grounding Behavior

- Preserve the condition "heart failure with reduced ejection fraction."
- Do not generalize the recommendation to all heart failure patients.
- Do not add other medications.

---

## Example 7 — Do Not Remove Important Qualifiers

### Input

**User Question:**

Should patients with heart failure restrict salt and fluid intake?

**Retrieved Context:**

Source: NICE Heart Failure Guideline

Do not routinely advise people with heart failure to restrict
their sodium or fluid consumption.

Ask about salt and fluid consumption and, if needed:

- restrict fluids for people with dilutional hyponatraemia.
- reduce intake for people with high levels of salt or fluid
  consumption.

### Expected Output

The retrieved NICE guidance says not to routinely advise people
with heart failure to restrict sodium or fluid intake.

However, if needed:

- fluid restriction may be advised for people with dilutional
  hyponatraemia.
- intake may be reduced for people with high levels of salt or
  fluid consumption.

The need for restriction should be reviewed.

### Grounding Behavior

- Preserve the "do not routinely" qualifier.
- Preserve the specific exceptions.
- Do not turn a conditional recommendation into a general rule.

---

## Example 8 — Conflicting Guidelines

### Input

**User Question:**

What is the recommended treatment for heart failure with reduced
ejection fraction?

**Retrieved Context:**

### NICE

Offer:

- ACE inhibitor
- beta-blocker
- MRA
- SGLT2 inhibitor

for people with heart failure with reduced ejection fraction.

### ESC

The retrieved ESC context contains a different treatment
recommendation.

### Expected Output

The retrieved guidelines should be considered separately.

**NICE:** The retrieved NICE guidance recommends an ACE inhibitor,
beta-blocker, MRA and SGLT2 inhibitor for people with heart failure
with reduced ejection fraction.

**ESC:** The retrieved ESC context contains a different treatment
recommendation, but the specific recommendation is not fully
included in the retrieved context.

Therefore, the available context does not support combining the
two guidelines into a single recommendation.

### Grounding Behavior

- Do not silently choose one guideline.
- Do not merge the two recommendations.
- Do not invent the missing ESC recommendation.
- Clearly distinguish between sources.

---

## Example 9 — Comparing Guidelines Only When Both Are Supported

### Input

**User Question:**

Do NICE and ESC recommend the same treatment?

**Retrieved Context:**

### NICE

For people with heart failure with reduced ejection fraction,
NICE recommends an ACE inhibitor, beta-blocker, MRA and SGLT2
inhibitor.

### ESC

The retrieved ESC chunk states a different recommendation for
the same clinical population.

### Expected Output

The retrieved context indicates that the NICE and ESC
recommendations differ.

**NICE:** recommends an ACE inhibitor, beta-blocker, MRA and
SGLT2 inhibitor for people with heart failure with reduced
ejection fraction.

**ESC:** the retrieved context indicates a different recommendation.

The specific ESC recommendation is not fully available in the
retrieved context, so a detailed comparison cannot be made
reliably.

### Grounding Behavior

- State that the sources differ only when the retrieved context
  explicitly supports this.
- Do not reconstruct the missing recommendation.
- Do not claim which guideline is better.

---

## Example 10 — Source Does Not Support Patient-Specific Advice

### Input

**User Question:**

My NT-proBNP is 1,500 ng/L. What medication should I start today?

**Retrieved Context:**

Source: NICE Heart Failure Guideline

An NT-proBNP level between 400 and 2,000 ng/L should lead to
specialist assessment and transthoracic echocardiography within
6 weeks.

### Expected Output

The retrieved NICE context states that an NT-proBNP level between
400 and 2,000 ng/L should lead to specialist assessment and
transthoracic echocardiography within 6 weeks.

The retrieved context does not provide enough information to
determine which medication should be started for an individual
patient.

### Grounding Behavior

- Use the supported guideline information.
- Do not prescribe a medication.
- Do not infer a treatment decision from the laboratory value alone.

---

## Example 11 — Question About Information Not Present in Retrieved Chunks

### Input

**User Question:**

What are the contraindications to SGLT2 inhibitors?

**Retrieved Context:**

Source: NICE Heart Failure Guideline

SGLT2 inhibitors are included in the treatment combination for
people with heart failure with reduced ejection fraction.

No contraindications are included in the retrieved context.

### Expected Output

The retrieved context supports the inclusion of SGLT2 inhibitors
in the treatment combination for people with heart failure with
reduced ejection fraction.

However, the retrieved context does not provide information about
contraindications to SGLT2 inhibitors.

I don't have enough information in the retrieved guideline context
to answer the contraindications question reliably.

### Grounding Behavior

- Do not invent contraindications.
- Do not use information outside the retrieved context.

---

## Example 12 — Retrieval Context Is Empty

### Input

**User Question:**

What is the recommended monitoring frequency for patients with
heart failure?

**Retrieved Context:**

No relevant chunks were retrieved.

### Expected Output

I don't have enough information in the retrieved guideline context
to answer this reliably.

No relevant guideline information was retrieved for this question.

### Grounding Behavior

- Never answer from general knowledge when no context is retrieved.
- Treat an empty retrieval result as insufficient evidence.

---

## Example 13 — Retrieval Context Is Irrelevant

### Input

**User Question:**

What is the recommended treatment for HFrEF?

**Retrieved Context:**

Source: NICE Heart Failure Guideline

The specialist heart failure multidisciplinary team should provide
information to people with newly diagnosed heart failure.

The retrieved chunks contain no treatment recommendation for HFrEF.

### Expected Output

The retrieved context does not contain enough information to
determine the recommended treatment for heart failure with reduced
ejection fraction.

I don't have enough information in the retrieved guideline context
to answer this reliably.

### Grounding Behavior

- Do not treat related information as evidence for the requested answer.
- Relevant-to-topic does not necessarily mean sufficient-to-answer.

---

## Example 14 — Follow-up Question Must Stay Grounded

### Input

**Conversation:**

User:
What does NICE recommend for people with HFrEF?

Assistant:
NICE recommends an ACE inhibitor, beta-blocker, MRA and SGLT2
inhibitor for people with HFrEF.

User:
What about the dose?

**Retrieved Context:**

The retrieved context contains the treatment combination but
does not contain medication doses.

### Expected Output

The retrieved context does not provide the doses of these medicines.

I don't have enough information in the retrieved guideline context
to provide reliable dosing information.

### Grounding Behavior

- The previous assistant answer does not become a source of new facts.
- The current retrieved context must support the answer.
- Do not invent doses based on the previous answer.

---

## Example 15 — Preserve Numerical Values and Units

### Input

**User Question:**

What is the threshold for urgent referral?

**Retrieved Context:**

NT-proBNP more than 2,000 nanogram per litre should lead to
urgent specialist assessment and transthoracic echocardiography
within 2 weeks.

### Expected Output

The threshold is **more than 2,000 ng/L**.

At this level, the retrieved NICE guidance recommends urgent
specialist assessment and transthoracic echocardiography within
**2 weeks**.

### Grounding Behavior

- Preserve the numerical threshold.
- Preserve the unit.
- Preserve the timeframe.
- Do not change "more than" into "equal to or greater than."

---

## Example 16 — Supported Answer with a Clear Source

### Input

**User Question:**

How often should stable people with heart failure be monitored?

**Retrieved Context:**

Chunk: `nice_hf_2018_chunk_0001`
Document: `NICE_HF_2018_Guideline.pdf`
Section: `1.8 Monitoring`
Page: 45
Guideline Family: `NICE_HF`
Guideline Year: `2018`
Guideline Type: `full_guideline`

For stable people with proven heart failure, monitor at least
every 6 months.

### Expected Output

For stable people with proven heart failure, the retrieved NICE
guidance recommends monitoring at least every 6 months.

**Source:** NICE HF 2018 (`NICE_HF_2018_Guideline.pdf`), section 1.8, page 45.

### Grounding Behavior

- Answer directly.
- Preserve the "stable" condition.
- Preserve "at least every 6 months."
- Include the source when metadata is available.

---

## Example 17 — ESC 2023 Focused Update

### Input

**User Question:**

What is the recommended treatment for HFrEF according to ESC?

**Retrieved Context:**

Chunk 1: `esc_hf_2021_chunk_0100`
Document: `ESC_HF_2021_Guideline.pdf`
Section: `4.2 Pharmacological treatment`
Guideline Family: `ESC_HF`
Guideline Year: `2021`
Guideline Type: `full_guideline`
Superseded By: `ESC_HF_2023_Focused_Update`

For HFrEF, recommend ACE inhibitor, beta-blocker, MRA, and SGLT2 inhibitor.

Chunk 2: `esc_hf_2023_focused_update_chunk_0020`
Document: `ESC_HF_2023_Focused_Update.pdf`
Section: `5.1 Updated pharmacological recommendations`
Guideline Family: `ESC_HF`
Guideline Year: `2023`
Guideline Type: `focused_update`
Parent Guideline: `ESC_HF_2021`

The 2023 focused update confirms the use of ACE inhibitor (or ARNI), beta-blocker, MRA, and SGLT2 inhibitor as foundational therapy for HFrEF.

### Expected Output

According to the retrieved context:

**ESC HF 2021** recommends ACE inhibitor, beta-blocker, MRA, and SGLT2 inhibitor for HFrEF.

**ESC HF 2023 Focused Update** confirms and updates this to: ACE inhibitor (or ARNI), beta-blocker, MRA, and SGLT2 inhibitor as foundational therapy.

The 2023 focused update supersedes the 2021 recommendation for this section.

**Source:** ESC HF 2023 Focused Update (`ESC_HF_2023_Focused_Update.pdf`), section 5.1.

### Grounding Behavior

- Distinguish between the original and the focused update.
- Present the updated recommendation from the focused update.
- Note that the focused update supersedes the 2021 recommendation.
- Use the `parent_guideline` and `superseded_by` metadata.

---

## Example 18 — Cross-Document Query

### Input

**User Question:**

How do NICE and ESC differ in their approach to heart failure diagnosis?

**Retrieved Context:**

Chunk 1: `nice_hf_2018_chunk_0005`
Document: `NICE_HF_2018_Guideline.pdf`
Section: `1.2 Diagnosing heart failure`
Guideline Family: `NICE_HF`
Guideline Year: `2018`
Guideline Type: `full_guideline`

Measure NT-proBNP in people with suspected heart failure.

Chunk 2: `esc_hf_2021_chunk_0050`
Document: `ESC_HF_2021_Guideline.pdf`
Section: `3.1 Diagnostic workup`
Guideline Family: `ESC_HF`
Guideline Year: `2021`
Guideline Type: `full_guideline`

Diagnosis of heart failure requires symptoms, signs, and objective evidence of cardiac dysfunction.

### Expected Output

The retrieved guidelines provide different diagnostic approaches:

**NICE HF 2018:** recommends measuring NT-proBNP in people with suspected heart failure as an initial step.

**ESC HF 2021:** requires symptoms, signs, and objective evidence of cardiac dysfunction for diagnosis.

The guidelines take different approaches: NICE emphasizes NT-proBNP testing, while ESC emphasizes a broader clinical assessment.

**Source:** NICE HF 2018 (`NICE_HF_2018_Guideline.pdf`), section 1.2; ESC HF 2021 (`ESC_HF_2021_Guideline.pdf`), section 3.1.

### Grounding Behavior

- Present each guideline's approach separately.
- Identify the source of each recommendation.
- Do not merge the approaches into a single recommendation.
- Do not claim which guideline is better.

---

## Example 19 — Superseded Guideline

### Input

**User Question:**

What does ESC recommend for SGLT2 inhibitors in HFrEF?

**Retrieved Context:**

Chunk 1: `esc_hf_2021_chunk_0080`
Document: `ESC_HF_2021_Guideline.pdf`
Section: `4.2.3 SGLT2 inhibitors`
Guideline Family: `ESC_HF`
Guideline Year: `2021`
Guideline Type: `full_guideline`
Superseded By: `ESC_HF_2023_Focused_Update`

SGLT2 inhibitors are recommended for HFrEF.

Chunk 2: `esc_hf_2023_focused_update_chunk_0015`
Document: `ESC_HF_2023_Focused_Update.pdf`
Section: `5.2 SGLT2 inhibitors update`
Guideline Family: `ESC_HF`
Guideline Year: `2023`
Guideline Type: `focused_update`
Parent Guideline: `ESC_HF_2021`

SGLT2 inhibitors (dapagliflozin or empagliflozin) are recommended for HFrEF regardless of diabetes status.

### Expected Output

The retrieved context contains both the original ESC 2021 recommendation and the 2023 focused update:

**ESC HF 2021:** recommends SGLT2 inhibitors for HFrEF.

**ESC HF 2023 Focused Update:** specifies dapagliflozin or empagliflozin, and extends the recommendation regardless of diabetes status.

The 2023 focused update provides more specific guidance and supersedes the 2021 recommendation for this section.

**Source:** ESC HF 2023 Focused Update (`ESC_HF_2023_Focused_Update.pdf`), section 5.2.

### Grounding Behavior

- Present the updated recommendation from the focused update.
- Note that the 2021 recommendation has been superseded.
- Use the `superseded_by` metadata to identify the relationship.
- Do not present the superseded recommendation as current.

---

## Example 20 — Focused Update Does Not Cover All Sections

### Input

**User Question:**

What does ESC recommend for device therapy in heart failure?

**Retrieved Context:**

Chunk 1: `esc_hf_2021_chunk_0200`
Document: `ESC_HF_2021_Guideline.pdf`
Section: `9. Device therapy`
Guideline Family: `ESC_HF`
Guideline Year: `2021`
Guideline Type: `full_guideline`

ICD and CRT are recommended for eligible patients with heart failure.

No chunks from the ESC 2023 Focused Update are retrieved for this section.

### Expected Output

According to the retrieved ESC HF 2021 guideline, ICD and CRT are recommended for eligible patients with heart failure.

Note: The ESC 2023 Focused Update does not cover device therapy, so the ESC 2021 recommendation remains the current reference for this section.

**Source:** ESC HF 2021 (`ESC_HF_2021_Guideline.pdf`), section 9.

### Grounding Behavior

- Answer based on the available ESC 2021 context.
- Note that the focused update does not cover this section.
- Do not assume that the focused update has changed this recommendation.

---

## Example 21 — Only Focused Update Retrieved

### Input

**User Question:**

What is the ESC recommendation for ARNI in HFrEF?

**Retrieved Context:**

Chunk 1: `esc_hf_2023_focused_update_chunk_0025`
Document: `ESC_HF_2023_Focused_Update.pdf`
Section: `5.3 ARNI recommendation`
Guideline Family: `ESC_HF`
Guideline Year: `2023`
Guideline Type: `focused_update`
Parent Guideline: `ESC_HF_2021`

ARNI is recommended as a replacement for ACE inhibitor in patients who remain symptomatic despite optimal therapy.

### Expected Output

According to the retrieved ESC HF 2023 Focused Update, ARNI is recommended as a replacement for ACE inhibitor in patients who remain symptomatic despite optimal therapy.

Note: This is from the 2023 focused update, which supplements the ESC HF 2021 guideline.

**Source:** ESC HF 2023 Focused Update (`ESC_HF_2023_Focused_Update.pdf`), section 5.3.

### Grounding Behavior

- Answer based on the focused update.
- Note that it is a focused update supplementing ESC 2021.
- Do not assume the focused update is a complete guideline.

---

## Example 22 — NICE vs ESC Conflict with Metadata

### Input

**User Question:**

Do NICE and ESC recommend the same threshold for NT-proBNP?

**Retrieved Context:**

Chunk 1: `nice_hf_2018_chunk_0003`
Document: `NICE_HF_2018_Guideline.pdf`
Section: `1.2.1 NT-proBNP thresholds`
Guideline Family: `NICE_HF`
Guideline Year: `2018`
Guideline Type: `full_guideline`

NT-proBNP more than 2,000 ng/L requires urgent specialist assessment.

Chunk 2: `esc_hf_2021_chunk_0030`
Document: `ESC_HF_2021_Guideline.pdf`
Section: `3.2 Biomarkers`
Guideline Family: `ESC_HF`
Guideline Year: `2021`
Guideline Type: `full_guideline`

NT-proBNP above 450 pg/mL (or BNP above 100 pg/mL) suggests heart failure.

### Expected Output

The retrieved guidelines use different thresholds:

**NICE HF 2018:** NT-proBNP more than 2,000 ng/L requires urgent specialist assessment.

**ESC HF 2021:** NT-proBNP above 450 pg/mL suggests heart failure.

Note: The thresholds are for different purposes (NICE for urgent referral, ESC for diagnosis suggestion) and use different units (ng/L vs pg/mL).

**Source:** NICE HF 2018 (`NICE_HF_2018_Guideline.pdf`), section 1.2.1; ESC HF 2021 (`ESC_HF_2021_Guideline.pdf`), section 3.2.

### Grounding Behavior

- Present each guideline's threshold separately.
- Note the different units and purposes.
- Do not combine the thresholds into a single recommendation.
- Preserve the exact numerical values and units.

---

## Summary of Grounding Behavior

The assistant should follow these rules for all examples:

1. Use only the retrieved context as the factual source.
2. Do not use outside or pretrained knowledge to fill missing information.
3. Do not fabricate medication doses, recommendations, thresholds,
   contraindications, statistics, or clinical decisions.
4. If the context is sufficient, answer directly.
5. If the context is partially sufficient, answer only the supported part.
6. If the context is insufficient or empty, explicitly abstain.
7. Preserve conditions, qualifiers, numerical values, units, and timeframes.
8. When multiple guidelines are retrieved, distinguish their
   recommendations instead of silently combining them.
9. Do not make patient-specific treatment decisions unless they are
   explicitly supported by the retrieved context.
10. When source metadata is available, identify the supporting source.
11. Handle focused updates (ESC 2023) as supplements to the parent
    guideline (ESC 2021), not as complete guidelines.
12. Use `superseded_by` and `parent_guideline` metadata to identify
    relationships between guidelines.
13. Note when a focused update does not cover a particular section.
14. Use the `document`, `guideline_family`, `guideline_year`, and
    `guideline_type` fields for accurate source attribution.