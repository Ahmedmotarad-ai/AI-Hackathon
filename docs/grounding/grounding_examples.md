# Grounding Generation — Input → Expected Output Examples

This file contains examples for evaluating grounded generation.
The assistant must answer using only the retrieved context.

---

## Example 1 — Fully Supported Question

### Input

**User Question:**

What NT-proBNP level requires urgent specialist assessment?

**Retrieved Context:**

Source: NICE Heart Failure Guideline  
Recommendation 1.2.3

People with suspected heart failure and an NT-proBNP level
more than 2,000 nanogram per litre should be referred urgently
for specialist assessment and transthoracic echocardiography
within 2 weeks.

### Expected Output

According to the retrieved NICE guidance, people with suspected
heart failure and an NT-proBNP level above 2,000 ng/L should be
referred urgently for specialist assessment and transthoracic
echocardiography within 2 weeks.

**Source:** NICE Heart Failure Guideline, recommendation 1.2.3.

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

Source: NICE Heart Failure Guideline  
Recommendation 1.4.1

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

**Source:** NICE Heart Failure Guideline, recommendation 1.4.1.

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

Source: NICE Heart Failure Guideline  
Recommendation 1.8.3

For stable people with proven heart failure, monitor at least
every 6 months.

### Expected Output

For stable people with proven heart failure, the retrieved NICE
guidance recommends monitoring at least every 6 months.

**Source:** NICE Heart Failure Guideline, recommendation 1.8.3.

### Grounding Behavior

- Answer directly.
- Preserve the "stable" condition.
- Preserve "at least every 6 months."
- Include the source when metadata is available.

---

# Summary of Grounding Behavior

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