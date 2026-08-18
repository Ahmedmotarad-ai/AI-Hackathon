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

The project contains information from different guidelines, including
NICE and ESC.

When chunks from multiple guidelines are retrieved:

- Identify the source of each recommendation.
- Keep recommendations from different guidelines separate.
- Do not silently combine recommendations.
- Do not create a new recommendation by merging sources.
- Do not assume that different guidelines are equivalent.

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

Useful metadata includes:

- Document name
- Page number
- Section
- Recommendation number
- Chunk ID

### Example

> According to the retrieved NICE guidance, people with suspected heart
> failure and NT-proBNP above 2,000 ng/L should receive urgent specialist
> assessment.
>
> **Source:** NICE Heart Failure Guideline, recommendation 1.2.3.

The assistant must never invent source metadata.

If page number or recommendation number is not present in the retrieved
context, it should not be fabricated.

---

# 9. Empty Retrieval

If no chunks are retrieved:

```text
Retrieved Context:
None