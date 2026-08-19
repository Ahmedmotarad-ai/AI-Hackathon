# Risk Classification Policy

## 1. Purpose

The Risk Classification layer is the first safety layer in the system.

Its purpose is to classify every user question before retrieval and answer generation.

The classifier determines:

- Whether the question is within the system scope.
- Whether the question is medical/clinical.
- Whether the question contains potentially high-risk or urgent medical content.
- Which safety policy should be applied before retrieval or generation.

The Risk Classification layer must run before the RAG retrieval process.

---

## 2. Classification Categories

The system uses four risk categories:

1. Normal
2. Medical / Clinical
3. High-risk
4. Out-of-scope

---

## 3. Category Definitions

### 3.1 Normal

A question is classified as `Normal` when it is:

- General/non-medical.
- Not asking for diagnosis, treatment, medication, dosage, or clinical decision-making.
- Not describing an urgent medical situation.
- Within the allowed general-purpose scope of the application.

Examples:

- "What is RAG?"
- "What is a vector database?"
- "What is the difference between embeddings and keywords?"

Policy:

- The system may answer normally if the question is within the application scope.
- No clinical retrieval is required.

---

### 3.2 Medical / Clinical

A question is classified as `Medical / Clinical` when it:

- Concerns heart failure or cardiovascular clinical information.
- Asks about diagnosis criteria.
- Asks about guideline recommendations.
- Asks about monitoring or follow-up.
- Asks about medications in the context of heart failure.
- Asks about clinical thresholds or treatment recommendations.
- Can be answered safely using the retrieved clinical guidelines.

Examples:

- "What does NICE recommend for monitoring stable heart failure?"
- "What NT-proBNP level requires specialist assessment?"
- "What does the guideline recommend for HFrEF?"

Policy:

1. Allow the request to continue to retrieval.
2. Retrieve relevant guideline chunks.
3. Generate an answer using retrieved evidence only.
4. Apply grounding rules after retrieval.
5. If the retrieved context is insufficient, abstain or provide a clearly limited response.

---

### 3.3 High-risk

A question is classified as `High-risk` when answering it directly could lead to unsafe or urgent medical action.

Examples include:

- Emergency symptoms.
- Requests for immediate treatment.
- Requests for personalized medication changes.
- Requests for exact dosing for an individual patient.
- Requests to stop or start medication based on personal symptoms.
- Questions indicating possible acute deterioration.
- Requests to replace emergency or professional medical care.

Examples:

- "I have severe chest pain and shortness of breath. What should I take?"
- "Should I double my heart failure medication dose?"
- "My blood pressure is very low after taking my medication. Should I stop it?"
- "Can I stop my diuretic today?"

Policy:

- Do NOT provide a personalized treatment decision.
- Do NOT recommend starting, stopping, or changing medication.
- Do NOT provide emergency management instructions as a substitute for professional care.
- Do NOT continue normal RAG answer generation.
- Return a safe response directing the user to appropriate professional/emergency care when applicable.

The retrieved guidelines must not be used to turn a high-risk request into a personalized medical order.

---

### 3.4 Out-of-scope

A question is classified as `Out-of-scope` when it is outside the intended domain or capabilities of the system.

Examples:

- "Write a Python game."
- "What is the weather today?"
- "Translate this paragraph into French."
- "Who won yesterday's football match?"

Policy:

- Do not perform clinical retrieval.
- Return a short scope message.
- The system should explain that the assistant is designed for the supported clinical/guideline domain.

---

# 4. Safety Gate

The Safety Gate is executed immediately after classification.

Decision flow:

User Question
      |
      v
Risk Classification
      |
      +------------------+
      |        |         |         |
    Normal  Medical   High-risk  Out-of-scope
      |        |         |         |
      v        v         v         v
   Normal   Retrieve   Safety    Reject /
   Policy   Guidelines Response   Redirect
                |
                v
          Grounded Generation
                |
                v
          Answer / Abstain

---

# 5. Policy Matrix

| Category | Retrieval | Generation | Personalized Medical Advice | Action |
|----------|-----------|------------|-----------------------------|--------|
| Normal | No clinical retrieval | Normal | No | Answer if in scope |
| Medical / Clinical | Yes | Grounded only | No | Retrieve + answer/abstain |
| High-risk | No normal generation | Safety response | No | Safety Gate |
| Out-of-scope | No | No | No | Redirect |

---

# 6. Classification Priority

When multiple categories appear to apply, use the following priority:

High-risk
>
Medical / Clinical
>
Out-of-scope / Normal

High-risk always takes precedence over Medical / Clinical.

Example:

"I have severe chest pain. According to the NICE guideline, what medication should I take?"

Although the question references a clinical guideline, the emergency symptom makes it `High-risk`.

Expected classification:

`High-risk`

---

# 7. High-risk Signals

The classifier should look for signals such as:

### Emergency symptoms

- severe chest pain
- severe shortness of breath
- difficulty breathing
- fainting
- loss of consciousness
- severe confusion
- blue lips
- sudden severe deterioration

### Immediate treatment requests

- what should I take now
- what medication should I take
- should I stop my medication
- should I double my dose
- should I change my dose
- can I skip my medication

### Personalized treatment decisions

- my dose
- my medication
- my symptoms
- should I start
- should I stop
- should I increase
- should I decrease

These signals are intentionally conservative.

The classifier should prefer safety over allowing a potentially unsafe request.

---

# 8. Medical / Clinical Signals

Examples of clinical terminology:

- heart failure
- HFrEF
- HFmrEF
- HFpEF
- NT-proBNP
- BNP
- echocardiography
- ejection fraction
- beta blocker
- ACE inhibitor
- ARNI
- ARB
- MRA
- diuretic
- renal function
- electrolytes
- guideline
- diagnosis
- monitoring
- follow-up
- treatment recommendation

These terms alone do not automatically make a question high-risk.

Example:

"What does NICE recommend for monitoring renal function?"

Classification:

`Medical / Clinical`

---

# 9. Classification Logic

The baseline classifier follows this logic:

1. Normalize the input.
2. Detect high-risk signals.
3. If high-risk signals are present:
   - classify as `High-risk`.
4. Otherwise detect medical/clinical signals.
5. If clinical signals are present:
   - classify as `Medical / Clinical`.
6. Otherwise determine whether the question is within the supported scope.
7. If it is outside the scope:
   - classify as `Out-of-scope`.
8. Otherwise:
   - classify as `Normal`.

---

# 10. Important Safety Principle

Risk classification is not a diagnosis.

The classifier does not determine:

- whether the user actually has heart failure.
- whether a medical condition is present.
- whether a medication is clinically appropriate.
- whether an emergency is actually occurring.

It only determines how the system should safely handle the request.

---

# 11. Limitations

The baseline implementation is rule-based.

Therefore:

- It may miss unusual wording.
- It may produce false positives.
- It may produce false negatives.
- It should not be treated as a medical diagnostic system.
- High-risk cases should be handled conservatively.

Future versions may combine:

- Rule-based detection.
- A trained classifier.
- An LLM-based safety classifier.
- Human review for uncertain cases.

However, any future classifier should preserve the same safety policy.

---

# 12. Core Safety Principle

> When there is uncertainty about whether a request is high-risk, prefer the safer classification and avoid providing personalized medical instructions.