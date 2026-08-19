# Person 2: Retrieval Confidence + Evidence Sufficiency

## Purpose

This integration adds the decision layer between retrieval/reranking and answer
generation.

```text
Query
  ↓
Retriever
  ↓
Reranker
  ↓
Confidence + Evidence Sufficiency
  ├── High   → Answer
  ├── Medium → Caution
  └── Low    → Refuse
```

## Confidence model

Final confidence is a weighted combination:

- Retrieval quality: 30%
- Evidence coverage: 30%
- Score agreement: 20%
- Source consistency: 20%

Thresholds:

- High: >= 0.75
- Medium: 0.50–0.74
- Low: < 0.50

These are operational policy thresholds, not calibrated probabilities.

## Evidence gates

High confidence additionally requires:

- at least 2 supporting chunks
- coverage >= 0.80
- no unresolved source conflict

Low confidence is forced when:

- there are no supporting chunks
- coverage < 0.50
- weighted confidence < 0.50

Conflicting guideline/source versions force Caution so recommendations are not
silently merged.

## Integration API

Use `calculate_confidence(EvidenceInput(...))` after retrieval/reranking and
before generation. The returned `decision` is one of:

- `answer`
- `caution`
- `refuse`

The implementation intentionally does not interpret a raw cross-encoder score
as a probability.
