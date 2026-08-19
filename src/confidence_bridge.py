"""
Confidence Bridge — Convert retrieval output into EvidenceInput.

Bridges the gap between the retrieval pipeline (chunk_ids + CE scores)
and the confidence policy module (EvidenceInput dataclass).

DO NOT MODIFY confidence_policy.py.
"""
import statistics
from typing import Dict, List, Optional

from confidence_policy import EvidenceInput


# CE score normalization constants.
# cross-encoder/ms-marco-MiniLM-L-6-v2 outputs logits in roughly [-5, 12].
# These bounds are empirical from the locked hyperparameter search.
_CE_SCORE_MIN = -5.0
_CE_SCORE_MAX = 12.0
_CE_SCORE_RANGE = _CE_SCORE_MAX - _CE_SCORE_MIN

# Threshold above which a CE score is considered "supporting" (relevant).
_CE_RELEVANCE_THRESHOLD = 3.0


def _normalize_ce(score: float) -> float:
    """Normalize a raw CE logit to [0, 1]."""
    return max(0.0, min(1.0, (score - _CE_SCORE_MIN) / _CE_SCORE_RANGE))


def build_evidence_input(
    chunk_ids: List[str],
    ce_scores: List[float],
    full_chunks: Dict[str, dict],
    top_k: int = 3,
) -> EvidenceInput:
    """Build an EvidenceInput from retrieval output for the confidence policy.

    Args:
        chunk_ids: Ranked chunk IDs from the reranker (all 20).
        ce_scores: Corresponding CE scores (all 20).
        full_chunks: dict mapping chunk_id → full chunk metadata dict
            (must include at least "guideline_family", "superseded_by").
        top_k: Number of top chunks to consider (default 3, fallback 5).

    Returns:
        EvidenceInput ready for calculate_confidence().
    """
    n = min(top_k, len(chunk_ids))
    ids = chunk_ids[:n]
    scores = ce_scores[:n]

    # ── Signal 1: Retrieval quality ──────────────────────────────
    # Mean normalized CE score of top-k.
    if n == 0:
        retrieval_quality = 0.0
    else:
        retrieval_quality = sum(_normalize_ce(s) for s in scores) / n

    # ── Signal 2: Evidence coverage ──────────────────────────────
    # Fraction of top-k chunks that exceed the relevance threshold.
    if n == 0:
        coverage = 0.0
    else:
        supporting = sum(1 for s in scores if s > _CE_RELEVANCE_THRESHOLD)
        coverage = supporting / n

    # ── Signal 3: Score agreement ────────────────────────────────
    # Inverse variability. Low stdev → high agreement.
    # If all scores are identical, agreement = 1.0.
    if n <= 1:
        score_agreement = 1.0
    else:
        mean_abs = statistics.mean(abs(s) for s in scores)
        if mean_abs < 1e-9:
            score_agreement = 0.0
        else:
            stdev = statistics.stdev(scores)
            score_agreement = max(0.0, 1.0 - (stdev / mean_abs))

    # ── Signal 4: Source consistency ─────────────────────────────
    # Fraction of top-k chunks from the same guideline family.
    families = []
    for cid in ids:
        cm = full_chunks.get(cid, {})
        families.append(cm.get("guideline_family", "unknown"))

    if n == 0:
        source_consistency = 0.0
    else:
        from collections import Counter
        counts = Counter(families)
        most_common_count = counts.most_common(1)[0][1]
        source_consistency = most_common_count / n

    # ── Supporting chunks count ──────────────────────────────────
    supporting_chunks = sum(1 for s in scores if s > _CE_RELEVANCE_THRESHOLD)

    # ── Conflicting sources ──────────────────────────────────────
    # True if any retrieved chunk supersedes another retrieved chunk,
    # or if chunks come from conflicting guideline versions.
    conflicting = False
    superseded_chunks = set()
    parent_chunks = set()
    for cid in ids:
        cm = full_chunks.get(cid, {})
        sup = cm.get("superseded_by", "")
        par = cm.get("parent_guideline", "")
        if sup:
            superseded_chunks.add(cid)
        if par:
            parent_chunks.add(cid)

    # If we have chunks that are superseded AND chunks that are their
    # successors, flag conflict.
    if superseded_chunks and parent_chunks:
        conflicting = True

    # Also flag if chunks come from >2 distinct guideline families.
    unique_families = set(families)
    if len(unique_families) > 2:
        conflicting = True

    return EvidenceInput(
        retrieval_quality=round(retrieval_quality, 4),
        coverage=round(coverage, 4),
        score_agreement=round(score_agreement, 4),
        source_consistency=round(source_consistency, 4),
        supporting_chunks=supporting_chunks,
        conflicting_sources=conflicting,
    )
