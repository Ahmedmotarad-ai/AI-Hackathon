"""
Person 2: Retrieval Confidence + Evidence Sufficiency

Integration layer for the RAG pipeline.

Important: reranker / retrieval scores are relevance signals, not calibrated
probabilities. The policy combines multiple evidence signals and applies hard
gates before allowing an answer.
"""

import json
import os
from dataclasses import dataclass
from typing import List, Optional


# Default location of the JSON config shipped alongside this module. Callers
# can override with an explicit path when calling load_config_from_json().
_DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "confidence_policy_config.json"
)


@dataclass
class ConfidenceConfig:
    high_threshold: float = 0.75
    medium_threshold: float = 0.50
    retrieval_weight: float = 0.30
    coverage_weight: float = 0.30
    agreement_weight: float = 0.20
    consistency_weight: float = 0.20
    min_chunks_for_high: int = 2
    min_coverage_for_high: float = 0.80
    min_coverage_for_medium: float = 0.50


def load_config_from_json(path: Optional[str] = None) -> ConfidenceConfig:
    """Build a ConfidenceConfig from confidence_policy_config.json.

    This makes the JSON file the actual source of truth for thresholds and
    weights instead of just documentation that can drift out of sync with the
    hardcoded dataclass defaults.

    Args:
        path: Optional path to a config JSON file. Defaults to
            confidence_policy_config.json next to this module.

    Raises:
        FileNotFoundError: if the config file does not exist.
        ValueError: if the config file is missing required keys or the
            weights don't sum to 1.0 (within floating point tolerance).
    """
    path = path or _DEFAULT_CONFIG_PATH

    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    try:
        weights = raw["weights"]
        evidence = raw["evidence"]
        config = ConfidenceConfig(
            high_threshold=raw["high_threshold"],
            medium_threshold=raw["medium_threshold"],
            retrieval_weight=weights["retrieval_quality"],
            coverage_weight=weights["evidence_coverage"],
            agreement_weight=weights["score_agreement"],
            consistency_weight=weights["source_consistency"],
            min_chunks_for_high=evidence["min_chunks_for_high"],
            min_coverage_for_high=evidence["min_coverage_for_high"],
            min_coverage_for_medium=evidence["min_coverage_for_medium"],
        )
    except KeyError as exc:
        raise ValueError(f"Config file {path} is missing required key: {exc}") from exc

    weight_sum = (
        config.retrieval_weight
        + config.coverage_weight
        + config.agreement_weight
        + config.consistency_weight
    )
    if abs(weight_sum - 1.0) > 1e-6:
        raise ValueError(
            f"Confidence weights in {path} must sum to 1.0, got {weight_sum}"
        )

    return config


@dataclass
class EvidenceInput:
    retrieval_quality: float
    coverage: float
    score_agreement: float
    source_consistency: float
    supporting_chunks: int
    conflicting_sources: bool = False


@dataclass
class ConfidenceResult:
    score: float
    level: str
    decision: str
    sufficient: bool
    reasons: List[str]


def calculate_confidence(
    evidence: EvidenceInput,
    config: ConfidenceConfig | None = None,
) -> ConfidenceResult:
    if config is None:
        try:
            config = load_config_from_json()
        except (FileNotFoundError, ValueError):
            # Fall back to hardcoded defaults if the JSON isn't shipped
            # alongside the module (e.g. module used standalone).
            config = ConfidenceConfig()

    score = (
        config.retrieval_weight * evidence.retrieval_quality
        + config.coverage_weight * evidence.coverage
        + config.agreement_weight * evidence.score_agreement
        + config.consistency_weight * evidence.source_consistency
    )
    score = max(0.0, min(1.0, score))

    if evidence.supporting_chunks <= 0:
        return ConfidenceResult(
            score, "low", "refuse", False,
            ["No supporting evidence was retrieved."]
        )

    if evidence.coverage < config.min_coverage_for_medium:
        return ConfidenceResult(
            score, "low", "refuse", False,
            ["Evidence coverage is below the minimum medium-confidence threshold."]
        )

    if evidence.conflicting_sources:
        return ConfidenceResult(
            min(score, config.medium_threshold - 0.001),
            "medium",
            "caution",
            True,
            ["Conflicting source versions require a cautious, source-separated answer."]
        )

    if (
        score >= config.high_threshold
        and evidence.supporting_chunks >= config.min_chunks_for_high
        and evidence.coverage >= config.min_coverage_for_high
    ):
        return ConfidenceResult(
            score, "high", "answer", True,
            [
                "Multiple supporting chunks are available.",
                "Evidence coverage meets the high-confidence threshold.",
                "No unresolved source conflict was detected.",
            ],
        )

    if score >= config.medium_threshold:
        return ConfidenceResult(
            score, "medium", "caution", True,
            ["Evidence is useful but does not satisfy all high-confidence gates."]
        )

    return ConfidenceResult(
        score, "low", "refuse", False,
        ["Weighted confidence is below the medium threshold."]
    )
