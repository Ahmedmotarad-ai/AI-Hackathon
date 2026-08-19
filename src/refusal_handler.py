"""
Refusal Handler — Post-retrieval answer/caution/refuse decision.

Consumes ConfidenceResult from confidence_policy and produces a
 RefusalDecision that tells the pipeline whether to proceed with
LLM generation or return a safe fallback.
"""
from dataclasses import dataclass
from typing import Optional

from confidence_policy import ConfidenceResult


_REFUSAL_MESSAGE = (
    "I don't have enough information in the retrieved guideline context "
    "to answer that reliably."
)

_CAUTION_PREFIX = (
    "The following answer is based on limited or partially conflicting "
    "guideline evidence. Please verify with the original source documents.\n\n"
)


@dataclass
class RefusalDecision:
    action: str                     # "answer" | "caution" | "refuse"
    proceed_to_llm: bool            # True → generate, False → skip LLM
    message: Optional[str]          # Refusal/caution message (None for answer)
    confidence_level: str           # Forwarded from ConfidenceResult
    confidence_score: float         # Forwarded from ConfidenceResult
    reasons: list                   # Forwarded from ConfidenceResult


def get_refusal_decision(confidence_result: ConfidenceResult) -> RefusalDecision:
    """Decide whether to answer, caution, or refuse based on confidence.

    Args:
        confidence_result: Output from confidence_policy.calculate_confidence().

    Returns:
        RefusalDecision with action, LLM routing, and messages.
    """
    if confidence_result.decision == "answer":
        return RefusalDecision(
            action="answer",
            proceed_to_llm=True,
            message=None,
            confidence_level=confidence_result.level,
            confidence_score=confidence_result.score,
            reasons=confidence_result.reasons,
        )

    if confidence_result.decision == "caution":
        return RefusalDecision(
            action="caution",
            proceed_to_llm=True,
            message=_CAUTION_PREFIX,
            confidence_level=confidence_result.level,
            confidence_score=confidence_result.score,
            reasons=confidence_result.reasons,
        )

    # decision == "refuse"
    return RefusalDecision(
        action="refuse",
        proceed_to_llm=False,
        message=_REFUSAL_MESSAGE,
        confidence_level=confidence_result.level,
        confidence_score=confidence_result.score,
        reasons=confidence_result.reasons,
    )
