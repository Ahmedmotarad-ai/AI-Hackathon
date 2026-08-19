"""
Risk Classification and Safety Gate

This module provides a simple rule-based baseline classifier
for the Heart Failure Clinical RAG system.

Categories:
    - Normal
    - Medical / Clinical
    - High-risk
    - Out-of-scope
"""

import re
from enum import Enum
from dataclasses import dataclass
from typing import List


class RiskCategory(str, Enum):
    NORMAL = "Normal"
    MEDICAL = "Medical / Clinical"
    HIGH_RISK = "High-risk"
    OUT_OF_SCOPE = "Out-of-scope"


class Action(str, Enum):
    NORMAL_RESPONSE = "Answer normally"
    RETRIEVE_AND_GROUND = "Retrieve + grounded answer"
    SAFETY_RESPONSE = "Safety response"
    REDIRECT = "Reject / redirect"


@dataclass
class ClassificationResult:
    category: RiskCategory
    action: Action
    matched_signals: List[str]
    reason: str


# ---------------------------------------------------------
# High-risk signals
# ---------------------------------------------------------

HIGH_RISK_PATTERNS = [
    r"\bsevere chest pain\b",
    r"\bchest pain\b",
    r"\bsevere shortness of breath\b",
    r"\bdifficulty breathing\b",
    r"\bcan'?t breathe\b",
    r"\bshortness of breath\b",
    r"\bloss of consciousness\b",
    r"\bfainted\b",
    r"\bfainting\b",
    r"\bblue lips\b",
    r"\bsudden(ly)? (worse|deteriorat)\b",

    # Personalized medication changes
    r"\bshould i (stop|skip|double|increase|decrease|change)\b.*\b(medication|medicine|dose|drug)\b",
    r"\bcan i (stop|skip|double|increase|decrease|change)\b.*\b(medication|medicine|dose|drug)\b",
    r"\bmy (medication|medicine|dose)\b.*\b(stop|skip|increase|decrease|change)\b",
    r"\bwhat should i take right now\b",
    r"\bwhat medication should i take\b",
]


# ---------------------------------------------------------
# Medical / Clinical signals
# ---------------------------------------------------------

MEDICAL_PATTERNS = [
    r"\bheart failure\b",
    r"\bhfref\b",
    r"\bhfmr?ef\b",
    r"\bhfpef\b",
    r"\bnt[- ]?probnp\b",
    r"\bprobnp\b",
    r"\bbnp\b",
    r"\bejection fraction\b",
    r"\bechocardiograph(y|ic)\b",
    r"\bace inhibitor\b",
    r"\barni\b",
    r"\barb\b",
    r"\bmra\b",
    r"\bbeta[- ]?blocker\b",
    r"\bdiuretic\b",
    r"\brenal function\b",
    r"\belectrolytes?\b",
    r"\bclinical guideline\b",
    r"\bguideline\b",
    r"\bdiagnos(is|tic)\b",
    r"\bmonitor(ing)?\b",
    r"\bfollow[- ]?up\b",
    r"\btreatment recommendation\b",
]


# ---------------------------------------------------------
# General technical / supported signals
# ---------------------------------------------------------

NORMAL_PATTERNS = [
    r"\brag\b",
    r"\bretrieval\b",
    r"\bvector database\b",
    r"\bvector db\b",
    r"\bembedding(s)?\b",
    r"\bpython\b",
    r"\bmachine learning\b",
    r"\bartificial intelligence\b",
]


def normalize_text(text: str) -> str:
    """
    Normalize input text for simple pattern matching.
    """
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def find_matches(text: str, patterns: List[str]) -> List[str]:
    """
    Return the patterns that match the input.
    """
    matches = []

    for pattern in patterns:
        if re.search(pattern, text, flags=re.IGNORECASE):
            matches.append(pattern)

    return matches


def classify_risk(user_input: str) -> ClassificationResult:
    """
    Classify a user question according to the project safety policy.

    Priority:
        High-risk
        >
        Medical / Clinical
        >
        Normal / Out-of-scope
    """

    if not user_input or not user_input.strip():
        return ClassificationResult(
            category=RiskCategory.OUT_OF_SCOPE,
            action=Action.REDIRECT,
            matched_signals=[],
            reason="Empty or invalid input."
        )

    text = normalize_text(user_input)

    # -----------------------------------------------------
    # 1. High-risk detection
    # -----------------------------------------------------

    high_risk_matches = find_matches(text, HIGH_RISK_PATTERNS)

    if high_risk_matches:
        return ClassificationResult(
            category=RiskCategory.HIGH_RISK,
            action=Action.SAFETY_RESPONSE,
            matched_signals=high_risk_matches,
            reason=(
                "The input contains potential emergency symptoms "
                "or a personalized medication/treatment request."
            )
        )

    # -----------------------------------------------------
    # 2. Medical / Clinical detection
    # -----------------------------------------------------

    medical_matches = find_matches(text, MEDICAL_PATTERNS)

    if medical_matches:
        return ClassificationResult(
            category=RiskCategory.MEDICAL,
            action=Action.RETRIEVE_AND_GROUND,
            matched_signals=medical_matches,
            reason=(
                "The input contains clinical or heart-failure "
                "guideline-related content."
            )
        )

    # -----------------------------------------------------
    # 3. Normal detection
    # -----------------------------------------------------

    normal_matches = find_matches(text, NORMAL_PATTERNS)

    if normal_matches:
        return ClassificationResult(
            category=RiskCategory.NORMAL,
            action=Action.NORMAL_RESPONSE,
            matched_signals=normal_matches,
            reason=(
                "The input is a general technical question "
                "and does not contain high-risk clinical content."
            )
        )

    # -----------------------------------------------------
    # 4. Default: Out-of-scope
    # -----------------------------------------------------

    return ClassificationResult(
        category=RiskCategory.OUT_OF_SCOPE,
        action=Action.REDIRECT,
        matched_signals=[],
        reason=(
            "The input does not match the supported clinical "
            "or general technical scope."
        )
    )


def safety_gate(user_input: str) -> ClassificationResult:
    """
    Run classification and return the action that the system
    should take before retrieval/generation.
    """
    return classify_risk(user_input)


# ---------------------------------------------------------
# Example usage
# ---------------------------------------------------------

if __name__ == "__main__":

    test_inputs = [
        "What is RAG?",
        "What is a vector database?",
        "What does NICE recommend for monitoring stable heart failure?",
        "What does the guideline say about NT-proBNP?",
        "I have severe chest pain and difficulty breathing. What should I take right now?",
        "Should I double my heart failure medication dose?",
        "Write a Python game for me.",
        "What is the weather today?",
    ]

    for question in test_inputs:

        result = safety_gate(question)

        print("=" * 70)
        print(f"Input:    {question}")
        print(f"Category: {result.category.value}")
        print(f"Action:   {result.action.value}")
        print(f"Reason:   {result.reason}")

        if result.matched_signals:
            print(f"Signals:  {len(result.matched_signals)}")