"""
Safety Gate — Pre-retrieval query classification.

Rule-based MVP. Classifies incoming queries before they reach retrieval.
No LLM calls. No external dependencies beyond stdlib.
"""
from dataclasses import dataclass
from typing import List


@dataclass
class SafetyDecision:
    classification: str    # "normal" | "medical_clinical" | "high_risk" | "out_of_scope"
    safe: bool             # True → proceed to retrieval, False → block
    reason: str            # human-readable explanation
    blocked_response: str  # refusal message if safe=False, empty string if safe=True


# ── Keyword Sets ─────────────────────────────────────────────────
# Each set is lowercase. Matching is case-insensitive substring.

HIGH_RISK_KEYWORDS = [
    "overdose", "overdosing", "self-harm", "suicide", "suicidal",
    "kill myself", "end my life", "die", "dying",
    "how much", "what dose", "dose of", "dosage",
    "give me", "inject", "take too much", "took too much",
    "poisoning", "toxic", "emergency", "call ambulance",
    "acute mi", "cardiac arrest", "resuscitation",
    "what should i take", "recommend a dose",
    "how many mg", "how many tablets",
]

OUT_OF_SCOPE_KEYWORDS = [
    "diabetes", "insulin", "blood sugar",
    "cancer", "chemotherapy", "tumor",
    "pregnancy", "pregnant", "trimester",
    "asthma", "inhaler", "bronchitis",
    "antibiotic", "infection", "covid", "covid-19",
    "mental health", "depression", "antidepressant",
    "orthopedic", "fracture", "broken bone",
    "dermatology", "skin rash", "eczema",
    "pediatric", "infant",
]

AMBIGUOUS_KEYWORDS = [
    "tell me about", "what is", "explain", "describe",
    "give me information", "i want to know", "what are",
    "how does", "can you tell", "what do you know about",
]


def classify_query(query: str) -> SafetyDecision:
    """Classify a user query into a safety category.

    Args:
        query: The raw user query string.

    Returns:
        SafetyDecision with classification, safety flag, and optional
        refusal response.
    """
    q = query.lower().strip()

    if not q:
        return SafetyDecision(
            classification="out_of_scope",
            safe=False,
            reason="Empty query.",
            blocked_response="Please provide a question about heart failure guidelines.",
        )

    # 1. High-risk check (most restrictive — checked first)
    for kw in HIGH_RISK_KEYWORDS:
        if kw in q:
            return SafetyDecision(
                classification="high_risk",
                safe=False,
                reason=f"High-risk keyword detected: '{kw}'",
                blocked_response=(
                    "I cannot provide information about medication dosing, "
                    "overdose, self-harm, or emergency situations. "
                    "Please consult a qualified healthcare professional or "
                    "call emergency services if this is an urgent situation."
                ),
            )

    # 2. Out-of-scope check
    for kw in OUT_OF_SCOPE_KEYWORDS:
        if kw in q:
            return SafetyDecision(
                classification="out_of_scope",
                safe=False,
                reason=f"Out-of-scope keyword detected: '{kw}'",
                blocked_response=(
                    "I can only answer questions about heart failure diagnosis "
                    "and management based on the NICE and ESC guidelines in my "
                    "knowledge base. Please ask a heart failure-related question."
                ),
            )

    # 3. Medical/clinical vs normal
    medical_keywords = [
        "heart failure", "hf", "hfref", "hfpef", "hfmr ef",
        "ejection fraction", "ef", "nyha", "bnp", "nt-probnp",
        "ace inhibitor", "arb", "arni", "beta blocker", "bb",
        "mra", "spironolactone", "eplerenone", "sacubitril",
        "valsartan", "diuretic", "furosemide", " bumetanide",
        "digoxin", "hydralazine", "nitrate", "ivabradine",
        "CRT", "ICD", "device therapy",
        "cardiac", "myocardial", "ventricular", "atrial",
        "systolic", "diastolic", "remodeling", "fibrosis",
        "guideline", "nice", "esc", "classification",
        "diagnosis", "treatment", "management", "therapy",
        "prognosis", "comorbidity", "comorbidities",
        "chloride", "sodium", "renal", "kidney", "ckd",
        "iron", "ferritin", "transferrin", "anaemia", "anemia",
        "vaccination", "influenza", "pneumococcal",
        "palliative", "end of life", "advanced heart failure",
        "transplant", "LVAD", "mechanical support",
    ]

    is_medical = any(kw in q for kw in medical_keywords)

    if is_medical:
        # Check if it's also ambiguous
        is_ambiguous = any(kw in q for kw in AMBIGUOUS_KEYWORDS)
        classification = "medical_clinical"
        # Ambiguous medical queries are still safe — retrieval handles vagueness
        return SafetyDecision(
            classification=classification,
            safe=True,
            reason="Medical/clinical query about heart failure." + (
                " Query may be ambiguous." if is_ambiguous else ""
            ),
            blocked_response="",
        )

    # 4. Normal (non-medical but not harmful, not out-of-scope)
    return SafetyDecision(
        classification="normal",
        safe=True,
        reason="General query, not clearly medical or harmful.",
        blocked_response="",
    )
