"""
ML -> RAG Adapter

Sits between the ML clinical event pipeline and the RAG retrieval pipeline.
Transforms narrative ML clinical queries into retrieval-optimized guideline
questions that match the terminology indexed in the medical guideline corpus.

DO NOT MODIFY: rag_pipeline.py, finalize_retrieval.py, confidence_policy.py,
safety_gate.py, refusal_handler.py, or any retrieval/reranking components.
"""
import csv
import json
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional


# ── Path setup ──────────────────────────────────────────────────
_SRC_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = _SRC_DIR.parent
_ML_PROCESSED = _PROJECT_DIR / "ML" / "data" / "processed"


# ── Data Contracts ──────────────────────────────────────────────

@dataclass
class MLClinicalEvent:
    """Structured representation of one ML clinical event."""
    event_id: str
    subject_id: str
    event_type: str           # sustained_elevation | sustained_reduction | sudden_change |
                              # recurring_elevation | recurring_reduction | unusual_variability
    clinical_query: str       # Original ML-generated narrative query
    start_time: str = ""
    end_time: str = ""
    duration_minutes: float = 0.0
    personal_baseline_hr: float = 0.0
    contextual_baseline_hr: float = 0.0
    observed_mean_hr: float = 0.0
    observed_median_hr: float = 0.0
    observed_min_hr: float = 0.0
    observed_max_hr: float = 0.0
    deviation_bpm: float = 0.0
    relative_deviation_pct: float = 0.0
    peak_hr: float = 0.0
    peak_time: str = ""
    baseline_reliability: str = ""
    event_confidence: str = ""
    n_observations: int = 0
    n_occurrences: int = 0


@dataclass
class MLRAGResult:
    """Output of the ML->RAG adapter for a single event."""
    event_id: str
    subject_id: str
    event_type: str
    original_query: str
    retrieval_query: str
    rag_result: object = None      # PipelineResult from rag_pipeline
    integration_status: str = ""   # SUCCESS | LOW_CONFIDENCE | REFUSED | ERROR


# ── Query Templates ─────────────────────────────────────────────
# Terminology grounded in actual corpus analysis:
#   Indexed: "heart failure" (492), "tachycardia" (10), "bradycardia" (8),
#            "heart rate" (22), "rate control" (16), "arrhythmia" (62),
#            "atrial fibrillation" (44), "sustained" (12), "recurrent" (21),
#            "sudden" (37), "resting heart rate" (2)
#   NOT indexed (avoid): "heart rate variability" (0), "elevated heart rate" (0),
#                        "low heart rate" (0), "chronotropic" (0)

_QUERY_TEMPLATES = {
    "sustained_elevation": (
        "What clinical guidance addresses sustained tachycardia or persistently "
        "elevated resting heart rate in patients with heart failure, and what "
        "factors should be considered when evaluating sustained heart rate "
        "elevation?"
    ),
    "sustained_reduction": (
        "What clinical guidance addresses sustained bradycardia or persistently "
        "low resting heart rate in patients with heart failure, and what "
        "factors should be considered when evaluating sustained heart rate "
        "reduction?"
    ),
    "sudden_change": (
        "What clinical guidance addresses sudden changes in heart rate and "
        "their clinical significance in patients with heart failure, including "
        "hemodynamic assessment and risk stratification?"
    ),
    "recurring_elevation": (
        "What clinical guidance addresses recurrent episodes of tachycardia "
        "or elevated heart rate in patients with heart failure, and what "
        "management approach is recommended for recurring heart rate "
        "elevation?"
    ),
    "recurring_reduction": (
        "What clinical guidance addresses recurrent episodes of bradycardia "
        "or low heart rate in patients with heart failure, and what management "
        "approach is recommended for recurring heart rate reduction?"
    ),
    "unusual_variability": (
        "What clinical guidance addresses abnormal or irregular heart rate "
        "patterns in patients with heart failure, including arrhythmia "
        "assessment and rate control considerations?"
    ),
}


# ── Core Functions ──────────────────────────────────────────────

def _safe_float(val, default=0.0) -> float:
    """Convert a string or numeric value to float safely."""
    if val is None or val == "":
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def _safe_int(val, default=0) -> int:
    """Convert a string or numeric value to int safely."""
    if val is None or val == "":
        return default
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return default


def load_events_from_csv(
    queries_path: Optional[Path] = None,
    events_path: Optional[Path] = None,
) -> List[MLClinicalEvent]:
    """Load ML clinical events by joining clinical_queries + anomaly_events.

    Args:
        queries_path: Path to stage3_clinical_queries.csv.
        events_path: Path to stage3_anomaly_events.csv.

    Returns:
        List of MLClinicalEvent objects.
    """
    q_path = queries_path or (_ML_PROCESSED / "stage3_clinical_queries.csv")
    e_path = events_path or (_ML_PROCESSED / "stage3_anomaly_events.csv")

    # Load anomaly events keyed by event_id
    events_by_id = {}
    with open(e_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            events_by_id[row["event_id"]] = row

    # Load clinical queries and merge with event metadata
    results = []
    with open(q_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            eid = row["event_id"]
            evt = events_by_id.get(eid, {})
            results.append(MLClinicalEvent(
                event_id=eid,
                subject_id=row.get("subject_id", evt.get("subject_id", "")),
                event_type=row.get("event_type", evt.get("event_type", "")),
                clinical_query=row.get("query", ""),
                start_time=evt.get("start_time", ""),
                end_time=evt.get("end_time", ""),
                duration_minutes=_safe_float(evt.get("duration_minutes")),
                personal_baseline_hr=_safe_float(evt.get("personal_baseline_hr")),
                contextual_baseline_hr=_safe_float(evt.get("contextual_baseline_hr")),
                observed_mean_hr=_safe_float(evt.get("observed_mean_hr")),
                observed_median_hr=_safe_float(evt.get("observed_median_hr")),
                observed_min_hr=_safe_float(evt.get("observed_min_hr")),
                observed_max_hr=_safe_float(evt.get("observed_max_hr")),
                deviation_bpm=_safe_float(evt.get("deviation_bpm")),
                relative_deviation_pct=_safe_float(evt.get("relative_deviation_pct")),
                peak_hr=_safe_float(evt.get("peak_hr")),
                peak_time=evt.get("peak_time", ""),
                baseline_reliability=evt.get("baseline_reliability", ""),
                event_confidence=evt.get("event_confidence", ""),
                n_observations=_safe_int(evt.get("n_observations")),
                n_occurrences=_safe_int(evt.get("n_occurrences")),
            ))
    return results


def transform_query(event: MLClinicalEvent) -> str:
    """Transform an ML clinical event into a retrieval-optimized guideline query.

    Uses deterministic rule-based transformation. No LLM calls.

    The transformed query:
    - Uses terminology present in the indexed corpus (tachycardia, bradycardia,
      heart rate, arrhythmia, heart failure).
    - Preserves clinically relevant signal information (HR values, deviation).
    - Does NOT include patient identifiers, timestamps, or subject_id.
    - Does NOT diagnose or make clinical claims.

    Args:
        event: MLClinicalEvent from the ML pipeline.

    Returns:
        Retrieval-optimized query string.
    """
    base = _QUERY_TEMPLATES.get(event.event_type)
    if not base:
        # Fallback: use the original query as-is
        return event.clinical_query

    # Append structured clinical signal details when available.
    # Only include if values are physiologically meaningful.
    details = []
    if event.observed_mean_hr > 0:
        direction = "+" if event.deviation_bpm >= 0 else ""
        details.append(f"Observed heart rate: {event.observed_mean_hr:.0f} bpm")
    if event.personal_baseline_hr > 0:
        details.append(f"Personal baseline: {event.personal_baseline_hr:.0f} bpm")
    if abs(event.deviation_bpm) > 0:
        direction = "+" if event.deviation_bpm >= 0 else ""
        details.append(f"Deviation: {direction}{event.deviation_bpm:.0f} bpm")
    if event.duration_minutes > 0:
        details.append(f"Duration: {event.duration_minutes:.0f} minutes")

    if details:
        return base + "\n\n" + "\n".join(details)

    return base


def run_event(
    event: MLClinicalEvent,
    pipeline=None,
    config=None,
) -> MLRAGResult:
    """Transform one ML event and run it through the RAG pipeline.

    Args:
        event: MLClinicalEvent from the ML pipeline.
        pipeline: Optional pre-initialized RAGPipeline (avoids model reload).
        config: Optional PipelineConfig.

    Returns:
        MLRAGResult with the RAG output and integration metadata.
    """
    # Lazy import to avoid circular imports and allow testing without RAG loaded
    from rag_pipeline import RAGPipeline, PipelineConfig, PipelineResult

    retrieval_query = transform_query(event)

    result = MLRAGResult(
        event_id=event.event_id,
        subject_id=event.subject_id,
        event_type=event.event_type,
        original_query=event.clinical_query,
        retrieval_query=retrieval_query,
    )

    try:
        if pipeline is None:
            pipeline = RAGPipeline(config)
        rag_result = pipeline.run(retrieval_query)
        result.rag_result = rag_result

        # Determine integration status from RAG result
        if rag_result.error:
            result.integration_status = "ERROR"
        elif rag_result.safety_decision and not rag_result.safety_decision.safe:
            result.integration_status = "REFUSED"
        elif rag_result.refusal_decision and rag_result.refusal_decision.action == "refuse":
            result.integration_status = "REFUSED"
        elif rag_result.confidence_result and rag_result.confidence_result.level == "low":
            result.integration_status = "LOW_CONFIDENCE"
        else:
            result.integration_status = "SUCCESS"

    except Exception as exc:
        result.integration_status = "ERROR"
        result.rag_result = exc

    return result


def run_batch(
    events: List[MLClinicalEvent],
    pipeline=None,
    config=None,
    progress: bool = True,
) -> List[MLRAGResult]:
    """Run a batch of ML events through the adapter + RAG pipeline.

    Args:
        events: List of MLClinicalEvent objects.
        pipeline: Optional pre-initialized RAGPipeline (reused across batch).
        config: Optional PipelineConfig.
        progress: Print progress to stdout.

    Returns:
        List of MLRAGResult objects.
    """
    from rag_pipeline import RAGPipeline, PipelineConfig

    if pipeline is None:
        pipeline = RAGPipeline(config)

    results = []
    n = len(events)
    for i, event in enumerate(events):
        if progress and (i % 10 == 0 or i == n - 1):
            print(f"  [{i+1}/{n}] {event.event_id} [{event.event_type}]")
        results.append(run_event(event, pipeline=pipeline))

    return results


def run_batch_retrieval_only(
    events: List[MLClinicalEvent],
    pipeline=None,
    config=None,
    progress: bool = True,
) -> List[MLRAGResult]:
    """Run a batch through transformation + retrieval only (no LLM).

    This is the primary validation mode. It runs:
    ML event -> query transformation -> retrieval -> reranking -> confidence
    but stops before the LLM call.

    The RAG pipeline already supports this: when refusal.proceed_to_llm=False,
    it returns early without calling the LLM.

    Args:
        events: List of MLClinicalEvent objects.
        pipeline: Optional pre-initialized RAGPipeline.
        config: Optional PipelineConfig.
        progress: Print progress.

    Returns:
        List of MLRAGResult objects.
    """
    # Same as run_batch — the pipeline handles no-LLM mode internally
    return run_batch(events, pipeline=pipeline, config=config, progress=progress)


def results_to_json(results: List[MLRAGResult], path: Path) -> Path:
    """Save batch results to JSON.

    Extracts key fields from PipelineResult to keep the output clean.

    Args:
        results: List of MLRAGResult objects.
        path: Output JSON path.

    Returns:
        Path to saved file.
    """
    output = []
    for r in results:
        rec = {
            "event_id": r.event_id,
            "subject_id": r.subject_id,
            "event_type": r.event_type,
            "original_query": r.original_query,
            "retrieval_query": r.retrieval_query,
            "integration_status": r.integration_status,
        }
        # Extract safe fields from rag_result if available
        rr = r.rag_result
        if hasattr(rr, "retrieved_chunk_ids"):
            rec["retrieved_chunk_ids"] = rr.retrieved_chunk_ids[:5]
            rec["retrieved_ce_scores"] = [round(s, 4) for s in rr.retrieved_ce_scores[:5]]
        if hasattr(rr, "confidence_result") and rr.confidence_result:
            cr = rr.confidence_result
            rec["confidence_score"] = round(cr.score, 4)
            rec["confidence_level"] = cr.level
            rec["confidence_decision"] = cr.decision
            rec["confidence_reasons"] = cr.reasons
        if hasattr(rr, "safety_decision") and rr.safety_decision:
            rec["safety_classification"] = rr.safety_decision.classification
            rec["safety_safe"] = rr.safety_decision.safe
        if hasattr(rr, "refusal_decision") and rr.refusal_decision:
            rec["refusal_action"] = rr.refusal_decision.action
            rec["refusal_proceed_to_llm"] = rr.refusal_decision.proceed_to_llm
        if hasattr(rr, "retrieval_latency_ms"):
            rec["retrieval_latency_ms"] = round(rr.retrieval_latency_ms, 2)
        if hasattr(rr, "total_latency_ms"):
            rec["total_latency_ms"] = round(rr.total_latency_ms, 2)
        if hasattr(rr, "error"):
            rec["error"] = rr.error
        output.append(rec)

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    return path
