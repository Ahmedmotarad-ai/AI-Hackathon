"""
RAG Pipeline — End-to-end orchestrator.

Wires together all components:
  safety_gate → retrieval → confidence_bridge → confidence_policy →
  refusal_handler → context_builder → prompt_assembler → llm_client →
  citation_extractor

DO NOT MODIFY: finalize_retrieval.py, confidence_policy.py
"""
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder

from safety_gate import classify_query, SafetyDecision
from confidence_bridge import build_evidence_input
from confidence_policy import calculate_confidence, ConfidenceResult, load_config_from_json
from refusal_handler import get_refusal_decision, RefusalDecision
from context_builder import load_all_chunks, build_context
from prompt_assembler import build_messages, load_system_prompt
from llm_client import call_llm, LLMResponse
from citation_extractor import parse_llm_response, ParsedResponse


# ── Configuration (locked) ───────────────────────────────────
_SRC_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = _SRC_DIR.parent

DB_PATH = _PROJECT_DIR / "data" / "vector_db"
COLLECTION_NAME = "medical_guidelines"
BGE_MODEL = "BAAI/bge-small-en-v1.5"
CE_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
QUERY_PREFIX = "query: "
CANDIDATE_K = 20
DEFAULT_TOP_K = 3
FALLBACK_TOP_K = 5


@dataclass
class PipelineConfig:
    """Configuration for a single pipeline run."""
    top_k: int = DEFAULT_TOP_K
    fallback_top_k: int = FALLBACK_TOP_K
    llm_provider: str = "openai"      # "openai" | "anthropic"
    llm_model: Optional[str] = None   # None → provider default
    llm_max_tokens: int = 1024
    llm_temperature: float = 0.0
    llm_timeout: float = 30.0
    max_context_chars: Optional[int] = None  # None → no limit


@dataclass
class PipelineResult:
    """Full result from a single pipeline run."""
    query: str
    # Safety
    safety_decision: SafetyDecision = None
    # Retrieval
    retrieved_chunk_ids: List[str] = field(default_factory=list)
    retrieved_ce_scores: List[float] = field(default_factory=list)
    retrieval_latency_ms: float = 0.0
    # Confidence
    confidence_result: ConfidenceResult = None
    # Refusal
    refusal_decision: RefusalDecision = None
    # LLM
    llm_response: LLMResponse = None
    # Citation
    parsed_response: ParsedResponse = None
    # Timing
    total_latency_ms: float = 0.0
    # Error
    error: Optional[str] = None


class RAGPipeline:
    """End-to-end RAG pipeline for heart failure guideline queries."""

    def __init__(self, config: Optional[PipelineConfig] = None):
        """Initialize the pipeline. Loads ML models once.

        Args:
            config: Pipeline configuration. Uses defaults if None.
        """
        self.config = config or PipelineConfig()
        self._all_chunks = None
        self._bge = None
        self._reranker = None
        self._chroma_coll = None

    def _ensure_models_loaded(self):
        """Lazy-load ML models and data on first use."""
        if self._bge is not None:
            return

        print("Loading retrieval models...")
        self._bge = SentenceTransformer(BGE_MODEL, device="cpu")
        self._reranker = CrossEncoder(CE_MODEL, max_length=512)

        print("Loading chunks metadata...")
        self._all_chunks = load_all_chunks()

        print("Connecting to ChromaDB...")
        client = chromadb.PersistentClient(path=str(DB_PATH))
        self._chroma_coll = client.get_collection(name=COLLECTION_NAME)
        print(f"ChromaDB ready: {self._chroma_coll.count()} records")

    def _retrieve(self, query: str) -> tuple:
        """Run dense retrieval + CE reranking.

        Returns:
            (chunk_ids, ce_scores, latency_ms)
        """
        self._ensure_models_loaded()

        fq = QUERY_PREFIX + query
        t0 = time.perf_counter()
        emb = self._bge.encode(fq, normalize_embeddings=True).tolist()
        res = self._chroma_coll.query(
            query_embeddings=[emb],
            n_results=CANDIDATE_K,
            where={"section": {"$ne": "Front matter"}},
        )
        t1 = time.perf_counter()
        dense_ms = (t1 - t0) * 1000

        cand_ids = res["ids"][0]

        t2 = time.perf_counter()
        cand_texts = [self._all_chunks.get(cid, {}).get("text", "") for cid in cand_ids]
        pairs = [(query, ct) for ct in cand_texts]
        ce_scores = self._reranker.predict(pairs)
        t3 = time.perf_counter()
        rerank_ms = (t3 - t2) * 1000

        # Sort by CE score descending
        scored = list(zip(cand_ids, ce_scores))
        scored.sort(key=lambda x: -x[1])

        chunk_ids = [s[0] for s in scored]
        scores = [float(s[1]) for s in scored]
        total_ms = dense_ms + rerank_ms

        return chunk_ids, scores, total_ms

    def run(self, query: str) -> PipelineResult:
        """Execute the full RAG pipeline on a single query.

        Args:
            query: Clinical question (from ML pipeline or user).

        Returns:
            PipelineResult with all intermediate and final outputs.
        """
        pipeline_start = time.perf_counter()
        result = PipelineResult(query=query)

        # ── Step 1: Safety gate ────────────────────────────────
        safety = classify_query(query)
        result.safety_decision = safety
        if not safety.safe:
            result.total_latency_ms = (time.perf_counter() - pipeline_start) * 1000
            return result

        # ── Step 2: Retrieval ──────────────────────────────────
        try:
            chunk_ids, ce_scores, ret_ms = self._retrieve(query)
        except Exception as exc:
            result.error = f"Retrieval failed: {exc}"
            result.total_latency_ms = (time.perf_counter() - pipeline_start) * 1000
            return result

        result.retrieved_chunk_ids = chunk_ids
        result.retrieved_ce_scores = ce_scores
        result.retrieval_latency_ms = ret_ms

        # ── Step 3: Confidence assessment ──────────────────────
        top_k = self.config.top_k

        evidence_input = build_evidence_input(
            chunk_ids, ce_scores, self._all_chunks, top_k=top_k,
        )
        confidence = calculate_confidence(evidence_input)
        result.confidence_result = confidence

        # ── Step 4: Refusal decision ───────────────────────────
        refusal = get_refusal_decision(confidence)
        result.refusal_decision = refusal

        if not refusal.proceed_to_llm:
            result.total_latency_ms = (time.perf_counter() - pipeline_start) * 1000
            return result

        # ── Step 5: Context building ───────────────────────────
        # Use fallback_top_k if confidence is medium (caution)
        effective_top_k = top_k
        if refusal.action == "caution":
            effective_top_k = self.config.fallback_top_k

        context = build_context(
            chunk_ids, ce_scores, self._all_chunks,
            top_k=effective_top_k,
            max_chars=self.config.max_context_chars,
        )

        # ── Step 6: Prompt assembly ────────────────────────────
        caution_prefix = refusal.message if refusal.action == "caution" else None
        messages = build_messages(
            query=query,
            context=context,
            caution_prefix=caution_prefix,
        )

        # ── Step 7: LLM call ──────────────────────────────────
        llm_resp = call_llm(
            messages=messages,
            provider=self.config.llm_provider,
            model=self.config.llm_model,
            max_tokens=self.config.llm_max_tokens,
            temperature=self.config.llm_temperature,
            timeout=self.config.llm_timeout,
        )
        result.llm_response = llm_resp

        if llm_resp.error:
            result.error = llm_resp.error
            result.total_latency_ms = (time.perf_counter() - pipeline_start) * 1000
            return result

        # ── Step 8: Citation extraction ────────────────────────
        parsed = parse_llm_response(
            raw_text=llm_resp.text,
            retrieved_chunk_ids=chunk_ids[:effective_top_k],
            chunk_metadata=self._all_chunks,
        )
        result.parsed_response = parsed

        result.total_latency_ms = (time.perf_counter() - pipeline_start) * 1000
        return result


def run_pipeline(
    query: str,
    config: Optional[PipelineConfig] = None,
    pipeline: Optional[RAGPipeline] = None,
) -> PipelineResult:
    """Convenience function to run the pipeline on a single query.

    Args:
        query: Clinical question.
        config: Optional pipeline config.
        pipeline: Optional pre-initialized pipeline (avoids model reload).

    Returns:
        PipelineResult.
    """
    if pipeline is None:
        pipeline = RAGPipeline(config)
    return pipeline.run(query)


def run_pipeline_batch(
    queries: List[str],
    config: Optional[PipelineConfig] = None,
) -> List[PipelineResult]:
    """Run the pipeline on multiple queries, reusing loaded models.

    Args:
        queries: List of clinical questions.
        config: Optional pipeline config.

    Returns:
        List of PipelineResult objects.
    """
    pipe = RAGPipeline(config)
    results = []
    for i, q in enumerate(queries):
        print(f"  [{i+1}/{len(queries)}] {q[:60]}...")
        results.append(pipe.run(q))
    return results
