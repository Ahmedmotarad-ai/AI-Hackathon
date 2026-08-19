"""
Context Builder — Format retrieved chunks for LLM input.

Loads full chunk metadata from chunks.jsonl and formats the top-k
retrieved chunks into a structured context string for the prompt assembler.
"""
import json
from pathlib import Path
from typing import Dict, List, Optional


CHUNKS_FILE = Path(__file__).resolve().parent.parent / "data" / "chunks" / "chunks.jsonl"


def load_all_chunks(chunks_path: Optional[Path] = None) -> Dict[str, dict]:
    """Load all chunks from chunks.jsonl into a dict keyed by chunk_id.

    Returns:
        dict mapping chunk_id → full chunk metadata dict.
    """
    path = chunks_path or CHUNKS_FILE
    chunks = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                chunks[r["chunk_id"]] = r
    return chunks


def format_chunk_for_llm(chunk: dict, rank: int, ce_score: float) -> str:
    """Format a single chunk into a readable block for the LLM.

    Args:
        chunk: Full chunk metadata dict from chunks.jsonl.
        rank: 1-based rank in the reranked list.
        ce_score: Cross-encoder relevance score.

    Returns:
        Formatted string block.
    """
    doc = chunk.get("document", "Unknown")
    section = chunk.get("section", "Unknown")
    section_path = chunk.get("section_path", [])
    page_start = chunk.get("page_start", "?")
    page_end = chunk.get("page_end", "?")
    family = chunk.get("guideline_family", "")
    year = chunk.get("guideline_year", "")
    chunk_id = chunk.get("chunk_id", "unknown")
    text = chunk.get("text", "").strip()

    path_str = " > ".join(section_path) if isinstance(section_path, list) else section_path

    header = (
        f"[Chunk {rank}] {chunk_id}\n"
        f"Source: {doc} | {family} {year}\n"
        f"Section: {path_str} (pages {page_start}-{page_end})\n"
        f"Relevance score: {ce_score:.2f}\n"
    )
    return f"{header}\n{text}\n"


def build_context(
    chunk_ids: List[str],
    ce_scores: List[float],
    all_chunks: Dict[str, dict],
    top_k: int = 3,
    max_chars: Optional[int] = None,
) -> str:
    """Build a context string from reranked retrieval output for LLM input.

    Args:
        chunk_ids: Ranked chunk IDs from the reranker (all N).
        ce_scores: Corresponding CE scores (all N).
        all_chunks: dict mapping chunk_id → full chunk metadata.
        top_k: Number of top chunks to include (default 3, fallback 5).
        max_chars: Optional character limit for the total context.

    Returns:
        Formatted context string ready to embed in the prompt.
    """
    n = min(top_k, len(chunk_ids))
    blocks = []
    total_chars = 0

    for i in range(n):
        cid = chunk_ids[i]
        score = ce_scores[i]
        chunk = all_chunks.get(cid)
        if not chunk:
            continue

        block = format_chunk_for_llm(chunk, i + 1, score)

        if max_chars and total_chars + len(block) > max_chars:
            # Include partial block up to the limit
            remaining = max_chars - total_chars
            if remaining > 100:
                blocks.append(block[:remaining] + "\n[... truncated]")
            break

        blocks.append(block)
        total_chars += len(block)

    if not blocks:
        return "No relevant guideline context was retrieved."

    header = f"RETRIEVED GUIDELINE CONTEXT ({n} chunks):\n"
    separator = "\n" + "=" * 60 + "\n"
    return header + separator.join(blocks)
