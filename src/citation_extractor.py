"""
Citation Extractor — Parse and validate LLM output.

Extracts structured Answer / Evidence / Source fields from the LLM
response and validates cited chunk IDs against retrieved chunks.
"""
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class Citation:
    """A single cited source."""
    chunk_id: str
    document: str = ""
    section: str = ""
    page: str = ""
    is_valid: bool = True   # True if chunk_id exists in retrieved set


@dataclass
class ParsedResponse:
    """Structured parse of the LLM output."""
    raw_text: str
    answer: str = ""
    evidence: str = ""
    sources: List[Citation] = field(default_factory=list)
    cited_chunk_ids: List[str] = field(default_factory=list)
    invalid_citations: List[str] = field(default_factory=list)
    parse_success: bool = False


def extract_section(text: str, heading: str) -> str:
    """Extract text under a markdown-style heading (## or **Heading:**).

    Args:
        text: Full LLM response text.
        heading: Section name to look for (case-insensitive).

    Returns:
        Extracted section text, or empty string if not found.
    """
    patterns = [
        rf"##\s*{re.escape(heading)}\s*\n(.*?)(?=\n##|\Z)",
        rf"\*\*{re.escape(heading)}[:\*]*\s*\n?(.*?)(?=\n\*\*|\n##|\Z)",
        rf"{re.escape(heading)}:\s*\n?(.*?)(?=\n\w|\Z)",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE | re.DOTALL)
        if m:
            return m.group(1).strip()
    return ""


def extract_citations(text: str) -> List[str]:
    """Extract chunk_id references from the LLM response.

    Looks for patterns like:
    - nice_hf_2018_chunk_0042
    - esc_hf_2021_chunk_0001
    - chunk_id: nice_hf_2018_chunk_0042

    Returns:
        List of unique chunk_id strings found.
    """
    # Match typical chunk_id patterns
    pattern = r"(?:chunk_id[:\s]*)?((?:nice|esc)_(?:hf)_\d{4}_chunk_\d{4})"
    matches = re.findall(pattern, text, re.IGNORECASE)
    # Also match generic chunk IDs like evt-0001 or any chunk_XXXX pattern
    generic = re.findall(r"(chunk_\d{4})", text, re.IGNORECASE)
    all_ids = list(dict.fromkeys(matches + generic))  # dedupe, preserve order
    return all_ids


def parse_llm_response(
    raw_text: str,
    retrieved_chunk_ids: Optional[List[str]] = None,
    chunk_metadata: Optional[Dict[str, dict]] = None,
) -> ParsedResponse:
    """Parse the LLM response into structured fields and validate citations.

    Args:
        raw_text: Raw LLM output text.
        retrieved_chunk_ids: List of chunk IDs that were sent to the LLM.
        chunk_metadata: dict mapping chunk_id → metadata (for validation).

    Returns:
        ParsedResponse with answer, evidence, sources, and validation info.
    """
    answer = extract_section(raw_text, "Answer")
    evidence = extract_section(raw_text, "Evidence")
    sources_text = extract_section(raw_text, "Sources")

    # If structured parsing failed, use full text as answer
    if not answer:
        answer = raw_text.strip()

    # Extract cited chunk IDs
    cited_ids = extract_citations(raw_text)
    retrieved_set = set(retrieved_chunk_ids or [])

    # Build citation objects
    sources = []
    invalid = []
    for cid in cited_ids:
        meta = (chunk_metadata or {}).get(cid, {})
        is_valid = cid in retrieved_set if retrieved_set else True
        if not is_valid:
            invalid.append(cid)
        sources.append(Citation(
            chunk_id=cid,
            document=meta.get("document", ""),
            section=meta.get("section", ""),
            page=str(meta.get("page_start", "")),
            is_valid=is_valid,
        ))

    parse_success = bool(answer and answer != raw_text.strip())

    return ParsedResponse(
        raw_text=raw_text,
        answer=answer,
        evidence=evidence,
        sources=sources,
        cited_chunk_ids=cited_ids,
        invalid_citations=invalid,
        parse_success=parse_success,
    )
