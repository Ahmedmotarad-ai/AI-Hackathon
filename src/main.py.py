from typing import List, Dict, Any
from prompt_templates import SYSTEM_PROMPT

class MockChunk:
    """Mock class representing a retrieved document chunk with metadata."""
    def __init__(self, page_content: str, source: str, page: int):
        self.page_content = page_content
        self.metadata = {"source": source, "page": page}

def generate_answer_with_citation(user_question: str, retrieved_chunks: List[Any], llm_client=None) -> str:
    """
    Generates a structured answer with strict citation based ONLY on retrieved evidence.
    """
    # 1. Verification: Return safe fallback if no chunks are available
    if not retrieved_chunks:
        return "I don't have enough evidence from the provided guidelines to answer this question safely."

    # 2. Format Retrieved Evidence Context
    context_str = ""
    for idx, chunk in enumerate(retrieved_chunks, 1):
        context_str += f"\n--- Evidence Chunk {idx} ---\n"
        context_str += f"Document: {chunk.metadata.get('source', 'Official Guideline')}\n"
        context_str += f"Page: {chunk.metadata.get('page', 'Unknown')}\n"
        context_str += f"Content: {chunk.page_content}\n"

    # 3. Construct User Payload
    user_payload = f"""
USER QUESTION: {user_question}

RETRIEVED EVIDENCE:
{context_str}

Remember: Follow safety instructions strictly. No outside knowledge, no diagnosis, no unsupported recommendations.
"""

    # 4. LLM Call (Replace with real client integration as needed)
    if llm_client:
        return llm_client.generate(system_prompt=SYSTEM_PROMPT, prompt=user_payload, temperature=0.0)
    
    return f"System Prompt & Payload formatted successfully.\n\nPayload:\n{user_payload}"

if __name__ == "__main__":
    # Example execution
    sample_chunk = MockChunk(
        page_content="In patients with HFrEF, ACEi/ARNI, Beta-blockers, MRA, and SGLT2 inhibitors are recommended to reduce hospitalization and mortality.",
        source="ESC_HF_2021_Guideline.pdf",
        page=14
    )
    
    question = "What is the recommended first-line treatment for Heart Failure with reduced Ejection Fraction (HFrEF)?"
    result = generate_answer_with_citation(question, [sample_chunk])
    print(result)