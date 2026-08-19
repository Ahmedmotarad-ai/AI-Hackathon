from typing import Dict, Any

def generate_answer_with_citation(user_question: str, retrieved_chunks: list, llm_client) -> str:
    """
    Generates a structured answer with strict citation based ONLY on retrieved evidence.
    """
    # 1. Verification: If no chunks retrieved, return fallback immediately
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

    # 4. LLM Generation
    response = llm_client.generate(
        system_prompt=SYSTEM_PROMPT,
        prompt=user_payload,
        temperature=0.0 # Deterministic and factual
    )

    return response