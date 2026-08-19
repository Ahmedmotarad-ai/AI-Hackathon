"""
Prompt Assembler — Build OpenAI-format messages for the LLM.

Loads the system prompt and assembles system + user messages
with the retrieved context embedded.
"""
from pathlib import Path
from typing import List, Optional


SYSTEM_PROMPT_PATH = Path(__file__).resolve().parent.parent / "docs" / "grounding" / "system prompt.txt"


def load_system_prompt(path: Optional[Path] = None) -> str:
    """Load the system prompt from the grounding docs.

    Args:
        path: Optional override path. Defaults to docs/grounding/system prompt.txt.

    Returns:
        System prompt string.
    """
    p = path or SYSTEM_PROMPT_PATH
    with open(p, "r", encoding="utf-8") as f:
        return f.read().strip()


def build_messages(
    query: str,
    context: str,
    system_prompt: Optional[str] = None,
    prompt_path: Optional[Path] = None,
    caution_prefix: Optional[str] = None,
) -> List[dict]:
    """Assemble OpenAI-format messages for the LLM.

    Args:
        query: The user's clinical question.
        context: The formatted context string from context_builder.
        system_prompt: Optional pre-loaded system prompt string.
        prompt_path: Optional path to system prompt file.
        caution_prefix: Optional caution prefix to prepend to the user message.

    Returns:
        List of dicts with 'role' and 'content' keys (OpenAI chat format).
    """
    if system_prompt is None:
        system_prompt = load_system_prompt(prompt_path)

    # Build user message
    user_content = f"RETRIEVED CONTEXT:\n\n{context}\n\nUSER QUESTION:\n{query}"

    if caution_prefix:
        user_content = caution_prefix + "\n" + user_content

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
