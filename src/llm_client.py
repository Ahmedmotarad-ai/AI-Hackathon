"""
LLM Client — OpenAI / Anthropic wrapper with retry and timeout.

Reads API keys from environment variables. Supports both OpenAI and
Anthropic backends with a unified interface.
"""
import os
import time
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class LLMResponse:
    """Unified LLM response."""
    text: str
    model: str
    usage: dict          # {"prompt_tokens": int, "completion_tokens": int}
    latency_ms: float
    provider: str        # "openai" | "anthropic"
    error: Optional[str] = None


def call_openai(
    messages: List[dict],
    model: str = "gpt-4o-mini",
    api_key: Optional[str] = None,
    max_tokens: int = 1024,
    temperature: float = 0.0,
    timeout: float = 30.0,
    max_retries: int = 2,
    retry_delay: float = 1.0,
) -> LLMResponse:
    """Call the OpenAI Chat Completions API.

    Args:
        messages: OpenAI-format message list.
        model: Model name.
        api_key: OpenAI API key. Falls back to OPENAI_API_KEY env var.
        max_tokens: Maximum completion tokens.
        temperature: Sampling temperature.
        timeout: Request timeout in seconds.
        max_retries: Number of retries on failure.
        retry_delay: Base delay between retries (doubles each retry).

    Returns:
        LLMResponse with the completion text or error.
    """
    try:
        from openai import OpenAI
    except ImportError:
        return LLMResponse(
            text="", model=model, usage={}, latency_ms=0,
            provider="openai", error="openai package not installed. Run: pip install openai",
        )

    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        return LLMResponse(
            text="", model=model, usage={}, latency_ms=0,
            provider="openai", error="No OPENAI_API_KEY found. Set it in environment or .env file.",
        )

    client = OpenAI(api_key=key, timeout=timeout)

    last_error = None
    for attempt in range(max_retries + 1):
        t0 = time.perf_counter()
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            latency = (time.perf_counter() - t0) * 1000

            choice = resp.choices[0]
            usage = {
                "prompt_tokens": getattr(resp.usage, "prompt_tokens", 0),
                "completion_tokens": getattr(resp.usage, "completion_tokens", 0),
            }
            return LLMResponse(
                text=choice.message.content or "",
                model=model,
                usage=usage,
                latency_ms=round(latency, 2),
                provider="openai",
            )
        except Exception as exc:
            last_error = str(exc)
            if attempt < max_retries:
                time.sleep(retry_delay * (2 ** attempt))

    return LLMResponse(
        text="", model=model, usage={}, latency_ms=0,
        provider="openai", error=f"Failed after {max_retries + 1} attempts: {last_error}",
    )


def call_anthropic(
    messages: List[dict],
    model: str = "claude-sonnet-4-20250514",
    api_key: Optional[str] = None,
    max_tokens: int = 1024,
    temperature: float = 0.0,
    timeout: float = 30.0,
    max_retries: int = 2,
    retry_delay: float = 1.0,
) -> LLMResponse:
    """Call the Anthropic Messages API.

    Note: Anthropic uses a separate 'system' parameter, not a system message
    in the messages list. This function extracts the system message if present.

    Args:
        messages: OpenAI-format message list (system message extracted automatically).
        model: Model name.
        api_key: Anthropic API key. Falls back to ANTHROPIC_API_KEY env var.
        max_tokens: Maximum completion tokens.
        temperature: Sampling temperature.
        timeout: Request timeout in seconds.
        max_retries: Number of retries on failure.
        retry_delay: Base delay between retries.

    Returns:
        LLMResponse with the completion text or error.
    """
    try:
        import anthropic
    except ImportError:
        return LLMResponse(
            text="", model=model, usage={}, latency_ms=0,
            provider="anthropic", error="anthropic package not installed. Run: pip install anthropic",
        )

    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return LLMResponse(
            text="", model=model, usage={}, latency_ms=0,
            provider="anthropic", error="No ANTHROPIC_API_KEY found. Set it in environment or .env file.",
        )

    client = anthropic.Anthropic(api_key=key, timeout=timeout)

    # Extract system message for Anthropic's separate system parameter
    system_text = None
    user_messages = []
    for msg in messages:
        if msg["role"] == "system":
            system_text = msg["content"]
        else:
            user_messages.append(msg)

    last_error = None
    for attempt in range(max_retries + 1):
        t0 = time.perf_counter()
        try:
            kwargs = {
                "model": model,
                "messages": user_messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
            if system_text:
                kwargs["system"] = system_text

            resp = client.messages.create(**kwargs)
            latency = (time.perf_counter() - t0) * 1000

            text = ""
            for block in resp.content:
                if hasattr(block, "text"):
                    text += block.text

            usage = {
                "prompt_tokens": getattr(resp.usage, "input_tokens", 0),
                "completion_tokens": getattr(resp.usage, "output_tokens", 0),
            }
            return LLMResponse(
                text=text,
                model=model,
                usage=usage,
                latency_ms=round(latency, 2),
                provider="anthropic",
            )
        except Exception as exc:
            last_error = str(exc)
            if attempt < max_retries:
                time.sleep(retry_delay * (2 ** attempt))

    return LLMResponse(
        text="", model=model, usage={}, latency_ms=0,
        provider="anthropic", error=f"Failed after {max_retries + 1} attempts: {last_error}",
    )


def call_llm(
    messages: List[dict],
    provider: str = "openai",
    model: Optional[str] = None,
    **kwargs,
) -> LLMResponse:
    """Unified LLM call dispatcher.

    Args:
        messages: OpenAI-format message list.
        provider: "openai" or "anthropic".
        model: Model name override. Defaults to provider-specific default.
        **kwargs: Additional arguments passed to the provider function.

    Returns:
        LLMResponse.
    """
    if provider == "anthropic":
        default_model = "claude-sonnet-4-20250514"
        return call_anthropic(messages, model=model or default_model, **kwargs)
    else:
        default_model = "gpt-4o-mini"
        return call_openai(messages, model=model or default_model, **kwargs)
