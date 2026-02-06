from __future__ import annotations

import json
import logging
from typing import Any

import anthropic

logger = logging.getLogger(__name__)


class LLMClient:
    """Wrapper around Anthropic API with retry and JSON parsing.

    Provides a clean interface for the two LLM call patterns used by MemAlign:
    1. Extraction calls (Haiku) - cheap, for principle extraction and deduplication
    2. Judgment calls (Sonnet) - quality, for actual evaluation
    """

    def __init__(self, api_key: str) -> None:
        self._client = anthropic.Anthropic(api_key=api_key)

    async def call(
        self,
        system: str,
        user: str,
        model: str,
        max_tokens: int = 2048,
        temperature: float = 0.0,
    ) -> str:
        """Make an LLM call and return the text response.

        Args:
            system: System prompt.
            user: User prompt.
            model: Model ID (e.g., claude-haiku-4-5-20251001).
            max_tokens: Maximum tokens in response.
            temperature: Sampling temperature.

        Returns:
            Text response from the model.

        Raises:
            anthropic.APIError: On API failures after retries.
        """
        logger.debug("LLM call to %s (system: %d chars, user: %d chars)", model, len(system), len(user))

        response = self._client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
        )

        text = response.content[0].text
        logger.debug("LLM response: %d chars", len(text))
        return text

    async def call_json(
        self,
        system: str,
        user: str,
        model: str,
        max_tokens: int = 2048,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        """Make an LLM call and parse the response as JSON.

        Handles common issues like markdown code fences around JSON.

        Args:
            system: System prompt.
            user: User prompt.
            model: Model ID.
            max_tokens: Maximum tokens in response.
            temperature: Sampling temperature.

        Returns:
            Parsed JSON response as a dictionary.

        Raises:
            ValueError: If response cannot be parsed as JSON.
            anthropic.APIError: On API failures.
        """
        text = await self.call(
            system=system,
            user=user,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        return parse_json_response(text)


def parse_json_response(text: str) -> dict[str, Any]:
    """Parse a JSON response, handling common formatting issues.

    Handles:
    - Raw JSON
    - JSON wrapped in ```json ... ``` code fences
    - JSON wrapped in ``` ... ``` code fences
    - Leading/trailing whitespace

    Args:
        text: Raw text response from the LLM.

    Returns:
        Parsed dictionary.

    Raises:
        ValueError: If the text cannot be parsed as JSON.
    """
    cleaned = text.strip()

    # Remove markdown code fences
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        # Remove first line (```json or ```)
        lines = lines[1:]
        # Remove last line (```)
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()

    try:
        result = json.loads(cleaned)
        if not isinstance(result, dict):
            raise ValueError(f"Expected JSON object, got {type(result).__name__}")
        return result
    except json.JSONDecodeError as e:
        # Try to find JSON object in the text
        start = cleaned.find("{")
        end = cleaned.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                result = json.loads(cleaned[start:end])
                if isinstance(result, dict):
                    return result
            except json.JSONDecodeError:
                pass
        raise ValueError(
            f"Could not parse LLM response as JSON: {e}\n"
            f"Response text: {text[:500]}"
        ) from e
