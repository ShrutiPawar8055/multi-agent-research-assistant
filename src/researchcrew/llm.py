"""LLM interface. The pipeline only needs `complete(system, prompt) -> str`.

Every agent also works with `llm=None` (deterministic fallbacks), so the whole
project can be run and tested without any API key.
"""

from __future__ import annotations

from typing import Protocol


class LLM(Protocol):
    def complete(self, system: str, prompt: str) -> str: ...


class GeminiLLM:
    """Gemini through the `google-genai` SDK (Gemini API key or Vertex AI)."""

    def __init__(self, model: str = "gemini-2.5-flash", temperature: float = 0.2) -> None:
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise RuntimeError("Install the Gemini extra: pip install 'researchcrew[gemini]'") from exc
        self._types = types
        self._client = genai.Client()  # reads GOOGLE_API_KEY or Vertex env vars
        self._model = model
        self._temperature = temperature

    def complete(self, system: str, prompt: str) -> str:  # pragma: no cover - network call
        response = self._client.models.generate_content(
            model=self._model,
            contents=prompt,
            config=self._types.GenerateContentConfig(
                system_instruction=system, temperature=self._temperature
            ),
        )
        return (response.text or "").strip()
