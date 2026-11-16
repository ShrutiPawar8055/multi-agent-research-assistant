"""Wire a ResearchCrew from settings: pick the LLM and web provider that are available."""

from __future__ import annotations

from pathlib import Path

from .config import Settings
from .llm import LLM, GeminiLLM
from .pipeline import ResearchCrew
from .retrieval import KnowledgeBase
from .tools import TavilySearch


def build_crew(
    settings: Settings | None = None,
    docs: Path | str | None = None,
    llm_mode: str = "auto",  # "auto" | "gemini" | "none"
) -> ResearchCrew:
    settings = settings or Settings.from_env()
    kb = KnowledgeBase.from_directory(
        docs or settings.docs_dir, settings.chunk_words, settings.chunk_overlap
    )
    llm: LLM | None = None
    if llm_mode == "gemini" or (llm_mode == "auto" and settings.has_gemini_credentials):
        llm = GeminiLLM(settings.model)
    web = TavilySearch(settings.tavily_api_key) if settings.tavily_api_key else None
    return ResearchCrew(kb, web=web, llm=llm, settings=settings)
