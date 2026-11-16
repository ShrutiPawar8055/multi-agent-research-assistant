"""Runtime settings, read from environment variables (see .env.example)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    model: str = "gemini-2.5-flash"
    docs_dir: Path = Path("sample_docs")
    top_k: int = 3
    max_subquestions: int = 3
    max_revisions: int = 2
    max_evidence: int = 8
    chunk_words: int = 80
    chunk_overlap: int = 20
    tavily_api_key: str | None = None
    has_gemini_credentials: bool = False

    @classmethod
    def from_env(cls) -> "Settings":
        vertex = os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "").lower() in {"1", "true", "yes"}
        return cls(
            model=os.getenv("CREW_MODEL", cls.model),
            docs_dir=Path(os.getenv("CREW_DOCS_DIR", "sample_docs")),
            top_k=_int("CREW_TOP_K", cls.top_k),
            max_subquestions=_int("CREW_MAX_SUBQUESTIONS", cls.max_subquestions),
            max_revisions=_int("CREW_MAX_REVISIONS", cls.max_revisions),
            tavily_api_key=os.getenv("TAVILY_API_KEY") or None,
            has_gemini_credentials=bool(os.getenv("GOOGLE_API_KEY")) or vertex,
        )
