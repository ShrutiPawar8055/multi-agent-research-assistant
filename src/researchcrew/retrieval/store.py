"""Knowledge base: load documents, chunk them, index them, search them."""

from __future__ import annotations

import json
import re
from pathlib import Path

from ..models import Evidence
from .bm25 import BM25Index
from .chunking import Chunk, chunk_text

TEXT_SUFFIXES = {".md", ".txt"}


def read_document(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise RuntimeError("Install PDF support: pip install 'researchcrew[pdf]'") from exc
        return "\n".join((page.extract_text() or "") for page in PdfReader(str(path)).pages)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".md":  # "# Title" -> "Title." so it stays its own sentence
        text = re.sub(r"^#+\s*(.+?)\s*$", r"\1.", text, flags=re.MULTILINE)
    return text


class KnowledgeBase:
    def __init__(self, chunks: list[Chunk] | None = None) -> None:
        self._index = BM25Index()
        self.chunks: list[Chunk] = []
        if chunks:
            self.set_chunks(chunks)

    def __len__(self) -> int:
        return len(self.chunks)

    def set_chunks(self, chunks: list[Chunk]) -> None:
        self.chunks = list(chunks)
        self._index.build(self.chunks)

    @classmethod
    def from_directory(cls, directory: Path | str, size: int = 80, overlap: int = 20) -> "KnowledgeBase":
        root = Path(directory)
        if not root.is_dir():
            raise FileNotFoundError(f"Documents folder not found: {root}")
        suffixes = TEXT_SUFFIXES | {".pdf"}
        chunks: list[Chunk] = []
        for path in sorted(p for p in root.rglob("*") if p.suffix.lower() in suffixes):
            chunks.extend(chunk_text(read_document(path), path.relative_to(root).as_posix(), size, overlap))
        return cls(chunks)

    def search(self, query: str, k: int = 3) -> list[Evidence]:
        return [
            Evidence(origin="corpus", source=c.source, text=c.text, score=round(score, 4))
            for c, score in self._index.search(query, k)
        ]

    def save(self, path: Path | str) -> None:
        payload = [{"source": c.source, "position": c.position, "text": c.text} for c in self.chunks]
        Path(path).write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(cls, path: Path | str) -> "KnowledgeBase":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls([Chunk(**item) for item in payload])
