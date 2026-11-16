"""Split documents into overlapping word windows."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    source: str
    position: int
    text: str


def chunk_text(text: str, source: str, size: int = 80, overlap: int = 20) -> list[Chunk]:
    if size <= 0:
        raise ValueError("size must be positive")
    if not 0 <= overlap < size:
        raise ValueError("overlap must satisfy 0 <= overlap < size")
    words = text.split()
    if not words:
        return []
    step = size - overlap
    chunks: list[Chunk] = []
    for position, start in enumerate(range(0, len(words), step)):
        window = words[start : start + size]
        chunks.append(Chunk(source=source, position=position, text=" ".join(window)))
        if start + size >= len(words):
            break
    return chunks
