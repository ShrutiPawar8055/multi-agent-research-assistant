"""A small, dependency-free BM25 ranker."""

from __future__ import annotations

import math
from collections import Counter

from ..text import content_words
from .chunking import Chunk


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self._chunks: list[Chunk] = []
        self._tf: list[Counter[str]] = []
        self._len: list[int] = []
        self._df: Counter[str] = Counter()
        self._avg_len = 0.0

    def __len__(self) -> int:
        return len(self._chunks)

    def build(self, chunks: list[Chunk]) -> None:
        self._chunks = list(chunks)
        self._tf = [Counter(content_words(c.text)) for c in self._chunks]
        self._len = [sum(tf.values()) for tf in self._tf]
        self._df = Counter(term for tf in self._tf for term in tf)
        self._avg_len = sum(self._len) / len(self._len) if self._len else 0.0

    def _idf(self, term: str) -> float:
        n, df = len(self._chunks), self._df.get(term, 0)
        return math.log(1 + (n - df + 0.5) / (df + 0.5))

    def search(self, query: str, k: int = 3) -> list[tuple[Chunk, float]]:
        terms = set(content_words(query))
        if not terms or not self._chunks:
            return []
        scored: list[tuple[int, float]] = []
        for i, tf in enumerate(self._tf):
            norm = 1 - self.b + self.b * (self._len[i] / self._avg_len if self._avg_len else 0)
            score = 0.0
            for term in terms:
                freq = tf.get(term, 0)
                if freq:
                    score += self._idf(term) * freq * (self.k1 + 1) / (freq + self.k1 * norm)
            if score > 0:
                scored.append((i, score))
        scored.sort(key=lambda pair: (-pair[1], pair[0]))
        return [(self._chunks[i], s) for i, s in scored[:k]]
