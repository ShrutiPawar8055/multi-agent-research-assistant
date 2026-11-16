"""Small text utilities shared by retrieval, the writer and the critic."""

from __future__ import annotations

import re

_WORD = re.compile(r"[a-z0-9]+")
# A sentence boundary is whitespace after . ! ? (unless a citation like [2] follows,
# in which case the citation belongs to the sentence) or after a citation that
# is followed by a capitalised word.
_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+(?!\[\d)|(?<=\])\s+(?=[A-Z])")

CITATION = re.compile(r"\[(\d+)\]")

STOPWORDS = frozenset(
    """a an and are as at be been by for from how in into is it its of on or that the
    their this to was were what when which who why with does do did can could should
    would about between than then these those there here also not but if so such s"""
    .split()
)


def tokenize(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def stem(word: str) -> str:
    """Very light stemmer: enough to match plurals, not a linguistic stemmer."""
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def content_words(text: str) -> list[str]:
    """Lower-cased, stop-word-free, lightly stemmed tokens."""
    return [stem(t) for t in tokenize(text) if t not in STOPWORDS and len(t) > 1]


def split_sentences(text: str) -> list[str]:
    text = " ".join(text.split())
    if not text:
        return []
    return [s for s in _SENTENCE_BREAK.split(text) if s]


def strip_citations(text: str) -> str:
    return " ".join(CITATION.sub(" ", text).split())
