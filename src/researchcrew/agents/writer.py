"""Writer: turn numbered evidence into a cited answer."""

from __future__ import annotations

from ..llm import LLM
from ..models import NO_EVIDENCE, Evidence
from ..text import content_words, split_sentences

SYSTEM = """ROLE:writer
Answer the question using ONLY the numbered evidence. After every claim put its
evidence number in square brackets before the full stop, like: "Claim text [2]."
Never cite a number that is not in the list. If the evidence does not answer the
question, say so plainly instead of guessing. Be concise (under 150 words)."""


def format_evidence(evidence: list[Evidence]) -> str:
    return "\n\n".join(f"[{e.id}] ({e.source})\n{e.text}" for e in evidence)


def extractive_answer(question: str, evidence: list[Evidence], max_sentences: int = 4) -> str:
    """Deterministic fallback: quote the sentences that best match the question."""
    wanted = set(content_words(question))
    scored: list[tuple[float, int, int, str, int]] = []
    seen: set[str] = set()
    for rank, ev in enumerate(evidence):
        for pos, sentence in enumerate(split_sentences(ev.text)):
            key = sentence.lower()
            if key in seen or len(sentence.split()) < 4:
                continue
            seen.add(key)
            overlap = len(wanted & set(content_words(sentence)))
            if overlap:
                scored.append((-overlap, rank, pos, sentence, ev.id))
    if not scored:
        return NO_EVIDENCE
    best = sorted(scored)[:max_sentences]
    best.sort(key=lambda t: (t[1], t[2]))  # reading order: by evidence, then position
    return " ".join(f"{s.rstrip('.!? ')} [{ev_id}]." for _, _, _, s, ev_id in best)


class Writer:
    def __init__(self, llm: LLM | None = None) -> None:
        self.llm = llm

    def write(self, question: str, evidence: list[Evidence], feedback: list[str] | None = None) -> str:
        if not evidence:
            return NO_EVIDENCE
        if self.llm is None:
            return extractive_answer(question, evidence)
        prompt = f"Question: {question}\n\nEvidence:\n{format_evidence(evidence)}"
        if feedback:
            prompt += "\n\nYour previous draft was rejected. Fix these problems:\n- " + "\n- ".join(feedback)
        return self.llm.complete(SYSTEM, prompt).strip() or NO_EVIDENCE
