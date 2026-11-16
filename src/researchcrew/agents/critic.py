"""Critic: decide whether an answer is grounded in the evidence it cites.

Two layers:
1. Deterministic checks that always run (valid citations, no uncited claims,
   each cited sentence overlaps lexically with the evidence it cites).
2. An optional LLM judgment, which can only make the verdict stricter.
"""

from __future__ import annotations

from ..llm import LLM
from ..models import NO_EVIDENCE, Evidence, Review
from ..text import CITATION, content_words, split_sentences, strip_citations
from .writer import format_evidence

SYSTEM = """ROLE:critic
You verify groundedness. Check whether every claim in the answer is supported by the
numbered evidence it cites. Reply with exactly "PASS" or "FAIL: <one-line reason>"."""

MIN_CLAIM_WORDS = 6
MIN_SUPPORT = 0.3  # share of a sentence's content words found in its cited evidence


def grounding_issues(answer: str, evidence: list[Evidence]) -> list[str]:
    if answer.strip() == NO_EVIDENCE:
        return []  # an honest abstention is always grounded
    by_id = {e.id: e for e in evidence}
    issues: list[str] = []
    for sentence in split_sentences(answer):
        cited = [int(n) for n in CITATION.findall(sentence)]
        claim = strip_citations(sentence)
        if len(claim.split()) < MIN_CLAIM_WORDS and not cited:
            continue  # short connective text, not a claim
        if not cited:
            issues.append(f"Uncited claim: {claim[:80]!r}")
            continue
        unknown = [n for n in cited if n not in by_id]
        if unknown:
            issues.append(f"Cites missing evidence {unknown}: {claim[:60]!r}")
            continue
        words = set(content_words(claim))
        support = set()
        for n in cited:
            support |= set(content_words(by_id[n].text))
        if words and len(words & support) / len(words) < MIN_SUPPORT:
            issues.append(f"Not supported by cited evidence {cited}: {claim[:80]!r}")
    return issues


class Critic:
    def __init__(self, llm: LLM | None = None) -> None:
        self.llm = llm

    def review(self, question: str, answer: str, evidence: list[Evidence]) -> Review:
        issues = grounding_issues(answer, evidence)
        if self.llm is not None and not issues and answer.strip() != NO_EVIDENCE:
            prompt = f"Question: {question}\n\nAnswer:\n{answer}\n\nEvidence:\n{format_evidence(evidence)}"
            try:
                verdict = self.llm.complete(SYSTEM, prompt).strip()
            except Exception:
                verdict = "PASS"  # deterministic checks already passed; don't fail on outage
            if verdict.upper().startswith("FAIL"):
                issues.append(verdict.partition(":")[2].strip() or "LLM critic rejected the answer")
        return Review("FAIL" if issues else "PASS", issues)
