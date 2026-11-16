"""Researcher: gather evidence for every sub-question, in parallel."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from ..models import BOTH, CORPUS, WEB, Evidence
from ..retrieval import KnowledgeBase
from ..tools import WebSearch


@dataclass
class ResearchOutput:
    evidence: list[Evidence] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


class Researcher:
    def __init__(self, kb: KnowledgeBase, web: WebSearch | None = None, top_k: int = 3) -> None:
        self.kb, self.web, self.top_k = kb, web, top_k

    def _lookup(self, job: tuple[str, str]) -> tuple[list[Evidence], str | None]:
        kind, query = job
        try:
            if kind == "corpus":
                return self.kb.search(query, self.top_k), None
            assert self.web is not None
            return self.web.search(query, self.top_k), None
        except Exception as exc:  # one failing source must not sink the others
            return [], f"{kind} search failed for {query!r}: {exc}"

    def research(self, subquestions: list[str], route: str) -> ResearchOutput:
        jobs: list[tuple[str, str]] = []
        for q in subquestions:
            if route in (CORPUS, BOTH):
                jobs.append(("corpus", q))
            if route in (WEB, BOTH) and self.web is not None:
                jobs.append(("web", q))
        out = ResearchOutput()
        if not jobs:
            return out
        with ThreadPoolExecutor(max_workers=min(8, len(jobs))) as pool:
            for found, error in pool.map(self._lookup, jobs):  # map keeps job order
                out.evidence.extend(found)
                if error:
                    out.errors.append(error)
        return out
