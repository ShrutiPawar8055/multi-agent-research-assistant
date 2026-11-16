"""Plain data types passed between the stages of the pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

CORPUS = "CORPUS"
WEB = "WEB"
BOTH = "BOTH"
ROUTES = (CORPUS, WEB, BOTH)

NO_EVIDENCE = "Insufficient evidence to answer this question from the available sources."


@dataclass
class Evidence:
    """One retrieved passage. `id` is assigned once, globally, before writing."""

    origin: str  # "corpus" or "web"
    source: str  # file name or URL
    text: str
    score: float = 0.0
    id: int = 0


@dataclass
class Plan:
    route: str
    subquestions: list[str]


@dataclass
class Review:
    verdict: str  # "PASS" or "FAIL"
    issues: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.verdict == "PASS"


@dataclass
class Result:
    question: str
    route: str  # route the planner chose
    effective_route: str  # route actually used after availability checks
    subquestions: list[str]
    answer: str
    verdict: str
    revisions: int
    evidence: list[Evidence]
    issues: list[str]
    notes: list[str]
    trace: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
