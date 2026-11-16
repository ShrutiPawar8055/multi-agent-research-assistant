"""ResearchCrew: plan -> research -> write <-> critique (bounded) -> result."""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import replace
from typing import Any, Iterator

from .agents import Critic, Planner, Researcher, Writer
from .config import Settings
from .llm import LLM
from .models import BOTH, CORPUS, NO_EVIDENCE, WEB, Evidence, Result
from .retrieval import KnowledgeBase
from .tools import WebSearch


class Trace:
    """Collects one record per pipeline step with its duration."""

    def __init__(self) -> None:
        self.steps: list[dict[str, Any]] = []

    @contextmanager
    def step(self, name: str) -> Iterator[dict[str, Any]]:
        record: dict[str, Any] = {"step": name}
        start = time.perf_counter()
        try:
            yield record
        finally:
            record["seconds"] = round(time.perf_counter() - start, 4)
            self.steps.append(record)


def number_evidence(items: list[Evidence], limit: int) -> list[Evidence]:
    """Drop duplicates, cap the total, and assign stable 1-based ids."""
    seen: set[tuple[str, str]] = set()
    unique: list[Evidence] = []
    for item in items:
        key = (item.source, " ".join(item.text.split())[:120])
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return [replace(item, id=i) for i, item in enumerate(unique[:limit], start=1)]


class ResearchCrew:
    def __init__(
        self,
        kb: KnowledgeBase,
        web: WebSearch | None = None,
        llm: LLM | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or Settings()
        self.kb, self.web, self.llm = kb, web, llm
        s = self.settings
        self.planner = Planner(llm, s.max_subquestions)
        self.researcher = Researcher(kb, web, s.top_k)
        self.writer = Writer(llm)
        self.critic = Critic(llm)

    def _effective_route(self, route: str, notes: list[str]) -> str:
        if route in (WEB, BOTH) and self.web is None:
            notes.append(f"Planner chose {route} but no web search is configured; used the local corpus.")
            route = CORPUS
        if route == CORPUS and len(self.kb) == 0 and self.web is not None:
            notes.append("Local corpus is empty; used web search instead.")
            route = WEB
        return route

    def ask(self, question: str) -> Result:
        question = question.strip()
        if not question:
            raise ValueError("question must not be blank")
        trace, notes = Trace(), []

        with trace.step("plan") as rec:
            plan = self.planner.plan(question)
            rec.update(route=plan.route, subquestions=len(plan.subquestions))
        route = self._effective_route(plan.route, notes)

        with trace.step("research") as rec:
            output = self.researcher.research(plan.subquestions, route)
            evidence = number_evidence(output.evidence, self.settings.max_evidence)
            notes.extend(output.errors)
            rec.update(route=route, evidence=len(evidence))

        answer, issues, revisions, verdict = NO_EVIDENCE, [], 0, "PASS"
        feedback: list[str] | None = None
        for attempt in range(self.settings.max_revisions + 1):
            with trace.step(f"write#{attempt + 1}"):
                answer = self.writer.write(question, evidence, feedback)
            with trace.step(f"review#{attempt + 1}") as rec:
                review = self.critic.review(question, answer, evidence)
                rec.update(verdict=review.verdict, issues=len(review.issues))
            verdict, issues, revisions = review.verdict, review.issues, attempt
            if review.passed:
                break
            feedback = review.issues

        return Result(
            question=question,
            route=plan.route,
            effective_route=route,
            subquestions=plan.subquestions,
            answer=answer,
            verdict=verdict,
            revisions=revisions,
            evidence=evidence,
            issues=issues,
            notes=notes,
            trace=trace.steps,
        )
