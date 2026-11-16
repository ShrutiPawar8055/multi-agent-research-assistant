"""Planner: choose where to look (route) and split the question into sub-questions."""

from __future__ import annotations

import json
import re

from ..llm import LLM
from ..models import BOTH, CORPUS, ROUTES, WEB, Plan

SYSTEM = """ROLE:planner
You plan research. Given a question, reply with JSON only:
{"route": "CORPUS" | "WEB" | "BOTH", "subquestions": ["..."]}
CORPUS = stable concepts covered by the local document collection.
WEB = recent events or anything time-sensitive.
BOTH = needs foundations and current information.
Give 1 to {max_sub} self-contained sub-questions."""

_FRESH = re.compile(
    r"\b(latest|recent|recently|current|currently|today|news|this (?:week|month|year)|"
    r"newest|just released|20(?:2[4-9]|3\d))\b",
    re.IGNORECASE,
)
_CLAUSE_SPLIT = re.compile(r"\?\s+|;\s+")
_AND_SPLIT = re.compile(r"\s+and\s+(?=(?:how|why|what|when|which|who)\b)", re.IGNORECASE)


def heuristic_plan(question: str, max_sub: int = 3) -> Plan:
    route = WEB if _FRESH.search(question) else CORPUS
    parts: list[str] = []
    for clause in _CLAUSE_SPLIT.split(question.strip()):
        parts.extend(_AND_SPLIT.split(clause))
    subs = [p.strip(" ?;") + "?" for p in parts if len(p.split()) >= 3]
    return Plan(route=route, subquestions=(subs or [question.strip()])[:max_sub])


def _parse(raw: str, max_sub: int) -> Plan | None:
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    route = str(data.get("route", "")).upper()
    subs = [str(s).strip() for s in data.get("subquestions", []) if str(s).strip()]
    if route not in ROUTES or not subs:
        return None
    return Plan(route=route, subquestions=subs[:max_sub])


class Planner:
    def __init__(self, llm: LLM | None = None, max_subquestions: int = 3) -> None:
        self.llm = llm
        self.max_sub = max_subquestions

    def plan(self, question: str) -> Plan:
        if self.llm is not None:
            try:
                raw = self.llm.complete(SYSTEM.replace("{max_sub}", str(self.max_sub)), question)
            except Exception:  # model/network failure must not kill the request
                raw = ""
            parsed = _parse(raw, self.max_sub)
            if parsed:
                return parsed
        return heuristic_plan(question, self.max_sub)


__all__ = ["Planner", "heuristic_plan", "BOTH"]
